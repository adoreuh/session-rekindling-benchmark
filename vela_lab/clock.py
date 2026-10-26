"""The single virtual clock.

Contract §2: *only* explicit time events move the clock, and every time-aware
component reads this one object.  Nothing else in the harness may call
``datetime.now()`` for experiment semantics.

Wall-clock time exists, but only for run logs.  It is named ``wall_time`` and
is never reachable from model-visible content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .errors import ValidationError

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")
_ZONE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+/-]*$")


@dataclass(frozen=True)
class Clock:
    """A timezone-aware instant plus its canonical zone name.

    Immutable: advancing produces a new Clock.  Monotonicity is enforced by the
    reducer, not here — but ``advance`` refuses negative deltas so a caller
    cannot even express a rewind.
    """

    instant: str
    time_zone: str

    @staticmethod
    def of(instant: str, time_zone: str) -> "Clock":
        return Clock(
            instant=normalize_instant(instant),
            time_zone=normalize_zone(time_zone),
        )

    @property
    def dt(self) -> datetime:
        return parse_instant(self.instant)

    def advance(self, *, days: int = 0, hours: int = 0, minutes: int = 0,
                seconds: int = 0) -> "Clock":
        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        if delta < timedelta(0):
            raise ValidationError("clock cannot move backwards")
        return Clock(
            instant=normalize_instant(self.dt + delta),
            time_zone=self.time_zone,
        )

    def until(self, target: "Clock") -> timedelta:
        return target.dt - self.dt

    def to_json(self) -> dict:
        return {"instant": self.instant, "time_zone": self.time_zone}

    @staticmethod
    def from_json(payload: Any, what: str = "clock") -> "Clock":
        if not isinstance(payload, dict):
            raise ValidationError(f"{what}: must be a mapping")
        unknown = set(payload) - {"instant", "time_zone"}
        if unknown:
            raise ValidationError(f"{what}: unknown keys {sorted(unknown)}")
        try:
            instant = payload["instant"]
            zone = payload["time_zone"]
        except KeyError as exc:
            raise ValidationError(f"{what}: missing {exc.args[0]}") from exc
        return Clock.of(instant, zone)


def normalize_instant(value: Any, what: str = "instant") -> str:
    """Parse a tz-aware ISO-8601 instant and normalize it to UTC with 'Z'."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValidationError(f"{what}: naive datetime is ambiguous")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    parsed = parse_instant(value, what)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_instant(value: Any, what: str = "instant") -> datetime:
    """Parse a timezone-aware ISO-8601 string.  Naive input is rejected."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{what}: must be a non-empty ISO string")
    text = value
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"{what}: not a valid ISO-8601 instant: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValidationError(
            f"{what}: must carry a UTC offset or zone; naive timestamps are "
            "ambiguous and are rejected"
        )
    return parsed


def normalize_zone(value: Any, what: str = "time_zone") -> str:
    """Accept an IANA name (``Asia/Shanghai``) or a fixed offset (``+08:00``)."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{what}: must be a non-empty string")
    if _OFFSET_RE.match(value):
        return value
    if _ZONE_RE.match(value):
        return value
    raise ValidationError(f"{what}: not a recognized zone: {value!r}")


def format_human(clock: Clock) -> str:
    """The one model-visible rendering of the current time.

    Contract §4: this string must read like something a real assistant could
    legitimately see.  It deliberately does not say "virtual", "simulated",
    "controlled", or "harness".
    """
    local = clock.dt.astimezone(_fixed_offset(clock.time_zone))
    return local.isoformat(timespec="seconds")


def _fixed_offset(zone: str) -> timezone:
    match = _OFFSET_RE.match(zone)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3))
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(zone)
    except Exception as exc:  # pragma: no cover - depends on system tz database
        raise ValidationError(f"cannot resolve time zone {zone!r}: {exc}") from exc
