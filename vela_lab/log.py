"""Append-only, hash-chained event log with explicit ordinal.

Contract §1: the event stream is the single source of truth.  Anything that
affects the experiment must be an event, and only events may be replayed.

Two mechanism choices, both deliberate:

**Hash chain.** Every event carries ``prev_hash`` and its own ``hash``.  Every
read re-verifies the *entire* chain.  A truncated or edited log raises instead
of returning a plausible-looking prefix.

**Explicit ordinal.** ``seq`` is a monotonic integer, written into the event and
recorded in a resume marker.  It lets a new writer locate the tail in O(1)-ish
without trusting the file length, and it makes "prefix through ordinal N"
expressible — which is what fork boundaries and audit cutoffs are.

Honest limit (contract §6): a party who can rewrite the whole log can also
recompute the whole chain.  Local re-verification therefore proves internal
consistency, *not* that the run happened.  Pin the terminal hash of any run you
intend to publish to storage you do not control.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .errors import IntegrityError, StaleWriterError, ValidationError
from .jsonutil import SCHEMA_VERSION, canonical_json, check_json, json_copy, sha256_hex

EVENT_FIELDS = (
    "schema_version",
    "seq",
    "type",
    "data",
    "wall_time",
    "virtual_time",
    "prev_hash",
    "hash",
)

LOG_FILENAME = "events.jsonl"
LOCK_FILENAME = ".writer.lock"


def wall_now() -> str:
    """Wall-clock stamp for run logs only.  Never reaches model-visible text."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def event_hash(event: dict) -> str:
    """Hash over the event minus its own hash field."""
    body = {key: value for key, value in event.items() if key != "hash"}
    return sha256_hex(canonical_json(body))


def verify_chain(events: list) -> None:
    """Fail-closed verification of a parsed event list.

    Checks field presence, schema version, ordinal contiguity starting at 1,
    hash-chain linkage, and each event's own hash.  Raises ``IntegrityError``
    on the first problem; returns ``None`` when the chain is intact.

    The genesis event must carry ``prev_hash == ""``; every later event must
    name its predecessor's digest.
    """
    previous = ""
    for index, event in enumerate(events):
        where = f"event {index + 1}"
        if not isinstance(event, dict):
            raise IntegrityError(f"{where}: must be a mapping")
        missing = [field for field in EVENT_FIELDS if field not in event]
        if missing:
            raise IntegrityError(f"{where}: missing fields {missing}")
        if event["schema_version"] != SCHEMA_VERSION:
            raise IntegrityError(
                f"{where}: unsupported schema_version {event['schema_version']!r}"
            )
        if event["seq"] != index + 1:
            raise IntegrityError(
                f"{where}: ordinal mismatch (expected {index + 1}, got {event['seq']!r})"
            )
        if not isinstance(event["type"], str) or not event["type"]:
            raise IntegrityError(f"{where}: invalid type")
        try:
            check_json(event["data"], f"{where}.data")
        except ValidationError as exc:
            raise IntegrityError(str(exc)) from exc
        if event["prev_hash"] != previous:
            raise IntegrityError(f"{where}: prev_hash does not link to the previous event")
        if not isinstance(event["hash"], str) or len(event["hash"]) != 64:
            raise IntegrityError(f"{where}: hash must be a 64-hex digest")
        if event_hash(event) != event["hash"]:
            raise IntegrityError(f"{where}: hash does not match its own content")
        previous = event["hash"]


@dataclass(frozen=True)
class LogHead:
    """Where a log currently ends, and the chain head at that point."""

    seq: int
    hash: str
    byte_length: int


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class EventLog:
    """A hash-chained JSONL log rooted at ``root/events.jsonl``.

    Single-writer: an ``O_EXCL`` lock file guards appends, and every append
    re-reads and re-verifies the chain first, so a writer that fell behind
    raises ``StaleWriterError`` rather than forking the chain.
    """

    def __init__(self, root: os.PathLike | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / LOG_FILENAME
        self.lock_path = self.root / LOCK_FILENAME

    # ------------------------------------------------------------- reading

    def read(self) -> list:
        """Read and verify the whole log.  Absent file reads as empty."""
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        events = self._parse(raw)
        verify_chain(events)
        return events

    def head(self) -> LogHead:
        events = self.read()
        length = self.path.stat().st_size if self.path.exists() else 0
        if not events:
            return LogHead(seq=0, hash="", byte_length=0)
        return LogHead(seq=events[-1]["seq"], hash=events[-1]["hash"], byte_length=length)

    def _parse(self, raw: bytes) -> list:
        text = raw.decode("utf-8")
        events = []
        if not text:
            return events
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        elif lines and lines[-1] != "":
            # A trailing fragment means the write was torn mid-line.
            raise IntegrityError(
                "log ends mid-record (truncated write); refusing to guess at "
                "the missing bytes"
            )
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                raise IntegrityError(f"line {number}: blank line inside the log")
            try:
                events.append(json.loads(line))
            except Exception as exc:
                raise IntegrityError(f"line {number}: not valid JSON ({exc})") from exc
        return events

    # ------------------------------------------------------------- writing

    def append(self, event_type: str, data: dict, *, virtual_time: str) -> dict:
        """Append one event, returning the stored record."""
        check_json(data, "event data")
        with self._lock():
            events = self.read()
            head = events[-1]["hash"] if events else ""
            seq = (events[-1]["seq"] if events else 0) + 1
            event = {
                "schema_version": SCHEMA_VERSION,
                "seq": seq,
                "type": event_type,
                "data": json_copy(data, "event data"),
                "wall_time": wall_now(),
                "virtual_time": virtual_time,
                "prev_hash": head,
                "hash": "",
            }
            event["hash"] = event_hash(event)
            line = canonical_json(event) + "\n"
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_dir(self.root)
            return event

    def append_many(self, records: list, *, virtual_time: str) -> list:
        """Append several ``(type, data)`` pairs under one lock acquisition."""
        written = []
        for event_type, data in records:
            written.append(self.append(event_type, data, virtual_time=virtual_time))
        return written

    def _lock(self):
        return _WriteLock(self.lock_path)


class _WriteLock:
    """``O_EXCL`` lock file, refusing a second concurrent writer."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise StaleWriterError(
                f"another writer holds {self.path}; a second concurrent writer "
                "would fork the hash chain"
            ) from exc
        return self

    def __exit__(self, *exc_info):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            try:
                self.path.unlink()
            except OSError:
                pass
        return False


def copy_prefix(source: EventLog, target_root: os.PathLike | str) -> EventLog:
    """Byte-copy an entire log into a new root.

    Used by fork.  The copy is verified independently after writing, so a fork
    never inherits its parent's trust — it re-derives it.
    """
    target = EventLog(target_root)
    if target.path.exists():
        raise StaleWriterError(f"{target.path} already exists; refusing to overwrite")
    events = source.read()
    if not events:
        return target
    payload = "".join(canonical_json(event) + "\n" for event in events)
    with target.path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(target.root)
    verify_chain(target.read())
    return target
