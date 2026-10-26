"""Registered world events and deterministic scheduling.

Contract §3 requires that world facts change through registered events, not
through ad-hoc state writes.  The old harness let a scenario write
``set_state`` directly, which made "the world changed" indistinguishable from
"the operator nudged the world", and left no record of *when* things became
true.

The model here is deliberately small — this is a calendar, not a discrete-event
simulator.  A registered event declares three distinct moments:

* ``effective_at`` — when the fact becomes true in the world;
* ``published_at`` — when evidence of it becomes queryable;
* ``notify_at``    — when (if ever) a notification reaches the user or agent.

Keeping these apart matters.  "The appointment moved, but no notice has arrived
yet" is a legitimate world state, and treating it as a failure would be wrong —
it is precisely the case that produces stale memory.

Scheduling is deterministic: events are ordered by ``(effective_at, order)``
where ``order`` is an explicit tiebreak the scenario must supply for same-tick
events.  Nothing depends on dict iteration or wall time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .clock import Clock, parse_instant
from .errors import ContractViolation, ValidationError

EVENT_KINDS = frozenset(
    {
        "supersede_fact",
        "add_fact",
        "publish_evidence",
        "send_notification",
    }
)


@dataclass(frozen=True)
class WorldEvent:
    """One registered change to the world.

    ``at`` is the trigger tick.  Which of the three moments it drives is
    decided by ``kind``; ``notify_at`` may be ``None`` for events that
    deliberately never reach the conversation.
    """

    event_id: str
    kind: str
    at: str
    order: int = 0
    fact_id: str | None = None
    new_value: Any = None
    source: str | None = None
    evidence_id: str | None = None
    notify_at: str | None = None
    notification_text: str | None = None

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValidationError(f"unknown world event kind {self.kind!r}")
        parse_instant(self.at, f"event {self.event_id}.at")
        if self.notify_at:
            parse_instant(self.notify_at, f"event {self.event_id}.notify_at")
        if self.kind in ("supersede_fact", "add_fact") and not self.fact_id:
            raise ValidationError(f"event {self.event_id}: {self.kind} needs fact_id")
        if self.kind == "publish_evidence" and not self.evidence_id:
            raise ValidationError(f"event {self.event_id}: publish_evidence needs evidence_id")
        if self.kind == "send_notification" and not self.notification_text:
            raise ValidationError(
                f"event {self.event_id}: send_notification needs notification_text"
            )
        if self.kind == "send_notification" and not self.notify_at:
            raise ValidationError(
                f"event {self.event_id}: send_notification needs notify_at"
            )

    def trigger_tick(self) -> int:
        return int(parse_instant(self.at).timestamp())

    def sort_key(self):
        return (self.trigger_tick(), self.order, self.event_id)

    def to_json(self) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "at": self.at,
            "order": self.order,
            "fact_id": self.fact_id,
            "new_value": self.new_value,
            "source": self.source,
            "evidence_id": self.evidence_id,
            "notify_at": self.notify_at,
            "notification_text": self.notification_text,
        }

    @staticmethod
    def from_json(payload: dict) -> "WorldEvent":
        return WorldEvent(**payload)


class WorldSchedule:
    """A frozen table of registered events.

    Sealed after construction: the run cannot add an event mid-flight, which is
    what stops an operator from patching a failing manipulation by inventing a
    notification.  If the registered events do not produce the intended world,
    the correct response is to register a different table and re-run, not to
    intervene.
    """

    def __init__(self, events: Iterable[WorldEvent]):
        self._events = sorted(list(events), key=WorldEvent.sort_key)
        self._sealed = False
        self._due_cache: dict[str, list[WorldEvent]] = {}
        self._validate()

    def _validate(self) -> None:
        seen = set()
        for event in self._events:
            if event.event_id in seen:
                raise ValidationError(f"duplicate world event id {event.event_id!r}")
            seen.add(event.event_id)
        ticks: dict[int, list[str]] = {}
        for event in self._events:
            ticks.setdefault(event.trigger_tick(), []).append(str(event.order))
        for tick, orders in ticks.items():
            if len(orders) != len(set(orders)):
                raise ContractViolation(
                    f"world events at the same instant {tick} share an order "
                    f"value; deterministic ordering requires an explicit, unique "
                    f"tiebreak (got {orders})"
                )

    def seal(self) -> None:
        self._sealed = True

    def register(self, event: WorldEvent) -> None:
        if self._sealed:
            raise ContractViolation(
                "world schedule is sealed; no event may be registered during a run"
            )
        self._events.append(event)
        self._events.sort(key=WorldEvent.sort_key)
        self._due_cache.clear()

    def all_events(self) -> tuple[WorldEvent, ...]:
        return tuple(self._events)

    def due_between(self, start: Clock, end: Clock) -> list[WorldEvent]:
        """Events whose trigger instant falls in ``(start, end]``.

        Half-open on the left so an advance of zero does not re-fire the tick
        the clock is already sitting on.
        """
        start_tick = int(start.dt.timestamp())
        end_tick = int(end.dt.timestamp())
        if end_tick < start_tick:
            raise ContractViolation("schedule queried with a backwards interval")
        return [
            event
            for event in self._events
            if start_tick < event.trigger_tick() <= end_tick
        ]

    def to_json(self) -> dict:
        return {"events": [e.to_json() for e in self._events]}

    def digest(self) -> str:
        from .jsonutil import digest

        return digest(self.to_json())


def advance_to(
    schedule: WorldSchedule,
    start: Clock,
    target: Clock,
) -> list[tuple[WorldEvent, str]]:
    """Walk from ``start`` to ``target``, returning due events in order.

    Returns ``(event, moment)`` pairs where ``moment`` is one of
    ``"effective"`` (the fact changed), ``"published"`` (evidence became
    queryable), or ``"notified"`` (a notification is now deliverable).

    A single event can contribute more than one moment at the same tick; the
    ordering within a tick is effective → published → notified, so a
    notification never precedes the fact it announces.
    """
    due = schedule.due_between(start, target)
    if not due:
        return []
    # Re-order by instant so a notification scheduled later within the same
    # advance still lands after its own fact change.
    moments: list[tuple[int, int, WorldEvent, str]] = []
    for event in due:
        tick = event.trigger_tick()
        if event.kind in ("supersede_fact", "add_fact"):
            moments.append((tick, 0, event, "effective"))
        elif event.kind == "publish_evidence":
            moments.append((tick, 1, event, "published"))
        else:
            notify_tick = int(parse_instant(event.notify_at).timestamp())
            if start.dt.timestamp() < notify_tick <= target.dt.timestamp():
                moments.append((notify_tick, 2, event, "notified"))
    moments.sort(key=lambda item: (item[0], item[1], item[2].event_id))
    return [(event, moment) for _, _, event, moment in moments]
