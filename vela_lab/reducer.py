"""Event vocabulary and the pure fold.

Contract §1: the event stream is the only source of truth.  This module defines
what may enter that stream and how the stream folds back into experiment state.

Two rules that carry most of the weight:

**Only explicit delivery events append to history.**  ``user.message``,
``assistant.message``, ``tool.visible_result`` and ``observation.compiled``
extend the model-visible history.  Every control-plane event
(``clock.advance``, ``world.fact_superseded``, ``evidence.registered``,
``prompt.bound``, ``checkpoint.created``, ``branch.forked``) changes the
experiment but never becomes model-visible text.  That separation is what makes
"the harness moved the clock" unpublishable to the model by construction.

**Tool results arrive in strict order.**  ``tool.call`` →
``tool.raw_result`` → ``tool.visible_result`` → ``assistant.message`` (or the
tool message).  The reducer refuses any other order, so a fabricated tool
result cannot be spliced in.  The raw value is what the world produced; the
visible value is what the model saw; both are kept and cross-checked in audit.

Design note: ``KernelState`` is a plain frozen-ish record of JSON-safe values.
It holds no file handles, no clocks, and no provider objects, so folding the
same events always yields an equal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .clock import Clock, format_human
from .errors import ContractViolation, ValidationError
from .jsonutil import digest, json_copy

# ------------------------------------------------------------------ vocabulary

CONTROL_EVENTS = frozenset(
    {
        "session.started",
        "clock.advance",
        "world.fact_set",
        "world.fact_superseded",
        "world.event_applied",
        "evidence.registered",
        "prompt.bound",
        "turn.context",
        "checkpoint.created",
        "branch.forked",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "memory.retrieval",
        "observation.compiled",
        "observation.retired",
        "session.closed",
        "context.rebased",
        "session.opened",
        "stable_material.registered",
        "session.boundary",
        "model.request",
        "model.response",
        "model.failed",
        "tool.call",
        "tool.raw_result",
        "tool.visible_result",
        "fault.applied",
        # Exposure checking is control-plane by definition: it records what the
        # harness did *about* a request, not something the model said.  Placing
        # these here rather than among the history appenders is the enforcement —
        # a gate verdict can never be projected into the conversation.
        "exposure.checked",
        "exposure.parent_checked",
        "branch.gate_rejected",
    }
)

# Events that extend model-visible history, mapped to the role they carry.
# Note that ``tool.visible_result`` is *not* here: it records the projection on
# the control plane.  The history entry is the separate ``observation.compiled``
# event, so "what the tool returned" and "what entered the conversation" remain
# two distinguishable facts.
HISTORY_APPENDERS = {
    "user.message": "user",
    "assistant.message": "assistant",
    "observation.compiled": None,  # role read from the payload kind
}

# Visible fragment kind -> conversational role.
KIND_TO_ROLE = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "assistant_tool_call": "assistant",
    "tool_result": "tool",
    "memory_note": "user",
    "time_context": "user",
    "stable_material": "user",
}

VALID_EVENT_TYPES = CONTROL_EVENTS | set(HISTORY_APPENDERS)

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STABLE_STATUSES = (STATUS_IDLE,)


@dataclass
class PendingCall:
    call_id: str
    tool_name: str
    arguments: dict


@dataclass
class KernelState:
    """Everything the fold derives.  JSON-safe, comparable, copyable."""

    status: str = STATUS_IDLE
    clock: Clock | None = None
    facts: dict = field(default_factory=dict)
    applied_events: list = field(default_factory=list)
    history: list = field(default_factory=list)          # visible fragment records
    pending_calls: dict = field(default_factory=dict)     # call_id -> PendingCall dict
    raw_results: dict = field(default_factory=dict)       # call_id -> raw payload
    visible_results: dict = field(default_factory=dict)   # call_id -> visible payload
    retrievals: list = field(default_factory=list)
    requests: list = field(default_factory=list)          # model.request records
    responses: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    turn_index: int = 0
    bound_prompt: dict | None = None
    turn_contexts: list = field(default_factory=list)
    window_index: int = 0
    rebases: list = field(default_factory=list)
    stable_materials: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    last_error: str | None = None

    # ---------------------------------------------------------------- helpers

    def is_stable(self) -> bool:
        """A checkpoint boundary: no turn open, no tool call half-done."""
        return (
            self.status in STABLE_STATUSES
            and not self.pending_calls
            and all(call in self.visible_results for call in self.raw_results)
        )

    def message_records(self) -> list:
        return json_copy(self.history, "history")

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "clock": self.clock.to_json() if self.clock else None,
            "facts": self.facts,
            "applied_events": list(self.applied_events),
            "history": self.history,
            "pending_calls": self.pending_calls,
            "raw_results": self.raw_results,
            "visible_results": self.visible_results,
            "retrievals": self.retrievals,
            "requests": self.requests,
            "responses": self.responses,
            "failures": self.failures,
            "turn_index": self.turn_index,
            "bound_prompt": self.bound_prompt,
            "turn_contexts": self.turn_contexts,
            "window_index": self.window_index,
            "rebases": self.rebases,
            "stable_materials": self.stable_materials,
            "boundaries": self.boundaries,
            "last_error": self.last_error,
        }

    @staticmethod
    def from_json(payload: dict) -> "KernelState":
        clock = payload.get("clock")
        return KernelState(
            status=payload.get("status", STATUS_IDLE),
            clock=Clock.from_json(clock) if clock else None,
            facts=json_copy(payload.get("facts", {}), "facts"),
            applied_events=list(payload.get("applied_events", [])),
            history=json_copy(payload.get("history", []), "history"),
            pending_calls=json_copy(payload.get("pending_calls", {}), "pending_calls"),
            raw_results=json_copy(payload.get("raw_results", {}), "raw_results"),
            visible_results=json_copy(payload.get("visible_results", {}), "visible_results"),
            retrievals=json_copy(payload.get("retrievals", []), "retrievals"),
            requests=json_copy(payload.get("requests", []), "requests"),
            responses=json_copy(payload.get("responses", []), "responses"),
            failures=json_copy(payload.get("failures", []), "failures"),
            turn_index=payload.get("turn_index", 0),
            bound_prompt=json_copy(payload.get("bound_prompt"), "bound_prompt")
            if payload.get("bound_prompt")
            else None,
            turn_contexts=json_copy(payload.get("turn_contexts", []), "turn contexts"),
            window_index=payload.get("window_index", 0),
            rebases=json_copy(payload.get("rebases", []), "rebases"),
            stable_materials=json_copy(payload.get("stable_materials", []),
                                       "stable materials"),
            boundaries=json_copy(payload.get("boundaries", []), "boundaries"),
            last_error=payload.get("last_error"),
        )

    def digest(self) -> str:
        return digest(self.to_json())


def initial_state(clock: Clock) -> KernelState:
    return KernelState(status=STATUS_IDLE, clock=clock)


# ------------------------------------------------------------------- reducing


def reduce_event(state: KernelState, event: dict) -> KernelState:
    """Apply one event, returning a new state.

    Pure: does not mutate ``state``.  Raises ``ContractViolation`` when the
    event would break an experiment invariant.
    """
    event_type = event.get("type")
    if event_type not in VALID_EVENT_TYPES:
        raise ContractViolation(f"unknown event type {event_type!r}")
    if state.status == STATUS_FAILED:
        raise ContractViolation(
            f"session already failed ({state.last_error}); no further events may "
            "be applied — a failed run is terminated, not repaired"
        )

    next_state = _clone(state)

    if event_type == "session.started":
        return _on_started(next_state, event)
    if event_type == "clock.advance":
        return _on_clock(next_state, event)
    if event_type == "world.fact_set":
        return _on_fact_set(next_state, event)
    if event_type == "world.fact_superseded":
        return _on_fact_superseded(next_state, event)
    if event_type == "world.event_applied":
        return _on_event_applied(next_state, event)
    if event_type == "prompt.bound":
        return _on_prompt_bound(next_state, event)
    if event_type == "turn.context":
        next_state.turn_contexts.append(json_copy(event["data"], "turn context"))
        return next_state
    if event_type == "session.boundary":
        next_state.boundaries.append(json_copy(event["data"], "boundary"))
        return next_state
    if event_type == "context.rebased":
        return _on_rebased(next_state, event)
    if event_type == "session.closed":
        data = event["data"]
        if next_state.window_index != data.get("window"):
            raise ContractViolation(
                f"session.closed names window {data.get('window')!r} but the "
                f"session is in window {next_state.window_index}"
            )
        return next_state
    if event_type == "session.opened":
        data = event["data"]
        window = data.get("window")
        if not isinstance(window, int) or window != next_state.window_index + 1:
            raise ContractViolation(
                f"session.opened must open window {next_state.window_index + 1}, "
                f"got {window!r}; windows are sequential and may not be skipped"
            )
        next_state.window_index = window
        return next_state
    if event_type == "stable_material.registered":
        next_state.stable_materials.append(json_copy(event["data"], "stable material"))
        return next_state
    if event_type in ("tool.call", "tool.raw_result", "tool.visible_result"):
        return _on_tool(next_state, event, event_type)
    if event_type in HISTORY_APPENDERS:
        return _on_history(next_state, event, event_type)
    if event_type == "turn.started":
        next_state.status = STATUS_RUNNING
        next_state.turn_index += 1
        return next_state
    if event_type == "turn.completed":
        return _on_turn_completed(next_state, event)
    if event_type == "turn.failed":
        next_state.status = STATUS_FAILED
        next_state.last_error = event["data"].get("reason", "unspecified")
        return next_state
    if event_type == "memory.retrieval":
        next_state.retrievals.append(json_copy(event["data"], "retrieval"))
        return next_state
    if event_type == "model.request":
        next_state.requests.append(json_copy(event["data"], "request"))
        return next_state
    if event_type == "model.response":
        next_state.responses.append(json_copy(event["data"], "response"))
        return next_state
    if event_type == "model.failed":
        next_state.failures.append(json_copy(event["data"], "failure"))
        return next_state

    # observation.compiled, faults, checkpoints, forks: recorded, not replayed
    # into history.
    return next_state


def _clone(state: KernelState) -> KernelState:
    return KernelState.from_json(state.to_json())


def _on_started(state: KernelState, event: dict) -> KernelState:
    data = event["data"]
    if state.clock is None:
        raise ContractViolation("session.started requires a clock")
    if data.get("clock") is not None:
        state.clock = Clock.from_json(data["clock"])
    return state


def _on_clock(state: KernelState, event: dict) -> KernelState:
    data = event["data"]
    if state.clock is None:
        raise ContractViolation("clock.advance before session.started")
    if not state.is_stable():
        raise ContractViolation(
            "the clock may only advance at a stable boundary (no open turn, no "
            "pending tool call); advancing mid-turn would make one turn span "
            "two experiment conditions"
        )
    target = Clock.of(data["to"], data.get("time_zone", state.clock.time_zone))
    if target.dt < state.clock.dt:
        raise ContractViolation("clock cannot move backwards")
    state.clock = target
    return state


def _on_fact_set(state: KernelState, event: dict) -> KernelState:
    data = event["data"]
    fact_id = data["fact_id"]
    state.facts[fact_id] = json_copy(
        {**data, "revision": data.get("revision", 0)}, f"fact[{fact_id}]"
    )
    return state


def _on_fact_superseded(state: KernelState, event: dict) -> KernelState:
    data = event["data"]
    fact_id = data["fact_id"]
    existing = state.facts.get(fact_id, {})
    revision = existing.get("revision", 0) + 1
    history = list(existing.get("history", []))
    if existing:
        history.append(
            {
                "revision": existing.get("revision", 0),
                "value": existing.get("history_value", existing.get("value")),
                "source": existing.get("source"),
            }
        )
    state.facts[fact_id] = {
        "fact_id": fact_id,
        "value": data["new_value"],
        "source": data.get("source", "unspecified"),
        "effective_at": data.get("effective_at"),
        "revision": revision,
        "supersedes_revision": existing.get("revision", 0),
        "history": history,
        "history_value": data.get("new_value"),
    }
    return state


def _on_event_applied(state: KernelState, event: dict) -> KernelState:
    event_id = event["data"]["event_id"]
    if event_id in state.applied_events:
        raise ContractViolation(
            f"world event {event_id!r} applied twice; scheduling must be "
            "idempotent-by-refusal, not tolerant"
        )
    state.applied_events.append(event_id)
    return state


def _on_prompt_bound(state: KernelState, event: dict) -> KernelState:
    state.bound_prompt = json_copy(event["data"], "bound prompt")
    return state


def _on_tool(state: KernelState, event: dict, event_type: str) -> KernelState:
    data = event["data"]
    call_id = data.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ContractViolation(f"{event_type}: missing call_id")

    if event_type == "tool.call":
        if call_id in state.pending_calls or call_id in state.visible_results:
            raise ContractViolation(f"duplicate tool call id {call_id!r}")
        state.pending_calls[call_id] = {
            "call_id": call_id,
            "tool_name": data.get("tool_name"),
            "arguments": json_copy(data.get("arguments", {}), "arguments"),
        }
        return state

    if call_id not in state.pending_calls:
        raise ContractViolation(
            f"{event_type} for {call_id!r} without a preceding tool.call; a result "
            "that arrives before its call cannot be attributed to a real execution"
        )

    if event_type == "tool.raw_result":
        if call_id in state.raw_results:
            raise ContractViolation(f"duplicate raw result for {call_id!r}")
        state.raw_results[call_id] = json_copy(data.get("result"), "raw result")
        return state

    # tool.visible_result
    if call_id not in state.raw_results:
        raise ContractViolation(
            f"visible result for {call_id!r} before its raw result; the visible "
            "projection must be derived from a recorded raw value"
        )
    state.visible_results[call_id] = json_copy(data.get("result"), "visible result")
    return state


def _on_history(state: KernelState, event: dict, event_type: str) -> KernelState:
    data = event["data"]
    kind = data.get("kind", HISTORY_APPENDERS[event_type])
    role = HISTORY_APPENDERS[event_type] or KIND_TO_ROLE.get(kind)
    if role not in ("user", "assistant", "tool", "system"):
        raise ContractViolation(
            f"{event_type}: cannot determine role from kind {kind!r}"
        )
    text = data.get("text")
    if not isinstance(text, str):
        raise ValidationError(f"{event_type}: text must be a string")
    record = {
        "role": role,
        "kind": kind,
        "text": text,
        "provenance": json_copy(data.get("provenance", {}), "provenance"),
        "seq": event.get("seq"),
    }
    if data.get("meta"):
        record["meta"] = json_copy(data["meta"], "meta")
    if kind == "assistant_tool_call":
        calls = data.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise ContractViolation(
                "an assistant tool-call turn must carry at least one tool call; "
                "without it the following tool results answer nothing"
            )
        record["tool_calls"] = json_copy(calls, "tool calls")
    state.history.append(record)

    if role == "tool":
        call_id = data.get("call_id")
        if call_id not in state.visible_results:
            raise ContractViolation(
                f"tool message for {call_id!r} without a visible result; the "
                "history would claim an observation that never happened"
            )
        state.pending_calls.pop(call_id, None)
    return state


def _on_rebased(state: KernelState, event: dict) -> KernelState:
    """Record a window rebase and refuse an imprecise one.

    The manifest must enumerate what was retained and dropped.  A rebase that
    only says "recent content was kept" cannot be verified by anything
    downstream, so it is rejected here rather than discovered later.
    """
    data = event["data"]
    for required in ("policy", "window", "retained", "dropped"):
        if required not in data:
            raise ContractViolation(f"context.rebased is missing {required!r}")
    # The rebase manifest belongs to the window it opens, and is recorded before
    # ``session.opened`` so that the description of what was dropped is
    # attributed to the window it applies to rather than to the one it closed.
    expected = state.window_index + 1
    if data["window"] != expected:
        raise ContractViolation(
            f"context.rebased must name the window it opens ({expected}), got "
            f"{data['window']!r}"
        )
    for entry in data["retained"] + data["dropped"]:
        if not entry.get("content_digest") or not entry.get("ref"):
            raise ContractViolation(
                "context.rebased entries must carry a content digest and a "
                f"provenance ref; got {entry!r}"
            )
    state.rebases.append(json_copy(data, "rebase"))
    return state


def _on_turn_completed(state: KernelState, event: dict) -> KernelState:
    if state.pending_calls:
        raise ContractViolation(
            f"turn completed with unresolved tool calls {sorted(state.pending_calls)}"
        )
    state.status = STATUS_IDLE
    return state


def reduce_all(events: list, initial: KernelState) -> KernelState:
    state = initial
    for event in events:
        state = reduce_event(state, event)
    return state


def render_time_context(clock: Clock) -> str:
    """The one model-visible rendering of the current time (contract §4)."""
    return f"Current date and time: {format_human(clock)} ({clock.time_zone})"
