"""Kernel integrity: the log, the chain, the clock."""

from __future__ import annotations

import json

import pytest

from vela_lab.clock import Clock, format_human
from vela_lab.errors import ContractViolation, IntegrityError, StaleWriterError, ValidationError
from vela_lab.log import EventLog
from vela_lab.reducer import STATUS_FAILED, initial_state, reduce_event


def test_ordinals_are_contiguous_from_one(tmp_path):
    log = EventLog(tmp_path)
    for index in range(4):
        log.append("turn.started", {"i": index}, virtual_time="2026-01-01T00:00:00Z")
    assert [e["seq"] for e in log.read()] == [1, 2, 3, 4]


def test_genesis_links_to_empty_previous_hash(tmp_path):
    log = EventLog(tmp_path)
    log.append("session.started", {}, virtual_time="2026-01-01T00:00:00Z")
    assert log.read()[0]["prev_hash"] == ""


def test_edited_event_is_detected(tmp_path):
    log = EventLog(tmp_path)
    log.append("turn.started", {"value": 1}, virtual_time="2026-01-01T00:00:00Z")
    log.append("turn.completed", {}, virtual_time="2026-01-01T00:00:00Z")

    path = tmp_path / "events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["data"]["value"] = 2
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="hash does not match"):
        EventLog(tmp_path).read()


def test_truncated_write_is_detected(tmp_path):
    log = EventLog(tmp_path)
    log.append("turn.started", {}, virtual_time="2026-01-01T00:00:00Z")
    path = tmp_path / "events.jsonl"
    raw = path.read_bytes()
    path.write_bytes(raw[:-12])
    with pytest.raises(IntegrityError, match="truncated"):
        EventLog(tmp_path).read()


def test_reordered_events_are_detected(tmp_path):
    log = EventLog(tmp_path)
    log.append("a", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append("b", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append("c", {}, virtual_time="2026-01-01T00:00:00Z")
    path = tmp_path / "events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1], records[2] = records[2], records[1]
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError):
        EventLog(tmp_path).read()


def test_second_writer_is_refused(tmp_path):
    log = EventLog(tmp_path)
    (tmp_path / ".writer.lock").write_text("held", encoding="utf-8")
    with pytest.raises(StaleWriterError):
        log.append("turn.started", {}, virtual_time="2026-01-01T00:00:00Z")


def test_non_finite_numbers_are_rejected(tmp_path):
    log = EventLog(tmp_path)
    with pytest.raises(ValidationError, match="non-finite"):
        log.append("turn.started", {"x": float("nan")}, virtual_time="2026-01-01T00:00:00Z")


def test_clock_rejects_naive_and_backwards():
    with pytest.raises(ValidationError, match="naive"):
        Clock.of("2026-01-01T00:00:00", "UTC")
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    with pytest.raises(ValidationError, match="backwards"):
        clock.advance(days=-1)


def test_clock_normalizes_to_utc():
    clock = Clock.of("2026-03-01T09:00:00+08:00", "Asia/Shanghai")
    assert clock.instant == "2026-03-01T01:00:00Z"
    assert clock.time_zone == "Asia/Shanghai"


def test_rendered_time_carries_no_experiment_vocabulary():
    clock = Clock.of("2026-03-11T09:00:00+08:00", "Asia/Shanghai")
    rendered = format_human(clock)
    for term in ("virtual", "simulated", "controlled", "harness", "fixture"):
        assert term not in rendered.lower()


def test_advance_requires_a_stable_boundary():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    state = reduce_event(state, {"seq": 1, "type": "turn.started", "data": {}})
    with pytest.raises(ContractViolation, match="stable boundary"):
        reduce_event(
            state,
            {"seq": 2, "type": "clock.advance",
             "data": {"to": "2026-02-01T00:00:00Z", "time_zone": "UTC"}},
        )


def test_clock_cannot_move_backwards_through_the_reducer():
    clock = Clock.of("2026-02-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    with pytest.raises(ContractViolation, match="backwards"):
        reduce_event(
            state,
            {"seq": 1, "type": "clock.advance",
             "data": {"to": "2026-01-01T00:00:00Z", "time_zone": "UTC"}},
        )


def test_failed_session_is_terminal():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    state = reduce_event(state, {"seq": 1, "type": "turn.failed",
                                 "data": {"reason": "provider broke"}})
    assert state.status == STATUS_FAILED
    with pytest.raises(ContractViolation, match="already failed"):
        reduce_event(state, {"seq": 2, "type": "turn.started", "data": {}})


def test_tool_result_without_a_call_is_refused():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    with pytest.raises(ContractViolation, match="without a preceding tool.call"):
        reduce_event(
            state,
            {"seq": 1, "type": "tool.raw_result",
             "data": {"call_id": "c1", "result": {}}},
        )


def test_visible_result_without_raw_is_refused():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    state = reduce_event(
        state,
        {"seq": 1, "type": "tool.call",
         "data": {"call_id": "c1", "tool_name": "search", "arguments": {}}},
    )
    with pytest.raises(ContractViolation, match="before its raw result"):
        reduce_event(
            state,
            {"seq": 2, "type": "tool.visible_result",
             "data": {"call_id": "c1", "result": {}}},
        )


def test_tool_message_without_a_visible_result_is_refused():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    with pytest.raises(ContractViolation, match="without a visible result"):
        reduce_event(
            state,
            {"seq": 1, "type": "observation.compiled",
             "data": {"kind": "tool_result", "call_id": "c1", "text": "x"}},
        )


def test_turn_cannot_complete_with_open_tool_calls():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    state = reduce_event(
        state,
        {"seq": 1, "type": "tool.call",
         "data": {"call_id": "c1", "tool_name": "search", "arguments": {}}},
    )
    with pytest.raises(ContractViolation, match="unresolved tool calls"):
        reduce_event(state, {"seq": 2, "type": "turn.completed", "data": {}})


def test_world_event_cannot_be_applied_twice():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    state = reduce_event(
        state,
        {"seq": 1, "type": "world.event_applied", "data": {"event_id": "e1"}},
    )
    with pytest.raises(ContractViolation, match="applied twice"):
        reduce_event(
            state,
            {"seq": 2, "type": "world.event_applied", "data": {"event_id": "e1"}},
        )


def test_unknown_event_type_is_refused():
    clock = Clock.of("2026-01-01T00:00:00Z", "UTC")
    state = initial_state(clock)
    with pytest.raises(ContractViolation, match="unknown event type"):
        reduce_event(state, {"seq": 1, "type": "mystery.event", "data": {}})


def test_fold_is_deterministic_across_repeated_runs(tmp_path):
    log = EventLog(tmp_path)
    for index in range(5):
        log.append("turn.started", {"i": index}, virtual_time="2026-01-01T00:00:00Z")
    events = log.read()
    base = initial_state(Clock.of("1970-01-01T00:00:00Z", "UTC"))
    from vela_lab.reducer import reduce_all

    first = reduce_all(events, base)
    second = reduce_all(events, base)
    assert first.digest() == second.digest()
