"""Session windows: cross-session boundaries, rebasing, and leak detection.

Why this file exists.  Without a window switch, a fact established in an earlier
session stays in the raw conversation history, and the model can reach it without
going through memory at all.  An accessibility manipulation then changes nothing,
and every conclusion drawn from it is unsupported.  The project already paid for
this once, so the guarantee is tested directly rather than assumed.

The switch is also deliberately *not* per-branch.  It happens once on the shared
prefix, before the checkpoint, so all branches inherit the same rebased context;
otherwise the window itself would become a source of divergence.
"""

from __future__ import annotations

import json

import pytest

from vela_lab.audit import _audit_window_integrity
from vela_lab.clock import Clock
from vela_lab.domains import ObservationDomain, WorldDomain
from vela_lab.errors import ContractViolation, ValidationError
from vela_lab.evidence import ExposurePolicy
from vela_lab.log import EventLog
from vela_lab.provider import MockModel
from vela_lab.runner import (
    BranchSpec,
    ExperimentRunner,
    ExperimentSpec,
    WindowBoundary,
)
from vela_lab.session import Session

from conftest import DOCUMENTS, PREFIX, START, TZ, WORLD_EVENTS, bundle

PROBE = "我好像已经把签证面试错过了，是不是没救了？"
PREFIX_TURNS = ("我三月要办签证，怕自己看漏通知。", "最近事情有点多，忙得晕头转向。")


def _windowed_spec(**overrides) -> ExperimentSpec:
    fields = dict(
        experiment_id="window-test",
        clock_start=Clock.of(START, TZ),
        time_zone=TZ,
        prompt=bundle(),
        documents=DOCUMENTS,
        world_events=WORLD_EVENTS,
        enabled_tools=("world_time", "search"),
        prefix_turns=PREFIX_TURNS,
        window_boundaries=(WindowBoundary(after_prefix_turn=0, reason="next contact"),),
        stable_material=(("address", "她希望被称呼为小林。"),),
        branches=(
            BranchSpec("natural", ExposurePolicy(mode="full", top_k=2), PROBE),
            BranchSpec("oracle", ExposurePolicy(mode="full", top_k=2,
                                                force_include_ids=("notice_new",)), PROBE),
        ),
        gold_marker="notice_new",
        advance_days=10,
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


# ------------------------------------------------------- domain behaviour


def test_window_advances_and_rebase_drops_prior_window():
    obs = ObservationDomain()
    obs.begin_turn()
    obs.user("turn one", event_seq=1)
    assert obs.window == 0

    obs.begin_window()
    assert obs.window == 1
    manifest = obs.rebase()
    assert manifest["window"] == 1
    assert manifest["dropped"], "the rebase dropped nothing"


def test_frozen_material_survives_a_rebase():
    obs = ObservationDomain()
    obs.stable_material("她希望被称呼为小林。", material_id="address")
    obs.begin_turn()
    obs.user("turn one", event_seq=1)

    obs.begin_window()
    manifest = obs.rebase()
    retained = {entry["kind"] for entry in manifest["retained"]}
    assert "stable_material" in retained
    assert any(f.kind == "stable_material" for f in obs.fragments())


def test_rebase_manifest_enumerates_every_fragment():
    """A summary is not enough; the manifest must be machine-checkable."""
    obs = ObservationDomain()
    obs.begin_turn()
    obs.user("turn one", event_seq=1)
    obs.memory_note("evidence", retrieval_seq=2)

    obs.begin_window()
    manifest = obs.rebase()
    for entry in manifest["retained"] + manifest["dropped"]:
        assert entry["ref"], "manifest entry lacks a provenance ref"
        assert entry["content_digest"], "manifest entry lacks a content digest"
    accounted = (len(manifest["retained"]) + len(manifest["dropped"]))
    assert accounted == 2, f"manifest accounts for {accounted} of 2 fragments"


def test_dropped_content_is_still_recorded_as_exposed():
    """Retirement removes availability, not the record that exposure happened."""
    obs = ObservationDomain()
    obs.begin_turn()
    obs.user("the appointment is 3月1日", event_seq=1)

    obs.begin_window()
    obs.rebase()
    assert not any("3月1日" in f.text for f in obs.fragments())
    assert obs.was_exposed("3月1日") is True


def test_fragments_are_stamped_with_their_window():
    obs = ObservationDomain()
    obs.begin_turn()
    first = obs.user("window zero", event_seq=1)
    assert first.window == 0

    obs.begin_window()
    second = obs.user("window one", event_seq=2)
    assert second.window == 1


def test_stable_material_is_marked_frozen():
    obs = ObservationDomain()
    fragment = obs.stable_material("background", material_id="bg")
    assert fragment.frozen is True
    assert fragment.kind == "stable_material"


# ------------------------------------------------------- session behaviour


def _session(tmp_path) -> Session:
    return Session(
        tmp_path / "s",
        clock=Clock.of(START, TZ),
        observations=ObservationDomain(),
        world=WorldDomain(),
    )


def test_switch_window_emits_the_three_events(tmp_path):
    session = _session(tmp_path)
    session.add_time_context()
    session.add_user_turn("hello")
    session.observations.end_turn()

    session.switch_window()
    types = [e["type"] for e in session.events()]
    for expected in ("session.closed", "context.rebased", "session.opened"):
        assert expected in types, f"{expected} was not recorded"
    assert types.index("session.closed") < types.index("context.rebased")
    assert types.index("context.rebased") < types.index("session.opened")


def test_window_cannot_close_mid_turn(tmp_path):
    session = _session(tmp_path)
    session.add_user_turn("hello")
    with pytest.raises(ContractViolation, match="stable boundary"):
        session.close_window()


def test_windows_cannot_be_skipped(tmp_path):
    session = _session(tmp_path)
    session.observations.end_turn()
    session.close_window()
    with pytest.raises(ContractViolation, match="sequential"):
        session.append("session.opened", {"window": 5})


def test_rebase_must_be_recorded_before_the_window_opens(tmp_path):
    """The manifest belongs to the window it opens, not the one it closed."""
    session = _session(tmp_path)
    session.observations.end_turn()
    session.close_window()
    with pytest.raises(ContractViolation, match="must name the window it opens"):
        session.append("context.rebased",
                       {"policy": "p", "window": 0, "retained": [], "dropped": []})


def test_imprecise_rebase_manifest_is_refused(tmp_path):
    """A rebase that only says 'recent content was kept' is unverifiable."""
    session = _session(tmp_path)
    session.observations.end_turn()
    session.close_window()
    with pytest.raises(ContractViolation, match="missing"):
        session.append("context.rebased", {"policy": "p", "window": 1,
                                           "retained": []})


def test_rebase_entries_require_digest_and_ref(tmp_path):
    session = _session(tmp_path)
    session.observations.end_turn()
    session.close_window()
    with pytest.raises(ContractViolation, match="content digest"):
        session.append(
            "context.rebased",
            {"policy": "p", "window": 1,
             "retained": [{"kind": "user"}], "dropped": []},
        )


def test_register_stable_material_is_recorded(tmp_path):
    session = _session(tmp_path)
    session.register_stable_material("address", "她希望被称呼为小林。")
    events = session.events()
    assert any(e["type"] == "stable_material.registered" for e in events)


# ------------------------------------------------------- runner integration


def _run(tmp_path, spec) -> dict:
    return ExperimentRunner(spec, MockModel(), out_dir=tmp_path / "run").run()


def test_runner_forks_after_the_window_switch(tmp_path):
    """The switch must precede the checkpoint so all branches share it."""
    report = _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    types = [e["type"] for e in events]
    rebase_seq = types.index("context.rebased")
    fork_seq = types.index("branch.forked")
    assert rebase_seq < fork_seq, "the window switched after the fork"
    assert report["parent_equality"]["identical_parent"] is True


def test_all_branches_inherit_the_same_window(tmp_path):
    report = _run(tmp_path, _windowed_spec())
    windows = set()
    for branch in report["branches"]:
        events = EventLog(
            tmp_path / "run" / "branches" / branch["branch_id"]
        ).read()
        opened = [e["data"]["window"] for e in events if e["type"] == "session.opened"]
        windows.add(tuple(opened))
    assert len(windows) == 1, f"branches opened different windows: {windows}"


def test_prefix_fact_does_not_reach_the_probe_request(tmp_path):
    """The whole point: the earlier session's raw text must be gone."""
    _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    last_request = [e for e in events if e["type"] == "model.request"][-1]
    body = json.dumps(last_request["data"]["messages"], ensure_ascii=False)
    assert "怕自己看漏通知" not in body, (
        "the prefix's raw text survived into the probe request; the accessibility "
        "manipulation would be inert"
    )


def test_stable_material_reaches_the_probe_request(tmp_path):
    report = _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    last_request = [e for e in events if e["type"] == "model.request"][-1]
    body = json.dumps(last_request["data"]["messages"], ensure_ascii=False)
    assert "小林" in body, "pre-registered background was lost at the window switch"
    assert all(b["run_valid"] for b in report["branches"])


def test_windowed_run_passes_audit(tmp_path):
    report = _run(tmp_path, _windowed_spec())
    for branch in report["branches"]:
        assert branch["audit_findings"] == [], branch["audit_findings"]


def test_manifest_entries_name_their_window(tmp_path):
    _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    for event in events:
        if event["type"] != "model.request":
            continue
        for entry in event["data"]["context_manifest"]:
            assert "window" in entry, "manifest entry does not name its window"
            assert "frozen" in entry


# ------------------------------------------------------------- audit


def test_audit_flags_closed_window_content_in_a_later_request(tmp_path):
    """The concrete leak: a window-0 fragment delivered in window 1."""
    _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    last = max(i for i, e in enumerate(events) if e["type"] == "model.request")
    mutated = [dict(e) for e in events]
    target = dict(mutated[last])
    target["data"] = dict(target["data"])
    target["data"]["context_manifest"] = list(target["data"]["context_manifest"]) + [
        {
            "position": 99,
            "kind": "user",
            "source": "observation",
            "ref": "seq:5",
            "chars": 12,
            "content_digest": "deadbeef",
            "window": 0,
            "frozen": False,
        }
    ]
    mutated[last] = target

    findings = _audit_window_integrity(mutated)
    assert any(f["check"] == "closed_window_content_not_redelivered" for f in findings)


def test_audit_permits_frozen_material_across_windows(tmp_path):
    """Registered background is present by design and must not be flagged."""
    _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    last = max(i for i, e in enumerate(events) if e["type"] == "model.request")
    mutated = [dict(e) for e in events]
    target = dict(mutated[last])
    target["data"] = dict(target["data"])
    target["data"]["context_manifest"] = list(target["data"]["context_manifest"]) + [
        {
            "position": 99,
            "kind": "stable_material",
            "source": "template",
            "ref": "address",
            "chars": 12,
            "content_digest": "cafe",
            "window": 0,
            "frozen": True,
        }
    ]
    mutated[last] = target
    assert _audit_window_integrity(mutated) == []


def test_audit_flags_a_manifest_entry_without_a_window(tmp_path):
    _run(tmp_path, _windowed_spec())
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    last = max(i for i, e in enumerate(events) if e["type"] == "model.request")
    mutated = [dict(e) for e in events]
    target = dict(mutated[last])
    target["data"] = dict(target["data"])
    target["data"]["context_manifest"] = list(target["data"]["context_manifest"]) + [
        {"position": 99, "kind": "user", "source": "observation", "ref": "seq:9",
         "chars": 4, "content_digest": "beef"}
    ]
    mutated[last] = target
    findings = _audit_window_integrity(mutated)
    assert any(f["check"] == "manifest_entry_names_window" for f in findings)


def test_audit_flags_a_fragment_both_retained_and_dropped(tmp_path):
    session = _session(tmp_path)
    session.observations.end_turn()
    session.close_window()
    manifest = session.open_window()
    entry = {
        "kind": "user",
        "ref": "seq:1",
        "source": "observation",
        "content_digest": "abc",
    }
    events = session.events()
    events.append(
        {
            "seq": len(events) + 1,
            "type": "context.rebased",
            "data": {**manifest, "retained": [entry], "dropped": [entry]},
            "prev_hash": "",
            "hash": "",
        }
    )
    findings = _audit_window_integrity(events)
    assert any(f["check"] == "rebased_manifest_consistent" for f in findings)


# ------------------------------------------------------- preflight


def test_boundary_past_the_last_prefix_turn_is_refused(tmp_path):
    spec = _windowed_spec(
        window_boundaries=(WindowBoundary(after_prefix_turn=99, reason="x"),)
    )
    with pytest.raises(ValidationError, match="past the last prefix turn"):
        _run(tmp_path, spec)


def test_duplicate_boundaries_are_refused(tmp_path):
    spec = _windowed_spec(
        window_boundaries=(
            WindowBoundary(after_prefix_turn=0, reason="a"),
            WindowBoundary(after_prefix_turn=0, reason="b"),
        )
    )
    with pytest.raises(ValidationError, match="distinct turns"):
        _run(tmp_path, spec)


def test_boundary_must_state_a_reason():
    with pytest.raises(ValidationError, match="reason"):
        WindowBoundary(after_prefix_turn=0, reason="")


def test_window_boundary_enters_the_spec_digest(tmp_path):
    without = _windowed_spec(window_boundaries=())
    with_boundary = _windowed_spec()
    assert without.digest() != with_boundary.digest()


def test_stable_material_enters_the_spec_digest(tmp_path):
    plain = _windowed_spec(stable_material=())
    with_material = _windowed_spec()
    assert plain.digest() != with_material.digest()


def test_single_window_run_is_unchanged(tmp_path):
    """A spec without boundaries must behave exactly as before windows existed."""
    spec = _windowed_spec(
        prefix_turns=PREFIX,
        window_boundaries=(),
        stable_material=(),
    )
    report = _run(tmp_path, spec)
    events = EventLog(tmp_path / "run" / "branches" / "natural").read()
    assert not any(e["type"] == "context.rebased" for e in events)
    assert all(b["run_valid"] for b in report["branches"])
