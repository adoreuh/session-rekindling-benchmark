"""The frozen run policy and the resume consistency gate.

This is the narrow piece of GPT-6's "policy lifecycle" advice that survived
scrutiny.  A general per-turn policy-recovery engine was judged unnecessary for
a single-model, fixed-tool experiment; what *is* necessary is that the policy be
resolved once, frozen, and checked before a run continues.  These tests pin that
behaviour to the concrete failures it prevents.
"""

from __future__ import annotations

import json

import pytest

from vela_lab.clock import Clock
from vela_lab.errors import ContractViolation
from vela_lab.evidence import ExposurePolicy
from vela_lab.policy import RunPolicy, assert_resumable, compare_policies
from vela_lab.provider import MockModel
from vela_lab.runner import BranchSpec, ExperimentRunner, ExperimentSpec

from conftest import DOCUMENTS, PREFIX, START, TZ, WORLD_EVENTS, bundle

PROBE = "我好像已经把签证面试错过了，是不是没救了？"


def _spec(top_k: int = 2) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="policy-test",
        clock_start=Clock.of(START, TZ),
        time_zone=TZ,
        prompt=bundle(),
        documents=DOCUMENTS,
        world_events=WORLD_EVENTS,
        enabled_tools=("world_time", "search"),
        prefix_turns=PREFIX,
        branches=(BranchSpec("a", ExposurePolicy(mode="full", top_k=top_k), PROBE),),
        gold_marker="notice_new",
        advance_days=10,
    )


def _base_policy(**overrides) -> RunPolicy:
    fields = dict(
        prompt_digest="p",
        tool_schemas_digest="t",
        enabled_tools=("search",),
        phase_top_k={"prefix": 2, "a": 2},
        note_lifetime="turn_scoped",
        memory_wrapper_digest="w",
        tool_budget=4,
        request_params={},
        model_identity={"provider": "mock"},
    )
    fields.update(overrides)
    return RunPolicy(**fields)


# ------------------------------------------------------------- policy freeze


def test_run_writes_a_frozen_policy(tmp_path):
    ExperimentRunner(_spec(), MockModel(), out_dir=tmp_path / "run").run()
    payload = json.loads((tmp_path / "run" / "resolved_policy.json").read_text("utf-8"))
    for field in ("prompt_digest", "tool_schemas_digest", "enabled_tools",
                  "phase_top_k", "note_lifetime", "memory_wrapper_digest",
                  "tool_budget", "model_identity", "protocol_version"):
        assert field in payload, f"frozen policy lacks {field}"
    # The lifetime choice is part of the policy, not an internal detail: two runs
    # differing on it are not the same accessibility condition.
    assert payload["note_lifetime"] == "turn_scoped"


def test_policy_records_top_k_per_phase(tmp_path):
    ExperimentRunner(_spec(top_k=2), MockModel(), out_dir=tmp_path / "run").run()
    payload = json.loads((tmp_path / "run" / "resolved_policy.json").read_text("utf-8"))
    assert payload["phase_top_k"]["prefix"] == 2
    assert payload["phase_top_k"]["a"] == 2


def test_policy_digest_is_in_the_report(tmp_path):
    report = ExperimentRunner(
        _spec(), MockModel(), out_dir=tmp_path / "run"
    ).run()
    assert report["policy_digest"]
    recorded = json.loads(
        (tmp_path / "run" / "resolved_policy.json").read_text("utf-8")
    )
    assert RunPolicy.from_json(recorded).digest() == report["policy_digest"]


# --------------------------------------------------------- resume gate


def test_same_policy_may_continue(tmp_path):
    runner = ExperimentRunner(_spec(), MockModel(), out_dir=tmp_path / "run")
    runner.run()
    same = ExperimentRunner(_spec(), MockModel(), out_dir=tmp_path / "run")
    assert same.assert_continuable() is not None


def test_changed_top_k_refuses_to_continue(tmp_path):
    """The concrete failure: a key document at rank 3 becomes reachable.

    With top_k 2 the gold document is not returned; with top_k 4 it is.  The
    accessibility and attribution labels would both change, so the two cannot be
    counted as one experiment.
    """
    ExperimentRunner(_spec(top_k=2), MockModel(), out_dir=tmp_path / "run").run()
    widened = ExperimentRunner(_spec(top_k=4), MockModel(), out_dir=tmp_path / "run")
    with pytest.raises(ContractViolation, match="refusing to continue"):
        widened.assert_continuable()


def test_changed_budget_refuses_to_continue(tmp_path):
    spec = _spec()
    spec.tool_budget = 2
    ExperimentRunner(spec, MockModel(), out_dir=tmp_path / "run").run()

    spec2 = _spec()
    spec2.tool_budget = 4
    with pytest.raises(ContractViolation, match="tool_budget"):
        ExperimentRunner(spec2, MockModel(), out_dir=tmp_path / "run").assert_continuable()


def test_changed_model_refuses_to_continue(tmp_path):
    ExperimentRunner(_spec(), MockModel(model_name="mock-a"),
                     out_dir=tmp_path / "run").run()
    other = ExperimentRunner(_spec(), MockModel(model_name="mock-b"),
                             out_dir=tmp_path / "run")
    with pytest.raises(ContractViolation, match="model_identity"):
        other.assert_continuable()


def test_changed_memory_wrapper_refuses_to_continue(tmp_path):
    """The wrapper is part of the prompt identity, so changing it is an edit."""
    from vela_lab.prompt import PromptBundle

    spec = _spec()
    ExperimentRunner(spec, MockModel(), out_dir=tmp_path / "run").run()

    spec2 = _spec()
    spec2.prompt = PromptBundle(
        prompt_id=spec.prompt.prompt_id,
        system=spec.prompt.system,
        tool_descriptions=dict(spec.prompt.tool_descriptions),
        memory_wrapper="Notes:\n{entries}",
        notes="",
    )
    with pytest.raises(ContractViolation, match="memory_wrapper_digest"):
        ExperimentRunner(spec2, MockModel(), out_dir=tmp_path / "run").assert_continuable()


def test_changed_prompt_clause_refuses_to_continue(tmp_path):
    """A one-clause prompt edit flipped a real conclusion; it must not resume."""
    from vela_lab.prompt import PromptBundle

    spec = _spec()
    ExperimentRunner(spec, MockModel(), out_dir=tmp_path / "run").run()

    spec2 = _spec()
    spec2.prompt = PromptBundle(
        prompt_id=spec.prompt.prompt_id,
        system=spec.prompt.system + " When the user contradicts what you know, defer to the facts.",
        tool_descriptions=dict(spec.prompt.tool_descriptions),
    )
    with pytest.raises(ContractViolation, match="prompt_digest"):
        ExperimentRunner(spec2, MockModel(), out_dir=tmp_path / "run").assert_continuable()


def test_missing_policy_artifact_is_refused(tmp_path):
    runner = ExperimentRunner(_spec(), MockModel(), out_dir=tmp_path / "run")
    with pytest.raises(ContractViolation, match="missing"):
        runner.load_policy()


def test_policy_record_must_be_complete():
    with pytest.raises(ContractViolation, match="missing"):
        RunPolicy.from_json({"prompt_digest": "p"})


# --------------------------------------------------------- comparisons


def test_compare_policies_names_the_offending_field():
    left = _base_policy()
    right = _base_policy(note_lifetime="per_call")
    findings = compare_policies(left, right)
    assert [f["field"] for f in findings] == ["note_lifetime"]


def test_compare_policies_empty_when_identical():
    assert compare_policies(_base_policy(), _base_policy()) == []


def test_assert_resumable_says_what_changed():
    with pytest.raises(ContractViolation, match="tool_budget"):
        assert_resumable(_base_policy(), _base_policy(tool_budget=9))


def test_top_k_lookup_refuses_an_unknown_phase():
    policy = _base_policy()
    with pytest.raises(ContractViolation, match="no top_k registered"):
        policy.top_k_for("probe")
