"""The no-self-retrieval protocol is enforced, not requested.

The protocol has three separate mechanisms, and each is checked here because any
one of them alone would leave a route open:

1. the schema is absent, so the model is not told the tool exists;
2. the tool belt refuses the name at dispatch, so a model that asks anyway gets
   an error result rather than documents;
3. the request gate re-reads the serialized tool list, so a future change to the
   belt or the prompt cannot reintroduce the schema without the gate failing.

The third is the one that matters for a paid run: mechanisms one and two are the
implementation, and three is the check that the implementation is still in place
when the request actually goes out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.errors import ContractViolation  # noqa: E402
from vela_lab.evidence import EvidenceCorpus, ExposurePolicy  # noqa: E402
from vela_lab.provider import MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402
from vela_lab.tools import ToolBelt  # noqa: E402

from tools.v4_wiring import (  # noqa: E402
    FORBIDDEN_TOOL_NAMES,
    PROBE_TOOLS,
    build_v4_spec,
)


def _spec(elapsed_days: int = 15):
    return build_v4_spec(experiment_id="protocol-test", elapsed_days=elapsed_days)


# ------------------------------------------------------------ mechanism 1

def test_probe_turn_never_exposes_the_search_schema():
    spec = _spec()
    assert "search" not in spec.enabled_tools
    assert all("search" not in b.exposure.deliver_ids for b in spec.branches if b.exposure.deliver_ids)


@pytest.mark.parametrize("condition", ["oracle", "omit"])
def test_serialized_tool_list_carries_no_search(condition, tmp_path):
    """Read from the recorded request, not from the configuration."""
    import json

    spec = _spec()
    branch_id = f"claim_low_{condition}"
    assert branch_id in {b.branch_id for b in spec.branches}

    model = MockModel(scripted=["我不确定。"])
    out = tmp_path / "run"
    ExperimentRunner(spec, model, out_dir=out).run()

    log = out / "branches" / branch_id / "events.jsonl"
    names = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] != "model.request":
            continue
        for schema in event["data"]["payload"].get("tools", []):
            names.add(schema["function"]["name"])
    assert names == set(PROBE_TOOLS)
    assert not (names & set(FORBIDDEN_TOOL_NAMES))


# ------------------------------------------------------------ mechanism 2

def test_tool_belt_refuses_search_at_dispatch():
    """A model that asks anyway is refused, not served."""
    corpus = EvidenceCorpus([])
    corpus.seal()
    belt = ToolBelt(
        enabled=list(PROBE_TOOLS),
        corpus=corpus,
        world=None,
        exposure=ExposurePolicy(mode="full", top_k=1),
    )
    assert not belt.is_enabled("search")
    with pytest.raises(ContractViolation) as excinfo:
        belt.execute("c1", "search", {"query": "签证面试"}, clock=None)
    assert "not enabled" in str(excinfo.value)


def test_a_search_call_during_the_probe_is_returned_as_an_error(tmp_path):
    """The end-to-end behaviour: the call is answered with an error result.

    The model asked for a tool that is not in its schema.  A product would not
    silently serve it, and neither does this harness; the turn continues so the
    refusal is observable rather than fatal.
    """
    import json

    # The first scripted reply answers the prefix turn; the second is the probe
    # turn's request for the forbidden tool.
    probe_call = {
        "content": "",
        "tool_calls": [{
            "id": "p1", "type": "function",
            "function": {"name": "search", "arguments": '{"query": "签证面试"}'},
        }],
        "finish_reason": "tool_calls",
    }
    model = MockModel(scripted=["好，我们练习一下。", probe_call, "按 6 月 18 日到场就好。"])
    spec = _spec()
    out = tmp_path / "run"
    ExperimentRunner(spec, model, out_dir=out).run()

    log = out / "branches" / "claim_low_omit" / "events.jsonl"
    visible = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "tool.visible_result":
            visible.append(event["data"]["result"])
    assert visible, "the probe-turn search call was never recorded"
    assert all("error" in v for v in visible)
    assert all("search" in str(v["error"]) for v in visible)


# ------------------------------------------------------------ mechanism 3

def test_request_gate_rejects_a_reintroduced_search_schema():
    """The gate is the check that mechanisms 1 and 2 are still in place."""
    from tools import scenario_v4 as sc
    from tools.v4_wiring import make_request_gate

    contract = sc.exposure_contract()
    pair_manifest = __import__(
        "tools.v4_wiring", fromlist=["_manifests"]
    )._manifests()["oracle"]
    gate = make_request_gate(
        contract, pair_manifest,
        tuple(sc.PROBES.values()) + tuple(sc.STABLE_MATERIAL),
        tuple(sc.PROBES.values()),
    )
    payload = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [
            {"type": "function",
             "function": {"name": "search", "description": "d",
                          "parameters": {"type": "object", "properties": {}}}}
        ],
    }

    class _Compiled:
        pass

    compiled = _Compiled()
    compiled.payload = payload
    result = gate(compiled)
    assert result.status == "RUN_INVALID"
    assert "forbidden_tool_exposed" in result.reason_codes


# ------------------------------------------------- the accounting stays usable

def test_search_accounting_levels_are_reported_separately(tmp_path):
    """Zero must be interpretable: asked-and-refused is not never-asked."""
    spec = _spec()
    model = MockModel(scripted=["我理解。"])
    out = tmp_path / "run"
    report = ExperimentRunner(spec, model, out_dir=out).run()
    for branch in report["branches"]:
        accounting = branch["search_accounting"]
        assert set(accounting) == {
            "search_requested", "search_executed", "search_evidence_delivered"
        }
        # With the tool absent, execution is impossible by construction.
        assert accounting["search_executed"] == 0


def test_gate_verdicts_are_persisted_on_every_branch(tmp_path):
    """A verdict that exists only in a live object cannot be re-derived."""
    spec = _spec()
    model = MockModel(scripted=["我理解。"])
    out = tmp_path / "run"
    report = ExperimentRunner(spec, model, out_dir=out).run()
    for branch in report["branches"]:
        assert branch["exposure_verdicts"], (
            f"{branch['branch_id']} recorded no exposure check; a run with no "
            "gate events is indistinguishable from one whose gate found nothing"
        )
        assert branch["run_valid"] is True
        assert branch["invalid_reason"] == ""


# ------------------------------------------- name respellings close the route

def test_retrieval_capable_names_catches_respellings():
    """An exact-name check is not a check.

    ``"search" in FORBIDDEN`` blocks only that literal spelling, so
    ``search-notes``, ``search_notes`` and ``Search`` all reopened the retrieval
    route while the gate still reported PASS.  Normalising before the test is the
    fix; this pins it so the narrow version cannot come back.
    """
    from tools.v4_wiring import retrieval_capable_names

    for spelling in (
        "search", "search-notes", "search_notes", "Search", "SEARCH",
        " search", "search.notes", "websearch", "retrieve_notes",
        "memory_recall", "lookup", "rag_query",
    ):
        assert retrieval_capable_names([spelling]), (
            f"{spelling!r} would give the model a retrieval route but was allowed"
        )


def test_legitimate_probe_tools_are_not_flagged():
    """The fragment test must not fire on the tools the protocol does declare."""
    from tools.v4_wiring import PROBE_TOOLS, retrieval_capable_names

    assert retrieval_capable_names(PROBE_TOOLS) == []
    # A tool the harness legitimately exposes elsewhere must also pass: the
    # check is about retrieval capability, not about being unfamiliar.
    assert retrieval_capable_names(["world_time", "weather", "calendar_events"]) == []


def test_respelled_retrieval_tool_makes_the_run_invalid():
    """End to end: the gate rejects the payload, not just the helper."""
    from tools import scenario_v4 as sc
    from tools.v4_wiring import make_request_gate, _manifests

    contract = sc.exposure_contract()
    gate = make_request_gate(
        contract, _manifests()["oracle"],
        tuple(sc.PROBES.values()) + tuple(sc.STABLE_MATERIAL),
        tuple(sc.PROBES.values()),
    )
    for spelling in ("search-notes", "Search", "websearch"):
        payload = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {
                "name": spelling, "description": "d",
                "parameters": {"type": "object", "properties": {}}}}],
        }

        class _C:
            pass

        c = _C()
        c.payload = payload
        result = gate(c)
        assert result.status == "RUN_INVALID", spelling
        assert spelling in result.evidence["forbidden_tools"], spelling
