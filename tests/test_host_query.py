"""Regression tests for experiment-level host_query, not a harness implementation.

Integration assumptions for the external harness:
* tests.conftest exports Clock and MockModel as well as DOCUMENTS, START, TZ,
  and the ordinary bundle() factory. If those classes are not re-exported,
  replace their import sources below with their actual module paths.
* Document is a dataclass with doc_id, text, and is_gold fields. _documents()
  preserves the first existing document's remaining required metadata; adjust
  that helper if the local document constructor/body field differs.
* Each branch's events.jsonl is below a directory component equal to branch_id.
  Only _auto_events() needs adapting if the harness uses another path layout.

No harness imports or tests were executed in the isolated authoring workspace.
"""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest

# Adapted to the local tree: the isolated authoring workspace could not see it, so
# the import sources it flagged are corrected here.  conftest does not re-export
# Clock/MockModel, so they come from their own modules.
from conftest import DOCUMENTS, START, TZ, bundle
from vela_lab.clock import Clock
from vela_lab.evidence import ExposurePolicy
from vela_lab.provider import MockModel
from vela_lab.runner import BranchSpec, ExperimentRunner, ExperimentSpec


HOST_QUERY = "copperleaf"
OTHER_HOST_QUERY = "violetstone"
USER_TEXT_A = "我记得预约还是6月18日。copperleaf copperleaf copperleaf"
USER_TEXT_B = "预约是否已经改成6月28日？violetstone violetstone violetstone"
PREFIX_TURNS = ("请记住，我正在准备一次预约。",)
GOLD_ID = "hq-gold"
_OMITTED = object()


def _documents():
    template = DOCUMENTS[0]
    # Distinct repeated anchors make the legacy control sensitive to query text,
    # rather than accidentally testing a tie between unrelated documents.
    return (
        replace(
            template,
            doc_id="hq-copper-confirmation",
            text=(
                "copperleaf copperleaf copperleaf copperleaf copperleaf. "
                "原预约确认：预约日期为6月18日。"
            ),
            is_gold=False,
        ),
        replace(
            template,
            doc_id="hq-copper-preparation",
            text=(
                "copperleaf copperleaf copperleaf copperleaf copperleaf. "
                "原预约的准备事项与确认记录。"
            ),
            is_gold=False,
        ),
        replace(
            template,
            doc_id=GOLD_ID,
            text=(
                "violetstone violetstone violetstone violetstone violetstone. "
                "官方变更通知：预约已改为6月28日，原6月18日预约取消。"
            ),
            is_gold=True,
        ),
    )


def _branch(branch_id, user_text, *, top_k=1, force_include_ids=()):
    return BranchSpec(
        branch_id=branch_id,
        exposure=ExposurePolicy(
            top_k=top_k,
            force_include_ids=force_include_ids,
        ),
        user_text=user_text,
    )


def _spec(*, branches, host_query=_OMITTED):
    parameters = {
        "experiment_id": "host-query-regression",
        "clock_start": START if isinstance(START, Clock) else Clock.of(START, TZ),
        "time_zone": TZ,
        "prompt": bundle(),
        "documents": _documents(),
        "prefix_turns": PREFIX_TURNS,
        "branches": branches,
        "advance_days": 19,
        "tool_budget": 8,
    }
    # An omitted argument must exercise the constructor default, not a helper's
    # substitute for that default.
    if host_query is not _OMITTED:
        parameters["host_query"] = host_query
    return ExperimentSpec(**parameters)


def _run(spec, out_dir):
    ExperimentRunner(spec, MockModel(), out_dir=out_dir).run()


def _auto_events(out_dir, branch_id):
    paths = [
        path
        for path in Path(out_dir).rglob("events.jsonl")
        if branch_id in path.relative_to(out_dir).parts[:-1]
    ]
    assert len(paths) == 1, (
        f"Expected exactly one events.jsonl for {branch_id!r}; found {paths}"
    )
    events = []
    with paths[0].open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if (
                    event.get("type") == "memory.retrieval"
                    and event.get("data", {}).get("mode") == "auto"
                ):
                    events.append(event["data"])
    assert events, f"No host-driven retrieval recorded for {branch_id!r}"
    return events


def _last_auto(out_dir, branch_id):
    # Prefix turns may also retrieve; the last auto event belongs to the branch
    # probe, while later model-initiated searches use mode='probe'.
    return _auto_events(out_dir, branch_id)[-1]


def _doc_ids(event):
    assert isinstance(event["hits"], list)
    identifiers = [hit["doc_id"] for hit in event["hits"]]
    assert all(isinstance(identifier, str) and identifier for identifier in identifiers)
    return identifiers


def test_nonempty_host_query_overrides_user_text(tmp_path):
    branch = _branch("override", USER_TEXT_A)
    spec = _spec(branches=(branch,), host_query=HOST_QUERY)

    _run(spec, tmp_path)

    event = _last_auto(tmp_path, branch.branch_id)
    assert event["query"] == HOST_QUERY
    assert event["query"] != branch.user_text


def test_empty_host_query_falls_back_to_last_user_text_after_prefix(tmp_path):
    branch = _branch("fallback", USER_TEXT_A)
    spec = _spec(branches=(branch,), host_query="")
    assert spec.prefix_turns
    assert branch.user_text not in spec.prefix_turns

    _run(spec, tmp_path)

    event = _last_auto(tmp_path, branch.branch_id)
    assert event["query"] == branch.user_text
    assert event["query"] != spec.prefix_turns[-1]


@pytest.mark.parametrize(
    ("original_query", "changed_query"),
    [("", HOST_QUERY), (HOST_QUERY, OTHER_HOST_QUERY)],
)
def test_host_query_changes_digest_and_identical_value_preserves_it(
    original_query, changed_query
):
    original = _spec(
        branches=(_branch("digest", USER_TEXT_A),),
        host_query=original_query,
    )
    identical = replace(original, host_query=original_query)
    changed = replace(original, host_query=changed_query)

    assert original.digest() == identical.digest()
    assert original.digest() == original.digest()
    assert original.digest() != changed.digest()


def test_fixed_host_query_decouples_hits_from_probe_wording(tmp_path):
    branches = (
        _branch("fixed-copper", USER_TEXT_A, top_k=2),
        _branch("fixed-violet", USER_TEXT_B, top_k=2),
    )
    spec = _spec(branches=branches, host_query=HOST_QUERY)
    assert branches[0].user_text != branches[1].user_text
    assert branches[0].exposure == branches[1].exposure

    _run(spec, tmp_path)

    first = _last_auto(tmp_path, branches[0].branch_id)
    second = _last_auto(tmp_path, branches[1].branch_id)
    assert first["query"] == second["query"] == HOST_QUERY
    first_ids, second_ids = _doc_ids(first), _doc_ids(second)
    assert first_ids and second_ids, "Two empty results would be vacuous agreement"
    assert len(first_ids) == len(second_ids) == 2
    # A multiset ignores rank order without hiding dropped or duplicated hits.
    assert Counter(first_ids) == Counter(second_ids)


def test_legacy_fallback_keeps_hits_sensitive_to_probe_wording(tmp_path):
    branches = (
        _branch("legacy-copper", USER_TEXT_A),
        _branch("legacy-violet", USER_TEXT_B),
    )
    spec = _spec(branches=branches, host_query="")
    assert branches[0].exposure == branches[1].exposure

    _run(spec, tmp_path)

    first = _last_auto(tmp_path, branches[0].branch_id)
    second = _last_auto(tmp_path, branches[1].branch_id)
    assert first["query"] == branches[0].user_text
    assert second["query"] == branches[1].user_text
    first_ids, second_ids = _doc_ids(first), _doc_ids(second)
    assert len(first_ids) == len(second_ids) == 1
    # A rank-only change is not evidence that accessibility changed.
    assert Counter(first_ids) != Counter(second_ids)


def test_oracle_replaces_a_hit_without_increasing_top_k(tmp_path):
    top_k = 2
    branches = (
        _branch("natural", USER_TEXT_A, top_k=top_k),
        _branch("oracle", USER_TEXT_A, top_k=top_k, force_include_ids=(GOLD_ID,)),
    )
    spec = _spec(branches=branches, host_query=HOST_QUERY)

    _run(spec, tmp_path)

    natural = _last_auto(tmp_path, "natural")
    oracle = _last_auto(tmp_path, "oracle")
    assert natural["query"] == oracle["query"] == HOST_QUERY
    natural_ids, oracle_ids = _doc_ids(natural), _doc_ids(oracle)
    # If gold was already natural, this would not exercise forced replacement.
    assert GOLD_ID not in natural_ids
    assert GOLD_ID in oracle_ids
    assert len(natural_ids) == len(oracle_ids) == top_k
    assert len(set(natural_ids)) == len(set(oracle_ids)) == top_k
    assert any(hit["doc_id"] == GOLD_ID and hit["is_gold"] is True for hit in oracle["hits"])


def test_omitted_host_query_defaults_to_empty_but_logged_queries_are_nonempty(tmp_path):
    branch = _branch("omitted", USER_TEXT_B)
    spec = _spec(branches=(branch,))
    assert spec.host_query == ""
    assert spec.digest() == replace(spec, host_query="").digest()

    _run(spec, tmp_path)

    events = _auto_events(tmp_path, branch.branch_id)
    for event in events:
        assert isinstance(event["query"], str)
        assert event["query"].strip(), "An omitted host_query must not produce an empty query"
    assert events[-1]["query"] == branch.user_text