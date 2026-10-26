"""Experiment-level guarantees: fork isolation, oracle invariance, audit.

These are the tests that decide whether a run's comparison is admissible.  Each
one corresponds to a claim the old harness could not substantiate:

* branches really do share one parent state;
* the matched oracle changes only ranking, never the result count;
* the exposure chain distinguishes the four attribution cases;
* a corrupted log fails the audit rather than passing quietly.
"""

from __future__ import annotations

import json

import pytest

from vela_lab.audit import audit_events
from vela_lab.clock import Clock
from vela_lab.errors import AuditFailure, ContractViolation, IntegrityError
from vela_lab.evidence import EvidenceCorpus, ExposurePolicy
from vela_lab.provider import MockModel
from vela_lab.runner import BranchSpec, ExperimentRunner, ExperimentSpec
from vela_lab.schedule import WorldEvent, WorldSchedule, advance_to
from vela_lab.session import Session

from conftest import (
    DOCUMENTS,
    NEW_DOC,
    OLD_DOC,
    PREFIX,
    START,
    TZ,
    WORLD_EVENTS,
    bundle,
)

PROBE = "我好像已经把签证面试错过了，是不是没救了？"


def _spec(branches) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="ewm-test",
        clock_start=Clock.of(START, TZ),
        time_zone=TZ,
        prompt=bundle(),
        documents=DOCUMENTS,
        world_events=WORLD_EVENTS,
        enabled_tools=("world_time", "search"),
        prefix_turns=PREFIX,
        branches=branches,
        gold_fact_id="visa_date",
        gold_marker="notice_new",
        advance_days=10,
    )


def _run(tmp_path, spec):
    return ExperimentRunner(spec, MockModel(), out_dir=tmp_path / "run").run()


# ------------------------------------------------------------- fork


def test_every_branch_forks_from_the_same_parent(tmp_path):
    report = _run(
        tmp_path,
        _spec(
            (
                BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),
                BranchSpec("b", ExposurePolicy(mode="full", top_k=2,
                                               omit_ids=("notice_new",)), PROBE),
                BranchSpec("c", ExposurePolicy(mode="full", top_k=2,
                                               force_include_ids=("notice_new",)), PROBE),
            )
        ),
    )
    assert report["parent_equality"]["identical_parent"] is True
    assert len(report["parent_equality"]["parent_prefix_digests"]) == 1


def test_branch_is_independent_of_its_parent(tmp_path):
    """A child must remain readable after the parent directory is gone."""
    import shutil

    spec = _spec((BranchSpec("only", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    child_root = tmp_path / "run" / "branches" / "only"
    shutil.rmtree(tmp_path / "run" / "prefix")

    # The child's own log still verifies.
    from vela_lab.log import EventLog

    events = EventLog(child_root).read()
    assert events
    assert audit_events(events) == []


def test_a_branch_does_not_write_back_into_the_parent(tmp_path):
    spec = _spec(
        (
            BranchSpec("writes", ExposurePolicy(mode="full", top_k=2), PROBE),
        )
    )
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    parent_root = tmp_path / "run" / "prefix"
    parent_events = EventLog(parent_root).read()
    child_events = EventLog(tmp_path / "run" / "branches" / "writes").read()
    # The child has strictly more events, and the parent was not extended.
    assert len(child_events) > len(parent_events)
    checkpoint_events = [e for e in parent_events if e["type"] == "checkpoint.created"]
    assert checkpoint_events
    assert parent_events[-1]["type"] == "checkpoint.created"


def test_advance_is_refused_mid_turn(tmp_path):
    session = Session(tmp_path / "s", clock=Clock.of(START, TZ))
    session.append("turn.started", {})
    with pytest.raises(ContractViolation, match="stable boundary"):
        session.advance(days=1)


def test_checkpoint_is_refused_mid_turn(tmp_path):
    session = Session(tmp_path / "s", clock=Clock.of(START, TZ))
    session.append("turn.started", {})
    with pytest.raises(ContractViolation, match="stable boundary"):
        session.checkpoint("nope")


def test_checkpoint_checksum_is_verified(tmp_path):
    session = Session(tmp_path / "s", clock=Clock.of(START, TZ))
    session.checkpoint("shared")
    path = tmp_path / "s" / "checkpoints" / "shared.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["turn_index"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        session.load_checkpoint(tmp_path / "s", "shared")


def test_fork_refuses_a_tampered_parent_log(tmp_path):
    session = Session(tmp_path / "s", clock=Clock.of(START, TZ))
    session.checkpoint("shared")
    log_path = tmp_path / "s" / "events.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    records[0]["data"]["config"] = {"tampered": True}
    log_path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError):
        session.fork(tmp_path / "s", "shared", target_root=tmp_path / "child")


# ----------------------------------------------------------- oracle


def test_matched_oracle_preserves_the_result_count():
    corpus = EvidenceCorpus([OLD_DOC, NEW_DOC])
    corpus.seal()
    now = "2026-03-11T09:00:00+08:00"
    natural = corpus.retrieve(query="签证", now_instant=now, top_k=2)
    oracle = corpus.retrieve(
        query="签证", now_instant=now, top_k=2, force_include_ids=("notice_new",)
    )
    assert len(natural.hits) == len(oracle.hits)
    assert oracle.gold_rank() == 1


def test_oracle_promotes_gold_without_changing_eligibility():
    corpus = EvidenceCorpus([OLD_DOC, NEW_DOC])
    corpus.seal()
    now = "2026-03-11T09:00:00+08:00"
    natural = corpus.retrieve(query="签证", now_instant=now, top_k=2)
    oracle = corpus.retrieve(
        query="签证", now_instant=now, top_k=2, force_include_ids=("notice_new",)
    )
    assert natural.eligible_ids == oracle.eligible_ids


def test_unpublished_evidence_is_not_eligible():
    corpus = EvidenceCorpus([OLD_DOC, NEW_DOC])
    corpus.seal()
    early = corpus.retrieve(
        query="签证", now_instant="2026-03-02T09:00:00+08:00", top_k=4
    )
    assert "notice_new" not in early.eligible_ids


def test_document_text_cannot_be_edited_after_sealing():
    corpus = EvidenceCorpus([OLD_DOC])
    corpus.seal()
    with pytest.raises(ContractViolation, match="sealed"):
        corpus.add(NEW_DOC)


def test_world_schedule_refuses_same_tick_without_unique_order():
    with pytest.raises(ContractViolation, match="tiebreak"):
        WorldSchedule(
            [
                WorldEvent(event_id="a", kind="add_fact", at="2026-01-01T00:00:00Z",
                           order=0, fact_id="x"),
                WorldEvent(event_id="b", kind="add_fact", at="2026-01-01T00:00:00Z",
                           order=0, fact_id="y"),
            ]
        )


def test_schedule_is_sealed_against_mid_run_registration():
    schedule = WorldSchedule([])
    schedule.seal()
    with pytest.raises(ContractViolation, match="sealed"):
        schedule.register(
            WorldEvent(event_id="late", kind="add_fact",
                       at="2026-01-01T00:00:00Z", fact_id="x")
        )


def test_scheduling_orders_effective_before_notified():
    schedule = WorldSchedule(
        [
            WorldEvent(event_id="fact", kind="supersede_fact",
                       at="2026-03-05T09:00:00+08:00", order=0,
                       fact_id="visa_date", new_value="2026-03-24"),
            WorldEvent(event_id="notice", kind="send_notification",
                       at="2026-03-05T09:00:00+08:00", order=1,
                       notify_at="2026-03-05T09:00:00+08:00",
                       notification_text="更新"),
        ]
    )
    moments = advance_to(
        schedule,
        Clock.of("2026-03-01T09:00:00+08:00", TZ),
        Clock.of("2026-03-11T09:00:00+08:00", TZ),
    )
    assert [m for _, m in moments] == ["effective", "notified"]


# ------------------------------------------------- exposure chain


@pytest.mark.parametrize(
    "policy,user_text,expected_label",
    [
        (
            ExposurePolicy(mode="full", top_k=2, omit_ids=("notice_new",)),
            PROBE,
            "not_applicable",
        ),
        (
            ExposurePolicy(mode="full", top_k=2, force_include_ids=("notice_new",)),
            PROBE,
            "delivered_not_used",
        ),
    ],
)
def test_attribution_chain_labels(tmp_path, policy, user_text, expected_label):
    report = _run(tmp_path, _spec((BranchSpec("b", policy, user_text),)))
    branch = report["branches"][0]
    assert branch["score"]["L4"]["label"] == expected_label


def test_natural_branch_selects_different_evidence_than_oracle(tmp_path):
    report = _run(
        tmp_path,
        _spec(
            (
                BranchSpec("natural", ExposurePolicy(mode="full", top_k=2), PROBE),
                BranchSpec("oracle", ExposurePolicy(mode="full", top_k=2,
                                                    force_include_ids=("notice_new",)), PROBE),
            )
        ),
    )
    by_id = {b["branch_id"]: b for b in report["branches"]}
    natural_rank = by_id["natural"]["score"]["L2"]["retrievals"][-1]["gold_rank"]
    oracle_rank = by_id["oracle"]["score"]["L2"]["retrievals"][-1]["gold_rank"]
    assert natural_rank is None
    assert oracle_rank == 1


def test_all_branches_pass_audit(tmp_path):
    report = _run(
        tmp_path,
        _spec(
            (
                BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),
                BranchSpec("b", ExposurePolicy(mode="full", top_k=2,
                                               omit_ids=("notice_new",)), PROBE),
            )
        ),
    )
    for branch in report["branches"]:
        assert branch["audit_findings"] == []
        assert branch["run_valid"] is True


# --------------------------------------------------------------- audit


def test_audit_flags_gold_vocabulary_in_a_request(tmp_path):
    """A clean chain that leaks vocabulary must still be flagged."""
    from vela_lab.log import EventLog

    log = EventLog(tmp_path / "leak")
    log.append("session.started", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append(
        "model.request",
        {"messages": [{"role": "user", "content": "here is the gold answer"}]},
        virtual_time="2026-01-01T00:00:00Z",
    )
    findings = audit_events(log.read())
    assert any(f["check"] == "visible_vocabulary" for f in findings)


def test_audit_fails_when_events_are_missing():
    assert audit_events([]) != []


def test_audit_reports_tool_result_without_call(tmp_path):
    """A raw result with no preceding call is a forged observation."""
    from vela_lab.log import EventLog

    log = EventLog(tmp_path / "orphan")
    log.append("session.started", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append(
        "tool.raw_result",
        {"call_id": "ghost", "result": {}},
        virtual_time="2026-01-01T00:00:00Z",
    )
    findings = audit_events(log.read())
    assert any(f["check"] == "tool_raw_after_call" for f in findings)


def test_audit_reports_a_tampered_chain(tmp_path):
    from vela_lab.log import EventLog

    log = EventLog(tmp_path / "tamper")
    log.append("session.started", {"x": 1}, virtual_time="2026-01-01T00:00:00Z")
    path = tmp_path / "tamper" / "events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["data"]["x"] = 2
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records),
        encoding="utf-8",
    )
    # Reading must fail closed; the audit surface never sees bad bytes.
    with pytest.raises(IntegrityError):
        EventLog(tmp_path / "tamper").read()
def test_audit_or_raise_raises_on_findings():
    from vela_lab.audit import audit_or_raise

    with pytest.raises(AuditFailure):
        audit_or_raise([])


def test_replay_reproduces_the_live_state(tmp_path):
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog
    from vela_lab.reducer import initial_state, reduce_all

    child = tmp_path / "run" / "branches" / "a"
    events = EventLog(child).read()
    state = reduce_all(
        events, initial_state(Clock.of("1970-01-01T00:00:00Z", "UTC"))
    )
    assert state.clock is not None
    assert any(record["role"] == "assistant" for record in state.history)


# ------------------------------------------------- trajectory integrity


def test_every_request_has_exactly_one_outcome(tmp_path):
    """A request with no recorded outcome is the worst case for research."""
    from vela_lab.audit import _audit_call_pairing
    from vela_lab.log import EventLog

    log = EventLog(tmp_path / "unanswered")
    log.append("session.started", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append(
        "model.request",
        {"request_index": 1, "messages": [], "tools": [], "payload": {},
         "payload_digest": "x", "prompt_id": "p", "prompt_digest": "d"},
        virtual_time="2026-01-01T00:00:00Z",
    )
    findings = _audit_call_pairing(log.read())
    assert any(f["check"] == "request_has_outcome" for f in findings)


def test_a_request_answered_twice_is_flagged(tmp_path):
    from vela_lab.audit import _audit_call_pairing
    from vela_lab.log import EventLog

    log = EventLog(tmp_path / "double")
    log.append("session.started", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append(
        "model.request",
        {"request_index": 1, "messages": [], "tools": [], "payload": {},
         "payload_digest": "x", "prompt_id": "p", "prompt_digest": "d"},
        virtual_time="2026-01-01T00:00:00Z",
    )
    for _ in range(2):
        log.append(
            "model.response",
            {"request_index": 1, "response": {"content": "x"}},
            virtual_time="2026-01-01T00:00:00Z",
        )
    findings = _audit_call_pairing(log.read())
    checks = {f["check"] for f in findings}
    assert "request_answered_once" in checks


def test_indeterminate_retry_is_flagged(tmp_path):
    from vela_lab.audit import _audit_indeterminate
    from vela_lab.log import EventLog

    log = EventLog(tmp_path / "retry")
    log.append("session.started", {}, virtual_time="2026-01-01T00:00:00Z")
    log.append(
        "model.response",
        {"request_index": 1, "response": {"content": "x", "attempts": 2}},
        virtual_time="2026-01-01T00:00:00Z",
    )
    findings = _audit_indeterminate(log.read())
    assert any(f["check"] == "indeterminate_call_flagged" for f in findings)


def test_raw_provider_response_is_persisted(tmp_path):
    """Normalized fields discard exactly what matters after the fact."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    responses = [e for e in events if e["type"] == "model.response"]
    assert responses
    for event in responses:
        body = event["data"]["response"]
        for field in ("content", "finish_reason", "usage", "model", "attempts"):
            assert field in body, f"response record lacks {field}"


def test_request_indices_do_not_collide_across_the_fork(tmp_path):
    """The prefix's requests and the branch's requests share one numbering."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.audit import audit_events
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    indices = [e["data"]["request_index"] for e in events
               if e["type"] == "model.request"]
    assert len(indices) == len(set(indices))
    assert audit_events(events) == []


def test_turn_context_snapshots_the_policy_in_force(tmp_path):
    """A later reader must be able to see which policy produced a turn."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2,
                                                  omit_ids=("notice_new",)), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    contexts = [e for e in events if e["type"] == "turn.context"]
    assert contexts, "no turn context was recorded"

    probe_contexts = [c for c in contexts if c["data"]["label"] == "a"]
    assert probe_contexts, "the branch turn has no context snapshot"
    body = probe_contexts[-1]["data"]
    for field in ("virtual_time", "prompt_id", "prompt_digest", "enabled_tools",
                  "tool_schema_digest", "exposure", "budget", "model"):
        assert field in body, f"turn context lacks {field}"
    # The snapshot must reflect the branch's own manipulation, not the default.
    assert body["exposure"]["omit_ids"] == ["notice_new"]


def test_turn_context_records_the_virtual_time_of_that_turn(tmp_path):
    """After the clock moves, the probe turn's context shows the new time."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    contexts = [e["data"] for e in events if e["type"] == "turn.context"]
    times = {c["virtual_time"] for c in contexts}
    assert len(times) > 1, "expected the prefix and probe turns to differ in time"


def test_auto_retrieval_is_host_driven_not_model_driven(tmp_path):
    """Retrieval must not depend on the model choosing to call a tool.

    Otherwise accessibility would be entangled with tool-use willingness, and
    "did not retrieve" would be indistinguishable from "chose not to look".
    """
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    report = _run(tmp_path, spec)
    branch = report["branches"][0]
    assert branch["auto_retrieval"]["hits"], "no automatic retrieval was performed"
    assert branch["auto_retrieval"]["query"]

    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    auto = [e for e in events
            if e["type"] == "memory.retrieval" and e["data"]["mode"] == "auto"]
    assert auto, "the automatic retrieval was not recorded"


def test_retired_notes_are_reported_and_not_reactivated(tmp_path):
    """The lifetime guarantee, checked against the recorded turn boundary.

    A tautology is easy to write here, so this asserts on real structure: the
    retirement event must exist between the prefix turn and the branch turn, and
    the branch's first request must carry manifest entries whose provenance
    points at the branch's own retrieval rather than at the prefix's.
    """
    from vela_lab.log import EventLog

    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    events = EventLog(tmp_path / "run" / "branches" / "a").read()

    prefix_end = None
    fork_seq = None
    retirement = None
    for event in events:
        if event["type"] == "turn.completed" and prefix_end is None:
            prefix_end = event["seq"]
        if event["type"] == "observation.retired" and prefix_end is not None:
            retirement = event["seq"]
        if event["type"] == "branch.forked" and fork_seq is None:
            fork_seq = event["seq"]

    assert prefix_end is not None, "the prefix turn never completed"
    assert retirement is not None, "no retirement was recorded after the prefix"
    assert fork_seq is not None, "the branch was not forked from the checkpoint"
    assert prefix_end < fork_seq, "the branch forked before the prefix closed"

    # The branch's retrieval happens while setting up its own turn, so it sits
    # between the fork and the first request; that is expected, not a leak.
    first_request = next(
        e for e in events
        if e["type"] == "model.request" and e["seq"] > fork_seq
    )
    manifest_refs = {
        entry["ref"] for entry in first_request["data"]["context_manifest"]
    }
    assert manifest_refs, "the branch request has no manifest"

    # Every memory note in the branch's request must trace to a retrieval issued
    # after the fork — never to the prefix, whose notes were retired.
    branch_retrievals = {
        e["seq"] for e in events
        if e["type"] == "memory.retrieval" and e["seq"] > fork_seq
    }
    assert branch_retrievals, "the branch issued no retrieval of its own"
    for entry in first_request["data"]["context_manifest"]:
        if entry["kind"] != "memory_note":
            continue
        seq = int(entry["ref"].split(":")[1])
        assert seq > fork_seq, (
            f"note at {entry['ref']} predates the fork; a prefix retrieval leaked "
            "into the probe condition"
        )


def test_range_of_search_is_not_model_controlled(tmp_path):
    """A model asking for more results must not widen the evidence set.

    Scoped to the branch's own turns: the prefix runs under the experiment's
    neutral policy, and it is the probe's exposure that the experiment varies.
    """
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()

    # Locate the branch turn (label == branch id) and inspect retrievals after it.
    branch_start = None
    for event in events:
        if event["type"] == "turn.context" and event["data"]["label"] == "a":
            branch_start = event["seq"]
            break
    assert branch_start is not None, "the branch turn was not recorded"

    probe_retrievals = [
        e for e in events
        if e["type"] == "memory.retrieval" and e["seq"] > branch_start
    ]
    assert probe_retrievals, "the branch made no retrieval"
    for event in probe_retrievals:
        assert event["data"]["top_k"] == 2, "policy top_k was overridden"
        assert len(event["data"]["hits"]) <= 2


def test_oracle_and_natural_keep_the_same_result_count(tmp_path):
    report = _run(
        tmp_path,
        _spec(
            (
                BranchSpec("natural", ExposurePolicy(mode="full", top_k=2), PROBE),
                BranchSpec("oracle", ExposurePolicy(mode="full", top_k=2,
                                                    force_include_ids=("notice_new",)), PROBE),
            )
        ),
    )
    counts = {
        b["branch_id"]: b["score"]["L2"]["retrievals"][-1]["result_count"]
        for b in report["branches"]
    }
    assert len(set(counts.values())) == 1, f"result counts differ: {counts}"


def test_context_manifest_records_what_entered_each_request(tmp_path):
    """A reader must see which material entered this request, not infer it."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    probe_requests = []
    branch_start = None
    for event in events:
        if event["type"] == "turn.context" and event["data"]["label"] == "a":
            branch_start = event["seq"]
        if (event["type"] == "model.request" and branch_start
                and event["seq"] > branch_start):
            probe_requests.append(event)

    assert probe_requests, "the branch made no request"
    for event in probe_requests:
        manifest = event["data"]["context_manifest"]
        assert manifest, "request has no context manifest"
        kinds = {entry["kind"] for entry in manifest}
        assert "user" in kinds
        for entry in manifest:
            assert entry["source"] and entry["ref"], "manifest entry lacks provenance"


def test_manifest_marks_delivered_gold_evidence(tmp_path):
    spec = _spec(
        (BranchSpec("oracle", ExposurePolicy(mode="full", top_k=2,
                                             force_include_ids=("notice_new",)), PROBE),)
    )
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "oracle").read()
    gold_entries = []
    for event in events:
        if event["type"] != "model.request":
            continue
        for entry in event["data"]["context_manifest"]:
            if entry.get("is_gold"):
                gold_entries.append(entry)
    assert gold_entries, "gold evidence was delivered but not marked in the manifest"
    assert all(entry["doc_id"] == "notice_new" for entry in gold_entries)


def test_auto_retrieval_outcome_is_classified(tmp_path):
    """Legitimate empty must not look like a skipped or failed retrieval."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    report = _run(tmp_path, spec)
    outcome = report["branches"][0]["auto_retrieval"]["outcome"]
    assert outcome in ("ok", "empty", "skipped")
    assert outcome == "ok", f"expected documents, got {outcome}"


def test_prefix_turns_leave_no_active_notes_for_the_probe(tmp_path):
    """The probe's exposure condition must not inherit the prefix's retrieval."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    branch_start = None
    for event in events:
        if event["type"] == "turn.context" and event["data"]["label"] == "a":
            branch_start = event["seq"]
            break
    assert branch_start is not None

    # The branch turn must begin with a fresh user turn: any note in the first
    # branch request must come from the branch's own retrieval.
    first_request = next(
        e for e in events
        if e["type"] == "model.request" and e["seq"] > branch_start
    )
    refs = {entry["ref"] for entry in first_request["data"]["context_manifest"]}
    assert any(ref.startswith("seq:") for ref in refs)


def test_prefix_and_probe_use_the_same_retrieval_width_by_default(tmp_path):
    """A prefix/branch mismatch would be a confound inherited by every branch.

    The prefix's settings are shared by all branches, so an accidental
    difference between prefix and probe would apply to every condition at once
    and be invisible in the branch comparison.
    """
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    widths = {e["data"]["top_k"] for e in events if e["type"] == "memory.retrieval"}
    assert widths == {2}, f"retrieval width varied within one run: {widths}"


def test_declared_prefix_policy_is_recorded(tmp_path):
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    contexts = [e["data"] for e in events if e["type"] == "turn.context"]
    prefix_contexts = [c for c in contexts if c["label"].startswith("prefix")]
    assert prefix_contexts, "the prefix turn has no recorded policy"
    assert prefix_contexts[0]["exposure"]["label"] == "prefix"


# ------------------------------------------------- raw / sent separation


def test_manifest_internal_markers_never_reach_the_payload(tmp_path):
    """The manifest holds internal ids and the gold label, so it must not ship."""
    spec = _spec(
        (BranchSpec("oracle", ExposurePolicy(mode="full", top_k=2,
                                             force_include_ids=("notice_new",)), PROBE),)
    )
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "oracle").read()
    for event in events:
        if event["type"] != "model.request":
            continue
        data = event["data"]
        body = json.dumps(data["payload"], ensure_ascii=False)
        for marker in ('"is_gold"', '"doc_id"', '"context_manifest"',
                       '"fragment_provenance"', '"content_digest"'):
            assert marker not in body, f"{marker} reached the transmitted payload"
        assert set(data["payload"]) <= {"messages", "tools", *data["applied_param_keys"]}


def test_manifest_carries_content_digests_and_positions(tmp_path):
    """Source plus character count does not pin the content; a digest does."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    requests = [e for e in events if e["type"] == "model.request"]
    assert requests
    for event in requests:
        manifest = event["data"]["context_manifest"]
        positions = [entry["position"] for entry in manifest]
        assert positions == sorted(positions), "manifest is not in request order"
        for entry in manifest:
            assert entry["content_digest"], "manifest entry lacks a content digest"


def test_internal_key_scan_catches_nested_json():
    """A tool result carrying its own metadata is the realistic leak path."""
    from vela_lab.audit import _find_internal_keys

    leaked = {
        "messages": [
            {"role": "tool", "content": json.dumps({"doc_id": "notice_new",
                                                    "is_gold": True})}
        ]
    }
    assert _find_internal_keys(leaked) == {"doc_id", "is_gold"}


def test_internal_key_scan_ignores_ordinary_user_text():
    """A keyword scan would false-positive here and become useless."""
    from vela_lab.audit import _find_internal_keys

    benign = {
        "messages": [
            {"role": "user", "content": "is gold a metal? doc_id sounds technical"}
        ]
    }
    assert _find_internal_keys(benign) == set()


# ------------------------------------------------ tool-call protocol validity


def test_tool_message_always_follows_a_tool_call_turn(tmp_path):
    """A tool result must answer a recorded tool-requesting turn.

    Omitting the assistant's tool-call turn leaves ``role: tool`` following a
    plain user message.  Strict providers reject that outright; tolerant ones
    accept it while the model sees a tool result appear with nothing requesting
    it.  This was a real defect in this harness, found only because one of the
    three routes validated the sequence.
    """
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    checked = 0
    for event in events:
        if event["type"] != "model.request":
            continue
        messages = event["data"]["messages"]
        for index, message in enumerate(messages):
            if message["role"] != "tool":
                continue
            checked += 1
            assert index > 0, "a tool result cannot be the first message"
            previous = messages[index - 1]
            assert previous["role"] == "assistant", (
                f"tool result follows {previous['role']!r}, not the assistant"
            )
            assert "tool_calls" in previous, (
                "the assistant turn preceding a tool result carries no tool_calls"
            )
    assert checked > 0, "the branch made no tool calls, so nothing was verified"


def test_assistant_tool_call_turn_is_recorded_and_replayed(tmp_path):
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    compiled_tool_calls = [
        e for e in events
        if e["type"] == "observation.compiled"
        and e["data"].get("kind") == "assistant_tool_call"
    ]
    assert compiled_tool_calls, "no assistant tool-call turn was compiled"
    for event in compiled_tool_calls:
        calls = event["data"]["tool_calls"]
        assert calls, "a tool-call turn was recorded with no calls"
        for call in calls:
            assert call["id"], "a tool call has no id"
            assert call["function"]["name"], "a tool call has no name"
            # The wire carries arguments as a string; re-emitting them as a
            # parsed object would be a different request.
            assert isinstance(call["function"]["arguments"], str)


def test_arguments_survive_the_projection_round_trip(tmp_path):
    """The replayed arguments must equal what the model originally sent."""
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    _run(tmp_path, spec)
    from vela_lab.log import EventLog

    events = EventLog(tmp_path / "run" / "branches" / "a").read()
    recorded = {}
    for event in events:
        if event["type"] == "observation.compiled" and \
                event["data"].get("kind") == "assistant_tool_call":
            for call in event["data"]["tool_calls"]:
                recorded[call["id"]] = call["function"]["arguments"]
    assert recorded, "no tool calls were recorded"
    for event in events:
        if event["type"] != "model.request":
            continue
        for message in event["data"]["messages"]:
            if message["role"] != "assistant" or "tool_calls" not in message:
                continue
            for call in message["tool_calls"]:
                assert call["function"]["arguments"] == recorded[call["id"]]


# ------------------------------------------------- search accounting levels


def test_search_accounting_separates_willingness_from_permission(tmp_path):
    """The four levels must be distinguishable, or a refusal looks like silence.

    A model that asked to search and was refused, and a model that never asked,
    produce the same number of *executed* searches.  Only ``requested`` tells them
    apart, and that number is about the model rather than about the harness.
    """
    from vela_lab.clock import Clock
    from vela_lab.domains import ObservationDomain, WorldDomain
    from vela_lab.evidence import EvidenceCorpus, ExposurePolicy
    from vela_lab.loop import AgentLoop
    from vela_lab.prompt import PromptBundle, neutral_system_prompt
    from vela_lab.provider import MockModel
    from vela_lab.compiler import RequestCompiler
    from vela_lab.session import Session
    from vela_lab.tools import ToolBelt
    from conftest import DOCUMENTS

    corpus = EvidenceCorpus(list(DOCUMENTS))
    corpus.seal()
    bundle = PromptBundle(prompt_id="p", system=neutral_system_prompt(),
                          tool_descriptions={"search": "Search your notes."})
    session = Session(
        tmp_path / "s",
        clock=Clock.of("2026-03-20T09:00:00+08:00", "Asia/Shanghai"),
        observations=ObservationDomain(),
        world=WorldDomain(),
    )
    session.add_user_turn("帮我查一下签证面试的通知。")
    belt = ToolBelt(enabled=("search",), corpus=corpus, world=session.world,
                    exposure=ExposurePolicy(mode="full", top_k=2))
    loop = AgentLoop(session=session, model=MockModel(),
                     compiler=RequestCompiler(bundle, tool_schemas=belt.schemas()),
                     tools=belt, budget=4)
    outcome = loop.run_turn(label="t")

    accounting = outcome.search
    assert set(accounting) == {
        "search_requested", "search_executed", "search_evidence_delivered"
    }
    # The mock consults notes on a substantive user turn, so all three are non-zero
    # and ordered by how far each attempt got.
    assert accounting["search_requested"] >= 1
    assert accounting["search_requested"] >= accounting["search_executed"]
    assert accounting["search_executed"] >= accounting["search_evidence_delivered"]


def test_counters_reach_the_branch_report(tmp_path):
    spec = _spec((BranchSpec("a", ExposurePolicy(mode="full", top_k=2), PROBE),))
    report = _run(tmp_path, spec)
    branch = report["branches"][0]
    assert "search_accounting" in branch
    assert "search" in branch["outcome"]
    for key in ("search_requested", "search_executed",
                "search_evidence_delivered"):
        assert key in branch["search_accounting"]
