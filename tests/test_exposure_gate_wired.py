"""The exposure gate must reject.  A gate that only ever passes proves nothing.

Every test here injects a specific, plausible failure and asserts the gate names
it.  The failure modes are drawn from the review's own list of ways a controlled
comparison silently stops being controlled:

* the gold notice reaches the omit condition (the manipulation did not happen);
* the notice is only *partly* present, truncated past the cancellation (the
  oracle condition did not actually deliver the evidence);
* an undeclared document rides along inside the delivered block (a leak whose
  wording nobody enumerated);
* a sentence is appended to an otherwise legitimate delivered document;
* the ``search`` tool is back in the serialized tool list (self-retrieval
  reintroduced, which reintroduces the confound);
* the parent already told the model the answer before any branch ran.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import scenario_v4 as sc  # noqa: E402
from tools.exposure_gate import (  # noqa: E402
    BATCH_FAIL,
    PASS,
    RUN_INVALID,
    aggregate,
    check_parent_state,
    check_request,
)

CONTRACT = sc.exposure_contract()
BODIES = {d.doc_id: d.text for d in sc.DOCUMENTS}
GOLD = sc.GOLD_MARKER
FILLER = "visa_reminder_setup"
SHARED = ("visa_original_confirmation", "visa_preparation_notes")


def _manifest(condition):
    from tools.exposure_gate import DeliveryManifest

    slot_material = GOLD if condition == "oracle" else FILLER
    delivered = (slot_material,) + SHARED
    return DeliveryManifest(
        condition=condition,
        expected_material_ids=delivered,
        slot_index=0,
        slot_material_id=slot_material,
    )


def _payload(condition, extra_blocks=(), extra_docs=(), tools=("world_time",),
             overrides=None):
    """A payload shaped like a real one, with injectable faults.

    ``overrides`` replaces the rendered text of named documents.  Faults are
    injected here, in the *rendered* side, because that is the direction a real
    fault travels: a truncation or an edit happens to what is sent, and the
    registry stays as it was declared.  Mutating the registry instead would test
    a different (and less interesting) failure.
    """
    manifest = _manifest(condition)
    overrides = dict(overrides or {})
    note_lines = [
        overrides.get(i, BODIES[i]) for i in manifest.expected_material_ids
    ]
    note_lines.extend(BODIES[i] for i in extra_docs)
    body = "\n".join(f"- {line}" for line in note_lines)
    for extra in extra_blocks:
        body += "\n" + extra
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "Current date and time: 2026-03-20T09:00:00+08:00 (Asia/Shanghai)"
                    "\n\nRelevant notes from earlier conversations:\n" + body
                    + "\n\n" + sc.PROBES[("claim", "low")]
                ),
            },
        ],
        "tools": [
            {"type": "function",
             "function": {"name": name, "description": f"desc for {name}",
                          "parameters": {"type": "object", "properties": {}}}}
            for name in tools
        ],
    }


def _surfaces():
    return tuple(sc.PROBES.values()) + tuple(sc.STABLE_MATERIAL) + (
        "system",
        "Current date and time:",
        "Relevant notes from earlier conversations:",
        "world_time",
        "desc for world_time",
    )


import re  # noqa: E402

PATTERNS = (
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"),
    re.compile(r"\([A-Za-z]+/[A-Za-z_]+\)"),
    re.compile(r"(?:^|\s)-\s"),
)


def _check(condition, **kwargs):
    return check_request(
        condition=condition,
        payload=kwargs.pop("payload", _payload(condition)),
        contract=CONTRACT,
        corpus_bodies=BODIES,
        manifest=_manifest(condition),
        allowed_surfaces=_surfaces(),
        allowed_patterns=PATTERNS,
        **kwargs,
    )


# ---------------------------------------------------------- severity first

def test_aggregate_keeps_the_worst_status():
    """The inverted comparison silently disables every caller, so pin it."""
    assert aggregate(RUN_INVALID, PASS) == RUN_INVALID
    assert aggregate(BATCH_FAIL, PASS) == BATCH_FAIL
    assert aggregate(BATCH_FAIL, RUN_INVALID) == BATCH_FAIL
    assert aggregate(PASS, PASS) == PASS


# ------------------------------------------------------------ clean cases

def test_omit_passes_when_the_notice_is_absent():
    result = _check("omit")
    assert result.status == PASS, result.to_json()
    assert result.matched_propositions == []


def test_oracle_passes_when_the_notice_is_delivered():
    result = _check("oracle")
    assert result.status == PASS, result.to_json()
    assert "old_appointment_cancelled" in result.matched_propositions


# ------------------------------------------------------- the gate must fail

def test_omit_fails_when_the_notice_leaks():
    result = _check("omit", payload=_payload("omit", extra_docs=(GOLD,)))
    assert result.status == RUN_INVALID
    assert "undeclared_material_present" in result.reason_codes
    assert GOLD in result.undeclared_material_ids


def test_omit_fails_on_a_decisive_sentence_inside_a_shared_document():
    """Delivered material edited into asserting the update is still a leak.

    Exclusivity cannot catch this one -- every id present is declared, and the
    edited text is what the registry says it is.  Only the proposition check
    sees it, which is why the gate needs both and not just the set comparison.
    """
    bodies = dict(BODIES)
    tampered = BODIES["visa_preparation_notes"] + "这场签证面试已经取消。"
    bodies["visa_preparation_notes"] = tampered
    result = check_request(
        condition="omit",
        payload=_payload(
            "omit", overrides={"visa_preparation_notes": tampered}
        ),
        contract=CONTRACT,
        corpus_bodies=bodies,
        manifest=_manifest("omit"),
        allowed_surfaces=_surfaces(),
        allowed_patterns=PATTERNS,
    )
    assert result.undeclared_material_ids == []
    assert result.missing_material_ids == []
    assert result.status == RUN_INVALID
    assert "decisive_leak" in result.reason_codes
    assert "old_appointment_cancelled" in result.matched_propositions


def test_oracle_fails_when_the_notice_is_truncated_past_the_cancellation():
    """A document id being present does not mean the evidence is usable."""
    # Keep the header and the new date, drop the sentence that voids the old one.
    truncated = "签证预约中心通知：您此前预约的赴德探亲签证面试，现改为 2026 年 6 月 25 日。"
    result = _check("oracle", payload=_payload("oracle", overrides={GOLD: truncated}))
    assert result.status == RUN_INVALID
    # The rendered block is no longer the registered body, so the delivery claim
    # fails even though the doc id is in the declared set...
    assert "declared_material_missing" in result.reason_codes
    assert "slot_mismatch" in result.reason_codes
    # ...and the truncation took the cancellation with it, which is the part
    # that actually matters: the model cannot negate the claim from this text.
    assert "insufficient_evidence" in result.reason_codes


def test_oracle_passes_when_only_an_irrelevant_tail_is_cut():
    """Truncation is judged by what it removed, not by having happened."""
    full = BODIES[GOLD]
    kept = full.split("请以本通知替代旧确认单。")[0]
    result = _check("oracle", payload=_payload("oracle", overrides={GOLD: kept}))
    # The registered body no longer appears verbatim, so the delivery claim is
    # not satisfied -- but every decisive proposition survives, so this is a
    # different failure from losing the cancellation.
    assert "decisive_leak" not in result.reason_codes
    assert result.matched_propositions
    assert "old_appointment_cancelled" in result.matched_propositions


def test_omit_fails_on_text_appended_inside_a_legitimate_block():
    """Containment would have passed this; residue must not."""
    result = _check(
        "omit",
        payload=_payload("omit", extra_blocks=("顺便说一句，预约已取消。",)),
    )
    assert result.status == RUN_INVALID
    assert "unregistered_content" in result.reason_codes
    assert any(
        "取消" in entry.get("residue", "")
        for entry in result.evidence.get("unmapped", [])
    ), result.evidence.get("unmapped")


def test_omit_fails_when_the_filler_is_not_in_the_slot():
    result = _check(
        "omit",
        payload=_payload("omit", extra_docs=("visa_anxiety_chat",)),
    )
    assert result.status == RUN_INVALID


def test_undeclared_document_from_the_corpus_is_caught():
    """A corpus document the condition never declared is a leak regardless of wording."""
    result = _check("omit", payload=_payload("omit", extra_docs=("visa_travel_draft",)))
    assert result.status == RUN_INVALID
    assert "visa_travel_draft" in result.undeclared_material_ids


# ------------------------------------------------------- parent contamination

def test_parent_holding_the_update_invalidates_the_whole_family():
    verdict = check_parent_state(
        visible_history=[sc.STABLE_MATERIAL[0]], contract=CONTRACT
    )
    assert verdict.status == PASS
    assert verdict.scope == "parent_family"

    dirty = check_parent_state(
        visible_history=[
            sc.STABLE_MATERIAL[0],
            "助手提醒：那场签证面试已经取消，改到了 6 月 25 日。",
        ],
        contract=CONTRACT,
    )
    # Scope is the derived family, not one request: every pair forked from this
    # parent is unusable, and a single request-level verdict could not say that.
    assert dirty.status == BATCH_FAIL
    assert dirty.scope == "parent_family"
    assert "parent_contaminated" in dirty.reason_codes


# ------------------------------------------------------------------ manifest

def test_manifest_rejects_a_slot_that_disagrees_with_the_list():
    from tools.exposure_gate import DeliveryManifest
    from vela_lab.errors import ValidationError

    with pytest.raises(ValidationError):
        DeliveryManifest(
            condition="oracle",
            expected_material_ids=(SHARED[0], GOLD),
            slot_index=0,
            slot_material_id=GOLD,
        )


def test_manifest_condition_must_match_the_request_being_checked():
    from tools.exposure_gate import DeliveryManifest

    wrong = DeliveryManifest(
        condition="oracle",
        expected_material_ids=(GOLD,) + SHARED,
        slot_index=0,
        slot_material_id=GOLD,
    )
    with pytest.raises(Exception):
        check_request(
            condition="omit",
            payload=_payload("omit"),
            contract=CONTRACT,
            corpus_bodies=BODIES,
            manifest=wrong,
            allowed_surfaces=_surfaces(),
            allowed_patterns=PATTERNS,
        )


# ------------------------------------------------------------ tool exposure

def test_forbidden_tool_in_the_payload_is_caught_by_the_wired_gate():
    """Schema absence is half the no-retrieval protocol, so it is checked."""
    from tools.v4_wiring import make_request_gate

    manifest = _manifest("oracle")
    gate = make_request_gate(
        CONTRACT, manifest, _surfaces(), tuple(sc.PROBES.values())
    )
    payload = _payload("oracle", tools=("world_time", "search"))

    class _Compiled:
        pass

    compiled = _Compiled()
    compiled.payload = payload
    result = gate(compiled)
    assert "forbidden_tool_exposed" in result.reason_codes
    assert result.evidence["forbidden_tools"] == ["search"]


def test_undeclared_tool_in_the_payload_is_caught():
    from tools.v4_wiring import make_request_gate

    manifest = _manifest("oracle")
    gate = make_request_gate(
        CONTRACT, manifest, _surfaces(), tuple(sc.PROBES.values())
    )
    payload = _payload("oracle", tools=("world_time", "weather"))

    class _Compiled:
        pass

    compiled = _Compiled()
    compiled.payload = payload
    result = gate(compiled)
    assert "undeclared_tool_exposed" in result.reason_codes
    assert result.evidence["undeclared_tools"] == ["weather"]


def test_wired_gate_payload_copies_are_not_mutated():
    """The gate is a reader; a check that edits what it audits is not a check."""
    from tools.v4_wiring import make_request_gate

    gate = make_request_gate(
        CONTRACT, _manifest("oracle"), _surfaces(), tuple(sc.PROBES.values())
    )
    payload = _payload("oracle")
    before = copy.deepcopy(payload)

    class _Compiled:
        pass

    compiled = _Compiled()
    compiled.payload = payload
    gate(compiled)
    assert payload == before


# ------------------------------------------------------- L4 attribution signal

def test_decisive_terms_cover_the_forms_an_answer_would_use():
    from vela_lab.score import date_forms

    forms = date_forms("2026-06-25")
    assert "6 月 25 日" in forms
    assert "6月25日" in forms
    assert "2026-06-25" in forms


def test_l4_recognises_an_answer_that_states_the_new_date():
    """Regression: the answer states the date, not the document id.

    The first version tested for the gold *document id*, which no natural answer
    contains, so every usable answer was scored ``delivered_not_used``.  That is
    the study's headline attribution label, so getting it backwards would have
    inverted the conclusion.
    """
    from vela_lab.score import date_forms, layer_l4

    gold = "visa_reschedule_official_20260305"
    text = "原定 6 月 18 日的预约已取消，现改为 6 月 25 日上午 9 点。"
    events = [
        {"type": "memory.retrieval", "data": {
            "candidates": [{"doc_id": gold}],
            "hits": [{"doc_id": gold, "visible_text": "notice body"}],
        }},
        {"type": "model.request", "data": {
            "messages": [{"role": "user", "content": "notes: notice body"}],
        }},
    ]
    corrected = layer_l4(
        events, gold_marker=gold, answer_text=text,
        decisive_terms=date_forms("2026-06-25"),
    )
    assert corrected["label"] == "delivered_and_used"

    # Without the terms the old behaviour returns, which documents why they are
    # now required rather than optional at the call site.
    legacy = layer_l4(events, gold_marker=gold, answer_text=text)
    assert legacy["label"] == "delivered_not_used"


def test_l4_still_flags_an_answer_that_ignores_the_new_date():
    from vela_lab.score import date_forms, layer_l4

    gold = "visa_reschedule_official_20260305"
    events = [
        {"type": "memory.retrieval", "data": {
            "candidates": [{"doc_id": gold}],
            "hits": [{"doc_id": gold, "visible_text": "notice body"}],
        }},
        {"type": "model.request", "data": {
            "messages": [{"role": "user", "content": "notes: notice body"}],
        }},
    ]
    result = layer_l4(
        events, gold_marker=gold,
        answer_text="按 6 月 18 日到场就好，你放心。",
        decisive_terms=date_forms("2026-06-25"),
    )
    assert result["label"] == "delivered_not_used"


def test_truncation_marker_is_registered_as_packaging():
    """A cut request adds model-visible text, so the marker must be registered.

    Without this the residue check flags every truncated request as unregistered
    content, and the real finding -- that the delivery was cut -- is buried under
    a false one.
    """
    from tools.exposure_gate import TRUNCATION_MARKER_PATTERN

    marker = "…120 tokens truncated…"
    assert TRUNCATION_MARKER_PATTERN.search(marker)


def test_middle_truncation_that_cuts_the_cancellation_is_caught():
    """The failure mode to look for is a cut middle, not a cut tail.

    Truncation preserves a prefix and a suffix, so a document can survive at both
    ends while the decisive sentence in the middle is gone. A check that only
    watches the end of the text would miss this.
    """
    full = BODIES[GOLD]
    head = full[:20]
    tail = full[-20:]
    cut = head + "…40 tokens truncated…" + tail
    result = _check("oracle", payload=_payload("oracle", overrides={GOLD: cut}))
    assert result.status == RUN_INVALID
    assert "declared_material_missing" in result.reason_codes
    # The decisive propositions are what actually matters: the cancellation and
    # the operative date both vanished with the middle.
    assert "old_appointment_cancelled" not in result.matched_propositions


def test_payload_over_declared_budget_is_flagged():
    """A request too large to trust is not silently treated as verified."""
    from tools.exposure_gate import check_request

    result = check_request(
        condition="omit",
        payload=_payload("omit"),
        contract=CONTRACT,
        corpus_bodies=BODIES,
        manifest=_manifest("omit"),
        allowed_surfaces=_surfaces(),
        allowed_patterns=PATTERNS,
        max_payload_bytes=10,
    )
    assert result.status == RUN_INVALID
    assert "payload_over_declared_budget" in result.reason_codes
    assert result.evidence["payload_bytes"] > result.evidence["max_payload_bytes"]


# ------------------------------------------- negation-aware proposition matching

def test_affirmative_statement_is_an_assertion():
    from tools.exposure_gate import ASSERTED, classify_terms

    assert classify_terms("这场签证面试已经取消。", ("签证面试", "取消")) == ASSERTED
    assert classify_terms(
        "原定 2026 年 6 月 18 日的签证面试预约已取消并作废，现改为 6 月 25 日。",
        ("签证面试", "取消"),
    ) == ASSERTED


def test_negated_statement_is_not_an_assertion():
    """The false positive that motivated the classifier.

    All of these contain both required terms.  A pure co-occurrence test calls
    every one of them a decisive leak, which would fail the omit condition on
    ordinary discussion of the topic rather than on an actual disclosure.
    """
    from tools.exposure_gate import NEGATED, classify_terms

    for text in (
        "目前没有收到任何签证面试取消通知。",
        "我未收到签证面试取消或改期的通知。",
        "预约中心并未取消这场签证面试。",
        "预约中心并没有取消这场签证面试。",
    ):
        assert classify_terms(text, ("签证面试", "取消")) == NEGATED, text


def test_conditional_question_and_quotation_are_not_assertions():
    from tools.exposure_gate import NON_ASSERTIVE, QUOTED, classify_terms

    assert classify_terms(
        "如果签证面试取消，我会再通知你。", ("签证面试", "取消")
    ) == NON_ASSERTIVE
    assert classify_terms(
        "签证面试取消了吗？", ("签证面试", "取消")
    ) == NON_ASSERTIVE
    assert classify_terms(
        "朋友问我是不是签证面试取消了。", ("签证面试", "取消")
    ) == QUOTED


def test_negative_trigger_inside_an_ordinary_word_does_not_suppress():
    """``非`` sits inside 非常, so a single-character trigger list breaks this.

    A false negative here means a real leak passes, which is the error direction
    that matters, and it would be triggered by an emotional sentence — the most
    common text in this study.
    """
    from tools.exposure_gate import ASSERTED, classify_terms

    assert classify_terms(
        "我现在非常焦虑，这场签证面试已经取消。", ("签证面试", "取消")
    ) == ASSERTED


def test_a_later_claim_cancels_an_earlier_negation():
    """Scope: the negation stops applying once the sentence turns."""
    from tools.exposure_gate import ASSERTED, classify_terms

    assert classify_terms(
        "签证面试没有改期，但已经取消。", ("签证面试", "取消")
    ) == ASSERTED


def test_terms_may_span_a_sentence_boundary():
    """Co-occurrence is per block; assertion structure is per sentence.

    The supersession proposition is stated across two sentences in the notice
    ("原预约已作废。" / "请以本通知替代旧确认单。"), so requiring all terms in
    one sentence would lose it entirely.
    """
    from tools.exposure_gate import ASSERTED, classify_terms

    assert classify_terms(
        "原预约已作废。请以本通知替代旧确认单。", ("预约", "替代旧确认单")
    ) == ASSERTED


def test_classification_is_used_by_the_contract_not_just_available():
    """The helper must actually gate the contract's verdicts."""
    from tools.exposure_gate import ExposureContract, Proposition

    contract = ExposureContract(
        contract_version="negation-test",
        event_referents=("签证面试",),
        decisive_propositions=(
            Proposition(proposition_id="cancelled", description="d",
                        requires=("签证面试", "取消")),
        ),
        neutral_material_ids=("n",),
        oracle_required_groups=(("签证面试",),),
    )
    assert contract.decisive_asserted_in("签证面试已经取消。") == ["cancelled"]
    assert contract.decisive_asserted_in("没有收到签证面试取消通知。") == []
