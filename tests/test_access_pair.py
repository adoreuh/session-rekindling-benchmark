"""Controlled key-evidence delivery, and the confounds it has to exclude.

Design B replaces the natural/oracle contrast, which could not be constructed here:
the corpus holds exactly one document about the reschedule, so reachability and
topical aiming are inseparable.  The replacement delivers an identical shared
context with **one registered slot** differing — gold in the oracle condition, a
neutral filler in omit.

The tests below check the properties that make that comparison mean something, and
each corresponds to a way the pair could silently stop isolating delivery:

* the two lists differ outside the slot, or have different lengths;
* the filler asserts a date, making it misleading rather than neutral evidence;
* the filler is much shorter or longer, so "evidence supplied" is confounded with
  "more text supplied";
* the gold is already in the shared context, so the slot is not what makes it
  available;
* the slot index is outside the context, so the substitution would not land.
"""

from __future__ import annotations

import pytest

from tools.access_pair import (
    AccessPair,
    build_access_pair,
    exposure_for_condition,
    policies_for_pair,
    select_filler,
)
from vela_lab.errors import ValidationError

AVAILABLE = ("a", "b", "c", "d", "gold", "filler")


def _pair(**overrides) -> AccessPair:
    fields = dict(
        shared_ids=("a", "b"),
        gold_id="gold",
        filler_id="filler",
        slot=1,
        available_ids=AVAILABLE,
    )
    fields.update(overrides)
    return build_access_pair(**fields)


# ------------------------------------------------------------- construction


def test_pair_differs_only_in_the_slot():
    pair = _pair()
    assert pair.oracle_doc_ids == ("a", "gold", "b")
    assert pair.omit_doc_ids == ("a", "filler", "b")
    assert pair.differs_only_in_the_slot() is True


def test_both_conditions_have_the_same_document_count():
    pair = _pair()
    assert len(pair.oracle_doc_ids) == len(pair.omit_doc_ids)


def test_shared_documents_keep_their_positions():
    pair = _pair(slot=2)
    assert pair.oracle_doc_ids[:2] == pair.omit_doc_ids[:2]
    assert pair.oracle_doc_ids[3:] == pair.omit_doc_ids[3:]


def test_slot_zero_and_the_end_are_both_legal():
    assert _pair(slot=0).oracle_doc_ids[0] == "gold"
    assert _pair(slot=2).oracle_doc_ids[2] == "gold"


# --------------------------------------------------------- rejected inputs


def test_filler_equal_to_gold_is_refused():
    """Identical delivery would make the contrast empty."""
    with pytest.raises(ValidationError, match="must differ from gold_id"):
        _pair(filler_id="gold")


def test_duplicate_shared_ids_are_refused():
    with pytest.raises(ValidationError, match="duplicate document"):
        _pair(shared_ids=("a", "a"))


def test_gold_already_shared_is_refused():
    """If gold is already there, the slot is not what makes it available."""
    with pytest.raises(ValidationError, match="already in the shared context"):
        _pair(shared_ids=("a", "gold"))


def test_filler_already_shared_is_refused():
    with pytest.raises(ValidationError, match="filler is already in the shared"):
        _pair(shared_ids=("a", "filler"))


def test_unknown_document_is_refused():
    with pytest.raises(ValidationError, match="missing"):
        _pair(shared_ids=("a", "nope"))


@pytest.mark.parametrize("slot", [-1, 3, 99])
def test_slot_outside_the_context_is_refused(slot):
    with pytest.raises(ValidationError, match="outside the shared context"):
        _pair(slot=slot)


def test_non_integer_slot_is_refused():
    with pytest.raises(ValidationError, match="expected an integer"):
        _pair(slot=1.5)


# ------------------------------------------------------------ filler choice

DOCUMENTS = {
    "gold": "预约变更通知：原定 6 月 18 日的安排已取消，现改为 6 月 25 日。" * 3,
    "with_date": "我记着是 6 月 18 日那天，别忘了。",
    "neutral_close": "用户整理了几条日常提醒，设在了手机备忘录里，不需要联网。" * 3,
    "neutral_far": "短。",
    "neutral_medium": "用户在周末整理了一下书桌，把文件按类别分开放好。" * 2,
}


def test_filler_must_not_assert_a_date():
    """Naming a date makes it misleading evidence, not neutral."""
    filler = select_filler(
        gold_id="gold", documents=DOCUMENTS,
        slot_pool=["with_date", "neutral_close"],
    )
    assert filler == "neutral_close"


def test_filler_selection_prefers_the_closest_length():
    """Otherwise the oracle slot is confounded with more text."""
    filler = select_filler(
        gold_id="gold", documents=DOCUMENTS,
        slot_pool=["neutral_far", "neutral_close"],
    )
    assert filler == "neutral_close"


def test_filler_selection_raises_when_nothing_qualifies():
    """Refusing beats relaxing: a dated filler changes the condition's meaning."""
    with pytest.raises(ValidationError, match="no registered candidate qualifies"):
        select_filler(gold_id="gold", documents=DOCUMENTS, slot_pool=["with_date"])


def test_filler_length_ratio_is_within_tolerance():
    filler = select_filler(gold_id="gold", documents=DOCUMENTS,
                           slot_pool=list(DOCUMENTS))
    ratio = max(len(DOCUMENTS[filler]), len(DOCUMENTS["gold"])) / min(
        len(DOCUMENTS[filler]), len(DOCUMENTS["gold"])
    )
    assert ratio <= 3.0, f"selected filler ratio {ratio}"


# -------------------------------------------------------------- exposure


def test_exposure_delivers_exactly_the_registered_list():
    """The delivered list is stated outright, in order.

    `force_include_ids` only promotes within a ranking, so it fixes the set but not
    the order, and the slot's position would then differ between conditions.  The
    ordered path is what makes "fixed slot" expressible.
    """
    pair = _pair()
    for condition, expected in (("oracle", pair.oracle_doc_ids),
                                ("omit", pair.omit_doc_ids)):
        policy = exposure_for_condition(condition, doc_ids=expected, gold_id="gold")
        assert tuple(policy.deliver_ids) == expected
        assert policy.top_k == len(expected)


def test_ordered_delivery_and_force_include_cannot_be_combined():
    """They mean different things, so combining them has no defined behaviour."""
    from vela_lab.evidence import ExposurePolicy

    with pytest.raises(ValidationError, match="mutually exclusive"):
        ExposurePolicy(deliver_ids=("a",), force_include_ids=("a",))


def test_omit_excludes_the_gold_outright():
    """It cannot appear once the count is filled, but exclusion makes that explicit."""
    policy = exposure_for_condition("omit", doc_ids=("a", "filler"), gold_id="gold")
    assert "gold" in policy.omit_ids


def test_oracle_must_deliver_the_gold():
    with pytest.raises(ValidationError, match="must deliver the gold"):
        exposure_for_condition("oracle", doc_ids=("a", "b"), gold_id="gold")


def test_omit_must_not_deliver_the_gold():
    with pytest.raises(ValidationError, match="must not deliver the gold"):
        exposure_for_condition("omit", doc_ids=("a", "gold"), gold_id="gold")


def test_unknown_condition_is_refused():
    with pytest.raises(ValidationError, match="unknown condition"):
        exposure_for_condition("natural", doc_ids=("a",), gold_id="gold")


def test_empty_delivered_set_is_refused():
    with pytest.raises(ValidationError, match="empty"):
        exposure_for_condition("omit", doc_ids=(), gold_id="gold")


# ------------------------------------------------------- integration facts


def _scenario_corpus():
    import tools.scenario_v4 as sc
    from tools.scenario_loader import _to_document

    docs = {
        d.doc_id: _to_document(
            {"doc_id": d.doc_id, "title": getattr(d, "title", None), "text": d.text,
             "published_at": d.published_at, "source": getattr(d, "source", None)},
            gold_marker=sc.GOLD_MARKER,
        )
        for d in sc.DOCUMENTS
    }
    return sc, docs


def test_a_real_pair_delivers_balanced_text():
    """End-to-end on the scenario corpus: the two slots are comparable in size.

    The oracle slot holds the gold notice; if the filler were much shorter, the
    oracle condition would receive both the evidence and more text, and the
    comparison would measure context volume as well as delivery.
    """
    sc, docs = _scenario_corpus()
    texts = {k: v.text for k, v in docs.items()}
    filler = select_filler(gold_id=sc.GOLD_MARKER, documents=texts,
                           slot_pool=[d for d in docs if d != sc.GOLD_MARKER])
    shared = ("visa_original_confirmation", "visa_preparation_notes",
              "visa_material_checklist")
    pair = build_access_pair(shared_ids=shared, gold_id=sc.GOLD_MARKER,
                             filler_id=filler, slot=1, available_ids=list(docs))
    assert pair.differs_only_in_the_slot()
    oracle_total = sum(len(texts[i]) for i in pair.oracle_doc_ids)
    omit_total = sum(len(texts[i]) for i in pair.omit_doc_ids)
    ratio = oracle_total / omit_total
    assert 0.8 <= ratio <= 1.25, (
        f"delivered text differs by {ratio:.2f}x, so the oracle condition would "
        "receive more text as well as the evidence"
    )


def test_the_scenario_has_a_filler_that_asserts_no_date():
    """The scenario must provide one, or design B cannot be run at all."""
    sc, docs = _scenario_corpus()
    texts = {k: v.text for k, v in docs.items()}
    pool = [d for d in docs if d != sc.GOLD_MARKER]
    filler = select_filler(gold_id=sc.GOLD_MARKER, documents=texts, slot_pool=pool)
    assert filler != sc.GOLD_MARKER
    for hint in ("6 月", "18", "25", "改期", "取消", "变更", "作废"):
        assert hint not in texts[filler], (
            f"filler {filler!r} contains {hint!r}, so it is not neutral evidence"
        )


def test_the_delivered_slot_is_at_the_same_position_in_both_conditions():
    """The property that was broken: set-equality is not position-equality.

    Before ordered delivery existed, `force_include_ids` promoted the forced
    documents while preserving their natural ranking, so the gold landed at
    position 0 in oracle and the filler at position 3 in omit.  The two conditions
    then differed in the evidence's *location* as well as its content.
    """
    from datetime import datetime, timedelta

    from vela_lab.evidence import EvidenceCorpus

    sc, docs = _scenario_corpus()
    corpus = EvidenceCorpus(list(docs.values()))
    corpus.seal()
    texts = {k: v.text for k, v in docs.items()}
    filler = select_filler(gold_id=sc.GOLD_MARKER, documents=texts,
                           slot_pool=[d for d in docs if d != sc.GOLD_MARKER])
    shared = ("visa_original_confirmation", "visa_preparation_notes",
              "visa_material_checklist")
    pair = build_access_pair(shared_ids=shared, gold_id=sc.GOLD_MARKER,
                             filler_id=filler, slot=1, available_ids=list(docs))
    now = (datetime.fromisoformat(sc.PREFIX_START) + timedelta(days=19)).isoformat()

    positions = {}
    counts = {}
    for condition, expected in (("oracle", pair.oracle_doc_ids),
                                ("omit", pair.omit_doc_ids)):
        policy = exposure_for_condition(condition, doc_ids=expected,
                                        gold_id=sc.GOLD_MARKER)
        record = corpus.retrieve(
            query=sc.HOST_QUERY, now_instant=now, top_k=policy.top_k, mode="probe",
            deliver_ids=policy.deliver_ids, omit_ids=policy.omit_ids,
        )
        delivered = [h["doc_id"] for h in record.hits]
        assert delivered == list(expected), (
            f"{condition}: delivered {delivered}, registered {list(expected)}"
        )
        marker = sc.GOLD_MARKER if condition == "oracle" else filler
        positions[condition] = delivered.index(marker)
        counts[condition] = len(delivered)

    assert positions["oracle"] == positions["omit"] == pair.slot, positions
    assert counts["oracle"] == counts["omit"], counts


def test_delivery_refuses_a_document_not_yet_published():
    """Naming a document does not override the corpus's availability rules."""
    from datetime import datetime

    from vela_lab.evidence import EvidenceCorpus

    sc, docs = _scenario_corpus()
    corpus = EvidenceCorpus(list(docs.values()))
    corpus.seal()
    before_update = datetime.fromisoformat(sc.PREFIX_START).isoformat()
    with pytest.raises(Exception) as excinfo:
        corpus.retrieve(
            query=sc.HOST_QUERY, now_instant=before_update, top_k=2, mode="probe",
            deliver_ids=(sc.GOLD_MARKER, "visa_original_confirmation"),
        )
    assert "not deliverable" in str(excinfo.value)


def test_policies_for_pair_returns_both_conditions():
    pair = _pair()
    policies = policies_for_pair(pair)
    assert set(policies) == {"oracle", "omit"}
    assert tuple(policies["oracle"].deliver_ids) == pair.oracle_doc_ids
    assert tuple(policies["omit"].deliver_ids) == pair.omit_doc_ids
