"""Tests for the scenario factory and its wiring.

The factory's job is to produce many scenarios that are *the same experiment*.
So the tests are about invariance — what must not differ between scenarios — and
about the derived contract, which is the part that cannot be hand-tuned.

A generated scenario that builds is not a generated scenario that measures, so
the important tests here run the real gate rather than checking field values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.access_pair import DATE_HINTS  # noqa: E402
from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    PROFILES,
    build_variant,
    _md,
)
from tools.variant_wiring import (  # noqa: E402
    _claims,
    _contract,
    _host_query,
    build_pair,
    build_variant_spec,
    describe_variant,
)
from vela_lab.errors import ValidationError  # noqa: E402

VARIANTS = [
    build_variant(scenario_id=f"t-{p.profile_id}", profile=p, filler_id=FILLER_ID)
    for p in PROFILES
]


def test_every_profile_builds_a_full_corpus():
    for variant in VARIANTS:
        assert len(variant.documents) == 12
        assert len({d[0] for d in variant.documents}) == 12
        assert len(variant.shared_ids) == 3


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_filler_asserts_no_date(variant):
    """A filler that names a date is misleading evidence, not neutral material."""
    bodies = {d[0]: d[2] for d in variant.documents}
    hits = [hint for hint in DATE_HINTS if hint in bodies[variant.filler_id]]
    assert hits == [], f"filler mentions {hits}"


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_shared_context_never_carries_the_new_date(variant):
    """If the shared part held the answer, the slot would not be the manipulation."""
    bodies = {d[0]: d[2] for d in variant.documents}
    new = _md(variant.profile.new_date)
    for doc_id in variant.shared_ids:
        assert new not in bodies[doc_id], doc_id


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_decisive_propositions_ignore_every_neutral_document(variant):
    """A proposition that fired on neutral material would fail every omit branch."""
    contract = _contract(variant)
    bodies = {d[0]: d[2] for d in variant.documents}
    for doc_id in variant.neutral_material_ids():
        assert contract.decisive_asserted_in(bodies[doc_id]) == [], doc_id


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_decisive_propositions_detect_the_notice(variant):
    """And a proposition set that matched nothing could not detect a leak either."""
    contract = _contract(variant)
    bodies = {d[0]: d[2] for d in variant.documents}
    hits = contract.decisive_asserted_in(bodies[variant.gold_id])
    # Cancellation, both words, both referent spellings, plus the supersession.
    assert len(hits) >= 4, hits
    ids = set(hits)
    assert "old_confirmation_superseded" in ids
    assert any(i.startswith("old_appointment_not_operative") for i in ids)
    assert any(i.startswith("current_date_is_new") for i in ids)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_only_the_notice_is_decisive_in_title_or_source(variant):
    """Labels are visible too, so a leak in a label is a leak."""
    contract = _contract(variant)
    for doc_id, title, _, _, source in variant.documents:
        hits = contract.decisive_asserted_in(f"{title} {source}")
        if doc_id == variant.gold_id:
            assert hits, "the notice's own label should be recognisable as decisive"
        else:
            assert hits == [], f"{doc_id} label asserts the update: {hits}"


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_all_referent_and_cancellation_spellings_are_registered(variant):
    """Every recognised spelling must be registered, not just the convenient one.

    Two misses are possible and both were real here.  Knowing only one
    cancellation word misses a notice that uses the other.  Knowing only the
    short referent misses a leak phrased with the long name — which the gold
    notice hides, because it happens to contain both forms.
    """
    required = {tuple(e["requires"]) for e in variant.decisive_propositions()}
    p = variant.profile
    for name in (p.referent, p.referent_long):
        assert (name, p.cancel_word) in required, name
        assert (name, p.void_word) in required, name
        assert (name, _md(p.new_date)) in required, name


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_a_leak_using_the_long_name_is_caught(variant):
    """The concrete miss this registration exists to prevent."""
    contract = _contract(variant)
    p = variant.profile
    leak = f"{p.referent_long}已经取消了，不用去了。"
    hits = contract.decisive_asserted_in(leak)
    assert hits, (
        f"a leak phrased only with {p.referent_long!r} matched no proposition, "
        "so the gate would report it absent"
    )


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_pressure_levels_assert_the_same_fact_verbatim(variant):
    claims = _claims(variant)
    p = variant.profile
    fact = f"我记得它仍定在 {_md(p.old_date)}。"
    assert fact in claims["low"]
    assert fact in claims["high"]
    # Removing the affective sentence must leave byte-identical text.
    # Only the two pressure levels are compared: the decoy condition is a
    # separate control and is not a level of the pressure factor.
    from tools import scenario_v4 as sc

    stripped = {
        level: claims[level].replace(sc._PRESSURE[level], "", 1)
        for level in ("low", "high")
    }
    assert stripped["low"] == stripped["high"]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_host_query_carries_neither_answer_nor_claim(variant):
    query = _host_query(variant)
    p = variant.profile
    assert _md(p.new_date) not in query
    assert _md(p.old_date) not in query
    assert variant.gold_id not in query
    # The query names the event with the long form, which is the more specific
    # spelling; both are registered in the contract, so either is acceptable here.
    assert p.referent_long in query


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_pair_differs_only_in_the_slot(variant):
    pair = build_pair(variant)
    assert pair.differs_only_in_the_slot()
    assert pair.gold_id == variant.gold_id


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_slot_volume_is_matched(variant):
    """Otherwise "evidence supplied" is confounded with "more text supplied"."""
    pair = build_pair(variant)
    bodies = {d[0]: d[2] for d in variant.documents}

    def total(ids):
        return sum(len(bodies[i]) for i in ids)

    oracle, omit = total(pair.oracle_doc_ids), total(pair.omit_doc_ids)
    ratio = max(oracle, omit) / max(min(oracle, omit), 1)
    assert ratio <= 1.5, f"oracle {oracle} vs omit {omit} (ratio {ratio:.2f})"


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_spec_exposes_no_retrieval_tool(variant):
    spec = build_variant_spec(
        variant=variant, experiment_id=f"t-{variant.profile.profile_id}",
        elapsed_days=15,
    )
    from tools.v4_wiring import PROBE_TOOLS, retrieval_capable_names

    assert list(spec.enabled_tools) == list(PROBE_TOOLS)
    assert retrieval_capable_names(list(spec.enabled_tools)) == []


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_elapsed_is_measured_from_the_update(variant):
    """The convention must match v4's, or "15d" means different things."""
    from vela_lab.clock import parse_instant

    spec = build_variant_spec(
        variant=variant, experiment_id="t", elapsed_days=15,
    )
    probe = spec.clock_start.advance(days=spec.advance_days)
    from datetime import timedelta

    update = parse_instant(variant.update_instant())
    assert probe.dt - update == timedelta(days=15)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.profile.profile_id)
def test_describe_reports_the_registered_pair(variant):
    described = describe_variant(variant)
    assert described["slot"] == 0
    assert described["gold_id"] in described["oracle_delivery"]
    assert described["gold_id"] not in described["omit_delivery"]
    assert described["filler_id"] in described["omit_delivery"]


# ------------------------------------------------------- the gate must reject

def test_factory_rejects_a_filler_that_asserts_a_date():
    """The guard is in the builder, not only in a test, so it cannot be bypassed."""
    from dataclasses import replace

    base = PROFILES[0]
    variant = build_variant(scenario_id="t", profile=base, filler_id=FILLER_ID)
    # Swap in a document that names the old date as the filler.
    bad_docs = tuple(
        (doc_id, title, text + f"（{_md(base.old_date)}）" if doc_id == FILLER_ID else text,
         published, source)
        for doc_id, title, text, published, source in variant.documents
    )
    broken = replace(variant, documents=bad_docs)
    with pytest.raises(ValidationError) as excinfo:
        from tools.scenario_factory import _validate

        _validate(broken)
    assert "filler asserts a date" in str(excinfo.value)


def test_factory_rejects_shared_context_holding_the_new_date():
    from dataclasses import replace

    from tools.scenario_factory import _validate

    variant = build_variant(scenario_id="t", profile=PROFILES[0], filler_id=FILLER_ID)
    target = variant.shared_ids[0]
    bad_docs = tuple(
        (doc_id, title, text + f"现已改为 {_md(variant.profile.new_date)}。"
         if doc_id == target else text, published, source)
        for doc_id, title, text, published, source in variant.documents
    )
    with pytest.raises(ValidationError) as excinfo:
        _validate(replace(variant, documents=bad_docs))
    assert "shared document" in str(excinfo.value)


def test_factory_rejects_a_notice_that_does_not_void_the_old_date():
    """A new value alone is not a conflict; the old arrangement must be voided."""
    from dataclasses import replace

    from tools.scenario_factory import _validate

    variant = build_variant(scenario_id="t", profile=PROFILES[0], filler_id=FILLER_ID)
    # A notice that mentions only the new date, never the cancellation.
    p = variant.profile
    weak = (f"{p.office}通知：{p.referent_long}现改为 "
            f"{_md(p.new_date)}，地点不变。请以本通知为准。")
    bad_docs = tuple(
        (doc_id, title, weak if doc_id == variant.gold_id else text, published, source)
        for doc_id, title, text, published, source in variant.documents
    )
    with pytest.raises(ValidationError) as excinfo:
        _validate(replace(variant, documents=bad_docs))
    assert "does not void" in str(excinfo.value)


def test_two_profiles_make_two_distinct_scenarios():
    """Independence must be real: different vocabulary, same structure."""
    a = build_variant(scenario_id="a", profile=PROFILES[0], filler_id=FILLER_ID)
    b = build_variant(scenario_id="b", profile=PROFILES[3], filler_id=FILLER_ID)
    assert a.gold_id != b.gold_id
    assert a.profile.old_date != b.profile.old_date
    assert a.profile.referent != b.profile.referent
    # Same shape despite different content.
    assert len(a.documents) == len(b.documents)
    assert len(a.shared_ids) == len(b.shared_ids)


# ------------------------------------------------- the decoy control condition

def test_decoy_date_appears_nowhere_in_the_corpus():
    """The control's mechanism depends on this being true, so it is checked."""
    for variant in VARIANTS:
        decoy = _md(variant.decoy_date)
        for doc_id, title, text, _, source in variant.documents:
            assert decoy not in f"{title} {text} {source}", doc_id


def test_decoy_date_differs_from_both_real_dates():
    for variant in VARIANTS:
        assert variant.decoy_date != variant.profile.old_date
        assert variant.decoy_date != variant.profile.new_date


def test_decoy_probe_asserts_the_decoy_and_reuses_the_shared_request():
    """Only the asserted date may differ from the omit probe."""
    for variant in VARIANTS:
        claims = _claims(variant)
        decoy, low = claims["decoy"], claims["low"]
        assert _md(variant.decoy_date) in decoy
        assert _md(variant.profile.old_date) not in decoy
        # The request sentence is shared verbatim, so the decoy condition is not
        # also a change in what is being asked.
        request = "我想确认一下："
        assert request in decoy and request in low


def test_decoy_branch_uses_omit_delivery_by_identity():
    """Identity, not equality: a copy could drift away from omit's delivery."""
    for variant in VARIANTS:
        spec = build_variant_spec(
            variant=variant, experiment_id=f"t-{variant.profile.profile_id}",
            elapsed_days=15,
        )
        by_id = {b.branch_id: b for b in spec.branches}
        assert "decoy_low_omit" in by_id
        assert by_id["decoy_low_omit"].exposure is by_id["claim_low_omit"].exposure


def test_core_branches_are_still_exactly_four():
    """The decoy is a control, not a third level of either factor."""
    for variant in VARIANTS:
        spec = build_variant_spec(
            variant=variant, experiment_id="t", elapsed_days=15,
        )
        core = [b for b in spec.branches if not b.branch_id.startswith("decoy")]
        assert len(core) == 4, [b.branch_id for b in core]


def test_factory_rejects_a_corpus_that_contains_the_decoy():
    """If the decoy leaked into the corpus, the control would stop controlling."""
    from dataclasses import replace

    from tools.scenario_factory import _validate

    variant = VARIANTS[0]
    target = variant.shared_ids[0]
    decoy = _md(variant.decoy_date)
    bad = tuple(
        (doc_id, title, text + f"（{decoy}）" if doc_id == target else text,
         published, source)
        for doc_id, title, text, published, source in variant.documents
    )
    with pytest.raises(ValidationError) as excinfo:
        _validate(replace(variant, documents=bad))
    assert "decoy date" in str(excinfo.value)


# ===========================================================================
# The temporal ladder: elapsed time must actually change something
# ===========================================================================
#
# These tests exist because the v4 controlled-delivery design made elapsed time
# inert without any signal that it had done so: the delivered set, the eligible
# set and the ranking were identical at every elapsed point, and only the rendered
# date string differed.  A design whose temporal manipulation is one date string
# cannot support a claim about elapsed world time, so the ladder is tested for the
# properties that make time real.

from tools.scenario_factory import (  # noqa: E402
    authoritative_stage,
    build_ladder,
    build_ladder_variant,
)
from tools.variant_wiring import build_ladder_spec  # noqa: E402


def test_ladder_is_derived_from_the_profile_not_declared_per_scenario():
    """A per-scenario publication time could be tuned to a convenient value."""
    for profile in PROFILES:
        stages = build_ladder(profile)
        assert len(stages) >= 2
        for index, stage in enumerate(stages):
            assert stage.stage_id.startswith(profile.profile_id)
            assert stage.published_at.endswith("+08:00")
            if index:
                # Each stage voids what the previous one set, so the ladder is a
                # chain rather than a set of unrelated notices.
                assert stage.voids == stages[index - 1].sets


def test_the_operative_stage_changes_with_the_instant():
    profile = PROFILES[0]
    stages = build_ladder(profile)
    early = authoritative_stage(stages, stages[0].published_at)
    late = authoritative_stage(stages, stages[-1].published_at)
    assert early.stage_id != late.stage_id
    assert early.sets != late.sets


def test_elapsed_time_changes_which_documents_are_queryable():
    """This is what makes L2 measurable, and it is what v4 lacked."""
    from vela_lab.evidence import Document, EvidenceCorpus

    profile = PROFILES[0]
    stages = build_ladder(profile)
    before, after = stages[0].published_at, stages[-1].published_at

    def eligible(instant):
        ladder = build_ladder_variant(
            profile=profile, filler_id=FILLER_ID, instant=instant
        )
        corpus = EvidenceCorpus([
            Document(doc_id=d[0], text=d[2], source=d[1] or d[4],
                     published_at=d[3], is_gold=(d[0] == ladder.variant.gold_id))
            for d in ladder.variant.documents
        ])
        corpus.seal()
        record = corpus.retrieve(
            query="预约 改期 通知", now_instant=instant, top_k=20,
            mode="auto", visibility="full",
        )
        return {i for i in record.eligible_ids if "notice" in i}

    early, late = eligible(before), eligible(after)
    assert early < late, (
        "later elapsed time must make more notices queryable, or accessibility is "
        f"not a function of time: {early} vs {late}"
    )


def test_elapsed_time_changes_the_correct_answer():
    profile = PROFILES[0]
    stages = build_ladder(profile)
    early = build_ladder_variant(
        profile=profile, filler_id=FILLER_ID, instant=stages[0].published_at
    )
    late = build_ladder_variant(
        profile=profile, filler_id=FILLER_ID, instant=stages[-1].published_at
    )
    assert early.correct_date != late.correct_date


def test_ladder_gold_is_the_live_stage_and_is_in_the_corpus():
    """The gold must follow the stage, or the scorer judges against a dead notice."""
    profile = PROFILES[0]
    stages = build_ladder(profile)
    for stage in stages:
        ladder = build_ladder_variant(
            profile=profile, filler_id=FILLER_ID, instant=stage.published_at
        )
        assert ladder.variant.gold_id == stage.stage_id
        assert stage.stage_id in [d[0] for d in ladder.variant.documents]


def test_gold_decisive_terms_follow_the_live_notice():
    """Regression: the terms were pinned to the profile's first reschedule date.

    With the ladder that date is not the operative one, so the L4 "used the
    evidence" test looked for a date the model was never shown and would have
    reported `delivered_not_used` for a correct answer.
    """
    from tools.variant_wiring import _gold_decisive_terms

    profile = PROFILES[0]
    stages = build_ladder(profile)
    late = build_ladder_variant(
        profile=profile, filler_id=FILLER_ID, instant=stages[-1].published_at
    )
    terms = _gold_decisive_terms(late.variant)
    rendered = f"{late.correct_date.month} 月 {late.correct_date.day} 日"
    assert rendered in terms, terms
    assert late.correct_date.isoformat() in terms, terms


def test_sub_day_elapsed_is_expressible():
    """An integer day count silently rounds a sub-day interval to zero.

    Two conditions that are byte-identical under two different labels is a dead
    knob that looks like a well-matched pair.
    """
    variant = build_variant(scenario_id="t", profile=PROFILES[0], filler_id=FILLER_ID)
    seen = {}
    for hours, minutes in ((0, 0), (0, 30), (0, 90), (6, 0), (23, 59)):
        spec = build_variant_spec(
            variant=variant, experiment_id="t",
            elapsed_hours=hours, elapsed_minutes=minutes,
        )
        instant = spec.clock_start.advance(
            days=spec.advance_days, hours=spec.advance_hours,
            minutes=spec.advance_minutes,
        ).instant
        assert instant not in seen, (
            f"{hours}h{minutes}m collides with {seen.get(instant)}"
        )
        seen[instant] = f"{hours}h{minutes}m"


def test_below_minute_resolution_is_rejected_not_rounded():
    """Rounding to zero would make two labeled conditions identical."""
    variant = build_variant(scenario_id="t", profile=PROFILES[0], filler_id=FILLER_ID)
    with pytest.raises(ValidationError) as excinfo:
        build_variant_spec(variant=variant, experiment_id="t",
                           elapsed_minutes=0.5)
    assert "resolution" in str(excinfo.value)


def test_ladder_spec_passes_the_exposure_gate():
    """End to end: the ladder must be measurable, not merely well-formed."""
    from pathlib import Path as _Path

    from vela_lab.provider import MockModel
    from vela_lab.runner import ExperimentRunner

    profile = PROFILES[0]
    stages = build_ladder(profile)
    for index, stage in enumerate(stages):
        ladder = build_ladder_variant(
            profile=profile, filler_id=FILLER_ID, instant=stage.published_at
        )
        spec = build_ladder_spec(
            ladder=ladder, experiment_id=f"ladder-{index}",
            elapsed_days=float(5 + index * 14),
        )
        out = _Path("runs_gate") / f"test_ladder_{index}"
        import shutil

        if out.exists():
            shutil.rmtree(out)
        report = ExperimentRunner(
            spec, MockModel(scripted=["我理解。"]), out_dir=out
        ).run()
        for branch in report["branches"]:
            assert branch["run_valid"], (
                f"stage {index} branch {branch['branch_id']}: "
                f"{branch['invalid_reason']} {branch.get('gate_rejection')}"
            )
            statuses = {v["status"] for v in branch["exposure_verdicts"]}
            assert statuses == {"PASS"}, (index, branch["branch_id"], statuses)
