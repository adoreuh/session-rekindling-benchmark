"""The temporal manipulation must not be inert.

Every test here guards a way "world time elapsed" silently became a no-op.  Those
failures are dangerous precisely because they do not look like failures: the
conditions stay byte-identical across elapsed points, which reads as a
well-matched design rather than as a dead knob.

Three were found and fixed, and each has a test here:

1. an integer day count rounded a sub-day interval to zero;
2. the delivered set was pinned, so elapsed time changed no document at all;
3. the gold's decisive terms were pinned to one date, so the L4 attribution test
   looked for a date the model was never shown.

The decay path is covered too, because a uniform multiplicative decay cancels out
of the ranking: using it as the temporal manipulation would be inert by
mathematics rather than by accident.
"""

from __future__ import annotations

import shutil
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.clock import parse_instant  # noqa: E402
from vela_lab.decay import resolve_profile  # noqa: E402
from vela_lab.evidence import Document, EvidenceCorpus  # noqa: E402
from vela_lab.errors import ValidationError  # noqa: E402
from vela_lab.provider import MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    PROFILES,
    build_ladder,
    build_ladder_variant,
    build_variant,
)
from tools.variant_wiring import (  # noqa: E402
    _gold_decisive_terms,
    build_ladder_spec,
    build_variant_spec,
)

PROFILE = PROFILES[0]


def _corpus_and_ladder(instant):
    ladder = build_ladder_variant(
        profile=PROFILE, filler_id=FILLER_ID, instant=instant
    )
    corpus = EvidenceCorpus([
        Document(doc_id=d[0], text=d[2], source=d[1] or d[4],
                 published_at=d[3], is_gold=(d[0] == ladder.variant.gold_id))
        for d in ladder.variant.documents
    ])
    corpus.seal()
    return corpus, ladder


# ----------------------------------------------------- the manipulation is real

def test_elapsed_time_moves_the_clock_to_distinct_instants():
    """The base requirement: different elapsed, different probe instant."""
    variant = build_variant(scenario_id="t", profile=PROFILE, filler_id=FILLER_ID)
    instants = {}
    for days in (0, 1, 7, 15, 61):
        spec = build_variant_spec(variant=variant, experiment_id="t",
                                  elapsed_days=days)
        instant = spec.clock_start.advance(days=spec.advance_days).instant
        assert instant not in instants, f"{days}d collides with {instants[instant]}"
        instants[instant] = days


def test_time_changes_the_eligible_set_and_the_correct_answer():
    """Both must move, or a temporal claim has nothing to rest on.

    The eligible set moving is what makes L2 measurable; the correct answer moving
    is what makes L3 measurable.  If either is time-invariant, the corresponding
    layer cannot respond to elapsed time at all.
    """
    stages = build_ladder(PROFILE)

    def position(instant):
        corpus, ladder = _corpus_and_ladder(instant)
        record = corpus.retrieve(query="预约 改期 通知", now_instant=instant,
                                 top_k=20, mode="auto", visibility="full")
        return ({i for i in record.eligible_ids if "notice" in i},
                ladder.correct_date)

    early_eligible, early_correct = position(stages[0].published_at)
    late_eligible, late_correct = position(stages[-1].published_at)
    assert early_eligible != late_eligible
    assert early_correct != late_correct


def test_the_probe_instant_tracks_the_requested_elapsed_interval():
    """Elapsed is measured from the world update, and must actually land there."""
    variant = build_variant(scenario_id="t", profile=PROFILE, filler_id=FILLER_ID)
    update = parse_instant(PROFILE.update_day.isoformat() + "T09:00:00+08:00")
    for days in (1, 15, 61):
        spec = build_variant_spec(variant=variant, experiment_id="t",
                                  elapsed_days=days)
        probe = parse_instant(
            spec.clock_start.advance(days=spec.advance_days).instant
        )
        assert probe - update == timedelta(days=days)


# --------------------------------------------------- the decay path is not it

def test_uniform_decay_does_not_reorder_retrieval():
    """Why the ladder uses publication timing instead of decay weights.

    A multiplicative weight applied to every candidate is a common factor in
    every score, so it cancels out of the comparison.  Using decay as the temporal
    manipulation would therefore produce an inert knob -- a mathematical property,
    not an empirical accident.
    """
    variant = build_variant(scenario_id="t", profile=PROFILE, filler_id=FILLER_ID)
    corpus = EvidenceCorpus([
        Document(doc_id=d[0], text=d[2], source=d[1] or d[4],
                 published_at=d[3], is_gold=(d[0] == variant.gold_id))
        for d in variant.documents
    ])
    corpus.seal()
    query = ("我们之前一起准备的赴德探亲签证面试：当前有效预约日期、"
             "官方改期通知、原预约是否取消")
    decay = resolve_profile("m01_memorybank")

    orders = []
    for instant in ("2026-03-06T09:00:00+08:00", "2026-05-05T09:00:00+08:00",
                    "2026-09-01T09:00:00+08:00"):
        record = corpus.retrieve(query=query, now_instant=instant, top_k=12,
                                 mode="auto", visibility="full",
                                 temporal_decay=decay)
        orders.append(tuple(h["doc_id"] for h in record.hits))
    assert len(set(orders)) == 1, (
        "a uniform multiplicative decay changed the ranking; if that becomes true "
        "the invariance finding needs revisiting"
    )


# ------------------------------------------------- the ladder is not tunable

def test_ladder_publication_times_come_from_the_profile():
    """A per-scenario publication time could be tuned to a convenient value."""
    for profile in PROFILES:
        stages = build_ladder(profile)
        first = parse_instant(stages[0].published_at)
        assert (first.month, first.day) == (profile.update_day.month,
                                            profile.update_day.day)
        for index in range(1, len(stages)):
            gap = (parse_instant(stages[index].published_at)
                   - parse_instant(stages[index - 1].published_at))
            assert gap == timedelta(days=14), (
                "the ladder gap is a design parameter, not a per-scenario choice"
            )


def test_a_stage_before_publication_is_reported_as_not_live():
    """An instant before every notice must not silently claim a live stage."""
    stages = build_ladder(PROFILE)
    before = (parse_instant(stages[0].published_at)
              - timedelta(days=1)).isoformat()
    ladder = build_ladder_variant(profile=PROFILE, filler_id=FILLER_ID,
                                  instant=before)
    assert ladder.stage_live is False
    assert ladder.stage.stage_id == stages[0].stage_id


def test_gold_and_its_decisive_terms_follow_the_live_stage():
    """Regression: the terms were pinned to the profile's first date.

    Under the ladder that date is not the operative one, so the L4 test looked for
    a date the model was never shown and would have reported `delivered_not_used`
    for an answer that quoted the delivered notice correctly.
    """
    for stage in build_ladder(PROFILE):
        ladder = build_ladder_variant(profile=PROFILE, filler_id=FILLER_ID,
                                      instant=stage.published_at)
        assert ladder.variant.gold_id == stage.stage_id
        terms = _gold_decisive_terms(ladder.variant)
        assert stage.sets.isoformat() in terms, terms
        rendered = f"{stage.sets.month} 月 {stage.sets.day} 日"
        assert rendered in terms, terms


# ------------------------------------------------- both regimes pass the gate

def test_each_ladder_stage_is_measurable_end_to_end():
    """A well-formed ladder that fails the gate measures nothing."""
    for index, stage in enumerate(build_ladder(PROFILE)):
        ladder = build_ladder_variant(profile=PROFILE, filler_id=FILLER_ID,
                                      instant=stage.published_at)
        spec = build_ladder_spec(ladder=ladder, experiment_id=f"t-{index}",
                                 elapsed_days=float(5 + 14 * index))
        out = Path("runs_gate") / f"test_temporal_{index}"
        if out.exists():
            shutil.rmtree(out)
        report = ExperimentRunner(spec, MockModel(scripted=["我理解。"]),
                                  out_dir=out).run()
        assert report["parent_equality"]["identical_parent"], index
        for branch in report["branches"]:
            assert branch["run_valid"], (
                index, branch["branch_id"], branch["invalid_reason"],
                branch.get("gate_rejection"),
            )


def test_sub_day_interval_is_rejected_when_unrepresentable():
    variant = build_variant(scenario_id="t", profile=PROFILE, filler_id=FILLER_ID)
    with pytest.raises(ValidationError):
        build_variant_spec(variant=variant, experiment_id="t",
                           elapsed_minutes=0.5)


def test_negative_interval_is_rejected():
    variant = build_variant(scenario_id="t", profile=PROFILE, filler_id=FILLER_ID)
    with pytest.raises(ValidationError):
        build_variant_spec(variant=variant, experiment_id="t", elapsed_days=-1)


# ------------------------------------------------------- the chain is complete

def test_the_update_chain_is_readable_end_to_end():
    """A notice may not cite a state the delivered material does not contain.

    Delivering stage 2 alone produced a notice voiding a date the material never
    stated.  One route spotted the gap and asked the user to verify the record; the
    other adopted the notice.  Both are defensible readings of a broken record --
    and neither is the temporal update the condition claimed to test.
    """
    stages = build_ladder(PROFILE)
    for index, stage in enumerate(stages):
        ladder = build_ladder_variant(profile=PROFILE, filler_id=FILLER_ID,
                                      instant=stage.published_at)
        bodies = {d[0]: d[2] for d in ladder.variant.documents}
        if index == 0:
            # Nothing precedes the first stage, so there is nothing to cite.
            assert ladder.preceding_ids == ()
            continue
        assert ladder.preceding_ids, index
        # The state the live notice voids must be stated somewhere in the
        # delivered material: either by a preceding notice or by the background.
        voided = f"{stage.voids.month} 月 {stage.voids.day} 日"
        searchable = list(ladder.variant.shared_ids) + [ladder.variant.gold_id]
        assert any(voided in bodies[doc] for doc in searchable), (
            f"stage {index} voids {voided} but no delivered document states it"
        )


def test_exactly_one_delivered_document_carries_the_live_answer():
    """The chain adds an antecedent, not extra evidence for the current answer.

    Stage 2 delivers one more document than stage 1, which would confound the
    temporal contrast if that document carried the live answer.  It does not: it
    states the date the live notice supersedes.
    """
    for stage in build_ladder(PROFILE):
        ladder = build_ladder_variant(profile=PROFILE, filler_id=FILLER_ID,
                                      instant=stage.published_at)
        spec = build_ladder_spec(ladder=ladder, experiment_id="t",
                                 elapsed_days=15)
        bodies = {d[0]: d[2] for d in ladder.variant.documents}
        current = f"{stage.sets.month} 月 {stage.sets.day} 日"
        for branch in spec.branches:
            carriers = [d for d in (branch.exposure.deliver_ids or ())
                        if current in bodies[d]]
            assert len(carriers) == 1, (
                f"{stage.stage_id} {branch.branch_id}: {len(carriers)} documents "
                f"carry the live answer {current}: {carriers}"
            )


def test_ladder_spec_is_oracle_only():
    """Chain completeness and omission cleanliness cannot both hold.

    A preceding notice asserts the user's claimed date is void, so a
    chain-complete record is not an omission condition.  The two contrasts are
    therefore measured separately, and the ladder carries no omit branch.
    """
    ladder = build_ladder_variant(profile=PROFILE, filler_id=FILLER_ID,
                                  instant=build_ladder(PROFILE)[-1].published_at)
    spec = build_ladder_spec(ladder=ladder, experiment_id="t", elapsed_days=15)
    assert spec.branches, "ladder spec has no branches"
    assert all(b.branch_id.endswith("_oracle") for b in spec.branches), [
        b.branch_id for b in spec.branches
    ]
