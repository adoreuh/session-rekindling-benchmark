"""The expiry experiment must not become a control that controls nothing.

Every test here guards a way "the world never changed, the clock moved past the
event" quietly stopped being the manipulation.  Those failures are dangerous for
the same reason the temporal ones are: they leave a design that reads as balanced
while measuring nothing.

The ways it can happen, and where each is caught below:

* the six conditions drifting apart — a different document, a reworded probe, a
  delivery difference — so "only the clock varies" is no longer true;
* the clock landing somewhere other than the declared instant;
* the visibility arm wired to the wrong value, or the arm that is supposed to
  hide the clock actually delivering it;
* the reschedule notice surviving the edit, or a change-word surviving elsewhere
  in the corpus;
* the world moving anyway, because the base variant's ``supersede_fact`` was kept;
* the probe re-asserting the user's state, naming the event, or carrying a date;
* the gate rejecting the corpus, which would mean the six conditions could not be
  run at all.

The two checks that could pass vacuously — "no supersession in the corpus" and
"the world never moves" — each carry a **negative control** that puts the notice
back and asserts the check fires.  A check with no failing input is not evidence.
"""

from __future__ import annotations

import re
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.clock import Clock  # noqa: E402
from vela_lab.errors import ValidationError  # noqa: E402
from vela_lab.provider import MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.expiry_experiment import (  # noqa: E402
    ALLOWED_FIELD_DIFFERENCES,
    ARMS,
    FORBIDDEN_WORLD_KINDS,
    PROBE,
    VISIBILITIES,
    assert_corpus_identical,
    assert_no_supersession,
    assert_world_static,
    audit_no_supersession,
    build_conditions,
    build_expiry_contract,
    build_expiry_variant,
    differing_fields,
    proposition_map,
)
from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    PROFILES,
    _md,
    _md_full,
    build_variant,
)
from tools.variant_wiring import build_variant_spec  # noqa: E402

PROFILE = PROFILES[0]
ROUTE = "hy3"


def _conditions(profile=PROFILE):
    variant = build_expiry_variant(profile=profile)
    return variant, build_conditions(variant=variant, route=ROUTE)


# ------------------------------------------------------------ the six-way identity


def test_the_six_conditions_share_one_corpus():
    """Only the clock and the visibility may vary; everything else is the corpus."""
    _, conditions = _conditions()
    identity = assert_corpus_identical(conditions)
    assert identity["unexpected"] == []
    assert set(identity["differing_fields"]) <= ALLOWED_FIELD_DIFFERENCES
    # ... and the allowed fields really do differ, or "six conditions" is a label.
    assert {"advance_days", "observe_time"} <= set(identity["differing_fields"])
    assert len(conditions) == len(ARMS) * len(VISIBILITIES)
    first = conditions[0].spec
    assert len({tuple(condition.spec.documents) for condition in conditions}) == 1
    assert {condition.spec.branches[0].user_text for condition in conditions} == {PROBE}
    for condition in conditions[1:]:
        spec = condition.spec
        # Compared by equality rather than by hashing a subset: these are the same
        # objects in the six specs, which is what makes "identical" a property of
        # the build rather than a coincidence of copying.
        assert spec.branches == first.branches
        assert spec.exposure_contract == first.exposure_contract
        assert spec.documents == first.documents
        assert spec.stable_material == first.stable_material
        assert spec.world_events == first.world_events
        assert spec.prefix_turns == first.prefix_turns
        assert spec.host_query == first.host_query
        assert spec.prefix_exposure == first.prefix_exposure
        assert spec.window_boundaries == first.window_boundaries
        assert spec.prompt == first.prompt
        assert spec.gold_marker == first.gold_marker
        assert spec.gold_fact_id == first.gold_fact_id
        assert spec.gold_decisive_terms == first.gold_decisive_terms
        assert spec.enabled_tools == first.enabled_tools
        assert spec.tool_budget == first.tool_budget
        assert spec.clock_start == first.clock_start
        assert spec.request_gate_factory is first.request_gate_factory
        assert spec.parent_visible_gate is first.parent_visible_gate


def test_the_identity_check_notices_a_changed_document():
    """A negative control: the six-way check must fail on a drifted corpus."""
    _, conditions = _conditions()
    specs = [condition.spec for condition in conditions]
    original = list(specs[1].documents)
    changed = list(original)
    first = changed[0]
    changed[0] = type(first)(
        doc_id=first.doc_id, text=first.text + "补充一句。", source=first.source,
        published_at=first.published_at, is_gold=first.is_gold,
    )
    specs[1].documents = tuple(changed)
    try:
        assert "documents" in differing_fields(specs)
        with pytest.raises(ValidationError):
            assert_corpus_identical(conditions)
    finally:
        specs[1].documents = tuple(original)
    # ... and the check passes again once the drift is undone, so the failure above
    # was the drift and not the mutation itself.
    assert assert_corpus_identical(conditions)


def test_the_identity_check_notices_a_reworded_probe():
    _, conditions = _conditions()
    specs = [condition.spec for condition in conditions]
    branch = specs[2].branches[0]
    reworded = type(branch)(
        branch_id=branch.branch_id, exposure=branch.exposure,
        user_text=branch.user_text + "顺便说一下。", label=branch.label,
    )
    original = specs[2].branches
    specs[2].branches = (reworded,)
    try:
        assert "branches" in differing_fields(specs)
        with pytest.raises(ValidationError):
            assert_corpus_identical(conditions)
    finally:
        specs[2].branches = original


# ---------------------------------------------------------------- the instants


def test_each_condition_lands_on_its_declared_instant():
    """Whole days from the event date, at the prefix's own time of day."""
    _, conditions = _conditions()
    expected = {
        "pre": PROFILE.old_date - timedelta(days=1),
        "post": PROFILE.old_date + timedelta(days=30),
        "late": PROFILE.old_date + timedelta(days=120),
    }
    for condition in conditions:
        spec = condition.spec
        assert condition.target == (
            f"{expected[condition.arm].isoformat()}T09:00:00+08:00"
        )
        landed = spec.clock_start.advance(
            days=spec.advance_days, hours=spec.advance_hours,
            minutes=spec.advance_minutes,
        )
        assert landed.instant == Clock.of(condition.target, spec.time_zone).instant
        assert spec.advance_hours == 0 and spec.advance_minutes == 0
    advances = {condition.arm: condition.spec.advance_days for condition in conditions}
    assert advances["pre"] < advances["post"] < advances["late"]
    # The three instants are exactly 31 and 90 days further out, so the arms differ
    # by a stated interval rather than by whatever the date arithmetic happened to
    # produce.
    assert advances["post"] - advances["pre"] == 31
    assert advances["late"] - advances["post"] == 90


def test_a_non_minute_aligned_instant_is_refused():
    """The clock has one-minute resolution; an instant it cannot hold is refused."""
    from tools.expiry_experiment import _advance_from

    with pytest.raises(ValidationError):
        _advance_from("2026-03-01T09:00:00+08:00", "2026-03-01T09:00:30+08:00")
    with pytest.raises(ValidationError):
        _advance_from("2026-03-01T09:00:00+08:00", "2026-02-01T09:00:00+08:00")


# --------------------------------------------------------------- the visibility


def test_observe_time_is_the_visibility_arm():
    """``delivered`` renders the clock, ``natural`` does not — and neither removes it.

    The rendered half of this claim is read off the real compiled requests in
    ``test_all_six_conditions_pass_the_gate_offline``; this is the structural half.
    """
    _, conditions = _conditions()
    for condition in conditions:
        assert condition.spec.observe_time is (condition.visibility == "delivered")
    assert {condition.visibility for condition in conditions} == {"delivered", "natural"}
    natural = [c for c in conditions if c.visibility == "natural"]
    delivered = [c for c in conditions if c.visibility == "delivered"]
    assert len(natural) == len(ARMS) and len(delivered) == len(ARMS)
    # The clock stays reachable in both arms: the natural arm hides the rendered
    # instant, it does not remove the route to it.
    assert all("world_time" in condition.spec.enabled_tools for condition in conditions)
    assert sorted(condition.name for condition in conditions) == [
        "late_delivered", "late_natural", "post_delivered", "post_natural",
        "pre_delivered", "pre_natural",
    ]


# ---------------------------------------------------------------- the corpus


def test_the_notice_is_gone_and_with_it_the_change():
    variant = build_expiry_variant(profile=PROFILE)
    audit = assert_no_supersession(variant)
    assert audit["notice_absent"] and audit["clean"]
    assert audit["surface_hits"] == {}
    assert audit["dialogue_surface_hits"] == {}
    assert audit["asserted"] == {}
    assert f"{PROFILE.profile_id}_reschedule_notice" not in {
        doc_id for doc_id, *_ in variant.documents
    }
    # Negative control: the base variant, which does carry the notice, must fail the
    # same audit.  Without this the check could pass because it looks at nothing.
    base = build_variant(
        scenario_id="control", profile=PROFILE, filler_id=FILLER_ID
    )
    control = audit_no_supersession(base)
    assert not control["clean"]
    assert not control["notice_absent"]
    assert f"{PROFILE.profile_id}_reschedule_notice" in control["surface_hits"]
    assert control["asserted"]


def test_the_corpus_is_the_base_variant_minus_the_notice():
    variant = build_expiry_variant(profile=PROFILE)
    base = build_variant(scenario_id="control", profile=PROFILE, filler_id=FILLER_ID)
    expected = tuple(
        doc for doc in base.documents if doc[0] != f"{PROFILE.profile_id}_reschedule_notice"
    )
    assert tuple(variant.documents) == expected
    # The prefix and the frozen background are untouched: the record's origin and
    # the background must stay comparable with the other experiments.
    assert tuple(variant.prefix_turns) == tuple(base.prefix_turns)
    assert tuple(variant.stable_material) == tuple(base.stable_material)
    assert variant.prefix_start == base.prefix_start
    # The operative date is the old one, because nothing supersedes it.
    assert variant.operative_date == PROFILE.old_date
    assert variant.gold_id == f"{PROFILE.profile_id}_original_confirmation"


def test_the_world_never_moves():
    variant, conditions = _conditions()
    spec = conditions[0].spec
    world = assert_world_static(spec, PROFILE)
    assert world["kinds"] == ["add_fact", "publish_evidence"]
    assert world["fact_values"] == [PROFILE.old_date.isoformat()]
    assert not [e for e in spec.world_events if e.kind in FORBIDDEN_WORLD_KINDS]
    # Every publication names a document that is actually in the corpus, so the
    # schedule cannot refer to the removed notice.
    ids = {doc_id for doc_id, *_ in variant.documents}
    assert {e.evidence_id for e in spec.world_events if e.evidence_id} <= ids
    # Negative control: the base variant's schedule does move the world, so the
    # check above is not passing because it inspects nothing.
    control = build_variant_spec(
        variant=build_variant(
            scenario_id="control", profile=PROFILE, filler_id=FILLER_ID
        ),
        experiment_id="control",
        elapsed_days=15,
    )
    with pytest.raises(ValidationError):
        assert_world_static(control, PROFILE)


# ------------------------------------------------------------------ the probe


def test_the_probe_does_not_re_assert_the_state_or_name_the_event():
    """A probe that refreshed the user's state would destroy the manipulation."""
    for token in (
        PROFILE.referent, PROFILE.referent_long, PROFILE.holder, PROFILE.office,
        PROFILE.office_short, _md(PROFILE.old_date), _md_full(PROFILE.old_date),
        _md(PROFILE.new_date),
    ):
        assert token not in PROBE
    assert not re.search(r"\d", PROBE)
    assert "紧" not in PROBE and "怕" not in PROBE and "担心" not in PROBE
    _, conditions = _conditions()
    assert {condition.spec.branches[0].user_text for condition in conditions} == {PROBE}


def test_the_decisive_proposition_is_the_records_own_rendering():
    """Why the decisive propositions are anchored the way they are.

    The parent check reads the whole pre-fork window, and that window contains the
    base variant's prefix turn, where the *user* quotes the date in the first
    person.  A decisive proposition keyed on that rendering would therefore mark
    every run as contaminated at the fork.  So the decisive rendering is the
    record's own (the issuer confirms the holder's appointment is on D), the
    first-person renderings stay registered as documentation, and this test pins
    both halves down.
    """
    variant = build_expiry_variant(profile=PROFILE)
    contract = build_expiry_contract(variant)
    decisive = [p for p in contract.decisive_propositions if p.decisive]
    assert decisive
    confirmation = next(
        text for doc_id, _, text, _, _ in variant.documents
        if doc_id == variant.gold_id
    )
    for proposition in decisive:
        assert proposition.asserted_in(confirmation)
        assert not proposition.asserted_in(PROBE)
        assert not proposition.asserted_in(variant.prefix_turns[0])
        assert not any(
            proposition.asserted_in(text) for text in variant.stable_material
        )
    documented = [p for p in contract.decisive_propositions if not p.decisive]
    assert documented
    assert any(p.asserted_in(variant.prefix_turns[0]) for p in documented)
    # And the documentation really is inert: the gate only reads the decisive ones.
    assert not contract.decisive_asserted_in(variant.prefix_turns[0])
    assert contract.decisive_asserted_in(confirmation)
    fired = proposition_map(variant, contract)
    assert "probe" not in fired["record_places_event_on_d"]["asserted_in"]
    assert variant.gold_id in fired["record_places_event_on_d"]["asserted_in"]
    assert fired["event_named_with_d"]["decisive"] is False
    assert "prefix_turn[0]" in fired["event_named_with_d"]["asserted_in"]


# ------------------------------------------------------------- the delivery


def test_the_delivered_record_is_sufficient_by_content():
    variant, conditions = _conditions()
    spec = conditions[0].spec
    contract = spec.exposure_contract
    manifest = spec.delivery["manifests"]["oracle"]
    bodies = {doc_id: text for doc_id, _, text, _, _ in variant.documents}
    joined = "\n".join(bodies[doc_id] for doc_id in manifest["expected_material_ids"])
    for group in contract.oracle_required_groups:
        assert any(term in joined for term in group)
    assert manifest["condition"] == "oracle"
    assert manifest["slot_material_id"] == variant.gold_id
    assert manifest["expected_material_ids"][manifest["slot_index"]] == variant.gold_id
    # The filler stays in the corpus and out of the delivery: if it were delivered
    # the exclusivity check would flag an undeclared document rather than pass.
    assert FILLER_ID in bodies
    assert FILLER_ID not in manifest["expected_material_ids"]
    assert variant.gold_id in spec.delivery["pair"]["oracle_doc_ids"]


def test_all_six_conditions_pass_the_gate_offline(tmp_path):
    """The end-to-end check: six conditions, six PASSing gates, no provider call.

    Also the rendered half of the visibility check, because it is read off the real
    compiled requests: with ``observe_time=False`` nothing renders the clock, so the
    three natural conditions send byte-identical requests.  Any difference between
    ``pre`` and ``post`` in that arm can therefore only come from the model asking
    ``world_time`` (or from provider non-determinism) — which is what makes the
    natural arm a test of whether the model consults the clock at all.  The
    delivered arm's three requests must differ, or the other arm's sameness would
    also hold in a harness that had stopped rendering the clock everywhere.
    """
    _, conditions = _conditions()
    record = list(conditions[0].spec.delivery["pair"]["oracle_doc_ids"])
    digests = {}
    for condition in conditions:
        report = ExperimentRunner(
            condition.spec, MockModel(), out_dir=tmp_path / condition.name,
        ).run()
        branch = report["branches"][0]
        assert branch["run_valid"], branch.get("invalid_reason", "")
        verdicts = branch["exposure_verdicts"]
        assert [v["condition"] for v in verdicts] == ["oracle"]
        assert [v["status"] for v in verdicts] == ["PASS"]
        assert verdicts[0]["reason_codes"] == ["provenance_complete"]
        assert verdicts[0]["present_material_ids"] == sorted(record)
        assert verdicts[0]["undeclared_material_ids"] == []
        assert verdicts[0]["missing_material_ids"] == []
        assert verdicts[0]["missing_required_groups"] == []
        assert branch["auto_retrieval"]["gold_delivered"] is True
        assert report["exposure"]["parent_check"]["status"] == "PASS"
        assert report["exposure"]["parent_check"]["reason_codes"] == ["parent_clean"]
        digests[condition.name] = verdicts[0]["payload_digest"]
    assert len({digests[f"{arm}_natural"] for arm, _ in ARMS}) == 1
    assert len({digests[f"{arm}_delivered"] for arm, _ in ARMS}) == len(ARMS)
