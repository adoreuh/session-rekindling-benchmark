"""The naturally-expired experiment: one unchanged record, three clocks, two visibilities.

What this adds to the contrasts already in the harness.

The delivery and temporal experiments both cover **supersession**: the world
changes, a notice is issued, and the question is whether the model tracks the
*new* arrangement.  They say nothing about the other way a record stops being a
description of the present — the world never moves, and the clock crosses the date
the record names.  Nothing is cancelled; the same sentence that described the
user's week becomes a description of last month.  This experiment isolates that.

The manipulation:

* **corpus** — the factory's base variant with the reschedule notice *removed*
  (``{profile_id}_reschedule_notice``).  Removing it is what makes the world
  static: no document asserts a change, and the world schedule carries no
  ``supersede_fact`` event, so the fact store holds ``D = profile.old_date`` from
  the prefix to the probe.  The operative date is the profile's ``old_date``,
  which ``{profile_id}_original_confirmation`` asserts — and which the base
  variant's own prefix turn quotes, unchanged;
* **instant** — three positions relative to ``D``: ``pre`` = D-1 (the event is
  tomorrow, so the record is current), ``post`` = D+30 (a month past), ``late`` =
  D+120 (plainly historical);
* **time visibility** — ``delivered`` (``observe_time=True``, which is the
  harness's existing behaviour: the current instant is rendered into the request
  as ``Current date and time: ...``) and ``natural`` (``observe_time=False``, a
  *new* condition: nothing puts the clock on screen, so the only route to it is
  the ``world_time`` tool, which stays enabled).

3 instants x 2 visibilities x 1 branch = 6 runs per (route, profile).  The corpus
is byte-identical across all six: the only fields that may differ are
``advance_days/hours/minutes``, ``observe_time`` and the identifiers.  ``--dry-run``
proves that by diffing the specs **field by field** and failing on any difference
outside the declared allow-list, rather than by digesting a hand-picked subset
and hoping the subset was complete.

Why one branch per run, and why that branch is registered as ``oracle``.

The manipulated variable is the clock, not delivery, so there is no omit condition
to pair against: all six conditions deliver the same record.  The gate's manifest
vocabulary has exactly two conditions, and ``oracle`` is the one that means "the
decisive material is delivered and the delivery must be sufficient" — which is
true in all six.  Labelling the branch ``omit`` would assert the opposite and the
gate would (correctly) reject the record as a ``decisive_leak``.

The delivered record, and the slot.

The probe delivers the *record of the event*: the confirmation (the slot, at the
head, where the harness declares the decisive material), the preparation notes,
the unrelated invitation, and the chat where the user says the event makes them
anxious.  The first three are the base variant's shared context, with the
confirmation moved from the shared list into the slot; the anxiety chat is added
to the delivery because it is the corpus's only statement of the user's *state*,
which is the other thing the clock makes stale.  The base variant's filler stays
in the corpus but is never delivered; if it ever were, the exclusivity check would
flag an undeclared document rather than pass quietly.

The probe is state-neutral by construction (see ``PROBE_RATIONALE``): it never
names the event, the date, or the user's feelings.  Saying "I am still nervous
about the interview" would re-assert the user's state in the present tense and
destroy the manipulation — the model would be answering about the state the probe
just refreshed rather than about the state it remembers.  Any wording that keeps
the six conditions comparable also has to be answerable both before and after the
event, which is why it asks for a plan for tonight and lets the model decide
whether the record is still relevant to tonight.

Honest limits, stated rather than left implicit:

* the corpus cannot prove *absence of synonymic* change-language; what it proves
  is that the notice is gone and that a listed transition vocabulary appears
  nowhere (``audit_no_supersession``).  The vocabulary is a constant so that
  widening it is a visible edit;
* the parent gate reads the whole pre-fork window, including the base variant's
  prefix turn, which quotes ``D`` in the *user's* voice.  So the decisive
  propositions are anchored on the record's own third-person wording ("the
  holder's appointment is on D"); the first-person renderings are registered
  and explicitly marked non-decisive, and ``--dry-run`` reports where they fire,
  so the exclusion is visible instead of silent;
* six conditions per (route, profile) is a small design.  It locates whether the
  record is treated as current; it does not measure the mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.clock import Clock, parse_instant  # noqa: E402
from vela_lab.errors import ValidationError  # noqa: E402
from vela_lab.evidence import Document, ExposurePolicy  # noqa: E402
from vela_lab.jsonutil import digest  # noqa: E402
from vela_lab.prompt import PromptBundle, neutral_system_prompt  # noqa: E402
from vela_lab.provider import MockModel, OpenAICompatibleModel  # noqa: E402
from vela_lab.runner import (  # noqa: E402
    BranchSpec,
    ExperimentRunner,
    ExperimentSpec,
    WindowBoundary,
)
from vela_lab.schedule import WorldEvent  # noqa: E402

from tools.access_pair import build_access_pair, exposure_for_condition  # noqa: E402
from tools.exposure_gate import (  # noqa: E402
    ASSERTED,
    DeliveryManifest,
    ExposureContract,
    ExposureResult,
    Proposition,
    classify_terms,
    check_parent_state,
)
from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    PROFILES,
    ScenarioVariant,
    _md,
    _md_full,
    build_variant,
)
from tools.scope import OutOfScopeRoute, SUBJECT_ROUTES, assert_in_scope  # noqa: E402
from tools.v4_wiring import (  # noqa: E402
    FORBIDDEN_TOOL_NAMES,
    PROBE_TOOLS,
    SLOT_INDEX,
    declared_tool_surfaces,
    make_request_gate,
)
from tools.variant_wiring import _gold_decisive_terms  # noqa: E402

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"

#: Names this scenario's contract.  Bumped whenever a proposition, a sufficiency
#: group or the neutrality rule changes, because each of those changes what
#: counts as evidence and therefore invalidates prior audits.
CONTRACT_VERSION = "expiry-natural-v1"

#: The one branch every run has.  Deliberately the *same* id in all six
#: conditions: the condition is a property of the spec (clock, visibility), not of
#: the branch, and keeping the id fixed is what lets the corpus comparison cover
#: the branch's user text and exposure policy as well.  The run directory
#: (``<profile>_<arm>_<visibility>``) is what tells the six apart.
BRANCH_ID = "probe"
BRANCH_LABEL = "expiry|record"

#: The three probe instants, as whole days relative to the event date ``D``.
#: ``pre`` is the control: the record describes tomorrow.  ``post`` and ``late``
#: differ only in how obviously historical the record has become, so a monotone
#: effect and a step effect are distinguishable.
ARMS = (("pre", -1), ("post", 30), ("late", 120))

#: The two time visibilities.  ``delivered`` is the harness's existing behaviour
#: and is therefore the arm that links this experiment to every run already on
#: disk; ``natural`` is the new condition, where the clock is reachable only by
#: asking the tool.
VISIBILITIES = (("delivered", True), ("natural", False))

#: The probe.  One sentence, verbatim in all six conditions.
PROBE = "结合你之前了解的情况，你觉得我今晚应该怎么安排？"

#: Why this probe, and what each word is doing.
#:
#: * it never names the event, the date, or the user's feelings.  Re-asserting the
#:   user's state ("I am still worried about the interview") would hand the user's
#:   state back to the model in the present tense, which is exactly the variable
#:   under test: measuring it after refreshing it measures the refresh;
#: * it is answerable both before and after the event.  A probe that presupposes
#:   the event is upcoming ("how should I prepare for tomorrow?") would be false in
#:   ``post``/``late`` and would measure the probe's own falsehood; a probe that
#:   presupposes it is past ("how did it go?") would be false in ``pre``.  Asking
#:   for a plan for tonight leaves the model to decide whether the record still
#:   bears on tonight — which is the measurement;
#: * it is short and free of any date rendering, so no decisive proposition's
#:   terms and no sufficiency group can match inside it;
#: * it is registered as dialogue, so the gate's provenance check is satisfied by
#:   registration rather than by the probe happening to look innocuous.
PROBE_RATIONALE = (
    "state-neutral, event-neutral, date-free, and answerable on both sides of D"
)

#: Visibilities minted by the intervention drivers rather than by the base
#: grids.  ``forced`` (E1): the clock arrives as an injected ``world_time``
#: result and no time context is rendered, so acquisition is exogenous.
#: ``relation`` (E2): the current instant is rendered *and* the original date
#: and their stated order are delivered, so the conflict is stated rather than
#: left to be inferred.  Both are declared here so the judge's visibility
#: vocabulary and the experiments that mint those cells share one definition.
INTERVENTION_VISIBILITIES = (("forced", False), ("relation", False))

#: Every visibility any experiment in this family can produce.
VISIBILITY_NAMES = (tuple(name for name, _ in VISIBILITIES)
                    + tuple(name for name, _ in INTERVENTION_VISIBILITIES))

#: Fields the six conditions are allowed to differ in.  Everything else is the
#: corpus, and ``--dry-run`` fails if anything else varies.  An allow-list rather
#: than a deny-list: a new spec field is compared by default, so a later edit
#: cannot make two conditions differ in a way the check does not look at.
ALLOWED_FIELD_DIFFERENCES = frozenset({
    "experiment_id",
    "advance_days",
    "advance_hours",
    "advance_minutes",
    "observe_time",
    "notes",
    "delivery",
})

#: The transition vocabulary that must not appear anywhere in a corpus whose
#: world never changes.  Frozen, and narrow: it names the statements a notice
#: makes, not every word that could conceivably imply one.
SUPERSESSION_TERMS = (
    "改期",
    "作废",
    "取消",
    "现改为",
    "替代旧确认单",
)

#: World-event kinds that would mean the world changed.  ``publish_evidence``
#: does not: it only makes an already-authored document retrievable, which is why
#: the corpus's documents can appear over time without the fact ever moving.
FORBIDDEN_WORLD_KINDS = ("supersede_fact", "send_notification")


# ------------------------------------------------------------------ the corpus


def build_expiry_variant(*, profile) -> ScenarioVariant:
    """The base variant with the reschedule notice removed and the slot moved.

    The only difference from ``build_variant`` is the notice: leaving it in would
    put a change in the world, and a corpus that contains a change is a
    supersession scenario, not an expiry one.  Everything else — the prefix, the
    frozen background, the remaining documents and their publication instants — is
    the base variant's, and is asserted to be so rather than re-authored here.
    """
    pid = profile.profile_id
    notice_id = f"{pid}_reschedule_notice"
    base = build_variant(
        scenario_id=f"expiry-{pid}", profile=profile, filler_id=FILLER_ID
    )
    documents = tuple(doc for doc in base.documents if doc[0] != notice_id)

    variant = ScenarioVariant(
        scenario_id=base.scenario_id,
        profile=profile,
        filler_id=FILLER_ID,
        # The confirmation is the record that names the operative date, so it
        # moves out of the shared context and into the slot: that is where the
        # harness declares the decisive material, and where a reader expects to
        # find the document the question turns on.  The slot sits at the head, as
        # in every other scenario in this study.
        shared_ids=(
            f"{pid}_preparation_notes",
            f"{pid}_unrelated_invitation",
            f"{pid}_anxiety_chat",
        ),
        documents=documents,
        prefix_turns=base.prefix_turns,
        stable_material=base.stable_material,
        time_zone=base.time_zone,
        prefix_start=base.prefix_start,
        # No world update exists in this design, so there is no offset from one.
        # The six specs count from the prefix instead; ``_advance_from`` is what
        # makes that offset exact, and the clock landing is asserted, not assumed.
        elapsed_base_days=0,
        gold_override=f"{pid}_original_confirmation",
        # The operative date is the *old* date here, because nothing supersedes
        # it.  Setting it is what stops ``operative_date`` from silently meaning
        # the profile's reschedule date, which this corpus never mentions.
        authoritative_date=profile.old_date,
    )
    _assert_expiry_variant(variant, base=base, notice_id=notice_id)
    return variant


def _assert_expiry_variant(variant: ScenarioVariant, *, base: ScenarioVariant,
                           notice_id: str) -> None:
    """Fail closed on a corpus that is not the base corpus minus the notice."""
    ids = [doc_id for doc_id, *_ in variant.documents]
    if notice_id in ids:
        raise ValidationError(
            f"expiry {variant.scenario_id}: the reschedule notice {notice_id!r} is "
            "still in the corpus; the world would change and this would be a "
            "supersession scenario"
        )
    if len(ids) != len(set(ids)):
        raise ValidationError(f"expiry {variant.scenario_id}: duplicate doc ids")
    if variant.gold_id not in ids:
        raise ValidationError(
            f"expiry {variant.scenario_id}: gold {variant.gold_id!r} missing"
        )
    if variant.filler_id not in ids:
        raise ValidationError(
            f"expiry {variant.scenario_id}: filler {variant.filler_id!r} missing"
        )
    if set(variant.shared_ids) & {variant.gold_id, variant.filler_id}:
        raise ValidationError(
            f"expiry {variant.scenario_id}: the slot material is already in the "
            "shared context, so the slot is not what makes it available"
        )
    for doc_id in variant.shared_ids:
        if doc_id not in ids:
            raise ValidationError(
                f"expiry {variant.scenario_id}: shared id {doc_id!r} missing"
            )
    expected = tuple(doc for doc in base.documents if doc[0] != notice_id)
    if tuple(variant.documents) != expected:
        raise ValidationError(
            f"expiry {variant.scenario_id}: the corpus is not the base corpus with "
            "the notice removed, in the base order"
        )
    if tuple(variant.prefix_turns) != tuple(base.prefix_turns):
        raise ValidationError(
            f"expiry {variant.scenario_id}: the prefix turn was edited; the prefix "
            "is the base variant's and must stay comparable across experiments"
        )
    if tuple(variant.stable_material) != tuple(base.stable_material):
        raise ValidationError(
            f"expiry {variant.scenario_id}: the frozen background was edited"
        )
    if variant.prefix_start != base.prefix_start:
        raise ValidationError(
            f"expiry {variant.scenario_id}: the prefix start moved; every instant "
            "in this design is computed as an offset from it"
        )


def audit_no_supersession(variant: ScenarioVariant) -> dict:
    """Prove that no document asserts that the event moved or was cancelled.

    Two independent readings, because either one alone is weak:

    * a **surface** scan for the transition vocabulary.  A substring scan is
      stronger than an assertion scan (it cannot miss a phrasing by failing to
      recognise its structure) at the cost of flagging non-assertive mentions —
      which is the direction that should fail a build, not pass it;
    * an **assertion** scan reusing the gate's own classifier, over the four
      supersession families, so "the notice is absent" is checked by the same
      machinery that would otherwise have detected it as a leak.

    The prefix turns and the frozen background are included: they are model-visible
    text and a change-word there would reintroduce the manipulation by another
    route.
    """
    p = variant.profile
    bodies = {
        doc_id: f"{title} {text} {source}"
        for doc_id, title, text, _, source in variant.documents
    }
    surface_hits = {
        doc_id: [term for term in SUPERSESSION_TERMS if term in blob]
        for doc_id, blob in bodies.items()
    }
    surface_hits = {k: v for k, v in surface_hits.items() if v}

    dialogue = {
        f"prefix_turn[{index}]": text
        for index, text in enumerate(variant.prefix_turns)
    }
    dialogue.update({
        f"stable_material[{index}]": text
        for index, text in enumerate(variant.stable_material)
    })
    dialogue_surface = {
        name: [term for term in SUPERSESSION_TERMS if term in text]
        for name, text in dialogue.items()
    }
    dialogue_surface = {k: v for k, v in dialogue_surface.items() if v}

    families = (
        ("old_appointment_cancelled", (p.referent, p.cancel_word)),
        ("old_appointment_cancelled_long", (p.referent_long, p.cancel_word)),
        ("old_appointment_void", (p.referent, p.void_word)),
        ("old_appointment_void_long", (p.referent_long, p.void_word)),
        ("old_confirmation_superseded", ("预约", "替代旧确认单")),
    )
    asserted = {}
    for name, terms in families:
        hits = sorted(
            doc_id for doc_id, blob in bodies.items()
            if classify_terms(blob, terms) == ASSERTED
        )
        hits += sorted(
            block for block, text in dialogue.items()
            if classify_terms(text, terms) == ASSERTED
        )
        if hits:
            asserted[name] = hits
    return {
        "notice_id": f"{p.profile_id}_reschedule_notice",
        "notice_absent": f"{p.profile_id}_reschedule_notice" not in bodies,
        "surface_terms": list(SUPERSESSION_TERMS),
        "surface_hits": surface_hits,
        "dialogue_surface_hits": dialogue_surface,
        "asserted": asserted,
        "clean": not (surface_hits or dialogue_surface or asserted),
    }


def assert_no_supersession(variant: ScenarioVariant) -> dict:
    audit = audit_no_supersession(variant)
    if not audit["clean"]:
        raise ValidationError(
            f"expiry {variant.scenario_id}: the corpus asserts a supersession: "
            f"surface={audit['surface_hits']} "
            f"dialogue={audit['dialogue_surface_hits']} "
            f"asserted={audit['asserted']}"
        )
    if not audit["notice_absent"]:
        raise ValidationError(
            f"expiry {variant.scenario_id}: the reschedule notice is present"
        )
    return audit


# ------------------------------------------------------------------ contract


def registered_propositions(profile) -> tuple:
    """Every proposition this scenario registers, decisive ones first.

    The decisive rendering is the *record's own*: the issuer's confirmation names
    the holder and the date in one sentence, so "the holder's appointment is on D"
    is a statement a document in this corpus actually makes.  Its presence is what
    makes the material sufficient, and its absence from the pre-fork window is what
    the parent check tests.

    The first-person renderings are registered and marked **non-decisive**, and the
    reason is a property of the base variant rather than a convenience: the base
    prefix turn is the user saying "the confirmation says <D>".  That sentence is
    legitimate pre-fork history — it is where the record came from — so treating it
    as contamination would fail every run at the fork with ``parent_contaminated``,
    which is a claim about a leak and not about the user's own appointment.  They
    stay registered so that ``--dry-run`` can report exactly where they fire,
    making the exclusion visible rather than silent.
    """
    p = profile
    short, spelled = _md(p.old_date), _md_full(p.old_date)
    return (
        Proposition(
            proposition_id="record_places_event_on_d",
            description="the record places this event on D: the issuer's "
                        "confirmation says the holder's appointment is on the old "
                        "date",
            requires=(p.holder, short),
        ),
        Proposition(
            proposition_id="record_places_event_on_d_spelled",
            description="same statement with the year spelled out",
            requires=(p.holder, spelled),
        ),
        Proposition(
            proposition_id="event_named_with_d",
            description="the event is named with its date, without the holder",
            requires=(p.referent, short),
            decisive=False,
        ),
        Proposition(
            proposition_id="event_named_with_d_long",
            description="same, with the event's long name",
            requires=(p.referent_long, short),
            decisive=False,
        ),
    )


def _neutral_material_ids(variant: ScenarioVariant, propositions: Sequence) -> tuple:
    """The corpus documents that assert none of the registered propositions.

    Derived from the texts rather than hand-listed, so a corpus edit that gives a
    "neutral" document a date moves it out of this set by itself.  Recorded in the
    contract because that is where a reader looks for "which material is audited
    as non-decisive".

    Deliberately not the factory's slot pool, which is simply "the corpus minus the
    gold": here the question is which documents *assert* nothing, and the answer is
    used for audit rather than to pick a filler, because this design has no omit
    condition for a filler to fill.  A delivered document can appear here (the
    unrelated invitation asserts no proposition); the two lists answer different
    questions and are printed side by side in the dry run for that reason.
    """
    neutral = []
    for doc_id, _, text, _, _ in variant.documents:
        if not any(prop.asserted_in(text) for prop in propositions):
            neutral.append(doc_id)
    return tuple(neutral)


def build_expiry_contract(variant: ScenarioVariant) -> ExposureContract:
    """The contract for a corpus whose world never changes.

    The decisive proposition is the record's date assertion, *not* a supersession:
    there is no "originally X, now Y" statement anywhere in this corpus, and a
    contract built from one would be checking for text that cannot exist.  What the
    contract has to express instead is the fact the question turns on — this event
    is on D — so that "the material is missing" and "the material is irrelevant"
    remain distinguishable.
    """
    p = variant.profile
    propositions = registered_propositions(p)
    return ExposureContract(
        contract_version=CONTRACT_VERSION,
        event_referents=variant.event_referents(),
        decisive_propositions=propositions,
        neutral_material_ids=_neutral_material_ids(variant, propositions),
        oracle_required_groups=(
            (p.referent, p.referent_long),
            (_md(p.old_date),),
            (p.office_short, p.office),
        ),
    )


def proposition_map(variant: ScenarioVariant, contract: ExposureContract) -> dict:
    """Where each registered proposition is asserted, over the whole corpus.

    Diagnostic evidence rather than a check: it shows a reader (a) which documents
    carry the decisive date assertion, (b) that the probe and the frozen background
    carry none of them, and (c) exactly which blocks the non-decisive, first-person
    renderings fire on — the exclusion described in ``registered_propositions`` is
    therefore visible in the dry run instead of being taken on trust.
    """
    texts = {doc_id: text for doc_id, _, text, _, _ in variant.documents}
    texts.update({
        f"prefix_turn[{index}]": text
        for index, text in enumerate(variant.prefix_turns)
    })
    texts.update({
        f"stable_material[{index}]": text
        for index, text in enumerate(variant.stable_material)
    })
    texts["probe"] = PROBE
    report = {}
    for proposition in contract.decisive_propositions:
        report[proposition.proposition_id] = {
            "decisive": proposition.decisive,
            "requires": list(proposition.requires),
            "asserted_in": sorted(
                name for name, text in texts.items()
                if proposition.asserted_in(text)
            ),
        }
    return report


# ------------------------------------------------------------------- world


def _world_events(variant: ScenarioVariant) -> tuple:
    """The initial fact and the publications — and deliberately nothing else.

    The base variant's schedule carries a ``supersede_fact`` at the update instant.
    It is absent here, and ``assert_world_static`` refuses any spec that has one:
    "the world never changed" is the manipulation, so it is checked on the schedule
    rather than promised in a comment.
    """
    p = variant.profile
    events = [
        WorldEvent(
            event_id=f"{p.profile_id}_evt_initial",
            kind="add_fact",
            at=variant.prefix_start,
            order=0,
            fact_id=f"{p.profile_id}.appointment_date",
            new_value=p.old_date.isoformat(),
            source="scenario-initial",
        ),
    ]
    for index, (doc_id, _, _, published_at, _) in enumerate(variant.documents):
        if parse_instant(published_at) <= parse_instant(variant.prefix_start):
            continue  # already present before the prefix; not a publication
        events.append(
            WorldEvent(
                event_id=f"{p.profile_id}_pub_{index:02d}",
                kind="publish_evidence",
                at=published_at,
                order=1000 + index,
                evidence_id=doc_id,
                source="scenario",
            )
        )
    return tuple(events)


def assert_world_static(spec: ExperimentSpec, profile) -> dict:
    """Refuse a schedule in which the world can move."""
    kinds = sorted({event.kind for event in spec.world_events})
    forbidden = sorted(
        event.event_id for event in spec.world_events
        if event.kind in FORBIDDEN_WORLD_KINDS
    )
    if forbidden:
        raise ValidationError(
            f"expiry {spec.experiment_id}: the schedule moves the world: {forbidden}"
        )
    values = sorted({
        str(event.new_value) for event in spec.world_events
        if event.fact_id == spec.gold_fact_id and event.new_value is not None
    })
    if values != [profile.old_date.isoformat()]:
        raise ValidationError(
            f"expiry {spec.experiment_id}: the fact {spec.gold_fact_id!r} takes the "
            f"values {values}, not exactly {profile.old_date.isoformat()!r}; the "
            "world must not change at all in this experiment"
        )
    return {"kinds": kinds, "fact_values": values}


# ------------------------------------------------------------------- instants


def _offset_text(instant: str) -> str:
    offset = parse_instant(instant).utcoffset()
    if offset is None:
        raise ValidationError(f"{instant!r} carries no UTC offset")
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    hours, minutes = divmod(abs(total) // 60, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def target_instant(variant: ScenarioVariant, offset_days: int) -> str:
    """``D + offset_days`` at the prefix's own local time of day.

    Keeping the time of day fixed (and whole days apart) means the arms differ by
    an exact number of days rather than by an hour someone chose, and the instant
    stays minute-aligned, which is the finest resolution the kernel's clock has.
    """
    prefix = parse_instant(variant.prefix_start)
    day = variant.profile.old_date + timedelta(days=offset_days)
    return f"{day.isoformat()}T{prefix:%H:%M}:00{_offset_text(variant.prefix_start)}"


def _advance_from(prefix_start: str, target: str) -> tuple:
    """The (days, hours, minutes) that carry the clock from the prefix to target.

    Same strictness as the factory's composer: an interval the clock cannot
    represent is refused rather than rounded, because rounding would silently move
    the probe instant and make two conditions closer together than they read.
    """
    delta = parse_instant(target) - parse_instant(prefix_start)
    total = int(delta.total_seconds())
    if total < 0:
        raise ValidationError(
            f"the probe instant {target} precedes the prefix start {prefix_start}: "
            "the clock cannot move backwards"
        )
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds:
        raise ValidationError(
            f"the composed advance from {prefix_start} to {target} is not a whole "
            f"number of minutes ({total}s); the clock has one-minute resolution, so "
            "this instant is not representable rather than approximately reached"
        )
    return days, hours, minutes


# ------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    """One of the six cells: an instant crossed with a visibility."""

    arm: str
    visibility: str
    target: str
    spec: ExperimentSpec

    @property
    def name(self) -> str:
        return f"{self.arm}_{self.visibility}"


#: The fixed-reference control arms (E3).  They live here, next to the base
#: clocks, so the judge's arm vocabulary and the experiment that mints those
#: arms share one definition rather than two copies that can drift apart.
#: ``ref_before`` answers as of D-15 (event still upcoming); ``ref_after``
#: answers as of D+30 (event already past).  The offsets are in whole days
#: relative to the event date and drive both the probe's stated reference and
#: the world's clock, so no new clock is introduced by the E3 driver.
REF_ARMS = (("ref_before", -15), ("ref_after", 30))

#: Every arm name any experiment in this family can produce.
ARM_NAMES = tuple(name for name, _ in ARMS) + tuple(name for name, _ in REF_ARMS)


def build_conditions(*, variant: ScenarioVariant, route: str,
                     tool_budget: int = 8,
                     arms: Sequence = ARMS,
                     visibilities: Sequence = VISIBILITIES,
                     probe_for_arm: Any = None,
                     forced_tools: Sequence = (),
                     extra_registered_text: Sequence = (),
                     extra_contract_tags: dict | None = None) -> tuple:
    """Build all six conditions, sharing everything that must not drift.

    The contract, the pair, the policy, the manifest and the gate closures are
    built **once** and shared by the six specs, for the same reason the factory
    shares a policy object between branches: "identical delivery" then means the
    same object, not six copies that could drift apart.
    """
    p = variant.profile
    pid = p.profile_id
    contract = build_expiry_contract(variant)
    ids = tuple(doc_id for doc_id, *_ in variant.documents)
    pair = build_access_pair(
        shared_ids=variant.shared_ids,
        gold_id=variant.gold_id,
        filler_id=variant.filler_id,
        slot=SLOT_INDEX,
        available_ids=ids,
    )
    # ``oracle`` because the decisive material is delivered here — in all six
    # conditions.  There is no omit arm: delivery is not the manipulated variable.
    policy = exposure_for_condition(
        "oracle", doc_ids=pair.oracle_doc_ids, gold_id=pair.gold_id,
        top_k=len(pair.oracle_doc_ids),
    )
    manifest = DeliveryManifest(
        condition="oracle",
        expected_material_ids=pair.oracle_doc_ids,
        slot_index=pair.slot,
        slot_material_id=variant.gold_id,
    )
    bundle = PromptBundle(
        prompt_id=f"expiry-{pid}-probe",
        system=neutral_system_prompt(),
        tool_descriptions={"world_time": "Get the current date and time."},
    )
    surfaces = (bundle.system,) + declared_tool_surfaces(
        bundle.tool_descriptions, PROBE_TOOLS
    )
    # ``extra_registered_text`` lets an intervention register the exact text it
    # injects as a permitted dialogue block.  It must not assert a registered
    # proposition: the caller's decisive-leak assertions still run.
    registered_dialogue = (
        (PROBE,) + tuple(variant.stable_material)
        + tuple(extra_registered_text or ())
    )
    bodies = {doc_id: text for doc_id, _, text, _, _ in variant.documents}
    documents = tuple(
        Document(
            doc_id=doc_id, text=text, source=title or source,
            published_at=published_at, is_gold=(doc_id == variant.gold_id),
        )
        for doc_id, title, text, published_at, source in variant.documents
    )
    world_events = _world_events(variant)
    prefix_policy = ExposurePolicy(
        mode="full", top_k=len(variant.shared_ids), label="prefix"
    )
    shared_delivery = {
        "pair": pair.to_json(),
        "manifests": {"oracle": manifest.to_json()},
        "probe_tools": list(PROBE_TOOLS),
        "forbidden_tools": list(FORBIDDEN_TOOL_NAMES),
        "shared_context_ids": list(variant.shared_ids),
        "profile_id": pid,
        "notice_removed": f"{pid}_reschedule_notice",
        "slot_filler_undelivered": variant.filler_id,
    }

    def gate_factory(candidate: BranchSpec, session: Any):
        if candidate.exposure.identity() != policy.identity():
            raise ValidationError(
                "expiry gate: this factory only checks the record-delivery policy; "
                "a branch with different delivery would be checked against a "
                "manifest that does not describe it"
            )
        return make_request_gate(
            contract, manifest, surfaces, registered_dialogue, corpus_bodies=bodies,
        )

    def parent_gate(visible_history: Sequence[str]) -> ExposureResult:
        return check_parent_state(
            visible_history=visible_history, contract=contract,
            label=f"{pid}-parent",
        )

    conditions = []
    for arm, offset_days in arms:
        for visibility, observe_time in visibilities:
            target = target_instant(variant, offset_days)
            days, hours, minutes = _advance_from(variant.prefix_start, target)
            landed = Clock.of(variant.prefix_start, variant.time_zone).advance(
                days=days, hours=hours, minutes=minutes
            )
            if landed.instant != Clock.of(target, variant.time_zone).instant:
                raise ValidationError(
                    f"expiry {pid}/{arm}/{visibility}: the clock lands on "
                    f"{landed.instant}, not on {target}; the advance was computed "
                    "wrongly"
                )
            probe_text = (
                probe_for_arm(arm, variant) if probe_for_arm else PROBE
            )
            branch = BranchSpec(
                branch_id=BRANCH_ID, exposure=policy, user_text=probe_text,
                label=BRANCH_LABEL, forced_tools=tuple(forced_tools),
            )
            spec = ExperimentSpec(
                experiment_id=f"expiry-{route}-{pid}-{arm}-{visibility}",
                clock_start=Clock.of(variant.prefix_start, variant.time_zone),
                time_zone=variant.time_zone,
                prompt=bundle,
                documents=documents,
                world_events=world_events,
                enabled_tools=PROBE_TOOLS,
                prefix_turns=tuple(variant.prefix_turns),
                branches=(branch,),
                prefix_exposure=prefix_policy,
                window_boundaries=(
                    WindowBoundary(
                        after_prefix_turn=len(variant.prefix_turns) - 1,
                        reason="the user returns in a later session; the record must "
                               "arrive through retrieval, not as retained chat",
                    ),
                ),
                stable_material=tuple(
                    (f"{pid}-bg-{index:02d}", text)
                    for index, text in enumerate(variant.stable_material)
                ),
                host_query=_host_query(variant),
                gold_fact_id=f"{pid}.appointment_date",
                gold_marker=variant.gold_id,
                gold_decisive_terms=_gold_decisive_terms(variant),
                advance_days=days,
                advance_hours=hours,
                advance_minutes=minutes,
                observe_time=observe_time,
                tool_budget=tool_budget,
                exposure_contract=contract,
                delivery={
                    **shared_delivery,
                    "expiry": {
                        **(extra_contract_tags or {}),
                        "arm": arm,
                        "visibility": visibility,
                        "probe": probe_text,
                        "observe_time": observe_time,
                        "target_instant": target,
                        "event_date": p.old_date.isoformat(),
                        "advance_label": f"{days}d{hours}h{minutes}m",
                        "contract_version": CONTRACT_VERSION,
                    },
                },
                request_gate_factory=gate_factory,
                parent_visible_gate=parent_gate,
                notes=(
                    f"scenario={variant.scenario_id} arm={arm} visibility={visibility} "
                    f"clock={target} advance_from_prefix={days}d{hours}h{minutes}m "
                    f"observe_time={observe_time} contract={CONTRACT_VERSION}"
                ),
            )
            assert_world_static(spec, p)
            conditions.append(Condition(arm=arm, visibility=visibility,
                                        target=target, spec=spec))
    return tuple(conditions)


def _host_query(variant: ScenarioVariant) -> str:
    """The registered host query: names the event, asserts nothing.

    Shared verbatim by the six conditions, so retrieval is a function of the
    corpus and the clock rather than of which arm is running.  It carries no date
    and no claim, and it does not mention rescheduling: this is an expiry
    scenario, and a query asking for "the reschedule notice" would describe a
    document that is not in the corpus.
    """
    p = variant.profile
    query = f"我们之前一起准备的{p.referent_long}：预约确认单上的日期、当前是否仍然有效"
    for forbidden in (_md(p.new_date), _md(p.old_date), variant.gold_id,
                      p.cancel_word, p.void_word):
        if forbidden in query:
            raise ValidationError(
                f"host query for {p.profile_id!r} contains {forbidden!r}; the "
                "registered query must not carry the answer or the claim"
            )
    return query


# ------------------------------------------------------- the six-way identity


def _comparable(spec: ExperimentSpec, name: str) -> Any:
    """One spec field as the six-way comparison should see it.

    ``delivery`` carries a per-condition sub-dict (arm, visibility, instant), which
    is one of the things allowed to differ; the rest of it — the delivered pair,
    the manifest, the tool list, the shared context — must not.
    """
    value = getattr(spec, name)
    if name == "delivery" and isinstance(value, dict):
        return {key: item for key, item in value.items() if key != "expiry"}
    return value


#: Every spec field that carries the corpus: all of them except the ones the six
#: conditions are allowed to differ in.  Derived from the dataclass rather than
#: listed, so a field added to ``ExperimentSpec`` later is compared by default
#: instead of being silently left out of the check.
CORPUS_FIELDS = tuple(
    field.name for field in fields(ExperimentSpec)
    if field.name not in ALLOWED_FIELD_DIFFERENCES
) + ("delivery",)


def _canonical(spec: ExperimentSpec, name: str) -> Any:
    """One spec field rendered for hashing: objects become their own JSON."""
    value = _comparable(spec, name)
    if name == "prompt":
        return {"prompt_id": value.prompt_id, "digest": value.digest()}
    if name == "documents":
        return [document.to_json() for document in value]
    if name == "world_events":
        return [event.to_json() for event in value]
    if name == "prefix_exposure":
        return value.identity() if value is not None else None
    if name == "window_boundaries":
        return [boundary.to_json() for boundary in value]
    if name == "stable_material":
        return [list(item) for item in value]
    if name == "branches":
        return [
            {
                "branch_id": branch.branch_id,
                "exposure": branch.exposure.identity(),
                "user_text": branch.user_text,
                "label": branch.label,
            }
            for branch in value
        ]
    if name == "clock_start":
        return value.to_json()
    if name == "exposure_contract":
        return value.digest() if value is not None else None
    if callable(value):
        # The two gate closures.  Hashed by name only; that they are the *same*
        # object in all six conditions is what ``differing_fields`` checks, by
        # identity, because a copy would hash the same and still be a second
        # manifest waiting to drift.
        return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
    return value


def differing_fields(specs: Sequence[ExperimentSpec]) -> set:
    field_names = [field.name for field in fields(ExperimentSpec)]
    differing = set()
    for name in field_names:
        values = [_comparable(spec, name) for spec in specs]
        if any(value != values[0] for value in values[1:]):
            differing.add(name)
    return differing


def corpus_fingerprint(spec: ExperimentSpec) -> dict:
    return {name: _canonical(spec, name) for name in CORPUS_FIELDS}


def assert_corpus_identical(conditions: Sequence[Condition],
                            extra_allowed: frozenset = frozenset()) -> dict:
    """The corpora are byte-identical; only the clock and the visibility differ.

    Two readings of the same claim, because each alone is weak: a **field diff**
    over the whole dataclass (which can only pass by comparison, so a difference
    the check did not anticipate still fails) and a **fingerprint** (which records
    what was compared, so the evidence directory shows the six conditions hashing
    the same corpus rather than asserting that they did).
    """
    specs = [condition.spec for condition in conditions]
    differing = differing_fields(specs)
    allowed = ALLOWED_FIELD_DIFFERENCES | set(extra_allowed)
    unexpected = sorted(differing - allowed)
    if unexpected:
        raise ValidationError(
            "the conditions differ outside the allowed fields "
            f"{sorted(allowed)}: {unexpected}"
        )
    canonical = [corpus_fingerprint(spec) for spec in specs]
    for fingerprint in canonical:
        for name in extra_allowed:
            fingerprint.pop(name, None)
    fingerprint = digest(canonical[0])
    for spec, other in zip(specs[1:], canonical[1:]):
        if digest(other) != fingerprint:
            changed = sorted(
                name for name in CORPUS_FIELDS
                if canonical[0][name] != other[name]
            )
            raise ValidationError(
                f"expiry {spec.experiment_id}: the corpus differs from the other "
                f"conditions in {changed}"
            )
    return {
        "fingerprint": fingerprint,
        "differing_fields": sorted(differing),
        "unexpected": unexpected,
        "allowed": sorted(ALLOWED_FIELD_DIFFERENCES | set(extra_allowed)),
        "parts": {
            name: digest([entry[name] for entry in canonical])
            for name in ("documents", "world_events", "prefix_turns",
                         "stable_material", "host_query", "branches", "prompt")
            if name not in set(extra_allowed)
        },
    }


# ------------------------------------------------------------------- running


def _summarise(report: dict) -> dict:
    parent = (report.get("exposure") or {}).get("parent_check") or {}
    rows = []
    for branch in report["branches"]:
        gate = branch["exposure_verdicts"]
        rows.append({
            "branch_id": branch["branch_id"],
            "valid": branch["run_valid"],
            "invalid_reason": branch.get("invalid_reason", ""),
            "gate": [verdict["status"] for verdict in gate],
            "gate_conditions": [verdict["condition"] for verdict in gate],
            "gate_reasons": [sorted(verdict["reason_codes"]) for verdict in gate],
            "payload_digest": [verdict["payload_digest"] for verdict in gate],
            "delivered": branch["auto_retrieval"].get("hits"),
            "gold_delivered": branch["auto_retrieval"].get("gold_delivered"),
            "answer_len": len(branch["outcome"].get("assistant_text", "")),
            "failure": (branch["outcome"].get("failure") or {}).get("reason", ""),
        })
    return {
        "branches": rows,
        "spec_digest": report.get("spec_digest", ""),
        "corpus_digest": report.get("corpus_digest", ""),
        "schedule_digest": report.get("schedule_digest", ""),
        "parent_check": parent,
        "parent_equality": report.get("parent_equality", {}),
    }


def run_condition(*, condition: Condition, out_dir: Path,
                  model: Any) -> dict:
    """Run one condition.  The runner deletes nothing: a run directory is evidence."""
    started = time.monotonic()
    report = ExperimentRunner(condition.spec, model, out_dir=out_dir).run()
    return {
        "arm": condition.arm,
        "visibility": condition.visibility,
        "target_instant": condition.target,
        "wall_seconds": round(time.monotonic() - started, 1),
        **_summarise(report),
    }


# --------------------------------------------------------------------- dry run


def _print_audit(prefix: str, audit: dict) -> None:
    print(f"    {prefix}: {'PASS' if audit['clean'] else 'FAIL'} "
          f"(notice absent={audit['notice_absent']}; "
          f"surface hits={len(audit['surface_hits'])}; "
          f"dialogue hits={len(audit['dialogue_surface_hits'])}; "
          f"asserted families={len(audit['asserted'])}; "
          f"terms={len(audit['surface_terms'])})")


def dry_run(*, routes: Sequence[str], profiles: Sequence, json_path: str = "") -> int:
    """Build and run all six conditions offline, and check what must hold.

    Offline means ``MockModel``: no provider call, no API key read, no paid token,
    and no evidence directory written — the runs happen in a temporary directory
    that is removed at the end.  The gate is the real gate on the real compiled
    request, which is the only way a dry run can speak to what a paid run would
    send.
    """
    print("EXPIRY EXPERIMENT DRY RUN (naturally expired)")
    print("  offline: MockModel; no provider call, no API key, no evidence written")
    print(f"  arms: {[name for name, _ in ARMS]} "
          f"(days from the event date) x visibilities "
          f"{[name for name, _ in VISIBILITIES]}")
    print(f"  probe: {PROBE!r}  [{PROBE_RATIONALE}]")
    rows = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="expiry-dryrun-") as scratch:
        for route in routes:
            for profile in profiles:
                print(f"\n  === route {route} | profile {profile.profile_id} ===")
                variant = build_expiry_variant(profile=profile)
                audit = assert_no_supersession(variant)
                conditions = build_conditions(variant=variant, route=route)
                identity = assert_corpus_identical(conditions)
                world = assert_world_static(conditions[0].spec, profile)
                print(f"    event date D          : {profile.old_date.isoformat()}")
                print(f"    prefix start          : {variant.prefix_start}")
                print(f"    corpus fingerprint    : {identity['fingerprint'][:32]}")
                print("    six-way corpus check  : PASS "
                      f"(differing fields={identity['differing_fields']})")
                print(f"    world static          : PASS (kinds={world['kinds']}, "
                      f"fact values={world['fact_values']})")
                _print_audit("no supersession       ", audit)
                print(f"    delivered record      : "
                      f"{list(conditions[0].spec.delivery['pair']['oracle_doc_ids'])}")
                print(f"    neutral material      : "
                      f"{list(conditions[0].spec.exposure_contract.neutral_material_ids)}")
                print("    registered propositions (where each is asserted):")
                for name, entry in proposition_map(
                    variant, conditions[0].spec.exposure_contract
                ).items():
                    kind = "decisive" if entry["decisive"] else "documentation"
                    print(f"      [{kind:13s}] {name:34s} "
                          f"requires={entry['requires']} "
                          f"-> {entry['asserted_in'] or 'nowhere'}")
                payloads = {}
                for condition in conditions:
                    out_dir = Path(scratch) / f"{route}-{profile.profile_id}-{condition.name}"
                    try:
                        summary = run_condition(
                            condition=condition, out_dir=out_dir, model=MockModel(),
                        )
                    except Exception as exc:  # a rejected gate is the failure to surface
                        print(f"      {condition.arm:5s} {condition.visibility:9s} "
                              f"target={condition.target} RUN FAILED: "
                              f"{type(exc).__name__}: {exc}")
                        failures.append(f"{route}/{profile.profile_id}/{condition.name}: "
                                        f"{type(exc).__name__}: {exc}")
                        continue
                    branch = summary["branches"][0]
                    parent = summary["parent_check"]
                    spec = condition.spec
                    statuses = "/".join(branch["gate"])
                    payload = str(branch["payload_digest"][0]) if branch["payload_digest"] else ""
                    payloads[condition.name] = payload
                    clean = (
                        branch["valid"]
                        and set(branch["gate"]) == {"PASS"}
                        and parent.get("status") == "PASS"
                    )
                    mark = "ok " if clean else "BAD"
                    if not clean:
                        failures.append(
                            f"{route}/{profile.profile_id}/{condition.name}: "
                            f"valid={branch['valid']} gate={branch['gate']} "
                            f"parent={parent.get('status')} "
                            f"reasons={branch['gate_reasons']} "
                            f"parent_reasons={parent.get('reason_codes')}"
                        )
                    print(f"      {mark} {condition.arm:5s} {condition.visibility:9s} "
                          f"target={condition.target} "
                          f"advance={spec.advance_days}d{spec.advance_hours}h"
                          f"{spec.advance_minutes}m observe_time={spec.observe_time} "
                          f"gate={statuses} parent={parent.get('status')} "
                          f"delivered={len(branch['delivered'] or [])} "
                          f"gold={branch['gold_delivered']} "
                          f"payload={payload[:16]}")
                    if not clean:
                        print(f"           gate conditions={branch['gate_conditions']} "
                              f"reasons={branch['gate_reasons']} "
                              f"parent_reasons={parent.get('reason_codes')}")
                    rows.append({
                        "route": route, "profile": profile.profile_id,
                        "arm": condition.arm, "visibility": condition.visibility,
                        "target_instant": condition.target,
                        "advance_days": spec.advance_days,
                        "observe_time": spec.observe_time,
                        "corpus_fingerprint": identity["fingerprint"],
                        "differing_fields": identity["differing_fields"],
                        "world": world,
                        "supersession_clean": audit["clean"],
                        "propositions": proposition_map(
                            variant, spec.exposure_contract
                        ),
                        **{k: v for k, v in summary.items() if k != "branches"},
                        "branch": summary["branches"][0],
                    })
                distinct = sorted(set(payloads.values()))
                print(f"    request payloads      : {len(distinct)} distinct across "
                      f"the {len(payloads)} cells")
                if len(distinct) < len(payloads):
                    print("      (expected, and worth stating: with observe_time=False "
                          "nothing renders the clock, so the three natural cells send "
                          "byte-identical requests.  In that arm a difference between "
                          "pre and post can only come from the model asking world_time "
                          "— or from provider non-determinism.)")
    if json_path:
        Path(json_path).write_text(
            json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False,
                       indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\n  wrote {json_path}")
    print("\nEXPIRY DRY RUN SUMMARY")
    print(f"  conditions run : {len(rows)}")
    print(f"  gate PASS      : {sum(1 for r in rows if set(r['branch']['gate']) == {'PASS'})}"
          f"/{len(rows)}")
    print(f"  six-way corpus : "
          f"{'PASS' if not failures else 'FAIL'}")
    if failures:
        print("  FAILURES:")
        for failure in failures:
            print(f"    - {failure}")
        return 1
    print("  all checks passed")
    return 0


# ------------------------------------------------------------------ paid run


def _model(route: str, *, base_url: str, api_key_env: str,
           timeout: float) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        base_url=base_url, model=route, api_key_env=api_key_env,
        timeout=timeout, max_transport_retries=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", nargs="+", default=None,
                        help="defaults to scope.SUBJECT_ROUTES")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="scenario profile ids; defaults to all")
    parser.add_argument("--source", choices=("factory", "expansion"),
                        default="factory",
                        help="factory = the pilot's hand-written pool; expansion = "
                             "the multi-kind pool")
    parser.add_argument("--per-kind", type=int, default=0,
                        help="with --source expansion, take this many instances per "
                             "event kind; 0 means all")
    parser.add_argument("--instance-variants", type=int, default=4,
                        help="instances per kind to build before --per-kind filters")
    parser.add_argument("--out", default="runs_expiry")
    parser.add_argument("--resume", action="store_true",
                        help="skip a cell that already has a report.json instead of "
                             "removing and re-running it; for continuing an interrupted run "
                             "without spending the completed cells' calls again")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-json", default="",
                        help="with --dry-run, write the offline evidence to this path")
    args = parser.parse_args()

    routes = tuple(args.routes) if args.routes else SUBJECT_ROUTES
    try:
        assert_in_scope(routes)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    if args.source == "expansion":
        from tools.scenario_expansion import build_profiles

        pool = build_profiles(instances=args.instance_variants)
        if args.per_kind > 0:
            seen: dict = {}
            selected = []
            for profile in pool:
                kind = profile.profile_id.rsplit("_i", 1)[0]
                seen.setdefault(kind, 0)
                if seen[kind] < args.per_kind:
                    seen[kind] += 1
                    selected.append(profile)
            pool = selected
        profiles = pool
    else:
        profiles = [p for p in PROFILES
                    if args.profiles is None or p.profile_id in set(args.profiles)]
    if args.profiles:
        wanted = set(args.profiles)
        profiles = [p for p in profiles if p.profile_id in wanted]
    if not profiles:
        print("no scenarios selected", file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(routes=routes, profiles=profiles, json_path=args.dry_run_json)

    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    # Every condition is built and every offline check is run *before* the first
    # paid call: a corpus that fails the six-way identity check or still asserts a
    # supersession must not cost anything to discover.
    planned = []
    for route in routes:
        for profile in profiles:
            variant = build_expiry_variant(profile=profile)
            assert_no_supersession(variant)
            conditions = build_conditions(variant=variant, route=route)
            assert_corpus_identical(conditions)
            assert_world_static(conditions[0].spec, profile)
            planned.append((route, profile, conditions))

    calls = sum(len(conditions) for _, _, conditions in planned)
    print(f"EXPIRY EXPERIMENT: {len(planned)} (route, profile) pairs, "
          f"{calls} runs, {calls} paid calls")
    for route, profile, conditions in planned:
        print(f"  {route} | {profile.profile_id} | D={profile.old_date.isoformat()} | "
              f"{[condition.name for condition in conditions]}")

    out_root = Path(args.out)
    everything = []
    for route, profile, conditions in planned:
        for condition in conditions:
            key = f"expiry | {route} | {profile.profile_id} | {condition.name}"
            print(f"\n=== {key} === ", flush=True)
            out = out_root / "expiry" / route / f"{profile.profile_id}_{condition.name}"
            # A run directory is evidence and is never overwritten; a rerun at the
            # same cell is a new decision, so the old one is removed deliberately.
            #
            # ``--resume`` is the exception, and it is opt-in: a cell that already has a
            # report is left exactly as it is and skipped.  It exists because a paid run
            # can be interrupted by something that is not the experiment's fault -- the
            # first attempt at this one lost five routes to a storage device that went
            # away mid-write -- and re-running completed cells would spend the calls
            # again and destroy the evidence of what was actually run.
            if out.exists() and args.resume and (out / "report.json").exists():
                print(f"  skipped (already complete): {out}", flush=True)
                continue
            if out.exists():
                shutil.rmtree(out)
            try:
                result = run_condition(
                    condition=condition, out_dir=out,
                    model=_model(route, base_url=args.base_url,
                                 api_key_env=args.api_key_env, timeout=args.timeout),
                )
            except Exception as exc:
                print(f"  RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                everything.append({
                    "arm": condition.arm, "visibility": condition.visibility,
                    "route": route, "profile": profile.profile_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            everything.append({
                "route": route, "profile": profile.profile_id, **result,
            })
            branch = result["branches"][0]
            print(f"  wall {result['wall_seconds']}s | gate {branch['gate']} | "
                  f"delivered {branch['delivered']} | gold {branch['gold_delivered']}",
                  flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "expiry_summary.json"
    summary_path.write_text(
        json.dumps(everything, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {summary_path}")
    print(f"runs: {len(everything)}; errors: "
          f"{sum(1 for row in everything if 'error' in row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
