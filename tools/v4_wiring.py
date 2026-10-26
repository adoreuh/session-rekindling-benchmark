"""Wire the v4 scenario, its controlled-delivery pair and its exposure contract
into a runnable ``ExperimentSpec``.

This is the integration point, kept out of both sides on purpose.  The kernel
knows nothing about visa interviews; the scenario module knows nothing about
event logs or checkpoints.  Everything that has to be true *about this
experiment* lives here, so a reader asking "what did this run actually do?" has
one file to read instead of having to reconstruct it from two.

Four decisions are made here, each for a stated reason.

**The shared context is registered, not retrieved.**  The two conditions must
differ in exactly one slot.  If the shared part were produced by a live
retrieval, the two conditions' shared documents could drift apart through a
scoring change and the comparison would silently stop being paired.  Naming the
shared set makes the pairing a property of the registration.

**The slot sits at the head.**  The gold notice is the top-ranked natural hit for
the registered host query, so placing the slot first reproduces where the
evidence would have appeared.  The position is identical in both conditions
because the manifest records one slot index, and the gate re-checks the rendered
request against it.

**The probe turn has no ``search`` tool.**  This is the "A+" protocol: the
schema is absent *and* the tool belt refuses the name at dispatch, so removal is
enforced rather than requested.  It matters because the manipulation is now
*purely* host-controlled delivery.  Leaving the model a retrieval path would
reintroduce the confound the controlled pair exists to remove: under ``omit``
the model could recover the update on its own, and under ``oracle`` it would
still look like it needed to.

    The honest limit of this choice: it removes self-initiated verification
    from the design entirely, so this run cannot speak to *whether* the model
    would have gone looking.  That question needs the separate design in
    ``NEXT_ROUND_PLAN.md`` and must not be answered from this data.

**Both conditions get the same tools and the same budget.**  Removing ``search``
is a whole-run property, not a condition, so it cannot become a second
manipulated variable.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from vela_lab.clock import Clock
from vela_lab.errors import ValidationError
from vela_lab.evidence import Document, ExposurePolicy
from vela_lab.prompt import PromptBundle, neutral_system_prompt
from vela_lab.runner import BranchSpec, ExperimentSpec, WindowBoundary
from vela_lab.schedule import WorldEvent
from vela_lab.score import date_forms

from tools import scenario_v4 as sc
from tools.access_pair import AccessPair, build_access_pair, select_filler
from tools.exposure_gate import (
    TRUNCATION_MARKER_PATTERN,
    DeliveryManifest,
    ExposureResult,
    check_parent_state,
    check_request,
)

#: Ids that make up the shared half of both conditions.  Chosen from the
#: natural ranking for the registered host query and then frozen: the old
#: confirmation (which asserts the *wrong* date, so the model is not being set
#: up with a one-sided context), the user's own preparation note, and the
#: unrelated job interview that a topical match would drag in.  A shared context
#: that contained only the answer would make the omit condition a straw man.
SHARED_CONTEXT_IDS = (
    "visa_original_confirmation",
    "visa_preparation_notes",
    "job_interview_invitation",
)

#: The slot is first because the gold notice is the top-ranked natural hit.
SLOT_INDEX = 0

#: The probe turn's tool set.  ``search`` is absent by decision, not oversight.
PROBE_TOOLS = ("world_time",)

#: Anything named here must be absent from the serialized tool list.  Kept as a
#: separate declaration so that "the model cannot retrieve" is asserted against
#: the payload rather than inferred from the configuration that was passed in.
FORBIDDEN_TOOL_NAMES = ("search",)

#: Name fragments that make a tool retrieval-capable.
#:
#: Why a fragment test rather than an exact-name list.  The first version checked
#: ``name in FORBIDDEN_TOOL_NAMES``, which blocks the literal string ``search``
#: and nothing else: ``search-notes``, ``search_notes``, ``Search``, and
#: ``websearch`` all pass it.  Any of those would reopen the very retrieval route
#: the protocol exists to close, while the gate reported PASS.
#:
#: The lesson is borrowed from the mature harness's registry, which validates
#: tool names by *normalising* them (it rejects a new name whose ``-``/``_`` form
#: collides with an existing one) instead of comparing raw strings.  Same idea
#: here: normalise case and separators, then test for the capability, so a name
#: has to be genuinely non-retrieval to get through rather than merely spelled
#: differently.
RETRIEVAL_NAME_FRAGMENTS = (
    "search", "retriev", "lookup", "look_up", "query", "fetch", "recall",
    "find", "browse", "knowledge", "memory", "rag",
)


def retrieval_capable_names(names: Sequence[str]) -> list:
    """Names, from ``names``, that would give the model a retrieval route.

    Normalisation is deliberately aggressive — case-folded, separators removed —
    because the cost of a false positive here is a failed run, and the cost of a
    false negative is a silently invalid measurement.
    """
    flagged = []
    for name in names:
        normalised = str(name).strip().lower().replace("-", "").replace("_", "")
        normalised = normalised.replace(".", "").replace(" ", "")
        if any(frag.replace("_", "") in normalised
               for frag in RETRIEVAL_NAME_FRAGMENTS):
            flagged.append(name)
    return sorted(set(flagged))


def build_pair() -> AccessPair:
    """Mint the controlled pair, choosing the filler from the registration."""
    bodies = {document.doc_id: document.text for document in sc.DOCUMENTS}
    filler_id = select_filler(
        gold_id=sc.GOLD_MARKER,
        documents=bodies,
        slot_pool=tuple(sc.NEUTRAL_MATERIAL_IDS),
    )
    return build_access_pair(
        shared_ids=SHARED_CONTEXT_IDS,
        gold_id=sc.GOLD_MARKER,
        filler_id=filler_id,
        slot=SLOT_INDEX,
        available_ids=tuple(bodies),
    )


def _policies() -> dict:
    """One exposure policy per condition, delivering the pair's ordered lists."""
    pair = build_pair()
    return {
        "oracle": ExposurePolicy(
            mode="full", top_k=len(pair.oracle_doc_ids),
            deliver_ids=pair.oracle_doc_ids, label="oracle",
        ),
        "omit": ExposurePolicy(
            mode="full", top_k=len(pair.omit_doc_ids),
            deliver_ids=pair.omit_doc_ids,
            # Naming the gold here changes nothing once the delivered list is
            # fixed; it is a second lock so a later edit to the delivery path
            # cannot quietly reintroduce it.
            omit_ids=(sc.GOLD_MARKER,), label="omit",
        ),
    }


def _manifests() -> dict:
    pair = build_pair()
    return {
        "oracle": DeliveryManifest(
            condition="oracle",
            expected_material_ids=pair.oracle_doc_ids,
            slot_index=pair.slot,
            slot_material_id=pair.gold_id,
        ),
        "omit": DeliveryManifest(
            condition="omit",
            expected_material_ids=pair.omit_doc_ids,
            slot_index=pair.slot,
            slot_material_id=pair.filler_id,
        ),
    }


def _branches(policies: dict) -> tuple:
    """Four branches: 2 pressure x 2 delivery, all asserting the wrong date.

    The pressure levels share a verbatim factual template -- the scenario module
    asserts that at import -- so the only thing varying within a delivery
    condition is the emotional sentence.
    """
    out = []
    for pressure in ("low", "high"):
        for condition in ("omit", "oracle"):
            out.append(
                BranchSpec(
                    branch_id=f"claim_{pressure}_{condition}",
                    exposure=policies[condition],
                    user_text=sc.PROBES[("claim", pressure)],
                    label=f"claim|{pressure}|{condition}",
                )
            )
    return tuple(out)


def _bundle() -> PromptBundle:
    return PromptBundle(
        prompt_id=f"{sc.EXPOSURE_CONTRACT_VERSION}-probe",
        system=neutral_system_prompt(),
        tool_descriptions={"world_time": "Get the current date and time."},
    )


def _corpus_bodies() -> dict:
    """Every registered body, for the exclusivity half of the gate."""
    return {document.doc_id: document.text for document in sc.DOCUMENTS}


#: Non-evidence surfaces that legitimately appear in a request: the harness's own
#: packaging, registered so the residue check can tell it apart from content.
#: Everything here is either a fixed string or a shape only a registered renderer
#: produces.  Nothing here may name the event or a date -- a registered wrapper
#: is still model-visible text, so registering it does not make it safe to say
#: anything in it.
_PACKAGING_SURFACES = (
    "Current date and time:",
    "Relevant notes from earlier conversations:",
)

#: Rendered-value shapes.  The current time is derived from the virtual clock, so
#: it cannot be a literal; it is registered as the exact shape the clock renderer
#: emits, which is narrow enough that no date-bearing content could masquerade as
#: it.  The bullet marker is a pattern rather than the literal ``"- "`` so it can
#: only consume a dash at the start of a list entry, never a hyphen inside a date.
_PACKAGING_PATTERNS = (
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"),
    re.compile(r"\([A-Za-z]+/[A-Za-z_]+\)"),
    re.compile(r"(?:^|\s)-\s"),
    # A truncation marker is harness-authored text, so it is packaging rather
    # than content -- but it is still model-visible, which is why it is
    # registered here instead of being ignored.  Its presence means the request
    # was cut, and the truncation check below is what decides whether the
    # delivery claim survives that.
    TRUNCATION_MARKER_PATTERN,
)

#: How large a request may get before server-side truncation becomes a credible
#: explanation for missing evidence.  Context windows are far larger than this;
#: the bound is set well below any served window so that a request under it
#: cannot have been truncated *for length* by a normal endpoint.  Its purpose is
#: to make "the harness sent it" a strong claim rather than to model a real
#: limit — see the note on the check itself.
MAX_PAYLOAD_BYTES = 200_000


def declared_tool_surfaces(tool_descriptions: dict, enabled: Sequence[str]) -> tuple:
    """The non-evidence surfaces declared by the *spec*, not by the payload.

    Taken from the frozen bundle rather than read out of the request, so that a
    description or tool name that appears in a request without having been
    declared is residue rather than a surface.  Reading the surfaces out of the
    payload would make the check self-approving: anything sent would define what
    was allowed to be sent.
    """
    surfaces = list(_PACKAGING_SURFACES)
    for name in enabled:
        surfaces.append(name)
        description = tool_descriptions.get(name, "")
        if description:
            surfaces.append(description)
            # The residue check strips literal strings, and a trailing period is
            # not part of what the description *says*; both forms are registered
            # so a period cannot leave a one-character residue behind.
            surfaces.append(description.rstrip("."))
    pieces = []
    for entry in surfaces:
        entry = str(entry).strip()
        if entry:
            pieces.append(entry)
    return tuple(sorted(set(pieces)))


def make_request_gate(contract, manifest: DeliveryManifest, allowed_surfaces,
                      registered_dialogue, corpus_bodies=None,
                      max_payload_bytes: int = MAX_PAYLOAD_BYTES):
    """Build the per-request checker for one condition.

    The closure captures the manifest rather than receiving it per call, because
    the delivered set is a property of the *condition*: allowing it to vary per
    request would mean a run could switch delivery mid-turn and still be scored
    as one condition.

    ``corpus_bodies`` defaults to the hand-written v4 module's corpus.  It is a
    parameter because the gate must check the exclusivity rule against the corpus
    the *run* uses: reading the wrong corpus does not fail loudly, it silently
    reports every delivered document as absent while the run proceeds.  Callers
    driving a different scenario must pass their own.
    """
    bodies = dict(corpus_bodies) if corpus_bodies is not None else _corpus_bodies()

    def gate(compiled) -> ExposureResult:
        result = check_request(
            condition=manifest.condition,
            payload=compiled.payload,
            contract=contract,
            corpus_bodies=bodies,
            manifest=manifest,
            allowed_surfaces=allowed_surfaces,
            allowed_patterns=_PACKAGING_PATTERNS,
            registered_dialogue=registered_dialogue,
            max_payload_bytes=max_payload_bytes,
        )
        # The no-self-retrieval protocol, checked on the bytes that would be sent.
        # A tool present in the payload is reachable however the belt is
        # configured, so the belt's own refusal is not sufficient evidence.
        sent_tools = {
            (schema.get("function") or {}).get("name", "")
            for schema in compiled.payload.get("tools", []) or []
        }
        # Any retrieval-capable name, however spelled.  An exact-name check
        # (``"search" in forbidden``) passes ``search-notes`` and every other
        # respelling, which would reopen the route while the gate said PASS.
        forbidden = retrieval_capable_names(sorted(sent_tools))
        if forbidden:
            result.status = "RUN_INVALID"
            result.reason_codes.append("forbidden_tool_exposed")
            result.evidence["forbidden_tools"] = forbidden
        # A tool the spec never declared is also a violation: it would be a
        # second, unreviewed change to the protocol riding along with the first.
        undeclared = sorted(sent_tools - set(PROBE_TOOLS))
        if undeclared:
            result.status = "RUN_INVALID"
            result.reason_codes.append("undeclared_tool_exposed")
            result.evidence["undeclared_tools"] = undeclared
        result.evidence["probe_tools"] = sorted(sent_tools)
        return result

    return gate


def build_v4_spec(
    *,
    experiment_id: str,
    elapsed_days: int,
    tool_budget: int = 8,
    prefix_turns: Sequence[str] = (),
) -> ExperimentSpec:
    """A gated v4 experiment: shared prefix, then four paired probe branches."""
    if elapsed_days < 0:
        raise ValidationError("elapsed_days must not be negative")

    contract = sc.exposure_contract()
    policies = _policies()
    manifests = _manifests()
    pair = build_pair()
    bundle = _bundle()

    if not prefix_turns:
        prefix_turns = tuple(
            message.content for message in sc.bundle().prefix
            if message.role == "user"
        )

    prefix_start = sc.PREFIX_START
    update_at = sc.UPDATE_AT
    # The scenario counts elapsed from the world update; the kernel advances from
    # the prefix.  Converted here so "N days after the notice" means the same
    # thing on both sides.
    from vela_lab.clock import parse_instant
    offset = (
        parse_instant(update_at) - parse_instant(prefix_start)
    ).total_seconds() / 86400.0
    advance_days = int(round(offset + elapsed_days))

    # The world starts holding the *old* date, and only the registered update
    # event changes it.  Stating the initial value as an event rather than as an
    # initial-state shortcut keeps a single provenance for every fact: a reader
    # can point at the event that made the date 6 月 18 日, and at the one that
    # voided it, instead of inferring the first from a constructor argument.
    events = [
        WorldEvent(
            event_id="v4_evt_initial_appointment",
            kind="add_fact",
            at=prefix_start,
            order=0,
            fact_id=sc.FACT_KEY,
            new_value=sc.WRONG_DATE,
            source="scenario-initial",
        )
    ]
    for index, raw in enumerate(sc.WORLD_EVENTS, start=1):
        if raw.kind == "supersede_fact":
            events.append(
                WorldEvent(
                    event_id=f"v4_evt_{index:03d}_supersede",
                    kind="supersede_fact",
                    at=raw.at,
                    order=index,
                    fact_id=raw.fact_key,
                    new_value=raw.value,
                    source="scenario",
                )
            )
        elif raw.kind == "publish_evidence":
            events.append(
                WorldEvent(
                    event_id=f"v4_evt_{index:03d}_publish",
                    kind="publish_evidence",
                    at=raw.at,
                    order=index,
                    evidence_id=raw.doc_id,
                    source="scenario",
                )
            )

    documents = tuple(
        Document(
            doc_id=document.doc_id,
            text=document.text,
            source=document.title or document.source,
            published_at=document.published_at,
            is_gold=(document.doc_id == sc.GOLD_MARKER),
        )
        for document in sc.DOCUMENTS
    )

    allowed_surfaces = (bundle.system,) + declared_tool_surfaces(
        bundle.tool_descriptions, PROBE_TOOLS
    )
    # Scenario-authored visible text: every probe (both pressure levels, all
    # conditions) and the frozen background.  Registering the whole probe set
    # rather than only the branch's own probe keeps the registration a property
    # of the scenario, so a branch cannot be made valid by what it happens to
    # deliver.
    registered_dialogue = (
        tuple(sc.PROBES.values()) + tuple(sc.STABLE_MATERIAL)
    )

    def gate_factory(branch: BranchSpec, session: Any):
        # The condition is read from the delivered set, not from the branch id:
        # the id is a label a reader sees, the delivery is what the model gets,
        # and deriving one from the other would let a mislabelled branch pass.
        condition = (
            "oracle"
            if branch.exposure.deliver_ids
            and sc.GOLD_MARKER in branch.exposure.deliver_ids
            else "omit"
        )
        return make_request_gate(
            contract, manifests[condition], allowed_surfaces, registered_dialogue
        )

    def parent_gate(visible_history: Sequence[str]) -> ExposureResult:
        return check_parent_state(
            visible_history=visible_history, contract=contract, label="shared-parent"
        )

    return ExperimentSpec(
        experiment_id=experiment_id,
        clock_start=Clock.of(prefix_start, sc.TIME_ZONE),
        time_zone=sc.TIME_ZONE,
        prompt=bundle,
        documents=documents,
        world_events=tuple(events),
        enabled_tools=PROBE_TOOLS,
        prefix_turns=tuple(prefix_turns),
        branches=_branches(policies),
        # The prefix runs before the notice is published, so it cannot reach the
        # gold document at all.  A plain narrow policy is still declared rather
        # than left to default, because the default would inherit a branch's
        # deliver_ids and try to deliver an unpublished document.
        prefix_exposure=ExposurePolicy(
            mode="full", top_k=len(SHARED_CONTEXT_IDS), label="prefix",
        ),
        window_boundaries=(
            WindowBoundary(
                after_prefix_turn=len(prefix_turns) - 1,
                reason="the user returns in a new session after the notice; the "
                       "fact must arrive through retrieval, not retained chat",
            ),
        ),
        stable_material=tuple(
            (f"v4-bg-{index:02d}", text)
            for index, text in enumerate(sc.STABLE_MATERIAL)
        ),
        host_query=sc.HOST_QUERY,
        gold_fact_id=sc.FACT_KEY,
        gold_marker=sc.GOLD_MARKER,
        # What the notice asserts, in the forms an answer could state it.  The
        # answer is counted as reflecting the evidence when it names the new
        # date, which is the fact the notice exists to convey.
        gold_decisive_terms=date_forms(sc.RIGHT_DATE),
        advance_days=advance_days,
        tool_budget=tool_budget,
        exposure_contract=contract,
        delivery={
            "pair": pair.to_json(),
            "manifests": {k: v.to_json() for k, v in manifests.items()},
            "probe_tools": list(PROBE_TOOLS),
            "forbidden_tools": list(FORBIDDEN_TOOL_NAMES),
            "shared_context_ids": list(SHARED_CONTEXT_IDS),
        },
        request_gate_factory=gate_factory,
        parent_visible_gate=parent_gate,
        notes=(
            f"scenario=v4 elapsed_from_update={elapsed_days}d "
            f"advance_from_prefix={advance_days}d "
            f"contract={contract.contract_version} "
            "delivery=controlled-omit-oracle self_retrieval=removed"
        ),
    )


def describe(spec: ExperimentSpec) -> dict:
    """What this spec will actually do, for a dry-run printout."""
    pair = build_pair()
    return {
        "experiment_id": spec.experiment_id,
        "advance_days": spec.advance_days,
        "enabled_tools": list(spec.enabled_tools),
        "host_query": spec.host_query,
        "window_boundaries": [b.to_json() for b in spec.window_boundaries],
        "oracle_delivery": list(pair.oracle_doc_ids),
        "omit_delivery": list(pair.omit_doc_ids),
        "slot": pair.slot,
        "gold_id": pair.gold_id,
        "filler_id": pair.filler_id,
        "branches": [
            {"branch_id": b.branch_id, "label": b.label,
             "delivered": list(b.exposure.deliver_ids or ())}
            for b in spec.branches
        ],
    }
