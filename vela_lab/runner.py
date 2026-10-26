"""Experiment orchestration: prefix, checkpoint, fork, probe, score.

The shape of one experiment mirrors the contract exactly:

1. run the shared prefix **once**, at end-of-prefix checkpoint it;
2. fork each condition from that one checkpoint;
3. apply that condition's *only* registered difference (its exposure policy);
4. run the probe turn;
5. audit every branch, then score.

Why the prefix is shared rather than replayed per branch: re-running it would let
sampling differences between branches contaminate the comparison.  The branches
must differ in the manipulated variable and nothing else, and the cheapest way
to guarantee that is to have exactly one execution of everything upstream.

The runner refuses to start unless the schedule, corpus, prompt and tool set are
all frozen.  A run whose inputs could still change is not an experiment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .audit import audit_events
from .clock import Clock
from .compiler import RequestCompiler
from .domains import JudgeDomain, ObservationDomain, WorldDomain
from .errors import ContractViolation, GateRejection, ValidationError
from .evidence import Document, EvidenceCorpus, ExposurePolicy
from .jsonutil import digest
from .loop import AgentLoop, MAX_TOOL_ROUNDS_DEFAULT, TurnOutcome
from .policy import RunPolicy, assert_resumable
from .prompt import PromptBundle, PromptRegistry
from .provider import ModelAdapter

# Needed for the annotation on ``exposure_contract``; the object itself stays
# opaque, so nothing here depends on which scenario supplies it.
from typing import Any
from .schedule import WorldEvent, WorldSchedule, advance_to
from .score import score_run
from .session import Session
from .tools import ToolBelt


@dataclass
class BranchSpec:
    """One experimental condition.

    ``exposure`` is the manipulated variable.  Everything else — tools, prompt,
    user text, clock — is shared by construction, because those come from the
    experiment, not from the branch.
    """

    branch_id: str
    exposure: ExposurePolicy
    user_text: str
    label: str = ""
    # Experimenter-side tool calls injected into visible history before the
    # turn runs (``do(tool)`` interventions).  Entries are tool names executed
    # with empty arguments via ``AgentLoop.inject_tool_call``; every record is
    # flagged ``injected`` so the intervention stays separable from model
    # behaviour.
    forced_tools: Sequence[str] = ()


@dataclass(frozen=True)
class WindowBoundary:
    """A pre-registered session boundary.

    The boundary is declared with the scenario rather than decided at run time,
    for the same reason the prompt and the schedule are: a boundary chosen while
    watching the run — because the answer looked wrong, or retrieval missed, or
    the context was nearly full — is an intervention, not a design.

    ``after_prefix_turn`` is an index into ``prefix_turns``; the window closes
    once that turn has completed.
    """

    after_prefix_turn: int
    reason: str

    def __post_init__(self):
        if not isinstance(self.after_prefix_turn, int) or self.after_prefix_turn < 0:
            raise ValidationError("after_prefix_turn must be a non-negative integer")
        if not self.reason:
            raise ValidationError(
                "a window boundary must state its reason, so a reader can see why "
                "the session ended where it did"
            )

    def to_json(self) -> dict:
        return {"after_prefix_turn": self.after_prefix_turn, "reason": self.reason}


@dataclass
class ExperimentSpec:
    """A frozen description of one experiment."""

    experiment_id: str
    clock_start: Clock
    time_zone: str
    prompt: PromptBundle
    documents: Sequence[Document]
    world_events: Sequence[WorldEvent] = ()
    enabled_tools: Sequence[str] = ("world_time", "search")
    prefix_turns: Sequence[str] = ()
    branches: Sequence[BranchSpec] = ()
    # The prefix runs under its own policy.  Declaring it explicitly (rather
    # than letting it fall back to a default) matters because the prefix's
    # retrieval settings are part of the condition every branch inherits: they
    # are shared by all branches, so they must be chosen once, on purpose, and
    # recorded.  They need not match the branches' settings — in a longitudinal
    # design they usually should not — but the difference has to be a decision,
    # not an accident of a default argument.
    prefix_exposure: ExposurePolicy | None = None
    # The query the host uses for its own retrieval.
    #
    # Without this the host retrieves with the probe text, which couples two things
    # that must stay separate: the probe is what the model is asked, the query is
    # what the harness looks up.  A conflict probe that asserts a wrong date makes
    # the wrong date a strong lexical signal, so the host retrieval hits the very
    # document that mentions it and the oracle condition has nothing to change.
    #
    # A registered query keeps the evidence being reachable a property of the
    # corpus and the clock, not of how the user happened to phrase the probe.
    host_query: str = ""
    # Session windows.  Empty means one window for the whole run.
    window_boundaries: Sequence[WindowBoundary] = ()
    # Background registered as permanently known.  Anything placed here can never
    # be the object of an accessibility manipulation, so the list is normally
    # short and contains only material that is static by design.
    stable_material: Sequence[tuple] = ()
    gold_fact_id: str = ""
    gold_marker: str = ""
    # Renderings of what the gold evidence asserts, as an answer might state it.
    # The scorer cannot derive these from the document id, and testing for the id
    # silently made "the answer used the evidence" unreachable.
    gold_decisive_terms: Sequence[str] = ()
    judge_rules: dict = field(default_factory=dict)
    # How far the virtual clock moves before the probe.  Three units rather than
    # one day count, because a design that can only advance in whole days cannot
    # express a sub-day manipulation *at all*: an elapsed amount smaller than a
    # day rounds to zero and the two conditions become byte-identical, which
    # looks like a well-matched pair rather than a dead knob.
    #
    # ``advance_days`` is retained as the coarsest unit (and keeps older specs
    # readable); ``advance_hours`` and ``advance_minutes`` compose with it.
    advance_days: int = 0
    advance_hours: int = 0
    advance_minutes: int = 0
    observe_time: bool = True
    tool_budget: int = MAX_TOOL_ROUNDS_DEFAULT
    request_params: dict = field(default_factory=dict)
    notes: str = ""
    # The exposure contract, as an opaque scenario-supplied object, plus the
    # per-condition delivery description it is checked against.  Opaque because
    # "what counts as decisive evidence" is a property of the scenario: the
    # kernel has no business knowing which sentences would settle a question.
    exposure_contract: Any = None
    delivery: dict = field(default_factory=dict)
    # Factories, not checkers.  The kernel cannot import the scenario layer, so
    # the scenario supplies a callable that builds the checker for one branch.
    # Signature: ``(branch, session) -> callable(CompiledRequest) -> verdict``.
    request_gate_factory: Any = None
    # Pre-fork check on the parent's *model-visible* history.  Separate from the
    # per-request check because a contaminated parent cannot be repaired by
    # anything the branches do afterwards: deleting a retrieved document does not
    # un-say what the prefix already told the model.
    parent_visible_gate: Any = None

    @property
    def gated(self) -> bool:
        """Whether this spec asks for request-level exposure checking."""
        return self.request_gate_factory is not None

    def advance_total_seconds(self) -> int:
        """The whole elapsed interval, in seconds."""
        return (
            self.advance_days * 86400
            + self.advance_hours * 3600
            + self.advance_minutes * 60
        )

    def advance_label(self) -> str:
        """The interval as a short human string, for reporting."""
        total = self.advance_total_seconds()
        if total == 0:
            return "0m"
        if total % 86400 == 0:
            return f"{total // 86400}d"
        if total % 3600 == 0:
            return f"{total // 3600}h"
        if total % 60 == 0:
            return f"{total // 60}min"
        return f"{total}s"

    def digest(self) -> str:
        return digest(
            {
                "experiment_id": self.experiment_id,
                "clock_start": self.clock_start.to_json(),
                "prompt_digest": self.prompt.digest(),
                "documents": [d.to_json() for d in self.documents],
                "world_events": [e.to_json() for e in self.world_events],
                "enabled_tools": list(self.enabled_tools),
                "prefix_turns": list(self.prefix_turns),
                "prefix_exposure": (
                    self.prefix_exposure.identity()
                    if self.prefix_exposure else None
                ),
                "host_query": self.host_query,
                "window_boundaries": [b.to_json() for b in self.window_boundaries],
                "stable_material": [list(item) for item in self.stable_material],
                "branches": [
                    {
                        "branch_id": b.branch_id,
                        "exposure": b.exposure.identity(),
                        "user_text": b.user_text,
                        "forced_tools": list(b.forced_tools),
                    }
                    for b in self.branches
                ],
                "advance_days": self.advance_days,
                "advance_hours": self.advance_hours,
                "advance_minutes": self.advance_minutes,
                "observe_time": self.observe_time,
                "gold_marker": self.gold_marker,
                "gold_decisive_terms": list(self.gold_decisive_terms),
                "exposure_contract": (
                    self.exposure_contract.digest()
                    if self.exposure_contract is not None else None
                ),
                "delivery": self.delivery,
            }
        )


class ExperimentRunner:
    """Runs one experiment and returns a report."""

    def __init__(self, spec: ExperimentSpec, model: ModelAdapter, *,
                 out_dir: os.PathLike | str):
        self.spec = spec
        self.model = model
        self.out_dir = Path(out_dir)
        self.corpus = EvidenceCorpus(list(spec.documents))
        self.schedule = WorldSchedule(list(spec.world_events))
        self.registry = PromptRegistry()
        self.judge = JudgeDomain()

    def run(self) -> dict:
        self._preflight()
        self.corpus.seal()
        self.schedule.seal()
        self.registry.register(self.spec.prompt)
        self.registry.seal()
        if self.spec.judge_rules:
            self.judge.load_rules(self.spec.judge_rules)
        self.judge.load_gold("gold_marker", self.spec.gold_marker)
        self.judge.load_gold("gold_fact_id", self.spec.gold_fact_id)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        policy = self._resolve_policy()
        # Freeze before anything runs.  A policy that is resolved once and then
        # checked on any later continuation is what stops a resumed run from
        # silently picking up newer defaults mid-experiment.
        self._write_policy(policy)
        prefix_root = self.out_dir / "prefix"
        world = self._world_at_start()

        session = Session(
            prefix_root,
            clock=self.spec.clock_start,
            observations=ObservationDomain(),
            world=world,
            config={"experiment_id": self.spec.experiment_id,
                    "spec_digest": self.spec.digest()},
        )
        self._run_prefix(session)
        self._advance_and_apply(session)
        checkpoint = session.checkpoint("shared")
        # The parent check runs once, on the one shared state every branch forks
        # from, so a contaminated parent stops the run rather than being
        # rediscovered per branch (and possibly reported as one branch's fault).
        parent_verdict = self._check_parent(session)

        branches = []
        for branch in self.spec.branches:
            branches.append(self._run_branch(branch, session, checkpoint))

        report = {
            "experiment_id": self.spec.experiment_id,
            "spec_digest": self.spec.digest(),
            "prompt_digest": self.spec.prompt.digest(),
            "corpus_digest": self.corpus.digest(),
            "schedule_digest": self.schedule.digest(),
            "checkpoint": {
                "name": "shared",
                "prefix_count": checkpoint["prefix_count"],
                "prefix_digest": checkpoint["prefix_digest"],
                "checksum": checkpoint["checksum"],
            },
            "model": self.model.identity(),
            "policy_digest": self.load_policy().digest(),
            "branches": branches,
            "notes": self.spec.notes,
            "exposure": {
                "contract_version": (
                    self.spec.exposure_contract.contract_version
                    if self.spec.exposure_contract is not None else None
                ),
                "gated": self.spec.gated,
                "parent_check": (
                    parent_verdict.to_json() if parent_verdict is not None else None
                ),
                "delivery": self.spec.delivery,
            },
        }
        report["parent_equality"] = self._parent_equality(branches)
        self._write(report)
        return report

    # -------------------------------------------------------------- preflight

    def _preflight(self) -> None:
        spec = self.spec
        if not spec.branches:
            raise ValidationError("an experiment needs at least one branch")
        if not spec.prefix_turns:
            raise ValidationError("an experiment needs at least one prefix turn")
        ids = [b.branch_id for b in spec.branches]
        if len(ids) != len(set(ids)):
            raise ValidationError("branch ids must be unique")
        for branch in spec.branches:
            if not branch.user_text:
                raise ValidationError(f"branch {branch.branch_id}: empty probe text")
            if branch.exposure.temporal_decay is None and spec.advance_days:
                pass  # time may be neutral; not an error
        if spec.advance_total_seconds() < 0:
            raise ValidationError(
                "the elapsed interval must not be negative; the clock only moves "
                "forward, because a backwards jump would make 'elapsed time' "
                "ambiguous"
            )
        if not self.spec.enabled_tools:
            raise ValidationError("an experiment needs at least one tool")
        boundaries = sorted(b.after_prefix_turn for b in spec.window_boundaries)
        if len(boundaries) != len(set(boundaries)):
            raise ValidationError("window boundaries must be at distinct turns")
        for boundary in spec.window_boundaries:
            if boundary.after_prefix_turn >= len(spec.prefix_turns):
                raise ValidationError(
                    f"window boundary at turn {boundary.after_prefix_turn} is past "
                    f"the last prefix turn ({len(spec.prefix_turns) - 1})"
                )
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            raise ContractViolation(
                f"{self.out_dir} is not empty; refusing to overwrite a previous run. "
                "A run directory is evidence — start a new one."
            )

    def _world_at_start(self) -> WorldDomain:
        """Build the world, then apply only the events the schedule says are live.

        Nothing here invents a fact: every fact comes from a registered event or
        from the spec's initial world.
        """
        world = WorldDomain()
        for event in self.spec.world_events:
            if event.kind == "add_fact" and event.fact_id:
                world.put_fact(
                    event.fact_id,
                    value=event.new_value,
                    source=event.source or "registered-event",
                    effective_at=event.at,
                    revision=0,
                )
        return world

    # ---------------------------------------------------------------- prefix

    def _run_prefix(self, session: Session) -> None:
        # Background is registered before any interaction so that every branch
        # shares it, and so that the set of facts which can never be manipulated
        # is fixed before the run rather than accumulated during it.
        for material_id, text in self.spec.stable_material:
            session.register_stable_material(material_id, text)

        boundaries = {b.after_prefix_turn: b for b in self.spec.window_boundaries}
        for index, text in enumerate(self.spec.prefix_turns):
            if self.spec.observe_time:
                session.add_time_context()
            session.add_user_turn(text)
            loop = self._loop(session, self._prefix_exposure())
            outcome = loop.run_turn(label=f"prefix[{index}]")
            if outcome.failure:
                raise ContractViolation(
                    f"prefix turn failed: {outcome.failure['reason']}; every branch "
                    "would inherit the failure, so the experiment cannot proceed"
                )
            # A prefix turn must not leave retrieval notes active into the probe:
            # the probe's accessibility condition is what the experiment varies.
            session.observations.end_turn()

            boundary = boundaries.get(index)
            if boundary is not None:
                # The switch happens here, before the checkpoint, so every branch
                # inherits the same rebased context.  If it happened per branch,
                # the window itself would become a source of divergence and the
                # comparison would no longer isolate the manipulated variable.
                session.append("session.boundary", boundary.to_json())
                session.switch_window()

    def _advance_and_apply(self, session: Session) -> None:
        """Move the clock and apply exactly the registered events that fall due."""
        if not self.spec.advance_total_seconds():
            return
        before = session.clock
        after = session.advance(
            days=self.spec.advance_days,
            hours=self.spec.advance_hours,
            minutes=self.spec.advance_minutes,
        )
        for event, moment in advance_to(self.schedule, before, after):
            if moment == "effective":
                session.append(
                    "world.fact_superseded",
                    {
                        "event_id": event.event_id,
                        "fact_id": event.fact_id,
                        "new_value": event.new_value,
                        "source": event.source,
                        "effective_at": event.at,
                    },
                )
            elif moment == "published":
                session.append(
                    "world.event_applied",
                    {
                        "event_id": event.event_id,
                        "kind": moment,
                        "evidence_id": event.evidence_id,
                    },
                )
            elif moment == "notified":
                session.append(
                    "world.event_applied",
                    {
                        "event_id": event.event_id,
                        "kind": moment,
                        "text": event.notification_text,
                    },
                )

    # -------------------------------------------------------------- branches

    def _parent_visible_text(self, session: Session) -> list:
        """Everything in the parent that the model could have read.

        Read from the *observations*, not from the log: the log legitimately
        contains the correct date in the world-fact records, and treating the
        world store as contamination would fail every scenario that has a fact.
        What matters is reachability by the model, which is exactly what the
        observation domain tracks.
        """
        snapshot = session.observations.snapshot_with_expired()
        blocks = []
        # Both halves count.  Retired notes were delivered to the model when they
        # were active; retiring them stops future exposure but does not un-deliver
        # them, so a parent that exposed the update in an earlier window is still
        # contaminated.
        for fragment in list(snapshot.get("active", [])) + list(
            snapshot.get("expired", [])
        ):
            text = fragment.get("text") if isinstance(fragment, dict) else None
            if text:
                blocks.append(text)
        return blocks

    def _check_parent(self, session: Session):
        """Refuse to fork from a parent that already tells the model the answer."""
        if self.spec.parent_visible_gate is None:
            return None
        verdict = self.spec.parent_visible_gate(
            self._parent_visible_text(session)
        )
        session.append("exposure.parent_checked", verdict.to_json())
        if getattr(verdict, "status", "PASS") == "BATCH_FAIL":
            raise GateRejection(
                "the shared parent state already exposes the key update to the "
                f"model ({sorted(set(verdict.reason_codes))}); no clean pair can "
                "be forked from it",
                verdict=verdict,
                scope="parent_family",
            )
        return verdict

    def _run_branch(self, branch: BranchSpec, parent: Session,
                    checkpoint: dict) -> dict:
        branch_root = self.out_dir / "branches" / branch.branch_id
        child = parent.fork(parent.root, checkpoint["name"], target_root=branch_root)

        # The branch's only registered difference: how evidence is exposed.
        belt = ToolBelt(
            enabled=list(self.spec.enabled_tools),
            corpus=self.corpus,
            world=child.world,
            exposure=branch.exposure,
            tool_descriptions=dict(self.spec.prompt.tool_descriptions),
        )
        if self.spec.observe_time:
            child.add_time_context()
        child.add_user_turn(branch.user_text)
        auto = self._auto_retrieve(child, branch.exposure)
        request_gate = (
            self.spec.request_gate_factory(branch, child)
            if self.spec.request_gate_factory is not None else None
        )
        loop = AgentLoop(
            session=child,
            model=self.model,
            compiler=RequestCompiler(
                self.spec.prompt, tool_schemas=belt.schemas()
            ),
            tools=belt,
            budget=self.spec.tool_budget,
            request_params=self.spec.request_params,
            request_gate=request_gate,
        )
        for forced_name in branch.forced_tools:
            loop.inject_tool_call(forced_name, {})
        gate_rejection: dict | None = None
        try:
            outcome = loop.run_turn(label=branch.branch_id)
        except GateRejection as exc:
            # A gate rejection is a measurement failure, not a model behaviour.
            # Kept as a recorded branch with no answer, so the allocation stays
            # visible in the report instead of the branch vanishing from it.
            child.append(
                "branch.gate_rejected",
                {
                    "scope": exc.scope,
                    "reason_codes": sorted(set(exc.verdict.reason_codes)),
                    "verdict": exc.verdict.to_json(),
                },
            )
            gate_rejection = {
                "scope": exc.scope,
                "reason_codes": sorted(set(exc.verdict.reason_codes)),
                "verdict": exc.verdict.to_json(),
                "message": str(exc),
            }
            outcome = TurnOutcome(
                request_count=0, tool_calls=0, assistant_text="",
                finish_reasons=[], failure={"reason": f"exposure gate: {exc}"},
            )

        events = child.events()
        if gate_rejection is not None:
            # Count what was actually compiled before the rejection, so the
            # report shows how far the branch got instead of implying that no
            # request was ever built.
            outcome.request_count = sum(
                1 for e in events if e.get("type") == "model.request"
            )
        findings = audit_events(events)
        score = score_run(
            events,
            gold_marker=self.spec.gold_marker,
            answer_text=outcome.assistant_text,
            decisive_terms=self.spec.gold_decisive_terms,
        ) if self.spec.gold_marker else {}
        run_valid = not findings and not outcome.failure

        return {
            "branch_id": branch.branch_id,
            "label": branch.label or branch.branch_id,
            "exposure": branch.exposure.to_json(),
            "user_text": branch.user_text,
            "auto_retrieval": auto,
            "search_accounting": outcome.search,
            "outcome": outcome.to_json(),
            "audit_findings": findings,
            "run_valid": run_valid,
            # Why a branch is invalid, as a machine-readable reason.  "run_valid
            # is false" alone cannot separate a gate rejection from an audit
            # finding from a provider failure, and those three have different
            # remedies, so collapsing them loses the only actionable part.
            "invalid_reason": (
                "exposure_gate" if gate_rejection else
                "audit" if findings else
                "branch_failure" if outcome.failure else ""
            ),
            "gate_rejection": gate_rejection,
            "exposure_verdicts": self._exposure_verdicts(child),
            "score": score,
            "event_count": len(events),
        }

    @staticmethod
    def _exposure_verdicts(session: Session) -> list:
        """Every exposure check recorded on this branch, in order.

        Read back from the log rather than returned from the gate object, so the
        report's verdicts are provably the ones written at the time.  A verdict
        that exists only as a live return value cannot be re-derived from the
        evidence directory, which is the whole point of keeping one.
        """
        return [
            event.get("data", {})
            for event in session.events()
            if event.get("type") == "exposure.checked"
        ]

    def _prefix_exposure(self) -> ExposurePolicy:
        """The policy the shared prefix runs under.

        Defaults to the first branch's policy so that a single-condition
        experiment cannot silently use two different retrieval settings.  When a
        spec wants the prefix to differ (which is legitimate), it says so
        explicitly and the choice enters the spec digest.
        """
        if self.spec.prefix_exposure is not None:
            return self.spec.prefix_exposure
        if self.spec.branches:
            base = self.spec.branches[0].exposure
            return ExposurePolicy(
                mode=base.mode,
                top_k=base.top_k,
                temporal_decay=base.temporal_decay,
                label="prefix",
            )
        return ExposurePolicy(label="prefix")

    def _auto_retrieve(self, session: Session, exposure: ExposurePolicy) -> dict:
        """Perform the host-driven retrieval for this user turn.

        Retrieval is not left to whether the model decides to call a tool.  If it
        were, an accessibility manipulation would be entangled with tool-use
        willingness, and "did not retrieve" would be indistinguishable from
        "chose not to look" — two different findings.

        The query comes from the current user turn only.  Deriving it from
        earlier text that contains the answer would make the retrieval partly
        self-fulfilling.

        Three outcomes are kept distinct, because collapsing them would make a
        legitimate condition look like a broken one:

        ``ok``       the retrieval ran and returned documents
        ``empty``    the retrieval ran and legitimately returned nothing
        ``skipped``  no query was available, so no retrieval was attempted

        A projection or audit failure is none of these — it raises.
        """
        # Prefer the registered host query.  Falling back to the user's text is
        # allowed for the prefix, where no manipulation is in play, but a run that
        # manipulates accessibility must declare its query or the probe wording
        # leaks into retrieval.
        query = self.spec.host_query or session.observations.latest_user_text()
        if not query:
            return {"outcome": "skipped", "reason": "no query available"}

        record = self.corpus.retrieve(
            query=query,
            now_instant=session.clock.instant,
            top_k=exposure.top_k,
            mode="auto",
            visibility=exposure.mode,
            omit_ids=exposure.omit_ids,
            force_include_ids=exposure.force_include_ids,
            strip_fields=exposure.strip_fields,
            temporal_decay=exposure.temporal_decay,
            deliver_ids=exposure.deliver_ids,
        )
        # Annotate each hit with the exact text that will be delivered *before*
        # the retrieval is recorded.  Audit verifies delivery against this, so a
        # retrieval event written without it cannot be checked.
        for hit in record.hits:
            document = self.corpus.get(hit["doc_id"])
            hit["visible_text"] = document.text
        session.append("memory.retrieval", record.to_json())

        # Mint the notes.  They stay active for this user turn (including any
        # tool-continuation rounds) and retire when the turn closes.
        notes = []
        for hit in record.hits:
            document = self.corpus.get(hit["doc_id"])
            fragment = session.observations.memory_note(
                document.text,
                retrieval_seq=session.head_seq(),
                meta={
                    "doc_id": hit["doc_id"],
                    "is_gold": hit["is_gold"],
                    "rank": hit["rank"],
                    "corpus_digest": record.corpus_digest,
                },
            )
            notes.append(fragment.text)
        return {
            "outcome": "ok" if record.hits else "empty",
            "query": query,
            "hits": [h["doc_id"] for h in record.hits],
            "gold_delivered": record.gold_delivered(),
            "result_count": len(record.hits),
            "eligible_count": len(record.eligible_ids),
        }

    def _loop(self, session: Session, exposure: ExposurePolicy) -> AgentLoop:
        belt = ToolBelt(
            enabled=list(self.spec.enabled_tools),
            corpus=self.corpus,
            world=session.world,
            exposure=exposure,
            tool_descriptions=dict(self.spec.prompt.tool_descriptions),
        )
        return AgentLoop(
            session=session,
            model=self.model,
            compiler=RequestCompiler(self.spec.prompt, tool_schemas=belt.schemas()),
            tools=belt,
            budget=self.spec.tool_budget,
            request_params=self.spec.request_params,
        )

    # ------------------------------------------------------------ assertions

    def _resolve_policy(self) -> RunPolicy:
        """Resolve the run's policy once, from the frozen inputs."""
        from .jsonutil import digest
        from . import PROTOCOL_VERSION

        base = self._prefix_exposure()
        return RunPolicy(
            prompt_digest=self.spec.prompt.digest(),
            tool_schemas_digest=digest(
                [s for s in ToolBelt(
                    enabled=list(self.spec.enabled_tools),
                    corpus=self.corpus,
                    world=WorldDomain(),
                    exposure=base,
                ).schemas()]
            ),
            enabled_tools=tuple(self.spec.enabled_tools),
            phase_top_k={"prefix": base.top_k,
                         **{b.branch_id: b.exposure.top_k
                            for b in self.spec.branches}},
            note_lifetime="turn_scoped",
            memory_wrapper_digest=digest(self.spec.prompt.memory_wrapper),
            tool_budget=self.spec.tool_budget,
            request_params=dict(self.spec.request_params),
            model_identity=self.model.identity(),
            corpus_digest=self.corpus.digest(),
            schedule_digest=self.schedule.digest(),
            protocol_version=PROTOCOL_VERSION,
        )

    def _write_policy(self, policy: RunPolicy) -> None:
        path = self.out_dir / "resolved_policy.json"
        with open(path, "w", encoding="utf-8") as handle:

            json.dump(policy.to_json(), handle, ensure_ascii=False,
                      indent=2, sort_keys=True)

    def load_policy(self) -> RunPolicy:
        """Read this run's frozen policy, for a continuation to check against."""
        path = self.out_dir / "resolved_policy.json"
        if not path.exists():
            raise ContractViolation(f"{path} is missing; the run was not frozen")
        return RunPolicy.from_json(json.loads(path.read_text(encoding="utf-8")))

    def assert_continuable(self) -> RunPolicy:
        """Refuse to continue when the resolved policy no longer matches."""
        recorded = self.load_policy()
        current = self._resolve_policy()
        assert_resumable(recorded, current)
        return current

    def _parent_equality(self, branches: list) -> dict:
        """Confirm every branch forked from the same parent state.

        Reads the recorded ``branch.forked`` events.  If two branches name
        different parent prefix digests, the comparison is not a comparison.
        """
        digests = set()
        prefixes = set()
        for branch in branches:
            root = self.out_dir / "branches" / branch["branch_id"]
            log_path = root / "events.jsonl"
            if not log_path.exists():
                continue
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["type"] == "branch.forked":
                    digests.add(record["data"]["parent_prefix_digest"])
                    prefixes.add(record["data"]["parent_prefix_count"])
        return {
            "identical_parent": len(digests) <= 1,
            "parent_prefix_digests": sorted(digests),
            "parent_prefix_counts": sorted(prefixes),
        }

    def _write(self, report: dict) -> None:
        path = self.out_dir / "report.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        manifest = self.out_dir / "manifest.json"
        with manifest.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "experiment_id": self.spec.experiment_id,
                    "spec_digest": self.spec.digest(),
                    "prompt_digest": self.spec.prompt.digest(),
                    "corpus_digest": self.corpus.digest(),
                    "schedule_digest": self.schedule.digest(),
                    "model": self.model.identity(),
                    "branch_count": len(self.spec.branches),
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
