"""Four state domains, with a construction-level gate between them.

This module exists to make one specific bug impossible.

The project's bridging experiment failed for a reason that had nothing to do
with the model: the gold fact was still sitting in the conversation history, so
"manipulating evidence accessibility" manipulated nothing.  The fix cannot be a
convention ("remember to keep gold out of the context") because conventions do
not survive contact with a deadline.  It has to be structural.

The structure is a capability:

* ``JudgeDomain`` holds gold answers, scoring rules and condition labels.  It
  has **no method that produces a ``VisibleFragment``**.  There is no way to
  turn a judge-domain value into something the model can see.
* ``ObservationDomain`` is the only object that can mint ``VisibleFragment``.
  It mints from registered templates, delivered user turns, tool-visible
  results and projected memory notes.
* The request compiler accepts ``Sequence[VisibleFragment]`` and nothing else.
  So "the compiler cannot receive gold" is enforced by its type signature
  rather than by the author's memory.

Contract §4 is the normative statement; this is its implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import ContractViolation, ValidationError
from .jsonutil import digest, json_copy

# Sentinel that only this module holds.  ``VisibleFragment`` refuses to be
# constructed without it, so no other module can fabricate model-visible text.
_MINT = object()

VISIBLE_KINDS = frozenset(
    {
        "system",        # a registered system template
        "user",          # a delivered user turn
        "assistant",     # a prior assistant turn
        "tool_result",   # a tool's visible projection
        "memory_note",   # a projected memory note
        "time_context",  # the rendered current date/time
        "stable_material",  # pre-registered background, frozen across windows
        "assistant_tool_call",  # the assistant turn that requests tools
    }
)

# Terms that must never appear in model-visible text.  Contract §4 / §8.
FORBIDDEN_TERMS = (
    "gold",
    "oracle",
    "must_include",
    "condition",
    "branch",
    "harness",
    "virtual",
    "simulated",
    "fixture",
    "benchmark",
    "controlled",
    "experiment",
    "treatment",
    "ablation",
    "elapsed",
    "manipulat",
    "reinject",
)


@dataclass(frozen=True)
class Provenance:
    """Where a visible fragment came from.

    Audit requires that every model-visible fragment be traceable to either a
    registered artifact (a frozen template or prompt bundle, identified by
    digest) or a logged observation event.  "Unknown origin" is a hard failure,
    not a warning.
    """

    source: str  # "template" | "observation" | "tool" | "memory"
    ref: str  # template id, or the event seq that delivered it
    digest: str = ""

    def to_json(self) -> dict:
        return {"source": self.source, "ref": self.ref, "digest": self.digest}


@dataclass(frozen=True)
class VisibleFragment:
    """Text the model is permitted to see, plus its provenance.

    Cannot be constructed outside this module.  Build one with
    ``ObservationDomain.system/user/assistant/tool_result/memory_note``.
    """

    kind: str
    text: str
    provenance: Provenance
    meta: dict = field(default_factory=dict)
    # Which session window delivered this.  A window switch drops everything
    # from earlier windows, which is what stops a previous session's raw
    # history from carrying a fact past a memory manipulation.
    window: int = 0
    # Registered background material is exempt from that drop.  Marking it
    # explicitly (rather than by kind) forces the author to say which facts are
    # permanently present, and therefore which facts can never be the object of
    # an accessibility manipulation.
    frozen: bool = False

    def __post_init__(self):
        # ``__post_init__`` runs for every construction path; re-check
        # membership here so a subclass cannot smuggle a fragment either.
        if getattr(self, "_token", None) is not _MINT:
            raise ContractViolation(
                "VisibleFragment must be minted by ObservationDomain; constructing "
                "model-visible text elsewhere defeats the domain gate"
            )
        if self.kind not in VISIBLE_KINDS:
            raise ValidationError(f"unknown visible kind {self.kind!r}")

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "provenance": self.provenance.to_json(),
            "meta": json_copy(self.meta, "fragment meta"),
            "window": self.window,
            "frozen": self.frozen,
        }


def _mint(kind: str, text: str, provenance: Provenance, meta: dict | None = None,
          *, window: int = 0, frozen: bool = False):
    if not isinstance(text, str):
        raise ValidationError("visible text must be a string")
    if not isinstance(window, int) or window < 0:
        raise ValidationError("fragment window must be a non-negative integer")
    fragment = object.__new__(VisibleFragment)
    object.__setattr__(fragment, "kind", kind)
    object.__setattr__(fragment, "text", text)
    object.__setattr__(fragment, "provenance", provenance)
    object.__setattr__(fragment, "meta", dict(meta or {}))
    object.__setattr__(fragment, "window", window)
    object.__setattr__(fragment, "frozen", frozen)
    object.__setattr__(fragment, "_token", _MINT)
    VisibleFragment.__post_init__(fragment)
    return fragment


class JudgeDomain:
    """Gold answers, scoring rules, condition labels.  Sealed.

    Deliberately has *no* projection method.  Any code that wants to show
    something derived from gold to the model has to either (a) find another
    route, which the compiler rejects, or (b) turn into an observation through
    the retrieval policy, which is a logged, auditable event.
    """

    def __init__(self):
        self._gold: dict = {}
        self._rules: dict = {}
        self._conditions: dict = {}
        self._sealed = True

    def load_gold(self, key: str, value: Any) -> None:
        self._gold[key] = json_copy(value, f"gold[{key}]")

    def load_rules(self, rules: dict) -> None:
        self._rules = json_copy(rules, "scoring rules")

    def load_conditions(self, conditions: dict) -> None:
        self._conditions = json_copy(conditions, "condition labels")

    def gold(self, key: str, default: Any = None) -> Any:
        return self._gold.get(key, default)

    def all_gold(self) -> dict:
        return json_copy(self._gold, "gold")

    def rules(self) -> dict:
        return json_copy(self._rules, "rules")

    def conditions(self) -> dict:
        return json_copy(self._conditions, "conditions")

    def gold_ids(self) -> list:
        """Which evidence ids are gold, for the matched-oracle policy."""
        value = self._gold.get("evidence_ids", [])
        return list(value) if isinstance(value, (list, tuple)) else []

    def classify(self, **observations) -> dict:
        """Apply the registered rules.  Judge-side only; never projected."""
        return json_copy(
            {"observations": observations, "verdict": self._apply(observations)},
            "judge verdict",
        )

    def _apply(self, observations: dict) -> str:
        for rule in self._rules.get("rules", []):
            if all(observations.get(k) == v for k, v in rule.get("when", {}).items()):
                return rule.get("label", "UNRESOLVED")
        return self._rules.get("default_label", "UNRESOLVED")

    def digest(self) -> str:
        return digest(
            {
                "gold": self._gold,
                "rules": self._rules,
                "conditions": self._conditions,
            }
        )


class ObservationDomain:
    """The only minter of model-visible text.

    Holds an ordered record of everything the model was *allowed* to see, plus a
    stricter notion: what is allowed to see it **right now**.

    The distinction exists because of a bug class that is easy to reintroduce.
    A retrieval result injected into a request and then re-injected on every
    later request is not a memory that can decay out of reach — the evidence
    never becomes inaccessible, so "manipulating accessibility" manipulates
    nothing.  That is precisely the defect this harness was built to remove.

    So notes have a **turn-scoped lifetime**, following the user-turn semantics a
    real assistant has:

    * a user turn begins and one automatic selection produces an active note set;
    * every model call inside that turn carries the same active notes, so a tool
      continuation round does not silently lose the evidence;
    * the turn ends, and the notes stop being active.

    Expiry does **not** erase history.  An expired note moves to
    ``expired_observations``, which records that it was once exposed.  Audit and
    scoring read both, because "the model saw this earlier" and "the model can
    see this now" are different facts and only the second affects the
    accessibility manipulation.

    Honest limit: a note leaving the active set does not prove it is
    unreachable.  The model may have restated it in an earlier reply, put it in
    a tool argument, or written it to memory.  Summary injection removes the
    original text, not its consequences.
    """

    def __init__(self, *, window: int = 0):
        self._fragments: list[VisibleFragment] = []
        self._expired: list[VisibleFragment] = []
        self._turn_open = False
        self._window = window

    # ------------------------------------------------------------- lifecycle

    def begin_turn(self) -> None:
        """Open a user turn.  Idempotent within a turn."""
        self._turn_open = True

    def end_turn(self) -> None:
        """Close the user turn, retiring any turn-scoped notes.

        Retired notes are retained in ``expired_observations`` so that a reader
        can tell that exposure once happened, without those notes continuing to
        reach the model.
        """
        keep = []
        for fragment in self._fragments:
            if fragment.kind in ("memory_note", "time_context"):
                self._expired.append(fragment)
            else:
                keep.append(fragment)
        self._fragments = keep
        self._turn_open = False

    @property
    def window(self) -> int:
        return self._window

    @property
    def turn_open(self) -> bool:
        """Whether a user turn has begun and not yet closed.

        Distinct from the event stream's turn lifecycle: a user message can be
        delivered (so the turn is conceptually open) before the agent's
        ``turn.started`` is recorded.  Both must be closed before a boundary is
        legal, or a user turn would span two sessions.
        """
        return self._turn_open

    def begin_window(self) -> int:
        """Open a new session window.  Content from earlier windows stops being delivered.

        Deliberately paired with ``rebase``: advancing the window only relabels,
        while the rebase is what actually removes prior-window content and
        records precisely what was removed.
        """
        self._window += 1
        self._turn_open = False
        return self._window

    def rebase(self, *, policy: str = "zero_cross_window_raw_history") -> dict:
        """Drop everything not carried into the current window.

        The first version keeps **zero** raw interaction from earlier windows.
        That is a deliberate choice for an accessibility study rather than a
        general companion-agent default: retaining even the last few turns of
        the previous session leaves the established fact sitting in the raw
        history, reachable without going through memory at all, which makes the
        accessibility manipulation inert.  This is the same failure the project
        already paid for once.

        Continuity is instead provided by two things that are themselves
        auditable: pre-registered stable material, and retrieval from the
        controlled corpus.

        Returns a machine-readable manifest.  A summary such as "recent content
        was retained" is not acceptable, because nothing downstream could verify
        it; the manifest enumerates every retained and dropped fragment with its
        content digest and provenance.
        """
        retained, dropped = [], []
        for fragment in self._fragments:
            if fragment.frozen or fragment.window >= self._window:
                retained.append(fragment)
            else:
                dropped.append(fragment)
                self._expired.append(fragment)
        self._fragments = retained
        self._turn_open = False
        return {
            "policy": policy,
            "window": self._window,
            "retained": [
                {
                    "kind": f.kind,
                    "ref": f.provenance.ref,
                    "source": f.provenance.source,
                    "content_digest": digest(f.text),
                    "frozen": f.frozen,
                }
                for f in retained
            ],
            "dropped": [
                {
                    "kind": f.kind,
                    "ref": f.provenance.ref,
                    "source": f.provenance.source,
                    "content_digest": digest(f.text),
                }
                for f in dropped
            ],
        }

    def expired_observations(self) -> Sequence[VisibleFragment]:
        """Observations that were delivered earlier but are no longer active."""
        return tuple(self._expired)

    def was_exposed(self, needle: str) -> bool:
        """Whether ``needle`` appears in any active or expired observation."""
        for fragment in list(self._fragments) + list(self._expired):
            if needle in fragment.text:
                return True
        return False

    # ------------------------------------------------------------ minting

    def system(self, text: str, *, template_id: str, template_digest: str):
        fragment = _mint(
            "system",
            text,
            Provenance(source="template", ref=template_id, digest=template_digest),
            window=self._window,
            frozen=True,
        )
        self._fragments.append(fragment)
        return fragment

    def time_context(self, text: str, *, event_seq: int, digest_: str):
        fragment = _mint(
            "time_context",
            text,
            Provenance(source="observation", ref=f"seq:{event_seq}", digest=digest_),
            window=self._window,
        )
        self._fragments.append(fragment)
        return fragment

    def user(self, text: str, *, event_seq: int):
        fragment = _mint(
            "user", text, Provenance(source="observation", ref=f"seq:{event_seq}"),
            window=self._window,
        )
        self._fragments.append(fragment)
        return fragment

    def assistant(self, text: str, *, event_seq: int):
        fragment = _mint(
            "assistant", text, Provenance(source="observation", ref=f"seq:{event_seq}"),
            window=self._window,
        )
        self._fragments.append(fragment)
        return fragment

    def tool_result(self, text: str, *, call_id: str, event_seq: int, meta: dict | None = None):
        fragment = _mint(
            "tool_result",
            text,
            Provenance(source="tool", ref=call_id),
            meta=dict(meta or {}, observation_seq=event_seq),
            window=self._window,
        )
        self._fragments.append(fragment)
        return fragment

    def memory_note(self, text: str, *, retrieval_seq: int, meta: dict | None = None):
        fragment = _mint(
            "memory_note",
            text,
            Provenance(source="memory", ref=f"seq:{retrieval_seq}"),
            meta=dict(meta or {}),
            window=self._window,
        )
        self._fragments.append(fragment)
        return fragment

    def assistant_tool_call(self, *, tool_calls: list, event_seq: int,
                            text: str = "", reasoning: str = ""):
        """The assistant turn that requests one or more tools.

        This is a genuine model-visible exchange and must be in the projected
        conversation.  Leaving it out produces a ``tool`` message with no
        preceding ``tool_calls`` — a sequence that strict providers reject
        outright, and that a tolerant provider silently accepts while the model
        sees a tool result arriving from nowhere.

        ``tool_calls`` are stored already normalized to the request shape
        (``id`` / ``type`` / ``function.name`` / ``function.arguments`` as a JSON
        string), so the projection is a straight re-emission rather than a
        re-derivation.

        ``reasoning`` is the provider's hidden reasoning text when it returns one.
        It is carried because some routes (DeepSeek's thinking mode) require it
        echoed back on the following turn and fail the continuation without it —
        a provider-shaped failure that must not be mistaken for model behaviour.
        """
        fragment = _mint(
            "assistant_tool_call",
            text,
            Provenance(source="observation", ref=f"seq:{event_seq}"),
            meta={"tool_calls": json_copy(tool_calls, "tool calls"),
                  # Kept so the projection can replay it.  Some routes require
                  # their own reasoning text back on the next turn of a tool
                  # exchange and reject the request without it.
                  "reasoning_content": reasoning},
            window=self._window,
        )
        self._fragments.append(fragment)
        return fragment

    def stable_material(self, text: str, *, material_id: str):
        """Pre-registered background that survives every window.

        Reserved for material the scenario declares permanently known: how the
        user is addressed, a standing preference, a background relationship.

        Deliberately **not** a place for recent events, disputed facts, or
        promises awaiting verification.  Anything frozen here can never be the
        object of an accessibility manipulation, so freezing a fact silently
        removes the very thing the experiment varies.  Those belong in the
        controlled corpus, reached by retrieval.
        """
        fragment = _mint(
            "stable_material",
            text,
            Provenance(source="template", ref=material_id,
                       digest=digest(text)),
            window=self._window,
            frozen=True,
        )
        self._fragments.append(fragment)
        return fragment

    # ------------------------------------------------------------ reading

    def fragments(self) -> Sequence[VisibleFragment]:
        return tuple(self._fragments)

    def history(self) -> Sequence[VisibleFragment]:
        """Fragments that form conversational history (everything but system)."""
        return tuple(f for f in self._fragments if f.kind != "system")

    def snapshot(self) -> list:
        return [f.to_json() for f in self._fragments]

    def snapshot_with_expired(self) -> dict:
        return {
            "window": self._window,
            "active": [f.to_json() for f in self._fragments],
            "expired": [f.to_json() for f in self._expired],
        }

    def restore(self, payload: Iterable[dict]) -> None:
        """Rebuild from a snapshot.  Re-mints, so provenance is re-validated.

        The window number and the frozen flag are part of the snapshot: a
        restored branch must know which session window it is in, or a later
        rebase would either drop material it should keep or keep material it
        should have dropped.
        """
        self._fragments = []
        for record in payload:
            fragment = _mint(
                record["kind"],
                record["text"],
                Provenance(
                    source=record["provenance"]["source"],
                    ref=record["provenance"]["ref"],
                    digest=record["provenance"].get("digest", ""),
                ),
                meta=record.get("meta", {}),
                window=int(record.get("window", 0)),
                frozen=bool(record.get("frozen", False)),
            )
            self._fragments.append(fragment)

    def latest_user_text(self) -> str:
        for fragment in reversed(self._fragments):
            if fragment.kind == "user":
                return fragment.text
        return ""


class WorldDomain:
    """Facts, registered world events, and the evidence corpus.

    Ground truth lives here.  It is *not* model-visible: the only route out is
    the visibility policy, which compiles observations.  Holding this object is
    therefore not sufficient to leak it.
    """

    def __init__(self):
        self._facts: dict[str, dict] = {}
        self._events: list[dict] = []
        self._applied: list[str] = []

    # ------------------------------------------------------------ facts

    def put_fact(self, fact_id: str, **fields) -> None:
        record = {"fact_id": fact_id, **json_copy(fields, f"fact[{fact_id}]")}
        self._facts[fact_id] = record

    def supersede_fact(self, fact_id: str, *, new_value: Any, source: str,
                       effective_at: str, supersedes_revision: int) -> None:
        """Replace a fact's value, recording what it supersedes.

        ``revision`` increments so that "the calendar now says X, and this
        supersedes revision N" is expressible — which is what lets a run
        distinguish stale memory from current authority (contract §3).
        """
        existing = self._facts.get(fact_id)
        revision = (existing or {}).get("revision", 0) + 1
        history = list((existing or {}).get("history", []))
        if existing:
            history.append(
                {
                    "revision": existing.get("revision", 0),
                    "value": existing.get("value"),
                    "source": existing.get("source"),
                }
            )
        self._facts[fact_id] = {
            "fact_id": fact_id,
            "value": new_value,
            "source": source,
            "effective_at": effective_at,
            "revision": revision,
            "supersedes_revision": supersedes_revision,
            "history": history,
        }

    def fact(self, fact_id: str) -> dict | None:
        record = self._facts.get(fact_id)
        return json_copy(record, f"fact[{fact_id}]") if record else None

    def facts(self) -> dict:
        return json_copy(self._facts, "facts")

    # ------------------------------------------------------- applied events

    def mark_applied(self, event_id: str) -> None:
        if event_id in self._applied:
            raise ContractViolation(f"world event {event_id!r} applied twice")
        self._applied.append(event_id)

    def applied(self) -> list:
        return list(self._applied)

    def snapshot(self) -> dict:
        return {
            "facts": self._facts,
            "applied": self._applied,
        }

    def restore(self, payload: dict) -> None:
        self._facts = json_copy(payload.get("facts", {}), "world facts")
        self._applied = list(payload.get("applied", []))

    def restore_from_facts(self, facts: dict) -> None:
        """Rebuild purely from folded state.

        The event log is the single authority for facts; this brings the
        ``WorldDomain`` view back in step with it.  Keeping one authority is
        what stops "the log says X but the tool served Y" from ever arising.
        """
        self._facts = json_copy(facts, "state facts")

    def digest(self) -> str:
        return digest(self.snapshot())


def assert_no_forbidden_terms(fragments: Sequence[VisibleFragment], *,
                              extra: Sequence[str] = ()) -> None:
    """Hard gate: no experiment vocabulary in anything the model can see.

    This is a *backstop*, not the primary defence — the domain gate is.  A
    clean scan does not prove the absence of semantic leakage, which is why the
    contract also requires registered templates and provenance.
    """
    terms = list(FORBIDDEN_TERMS) + list(extra)
    hits = []
    for fragment in fragments:
        lowered = fragment.text.lower()
        for term in terms:
            if term.lower() in lowered:
                hits.append({"kind": fragment.kind, "term": term,
                             "provenance": fragment.provenance.ref})
    if hits:
        raise ContractViolation(f"model-visible text contains experiment vocabulary: {hits}")
