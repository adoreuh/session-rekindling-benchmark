"""The exposure gate: what the model could actually see, checked on the real request.

Why this is a separate gate from `event_gate.py`.

`event_gate` checks the **scenario**: that the probe's referent resolves to one
event and that the asserted value genuinely conflicts with the world.  It answers
"is the question well-posed?".

This gate checks the **serialized request**: that the condition actually delivered
what it claims.  It answers "did the model see it?".  The two fail independently —
a well-posed scenario can still deliver the wrong thing after wrapping, truncation,
inherited history or caching.

Two corrections to the obvious approach, both insisted on in review:

**A keyword blacklist cannot prove absence.** The absence check cannot be "the
omit request contains none of these words", because free text always admits another
phrasing.  Here is the sharp case that killed the naive rule::

    "这场签证面试让我紧张"            same topic, NOT decisive evidence
    "这场签证面试原来的预约已经作废"   decisive leak, and it names no correct date
    "签证面试改期通知"                not sufficient to negate, but not neutral either
    "6 月 25 日" alone               not necessarily about this event at all

So the rule is **provenance**: every model-visible block must map to a registered
piece of material, and a block that cannot be mapped makes the run
measurement-invalid rather than passing by default.  Pattern matching stays as a
backstop for finding omissions, never as the basis for a pass.

**A document id in the request does not prove sufficiency.** Gold can be present
and truncated past the point where it says anything usable.  Sufficiency is checked
against registered spans that must survive into the rendered content: the event
binding, the cancellation, the current date, and the authority.  Truncating an
irrelevant tail is fine; truncating the negation is not.

Honest limit, stated because the contract requires it: with free-form model output
no finite check can prove *absence of synonymic leak* in arbitrary text.  What can
be guaranteed is that only registered, audited material enters the measurement
request.  Unknown content is not a pass — it is a failure to verify.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from vela_lab.errors import ValidationError
from vela_lab.jsonutil import digest

# Severity vocabulary, shared with the event gate and the manipulation checks.
PASS = "PASS"
UNINFORMATIVE = "UNINFORMATIVE"
RUN_INVALID = "RUN_INVALID"
BATCH_FAIL = "BATCH_FAIL"

SEVERITY_ORDER = (BATCH_FAIL, RUN_INVALID, UNINFORMATIVE, PASS)

#: Truncation markers, following the pattern in the mature harness this project
#: draws on (``codex-rs/utils/string/src/truncate.rs``): truncation preserves a
#: prefix and a suffix and inserts a **visible marker** naming how much was cut.
#:
#: Two consequences worth stating, because both differ from the naive picture.
#:
#: * The dropped region is normally the *middle*, not the tail.  A check that only
#:   worries about the end of a document is modelling the wrong failure: with
#:   middle truncation the head and tail both survive, so a decisive sentence in
#:   the middle can vanish while the document still looks complete at both ends.
#: * The marker is model-visible text.  It must therefore be registered as a
#:   packaging surface or the residue check will flag every truncated request as
#:   unregistered content.
TRUNCATION_MARKER_PATTERN = re.compile(r"…\d+ (?:tokens|chars|bytes) truncated…")


def aggregate(*statuses: str) -> str:
    """Combine gate results, keeping the most severe.

    BATCH_FAIL outranks RUN_INVALID because a shared defect (a contaminated parent,
    an unauditable serialization path) invalidates every pair derived from it,
    whereas RUN_INVALID is local to one request.

    ``SEVERITY_ORDER`` is written most-severe-first, so the most severe is the
    **smallest** index.  Taking the minimum is the whole job, and getting the
    direction wrong is not a cosmetic mistake: an inverted comparison makes the
    function return PASS for every input, which silently disables every check
    that calls it.  That failure mode leaves no trace in a report — the gate
    looks like it ran and found nothing — so it is asserted directly below.
    """
    worst = len(SEVERITY_ORDER) - 1
    for status in statuses:
        if status not in SEVERITY_ORDER:
            raise ValidationError(f"unknown severity {status!r}")
        worst = min(worst, SEVERITY_ORDER.index(status))
    return SEVERITY_ORDER[worst]


# The one property that must hold or every caller is silently disabled.
assert aggregate(RUN_INVALID, PASS) == RUN_INVALID
assert aggregate(BATCH_FAIL, PASS) == BATCH_FAIL
assert aggregate(BATCH_FAIL, RUN_INVALID) == BATCH_FAIL
assert aggregate(PASS, PASS) == PASS


#: How a proposition's required terms appear in a piece of text.  The distinction
#: between the first and the rest is the whole point: "预约已取消" settles the
#: question, while "没有收到取消通知" mentions the same two words and settles
#: nothing.  Collapsing them makes the gate fire on every mention of the concept,
#: which is not a leak — it is ordinary discussion of the topic.
ASSERTED = "asserted"                 # affirmative: the proposition holds
NEGATED = "negated"                   # affirmative form with a preceding negation
NON_ASSERTIVE = "non_assertive"       # conditional, interrogative, or hypothetical
QUOTED = "quoted"                     # reported speech: someone else's claim
MENTIONED = "mentioned"               # terms present, no assertion structure found

#: Negation triggers.  A trigger only suppresses when it appears *before* the
#: proposition's predicate inside the same sentence — the classic preceding-trigger
#: rule.  Deliberately a frozen list rather than a learned detector: what counts as
#: negated is then inspectable, and a change to it is a visible edit to a constant
#: instead of a silent drift in behaviour.
#:
#: Precision matters more than coverage here, because the two error directions are
#: not symmetric.  Treating an assertion as negated **misses a real leak**, while
#: treating a negation as an assertion only raises a false alarm.  So bare single
#: characters that occur inside ordinary words are excluded: ``非`` appears in
#: 非常 ("very"), and a trigger list containing it would silently neutralise any
#: sentence that says 非常焦虑.  Multi-character forms only.
NEGATION_TRIGGERS = (
    "没有", "并没", "并未", "并没有", "尚未", "从未", "未曾", "还没",
    "不是", "不会", "并非", "不打算", "无意", "未", "没",
)

#: Scope terminators.  A negation stops applying once the sentence turns — the
#: clause begins making a new claim — so "签证面试没有改期，但已经取消" is an
#: assertion of cancellation despite the earlier negation.  Without these the
#: preceding-trigger rule silently swallows the real claim, which is the failure
#: direction that matters: a missed leak is worse than a false alarm.
NEGATION_TERMINATORS = (
    "但", "但是", "不过", "然而", "可是", "却", "反而", "已经", "已", "现",
    "确认", "通知", "声明", "表示", "决定", "作废", "取消", "改为", "变更",
)

#: Non-assertive markers.  A conditional or interrogative form states no fact.
NON_ASSERTIVE_MARKERS = (
    "如果", "假如", "若", "万一", "要是", "假设", "一旦", "是否", "吗", "呢",
    "会不会", "能不能", "能否", "？", "?",
)

#: Quotation / hearsay markers.  "朋友问我是不是取消了" reports a question; the
#: speaker asserts nothing.
QUOTATION_MARKERS = (
    "据说", "听说", "转述", "声称", "问我", "问：", "说是", "提到过",
)

SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+")


def _sentences(text: str) -> list:
    return [s for s in SENTENCE_SPLIT.split(text) if s.strip()]


def classify_terms(text: str, terms: Sequence[str]) -> str:
    """How ``terms`` appear in ``text``, one of the constants above.

    Sentence scope, with the predicate located by the term that carries the
    claim — the term whose negation changes the meaning.  For the propositions in
    this study that is the *last* required term (the cancellation word, or the
    date), because the first is the event name and negating a name is not a thing
    people do.

    ``mentioned`` is a real outcome, not a failure: it means the terms co-occur
    without any assertion structure the rules recognise.  Reporting it separately
    keeps a possible detection gap visible instead of folding it into "asserted".

    Honest limit: this is a bounded rule set, not a parser.  It handles preceding
    negation with terminators, conditionals, questions and reported speech.  It
    does not resolve arbitrary scope, anaphora, or double negation, and it must
    not be assumed to generalise beyond the phrasings it is tested on.  The
    guarantee for a measurement comes from the *source-level* audit — declared
    delivery set, residue, manifest — for which natural-language inference is a
    backstop rather than the basis.
    """
    if not terms or not all(term in text for term in terms):
        return ""

    # Co-occurrence is checked one level wider than the assertion analysis.  A
    # proposition's terms can legitimately span a sentence boundary — "请以本通知
    # 替代旧确认单" carries one term while the appointment is named in the
    # preceding sentence — so requiring all terms inside one sentence would lose
    # the supersession proposition entirely.  The *assertion structure* is then
    # read from the sentence holding the predicate, which is where negation and
    # conditionality actually attach.
    predicate = terms[-1]
    predicate_sentence = None
    for sentence in _sentences(text):
        if predicate in sentence:
            predicate_sentence = sentence
            if all(term in sentence for term in terms):
                break
    if predicate_sentence is None:
        return ""

    if any(marker in predicate_sentence for marker in QUOTATION_MARKERS):
        return QUOTED
    if any(marker in predicate_sentence for marker in NON_ASSERTIVE_MARKERS):
        return NON_ASSERTIVE

    at = predicate_sentence.rfind(predicate)
    before = predicate_sentence[:at]
    # Preceding negation, but only within the current clause: walk back from the
    # predicate and stop at the first scope terminator, so a later claim cancels
    # an earlier negation.  "签证面试没有改期，但已经取消" asserts the
    # cancellation, and a rule that ignored the terminator would call it negated
    # and miss the leak.
    cut = -1
    for terminator in NEGATION_TERMINATORS:
        found = before.rfind(terminator)
        if found > cut:
            cut = found
    window = before[cut + 1:] if cut >= 0 else before
    if any(trigger in window for trigger in NEGATION_TRIGGERS):
        return NEGATED
    return ASSERTED


@dataclass(frozen=True)
class Proposition:
    """A statement whose presence in a request is decisive.

    ``requires`` lists the terms that must **co-occur in one sentence** for the
    proposition to be considered asserted.  Requiring co-occurrence is what
    separates evidence from topical noise: "6 月 25 日" somewhere in the request and
    "签证面试" somewhere else do not together assert that the interview moved.

    Co-occurrence alone is not enough, though.  The same two words appear in
    "预约已取消" and in "没有收到取消通知", and only the first settles anything.
    ``classify_in`` reports which case it is rather than returning a bare boolean,
    so a caller can distinguish a leak from a mention and a reviewer can see which
    it was.
    """

    proposition_id: str
    description: str
    requires: tuple
    decisive: bool = True

    def __post_init__(self):
        if not self.requires:
            raise ValidationError(
                f"proposition {self.proposition_id!r} needs at least one required term"
            )

    def classify_in(self, text: str) -> str:
        return classify_terms(text, self.requires)

    def asserted_in(self, text: str) -> bool:
        """True only for an affirmative statement.

        Kept as the narrow question so callers asking "is this a leak?" cannot
        accidentally treat a negation or a question as one.
        """
        return self.classify_in(text) == ASSERTED


@dataclass(frozen=True)
class ExposureContract:
    """What the scenario registers about decisive versus neutral material.

    Registered once and versioned: a change to any of these changes what counts as
    evidence, so it must change the contract version and invalidate prior audits
    rather than being applied retroactively.
    """

    contract_version: str
    event_referents: tuple            # how the event can be named in visible text
    decisive_propositions: tuple      # Proposition, ordered
    neutral_material_ids: tuple       # audited non-decisive material
    oracle_required_groups: tuple     # tuple of tuples: groups that must all appear
    allowed_transforms: tuple = ("whitespace", "wrapper")

    def __post_init__(self):
        if not self.contract_version:
            raise ValidationError("exposure contract needs a version")
        if not self.decisive_propositions:
            raise ValidationError(
                "exposure contract needs at least one decisive proposition; without "
                "it, absence cannot be distinguished from irrelevance"
            )
        if not self.oracle_required_groups:
            raise ValidationError(
                "exposure contract needs the oracle sufficiency requirements"
            )

    def digest(self) -> str:
        return digest(
            {
                "contract_version": self.contract_version,
                "event_referents": list(self.event_referents),
                "decisive": [
                    {"id": p.proposition_id, "requires": list(p.requires)}
                    for p in self.decisive_propositions
                ],
                "neutral_material_ids": list(self.neutral_material_ids),
                "oracle_required_groups": [list(g) for g in self.oracle_required_groups],
            }
        )

    def decisive_asserted_in(self, text: str) -> list:
        """Which decisive propositions this block asserts, if any."""
        return [
            p.proposition_id
            for p in self.decisive_propositions
            if p.decisive and p.asserted_in(text)
        ]


@dataclass(frozen=True)
class DeliveryManifest:
    """Exactly which registered material this condition is supposed to deliver.

    This is what turns "check for leaks" from a keyword hunt into a set
    comparison.  The condition declares its delivered set up front; the gate then
    verifies the rendered request against that declaration in both directions —
    everything declared is present, and nothing undeclared is.  A keyword scan
    can only ever look for the phrasings someone thought of, whereas an
    undeclared document is detectable regardless of how it is worded.
    """

    condition: str
    expected_material_ids: tuple
    slot_index: int
    slot_material_id: str

    def __post_init__(self):
        if self.condition not in ("oracle", "omit"):
            raise ValidationError(
                f"delivery manifest: unknown condition {self.condition!r}"
            )
        if not self.expected_material_ids:
            raise ValidationError("delivery manifest: the delivered set is empty")
        if not 0 <= self.slot_index < len(self.expected_material_ids):
            raise ValidationError(
                f"delivery manifest: slot {self.slot_index} is outside the "
                f"delivered set of {len(self.expected_material_ids)}"
            )
        actual = self.expected_material_ids[self.slot_index]
        if actual != self.slot_material_id:
            raise ValidationError(
                f"delivery manifest: slot {self.slot_index} holds {actual!r}, not "
                f"the declared slot material {self.slot_material_id!r}"
            )

    def to_json(self) -> dict:
        return {
            "condition": self.condition,
            "expected_material_ids": list(self.expected_material_ids),
            "slot_index": self.slot_index,
            "slot_material_id": self.slot_material_id,
        }


@dataclass
class ExposureResult:
    """One condition's verdict, with the evidence for it."""

    condition: str
    status: str
    reason_codes: list = field(default_factory=list)
    scope: str = "request"
    checked_blocks: int = 0
    unmapped_blocks: int = 0
    present_material_ids: list = field(default_factory=list)
    undeclared_material_ids: list = field(default_factory=list)
    missing_material_ids: list = field(default_factory=list)
    matched_propositions: list = field(default_factory=list)
    missing_required_groups: list = field(default_factory=list)
    payload_digest: str = ""
    contract_version: str = ""
    parent_visible_history_digest: str = ""
    # Kept so a reviewer can re-derive the verdict instead of trusting it.
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "condition": self.condition,
            "status": self.status,
            "reason_codes": sorted(set(self.reason_codes)),
            "scope": self.scope,
            "checked_blocks": self.checked_blocks,
            "unmapped_blocks": self.unmapped_blocks,
            "present_material_ids": sorted(set(self.present_material_ids)),
            "undeclared_material_ids": sorted(set(self.undeclared_material_ids)),
            "missing_material_ids": sorted(set(self.missing_material_ids)),
            "matched_propositions": sorted(set(self.matched_propositions)),
            "missing_required_groups": [list(g) for g in self.missing_required_groups],
            "payload_digest": self.payload_digest,
            "contract_version": self.contract_version,
            "parent_visible_history_digest": self.parent_visible_history_digest,
            "evidence": self.evidence,
        }


# Provenance categories for a model-visible block.  Kept apart because they are
# subject to different checks, and pooling them would let whichever check is
# weakest decide for all of them.
ORIGIN_MESSAGE = "message"    # authored text: system prompt, user turns, notes
ORIGIN_TOOL = "tool"          # a declared tool's own output
ORIGIN_SCHEMA = "schema"      # tool names and descriptions


def visible_blocks(payload: Mapping) -> list:
    """Every block of text the model could read, flattened, with its origin.

    Deliberately covers the whole request rather than the retrieved-document list:
    system messages, tool results, inherited history, tool schemas and source
    labels are all reachable by the model, and a leak in any of them is a leak.

    Each block carries an origin so the *provenance* check can be scoped without
    weakening the *evidence* checks.  Tool output is generated rather than
    authored, so there is nothing to register as a literal — but it is still
    scanned for undeclared evidence and for decisive statements, which is what
    would catch a tool that returned the answer.  Allowing it to skip the residue
    check is therefore not a hole: the two checks that matter still apply.
    """
    blocks = []
    for message in payload.get("messages", []) or []:
        origin = ORIGIN_TOOL if message.get("role") == "tool" else ORIGIN_MESSAGE
        content = message.get("content")
        if isinstance(content, str):
            blocks.append({"text": content, "origin": origin})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    blocks.append({"text": part["text"], "origin": origin})
    for schema in payload.get("tools", []) or []:
        function = schema.get("function", {}) if isinstance(schema, dict) else {}
        block = " ".join(
            str(function.get(key, "")) for key in ("name", "description")
        )
        if block.strip():
            blocks.append({"text": block, "origin": ORIGIN_SCHEMA})
    return blocks


def trace_block(block: str, registered_texts: Sequence[str],
                wrappers: Sequence[str] = (),
                patterns: Sequence = (),
                dialogue_groups: Sequence = ()) -> tuple:
    """Map a visible block onto what it was built from.

    Returns ``(residue, matched_ids)`` where *residue* is what is left of the
    block once every registered body and every registered wrapper fragment has
    been removed.

    Why not simple containment.  The first version asked "does the block contain
    a registered body?".  That passes a block which is one registered document
    **plus anything else at all** — and since memory notes are folded into the
    user turn, every probe turn contains a registered body by construction, so
    the whole turn was whitelisted.  An unregistered sentence appended to a
    delivered block would have been invisible.  That is precisely the leak the
    gate exists to find.

    Stripping instead means the residue must be accounted for.  Wrapper text is
    registered separately from material because the two are different things: a
    delivered document is evidence, the line "Relevant notes from earlier
    conversations:" is packaging.
    """
    text = block
    matched = []
    # Patterns run first.  Order is load-bearing: the bullet surface is the
    # two-character string "- ", and stripping literals first would eat the
    # hyphens out of an ISO timestamp, leaving "2026 03 20T09:00:00+08:00" as
    # residue that no registered surface can explain.
    for pattern in patterns:
        text = pattern.sub(" ", text)
    # Longest first, so a shorter body that is a prefix of a longer one cannot
    # consume only part of it and leave the remainder behind as false residue.
    for index, body in enumerate(sorted(registered_texts, key=len, reverse=True)):
        if body and body in text:
            matched.append(index)
            text = text.replace(body, " ")
    for group in dialogue_groups:
        for body in sorted(group, key=len, reverse=True):
            if body and body in text:
                text = text.replace(body, " ")
    for wrapper in sorted(wrappers, key=len, reverse=True):
        if wrapper:
            text = text.replace(wrapper, " ")
    return " ".join(text.split()), matched


def _present_ids(blocks: Sequence, bodies: Mapping[str, str]) -> set:
    """Which registered bodies appear anywhere in the visible blocks."""
    joined = "\n".join(b["text"] for b in blocks)
    return {
        material_id
        for material_id, body in bodies.items()
        if body and body.strip() and body.strip() in joined
    }


def check_request(
    *,
    condition: str,
    payload: Mapping,
    contract: ExposureContract,
    corpus_bodies: Mapping[str, str],
    manifest: DeliveryManifest,
    allowed_surfaces: Sequence[str] = (),
    allowed_patterns: Sequence = (),
    registered_dialogue: Sequence = (),
    parent_visible_history: Sequence[str] = (),
    max_payload_bytes: int = 0,
) -> ExposureResult:
    """Check one serialized request against the contract and its manifest.

    ``corpus_bodies`` maps *every* material id in the corpus to its body, not only
    the allowed ones: exclusivity ("nothing undeclared is present") can only be
    checked against the full registry, and that check is what catches a leak whose
    wording nobody anticipated.

    Three kinds of thing may legitimately be on screen, and they are registered
    separately rather than pooled, because conflating them is how a leak gets
    waved through:

    ``corpus_bodies``
        Evidence.  Subject to the exclusivity check: only the manifest's
        declared set may appear.
    ``allowed_surfaces``
        The harness's own packaging — system prompt, wrapper text, rendered
        clock line, tool names and descriptions.  Registered as literal strings.
    ``registered_dialogue``
        Text the *scenario* authors for the model to read: the probe the user
        says, and the frozen background.  Kept distinct from evidence because it
        is not retrieved material and must never be counted toward the manifest,
        but it is equally subject to having been registered in advance.
    """
    if condition not in ("oracle", "omit"):
        raise ValidationError(f"exposure gate: unknown condition {condition!r}")
    if manifest.condition != condition:
        raise ValidationError(
            f"exposure gate: manifest declares {manifest.condition!r} but the "
            f"request is being checked as {condition!r}"
        )
    undeclared_in_manifest = set(manifest.expected_material_ids) - set(corpus_bodies)
    if undeclared_in_manifest:
        raise ValidationError(
            f"exposure gate: manifest names material absent from the corpus: "
            f"{sorted(undeclared_in_manifest)}"
        )

    result = ExposureResult(
        condition=condition,
        status=PASS,
        contract_version=contract.contract_version,
        payload_digest=digest(payload),
        parent_visible_history_digest=digest(list(parent_visible_history)),
    )
    blocks = visible_blocks(payload)
    result.checked_blocks = len(blocks)
    result.evidence["blocks_checked"] = len(blocks)

    # ---- size bound ----
    #
    # What this can and cannot do, stated because it would be easy to read it as
    # a guarantee it does not provide.  The harness can prove what it *sent*; it
    # cannot prove what the endpoint did with it.  A request far over the served
    # context window may be truncated server-side, silently and invisibly to
    # this check, and the truncated form is what the model reads.
    #
    # So the bound is a **precondition on trusting the delivery claim**: if the
    # request is small relative to the declared window, server-side truncation
    # cannot be the explanation for missing evidence, and the provenance and
    # exclusivity results below are load-bearing.  If it is large, they are not,
    # and the honest verdict is that the measurement is unverified.
    if max_payload_bytes:
        size = len(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        result.evidence["payload_bytes"] = size
        result.evidence["max_payload_bytes"] = max_payload_bytes
        if size > max_payload_bytes:
            result.status = aggregate(result.status, RUN_INVALID)
            result.reason_codes.append("payload_over_declared_budget")

    expected = set(manifest.expected_material_ids)
    all_known = list(corpus_bodies.values()) + list(allowed_surfaces)

    # ---- provenance: every block must be accounted for ----
    #
    # The test is on the *residue*: strip every registered body and wrapper, and
    # whatever is left is text nobody registered.  Note the asymmetry from a
    # keyword scan -- a scan asks whether known-bad phrases appear, so it passes
    # any bad phrase nobody listed, whereas requiring an empty residue means new
    # content cannot ride along inside an otherwise legitimate block.
    for index, block in enumerate(blocks):
        if block["origin"] == ORIGIN_TOOL:
            # Generated, not authored: nothing to register as a literal.  The
            # evidence checks below still read it, so a tool that returned the
            # update is caught there rather than here.
            continue
        residue, _ = trace_block(
            block["text"], all_known, allowed_surfaces, allowed_patterns,
            registered_dialogue,
        )
        if not residue:
            continue
        # Unknown content is not neutral.  It is a failure to verify, because the
        # whole point of registration is that nothing unregistered reaches the model.
        result.unmapped_blocks += 1
        result.reason_codes.append("unregistered_content")
        result.evidence.setdefault("unmapped", []).append(
            {"block_index": index, "residue": residue[:160]}
        )
    if result.unmapped_blocks:
        result.status = aggregate(result.status, RUN_INVALID)

    # ---- delivery, in both directions ----
    #
    # This is the strong check, and the reason the gate is a gate rather than a
    # lint.  Direction one: everything the manifest promises is actually in the
    # request, so the oracle condition cannot silently lose its evidence to a
    # truncation or a wrapper change.  Direction two: nothing else from the corpus
    # is, so the omit condition cannot receive the update under a phrasing no one
    # enumerated.  Neither direction can be satisfied by pattern-matching.
    present = _present_ids(blocks, corpus_bodies)
    result.present_material_ids = sorted(present)
    result.undeclared_material_ids = sorted(present - expected)
    result.missing_material_ids = sorted(expected - present)

    if result.undeclared_material_ids:
        result.status = aggregate(result.status, RUN_INVALID)
        result.reason_codes.append("undeclared_material_present")
        result.evidence["undeclared"] = result.undeclared_material_ids
    if result.missing_material_ids:
        result.status = aggregate(result.status, RUN_INVALID)
        result.reason_codes.append("declared_material_missing")
        result.evidence["missing"] = result.missing_material_ids

    # ---- the slot must hold the declared material ----
    slot_body = corpus_bodies.get(manifest.slot_material_id, "")
    if slot_body.strip() not in "\n".join(b["text"] for b in blocks):
        result.status = aggregate(result.status, RUN_INVALID)
        result.reason_codes.append("slot_mismatch")
        result.evidence["slot_material_id"] = manifest.slot_material_id
    result.evidence["slot_index"] = manifest.slot_index

    # ---- decisive propositions in the delivered blocks ----
    for index, block in enumerate(blocks):
        asserted = contract.decisive_asserted_in(block["text"])
        for proposition_id in asserted:
            result.matched_propositions.append(proposition_id)
            result.evidence.setdefault("proposition_hits", []).append(
                {"block_index": index, "origin": block["origin"],
                 "proposition": proposition_id}
            )

    if condition == "omit":
        # Any decisive proposition present is a leak, including one that names no
        # date: "原来的预约已经作废" settles the question without a date.  This
        # catches the case exclusivity cannot: a *registered* shared document that
        # was edited into saying something decisive.
        if result.matched_propositions:
            result.status = aggregate(result.status, RUN_INVALID)
            result.reason_codes.append("decisive_leak")
    else:
        # oracle: sufficiency is about content surviving into the request, not about
        # a document id being present somewhere.
        joined = "\n".join(b["text"] for b in blocks)
        # Each group is a set of **alternatives**: it names one requirement of the
        # delivery, and any registered phrasing satisfies it.  Requiring every term
        # of a group would make the gate test wording rather than content — the
        # authority requirement is met by "预约中心" or by "官方", and demanding both
        # would fail a perfectly adequate notice.
        #
        # Co-occurrence is enforced by the propositions instead: a proposition fires
        # only when all of its terms appear in one block, so "the cancellation, about
        # this event" is expressed there rather than here.
        for group in contract.oracle_required_groups:
            if not any(term in joined for term in group):
                result.missing_required_groups.append(group)
        if result.missing_required_groups:
            result.status = aggregate(result.status, RUN_INVALID)
            result.reason_codes.append("insufficient_evidence")

    # ---- parent contamination ----
    for block in parent_visible_history:
        for proposition_id in contract.decisive_asserted_in(block):
            result.reason_codes.append("parent_contaminated")
            result.evidence["parent_hit"] = {
                "proposition": proposition_id, "preview": block[:120]
            }
            # Scope is the whole derived family, not this one request: every pair
            # forked from that parent is unusable, so a single invalid request
            # cannot express the damage.
            result.status = aggregate(result.status, BATCH_FAIL)
            result.scope = "parent_family"
            break

    if result.status == PASS and result.unmapped_blocks == 0:
        # Explicitly recorded so a reader can see the verdict rests on a completed
        # check rather than on finding nothing.
        result.reason_codes.append("provenance_complete")
    return result


def check_parent_state(
    *,
    visible_history: Iterable[str],
    contract: ExposureContract,
    label: str = "parent",
) -> ExposureResult:
    """Pre-fork check: has the key update already reached model-visible history?

    A world fact store holding the correct date is normal and is not contamination.
    What matters is whether that fact is reachable by the model — through delivered
    messages, a summary, or an inherited reference.  Deleting the retrieved document
    afterwards would not undo it.
    """
    result = ExposureResult(condition=label, status=PASS, scope="parent_family",
                            contract_version=contract.contract_version)
    hits = []
    for block in visible_history:
        for proposition_id in contract.decisive_asserted_in(block):
            hits.append({"proposition": proposition_id, "preview": block[:120]})
    result.evidence["hits"] = hits
    if hits:
        result.status = BATCH_FAIL
        result.reason_codes.append("parent_contaminated")
        result.matched_propositions = [h["proposition"] for h in hits]
    else:
        result.reason_codes.append("parent_clean")
    result.parent_visible_history_digest = digest(list(visible_history))
    return result
