"""Controlled key-evidence delivery: the omit/oracle pair.

Why this replaced natural/oracle.

The original accessibility contrast was natural (whatever retrieval returns)
against oracle (gold forced in at the same count).  It could not be constructed
here: the corpus contains exactly one document about the reschedule, so every
query aimed at that topic reaches it and every query that misses it is asking
about something else.  Reachability and topical aiming are not separable, so the
contrast had no clean interpretation.

The reviewer's decision was to stop trying to make natural retrieval fail and use a
**controlled delivery** contrast instead:

* both conditions receive the same shared context, the same document count, the
  same order, and the same wrapper;
* exactly one registered slot differs:

  ``oracle``  the slot holds the key update
  ``omit``    the slot holds a pre-registered filler that asserts no date

What this does and does not claim.  It measures whether *supplying the decisive
evidence* changes the answer under otherwise identical delivery.  It is **not** a
measurement of natural retrieval failure, and not of whether a model goes looking
for evidence on its own.  Both of those are separate questions with separate
designs, and reporting this as either would be a category error.

The filler must **not assert any date**.  A filler that states the old date is not
"no evidence" but "misleading evidence", which is a different condition and would
tilt the comparison rather than isolate delivery.  `select_filler` enforces that
with a check rather than trusting the author's choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from vela_lab.errors import ValidationError

# Terms that would let a filler assert or hint at a date.  Any of these in the
# filler makes it an evidence-bearing document rather than a neutral placeholder.
DATE_HINTS = (
    "18 日", "18日", "25 日", "25日", "6 月", "6月",
    "改期", "取消", "变更", "作废", "最新", "调整为",
)


@dataclass(frozen=True)
class AccessPair:
    """The two conditions of one controlled-delivery comparison."""

    oracle_doc_ids: tuple
    omit_doc_ids: tuple
    slot: int
    gold_id: str
    filler_id: str
    shared_ids: tuple

    def differs_only_in_the_slot(self) -> bool:
        """The contract's central property, checked rather than assumed."""
        if len(self.oracle_doc_ids) != len(self.omit_doc_ids):
            return False
        for index, (left, right) in enumerate(
            zip(self.oracle_doc_ids, self.omit_doc_ids)
        ):
            if index == self.slot:
                if left != self.gold_id or right != self.filler_id:
                    return False
            elif left != right:
                return False
        return True

    def to_json(self) -> dict:
        return {
            "oracle_doc_ids": list(self.oracle_doc_ids),
            "omit_doc_ids": list(self.omit_doc_ids),
            "slot": self.slot,
            "gold_id": self.gold_id,
            "filler_id": self.filler_id,
            "shared_ids": list(self.shared_ids),
            "differs_only_in_the_slot": self.differs_only_in_the_slot(),
        }


def build_access_pair(
    *,
    shared_ids: Sequence[str],
    gold_id: str,
    filler_id: str,
    slot: int,
    available_ids: Sequence[str],
) -> AccessPair:
    """Mint the paired document lists with one registered slot differing.

    Follows the reviewer's reference construction.  Validation is deliberately
    strict and happens before any model call, because a silently mis-built pair
    produces a comparison that looks fine and measures nothing.
    """
    shared_ids = tuple(shared_ids)
    available = set(available_ids)

    if not gold_id or not filler_id:
        raise ValidationError("access: gold_id and filler_id are required")
    if gold_id == filler_id:
        raise ValidationError(
            "access.filler_id: must differ from gold_id, or the two conditions "
            "deliver identical evidence and the contrast is empty"
        )
    if len(set(shared_ids)) != len(shared_ids):
        raise ValidationError("access.shared_ids: contains a duplicate document")
    if gold_id in shared_ids:
        raise ValidationError(
            "access.shared_ids: the gold document is already in the shared context, "
            "so the slot is not what makes it available"
        )
    if filler_id in shared_ids:
        raise ValidationError(
            "access.shared_ids: the filler is already in the shared context"
        )
    if isinstance(slot, bool) or not isinstance(slot, int):
        raise ValidationError("access.slot: expected an integer")
    if not 0 <= slot <= len(shared_ids):
        raise ValidationError(
            f"access.slot: {slot} is outside the shared context "
            f"(0..{len(shared_ids)})"
        )
    for document_id in (*shared_ids, gold_id, filler_id):
        if document_id not in available:
            raise ValidationError(f"access.documents[{document_id!r}]: missing")

    oracle_ids = shared_ids[:slot] + (gold_id,) + shared_ids[slot:]
    omit_ids = shared_ids[:slot] + (filler_id,) + shared_ids[slot:]
    pair = AccessPair(
        oracle_doc_ids=oracle_ids,
        omit_doc_ids=omit_ids,
        slot=slot,
        gold_id=gold_id,
        filler_id=filler_id,
        shared_ids=shared_ids,
    )
    if not pair.differs_only_in_the_slot():
        raise ValidationError(
            "access: constructed pair differs outside the slot; refusing to use it"
        )
    return pair


def policies_for_pair(pair: AccessPair, *, top_k: int | None = None) -> dict:
    """The two exposure policies for a pair, as a condition -> policy mapping."""
    return {
        "oracle": exposure_for_condition("oracle", doc_ids=pair.oracle_doc_ids,
                                         gold_id=pair.gold_id, top_k=top_k),
        "omit": exposure_for_condition("omit", doc_ids=pair.omit_doc_ids,
                                       gold_id=pair.gold_id, top_k=top_k),
    }


def select_filler(
    *, gold_id: str, documents: Mapping[str, str], slot_pool: Sequence[str],
    length_tolerance: float = 3.0,
) -> str:
    """Choose a filler that asserts no date and is close in length to the gold.

    ``slot_pool`` is the registered set of candidates; the function picks from it
    rather than searching the whole corpus, so the choice is a property of the
    registration rather than of the run.  If none qualifies, it raises instead of
    falling back to something that would change the condition's meaning.
    """
    gold_text = documents[gold_id]
    gold_len = len(gold_text)
    rejecting = []
    qualifying = []
    for candidate in slot_pool:
        if candidate == gold_id:
            continue
        text = documents[candidate]
        hits = [hint for hint in DATE_HINTS if hint in text]
        if hits:
            # A filler that names a date is not neutral: it either supports the
            # user's claim or contradicts it, which is a different condition.
            rejecting.append((candidate, f"asserts or hints at a date ({hits})"))
            continue
        ratio = max(len(text), gold_len) / max(min(len(text), gold_len), 1)
        if ratio > length_tolerance:
            rejecting.append(
                (candidate, f"length {len(text)} vs gold {gold_len} exceeds "
                            f"tolerance {length_tolerance}x")
            )
            continue
        qualifying.append((ratio, candidate))

    if not qualifying:
        raise ValidationError(
            "access.filler_id: no registered candidate qualifies as a neutral filler; "
            f"rejected: {rejecting}. Register a filler that asserts no date and has a "
            "comparable length, rather than relaxing the rule."
        )
    # Closest in length wins.  Taking the first qualifying candidate would make the
    # delivered text volume depend on registration order, and the oracle slot is
    # where "more evidence" must not be confounded with "more text".
    qualifying.sort(key=lambda pair: (pair[0], pair[1]))
    return qualifying[0][1]


def exposure_for_condition(condition: str, *, doc_ids: Sequence[str],
                           gold_id: str, top_k: int | None = None):
    """Build the exposure policy that delivers exactly ``doc_ids``, in order.

    Mechanism: ``deliver_ids`` states the delivered list outright.  It is needed
    because ``force_include_ids`` decides the *set* but not the *order*: it promotes
    the forced documents while preserving their natural ranking, so the same set can
    come back reordered and the slot's **position** ends up differing between
    conditions.  A controlled comparison needs both the content and the position of
    the differing document to be identical, so the delivered list is stated rather
    than induced.

    For ``omit`` the gold is additionally listed in ``omit_ids``.  It cannot appear
    once the delivered list is fixed, but naming it means a later change to the
    delivery path could not silently reintroduce it.
    """
    from vela_lab.evidence import ExposurePolicy

    if condition not in ("oracle", "omit"):
        raise ValidationError(f"access.condition: unknown condition {condition!r}")
    delivered = tuple(doc_ids)
    if not delivered:
        raise ValidationError("access: the delivered set is empty")
    if condition == "oracle" and gold_id not in delivered:
        raise ValidationError(
            "access: the oracle condition must deliver the gold document"
        )
    if condition == "omit" and gold_id in delivered:
        raise ValidationError(
            "access: the omit condition must not deliver the gold document"
        )
    return ExposurePolicy(
        mode="full",
        top_k=top_k if top_k is not None else len(delivered),
        deliver_ids=delivered,
        omit_ids=(gold_id,) if condition == "omit" else (),
        label=condition,
    )
