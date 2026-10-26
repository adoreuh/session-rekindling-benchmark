"""Controlled evidence corpus and the exposure chain.

The corpus holds documents.  Ground truth does not live here — it lives in
``WorldDomain``.  Documents are *evidence about* the world, and the point of the
module is to keep the two apart while recording, for every retrieval, the full
chain from "what exists" to "what the model actually saw".

That chain is the L4 measurement (contract §11):

    exists and policy allows
      -> in candidate set
      -> selected by retrieval
      -> projected into the final request
      -> provider dispatched
      -> answer agrees with evidence

The old harness could only observe the first and roughly the last of those.  It
therefore could not tell "never retrieved" from "retrieved but not delivered"
from "delivered but not used", and its most-cited conclusion rested on the
middle distinction.  Recording every arrow is the fix.

Retrieval is lexical by default (character bigrams — CJK-aware and
deterministic, no tokenizer).  Embeddings can be plugged in, but the interface
requires recording the model id, the corpus digest, the query, and every score,
and requires a deterministic tiebreak.  A retrieval whose scores cannot be
reproduced offline is not admissible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import ContractViolation, ValidationError
from .jsonutil import canonical_json, digest, json_copy

VISIBILITY_MODES = frozenset(
    {
        "full",       # everything eligible is visible
        "omit",       # matching documents hidden
        "stale",      # superseded documents shown, current ones hidden
        "partial",    # fields stripped from the returned documents
        "none",       # nothing returned
    }
)


@dataclass(frozen=True)
class Document:
    """One unit of evidence."""

    doc_id: str
    text: str
    authority: str = "medium"        # low | medium | high
    source: str = ""
    published_at: str | None = None  # virtual time it becomes queryable
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    facts: tuple = ()
    stance: str = "neutral"          # supports_current | supports_old | neutral
    is_gold: bool = False

    def __post_init__(self):
        if not self.doc_id or not isinstance(self.doc_id, str):
            raise ValidationError("document requires a non-empty doc_id")
        if self.authority not in ("low", "medium", "high"):
            raise ValidationError(f"document {self.doc_id}: bad authority")

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "authority": self.authority,
            "source": self.source,
            "published_at": self.published_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "superseded_by": self.superseded_by,
            "facts": list(self.facts),
            "stance": self.stance,
            "is_gold": self.is_gold,
        }

    @staticmethod
    def from_json(payload: dict) -> "Document":
        data = dict(payload)
        data["facts"] = tuple(data.get("facts", ()))
        return Document(**data)

    def digest(self) -> str:
        return digest(self.to_json())


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float
    lexical: float
    temporal_weight: float
    rank: int
    is_gold: bool
    reason: str = "ranked"  # "ranked" | "oracle_forced"

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "lexical": self.lexical,
            "temporal_weight": self.temporal_weight,
            "rank": self.rank,
            "is_gold": self.is_gold,
            "reason": self.reason,
        }


@dataclass
class RetrievalRecord:
    """Everything needed to reconstruct a retrieval offline."""

    query: str
    mode: str
    top_k: int
    corpus_digest: str
    retriever_id: str
    eligible_ids: list = field(default_factory=list)
    candidates: list = field(default_factory=list)      # all scored, pre-truncation
    hits: list = field(default_factory=list)             # post-policy, ranked
    hidden_ids: list = field(default_factory=list)
    oracle_forced: list = field(default_factory=list)
    clock_instant: str = ""
    requested_top_k: int | None = None

    def retrieved_ids(self) -> list:
        return [h["doc_id"] if isinstance(h, dict) else h.doc_id for h in self.hits]

    def gold_rank(self) -> int | None:
        for hit in self.hits:
            record = hit if isinstance(hit, dict) else hit.to_json()
            if record["is_gold"]:
                return record["rank"]
        return None

    def gold_delivered(self) -> bool:
        return any(
            (h if isinstance(h, dict) else h.to_json())["is_gold"] for h in self.hits
        )

    def to_json(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "top_k": self.top_k,
            "corpus_digest": self.corpus_digest,
            "retriever_id": self.retriever_id,
            "eligible_ids": list(self.eligible_ids),
            "candidates": json_copy(self.candidates, "candidates"),
            "hits": [
                h.to_json() if isinstance(h, RetrievalHit) else json_copy(h, "hit")
                for h in self.hits
            ],
            "hidden_ids": list(self.hidden_ids),
            "oracle_forced": list(self.oracle_forced),
            "clock_instant": self.clock_instant,
            "requested_top_k": self.requested_top_k,
            "delivered_ids": [h["doc_id"] for h in self.hits],
        }


# ------------------------------------------------------------------ similarity


def _bigrams(text: str) -> set:
    """Character bigrams.  CJK-safe, no tokenizer, fully deterministic."""
    cleaned = "".join(ch.lower() for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def lexical_similarity(query: str, text: str) -> float:
    """Jaccard over character bigrams, in [0, 1]."""
    left, right = _bigrams(query), _bigrams(text)
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class LexicalRetriever:
    """Deterministic default retriever.

    Reports its own weakness honestly: the project found gold scored 0.0000 on
    one probe with this method.  That is a property of the retriever, not
    evidence about the model, and the exposure chain records the scores so the
    distinction stays visible.
    """

    retriever_id = "lexical-bigram-jaccard-v1"

    def score(self, query: str, doc: Document) -> float:
        return lexical_similarity(query, doc.text)

    def tiebreak(self, doc_id: str) -> str:
        return doc_id


# --------------------------------------------------------------------- corpus


class EvidenceCorpus:
    """Documents plus the visibility policy.  No ground truth here."""

    def __init__(self, documents: Iterable[Document], retriever: Any | None = None):
        self._documents: dict[str, Document] = {}
        for doc in documents:
            if doc.doc_id in self._documents:
                raise ValidationError(f"duplicate document id {doc.doc_id!r}")
            self._documents[doc.doc_id] = doc
        self.retriever = retriever or LexicalRetriever()
        self._sealed = False

    def seal(self) -> None:
        self._sealed = True

    def add(self, doc: Document) -> None:
        if self._sealed:
            raise ContractViolation(
                "evidence corpus is sealed; adding a document mid-run would change "
                "the candidate set between branches"
            )
        self._documents[doc.doc_id] = doc

    def get(self, doc_id: str) -> Document:
        doc = self._documents.get(doc_id)
        if doc is None:
            raise ValidationError(f"unknown document {doc_id!r}")
        return doc

    def all_documents(self) -> dict:
        return dict(self._documents)

    def digest(self) -> str:
        return digest({k: v.to_json() for k, v in sorted(self._documents.items())})

    def gold_ids(self) -> list:
        return sorted(d.doc_id for d in self._documents.values() if d.is_gold)

    # ------------------------------------------------------------- retrieval

    def retrieve(
        self,
        *,
        query: str,
        now_instant: str,
        top_k: int,
        mode: str = "auto",
        visibility: str = "full",
        omit_ids: Sequence[str] = (),
        force_include_ids: Sequence[str] = (),
        strip_fields: Sequence[str] = (),
        temporal_decay: Any | None = None,
        deliver_ids: Sequence[str] | None = None,
    ) -> RetrievalRecord:
        """Run one retrieval and record the whole exposure chain.

        ``mode`` is ``"auto"`` for registered system events or ``"probe"`` for
        probe-triggered retrieval; it is recorded so a reader can see which
        retrievals the harness caused and which the agent asked for.

        ``force_include_ids`` implements the matched oracle.  It promotes the
        named documents into the result set **while keeping the result count
        equal to the natural case**, by displacing the lowest-ranked natural
        hits rather than appending.  A hard filter would change the number of
        documents on screen and therefore confound accessibility with context
        volume — the mistake the old harness had to retire.
        """
        if visibility not in VISIBILITY_MODES:
            raise ValidationError(f"unknown visibility mode {visibility!r}")
        if top_k < 1:
            raise ValidationError("top_k must be at least 1")

        eligible = self._eligible(now_instant, visibility, omit_ids)
        scored = []
        for doc_id in eligible:
            doc = self._documents[doc_id]
            lexical = float(self.retriever.score(query, doc))
            weight = 1.0
            if temporal_decay is not None:
                weight = float(
                    temporal_decay(
                        observed_at=doc.published_at or now_instant,
                        now=now_instant,
                    )
                )
            scored.append(
                {
                    "doc_id": doc_id,
                    "lexical": lexical,
                    "temporal_weight": weight,
                    "score": lexical * weight,
                    "is_gold": doc.is_gold,
                }
            )
        # Deterministic ordering: score desc, then doc_id asc.  No reliance on
        # insertion order or float noise.
        scored.sort(key=lambda row: (-row["score"], row["doc_id"]))

        hits = []
        hidden = []
        for row in scored:
            doc = self._documents[row["doc_id"]]
            hits.append(
                {
                    "doc_id": row["doc_id"],
                    "score": row["score"],
                    "lexical": row["lexical"],
                    "temporal_weight": row["temporal_weight"],
                    "rank": len(hits) + 1,
                    "is_gold": doc.is_gold,
                    "reason": "ranked",
                }
            )

        oracle_forced: list[str] = []
        if deliver_ids is not None:
            # Explicit ordered delivery.  `force_include_ids` decides the *set* but
            # not the order: it promotes the forced documents while preserving their
            # natural ranking, so the same set can come back in different orders and
            # a "fixed slot" cannot be expressed.  A controlled-delivery comparison
            # needs the position of the differing document to be identical across
            # conditions, so this path states the delivered list outright.
            #
            # Eligibility still applies.  A document that is unpublished at the probe
            # instant, or deliberately omitted, cannot be delivered by naming it —
            # that would let a policy smuggle in evidence the corpus says is not yet
            # available.
            requested = list(deliver_ids)
            if len(set(requested)) != len(requested):
                raise ValidationError(f"deliver_ids contains a duplicate: {requested}")
            unknown = [d for d in requested if d not in self._documents]
            if unknown:
                raise ValidationError(f"deliver_ids names unknown documents {unknown}")
            not_eligible = [d for d in requested if d not in set(eligible)]
            if not_eligible:
                raise ContractViolation(
                    f"deliver_ids names documents that are not deliverable at this "
                    f"instant: {not_eligible}. A document that is unpublished or "
                    "omitted cannot be delivered by naming it."
                )
            by_id = {h["doc_id"]: h for h in hits}
            # `hits` was built from eligible documents, so every requested id is
            # present; the lookup is by id and the order comes from the request.
            selected = []
            for index, doc_id in enumerate(requested, start=1):
                hit = dict(by_id[doc_id])
                hit["rank"] = index
                selected.append(hit)
            hits = selected
        elif force_include_ids:
            forced = [did for did in force_include_ids if did in self._documents]
            missing = [did for did in force_include_ids if did not in self._documents]
            if missing:
                raise ValidationError(f"oracle requested unknown documents {missing}")
            # Promote forced docs, then fill remaining slots from the natural
            # ranking, truncating to the same top_k.
            promoted = [h for h in hits if h["doc_id"] in forced]
            rest = [h for h in hits if h["doc_id"] not in forced]
            selected = promoted + rest
            selected = selected[:top_k]
            for index, hit in enumerate(selected, start=1):
                hit = dict(hit)
                hit["rank"] = index
                if hit["doc_id"] in forced:
                    hit["reason"] = "oracle_forced"
                selected[index - 1] = hit
            hits = selected
            oracle_forced = sorted(forced)
        else:
            hits = hits[:top_k]

        chosen = {h["doc_id"] for h in hits}
        hidden = sorted(d for d in eligible if d not in chosen)

        if strip_fields:
            for hit in hits:
                hit["stripped"] = sorted(strip_fields)

        record = RetrievalRecord(
            query=query,
            mode=mode,
            top_k=top_k,
            corpus_digest=self.digest(),
            retriever_id=getattr(self.retriever, "retriever_id", "unknown"),
            eligible_ids=sorted(eligible),
            candidates=scored,
            hits=hits,
            hidden_ids=hidden,
            oracle_forced=oracle_forced,
            clock_instant=now_instant,
        )
        _check_oracle_invariance(record, force_include_ids, top_k)
        return record

    def _eligible(self, now_instant: str, visibility: str,
                  omit_ids: Sequence[str]) -> list:
        from .clock import parse_instant

        now = parse_instant(now_instant)
        omit = set(omit_ids)
        eligible = []
        for doc_id, doc in self._documents.items():
            if doc_id in omit:
                continue
            if doc.published_at:
                if parse_instant(doc.published_at) > now:
                    continue  # not yet published: exists but not queryable
            if visibility == "stale":
                if doc.superseded_by is not None:
                    # keep superseded docs, drop their successors
                    eligible.append(doc_id)
                continue
            eligible.append(doc_id)
        if visibility == "none":
            return []
        if visibility == "stale":
            # In stale mode, hide the current authority entirely.
            current = {
                d.superseded_by for d in self._documents.values()
                if d.superseded_by and d.doc_id not in omit
            }
            eligible = [d for d in eligible if d not in current]
        return sorted(eligible)


def _check_oracle_invariance(record: RetrievalRecord, forced: Sequence[str],
                             top_k: int) -> None:
    """A forced-inclusion retrieval must not change the result count."""
    if forced and len(record.hits) != min(top_k, len(record.eligible_ids)):
        raise ContractViolation(
            "oracle retrieval changed the number of visible documents; it must "
            "only reorder within the same top_k (contract §7)"
        )


class ExposurePolicy:
    """Visibility settings for one branch.

    Declarative and hashed, so "what was hidden in this condition" is data, not
    a branch of code.  The old harness once expressed a condition as a sentence
    inside the prompt; this is the alternative.
    """

    def __init__(
        self,
        *,
        mode: str = "full",
        omit_ids: Sequence[str] = (),
        force_include_ids: Sequence[str] = (),
        strip_fields: Sequence[str] = (),
        top_k: int = 4,
        temporal_decay: Any | None = None,
        label: str = "natural",
        deliver_ids: Sequence[str] | None = None,
    ):
        if mode not in VISIBILITY_MODES:
            raise ValidationError(f"unknown visibility mode {mode!r}")
        if deliver_ids is not None and force_include_ids:
            raise ValidationError(
                "deliver_ids and force_include_ids are mutually exclusive: the "
                "first states the delivered list outright, the second only promotes "
                "within a ranking, so combining them has no defined meaning"
            )
        self.mode = mode
        self.omit_ids = tuple(omit_ids)
        self.force_include_ids = tuple(force_include_ids)
        self.strip_fields = tuple(strip_fields)
        self.top_k = top_k
        self.temporal_decay = temporal_decay
        self.label = label
        self.deliver_ids = tuple(deliver_ids) if deliver_ids is not None else None

    def identity(self) -> dict:
        return {
            "mode": self.mode,
            "omit_ids": list(self.omit_ids),
            "force_include_ids": list(self.force_include_ids),
            "strip_fields": list(self.strip_fields),
            "top_k": self.top_k,
            "label": self.label,
            "deliver_ids": list(self.deliver_ids) if self.deliver_ids else None,
        }

    def digest(self) -> str:
        return digest(self.identity())

    def __eq__(self, other: object) -> bool:
        """Value equality, so two policies with the same settings are the same policy.

        Without this, `policy_a == policy_b` compares identity and is always false,
        which makes an invariant like "all branches share the same evidence policy"
        impossible to assert directly.  The harness already compares policies when
        checking fork isolation, so value semantics are the correct ones here.
        """
        if not isinstance(other, ExposurePolicy):
            return NotImplemented
        return self.identity() == other.identity()

    def __hash__(self) -> int:
        # Identity is a plain dict, so hash its canonical encoding.
        return hash(canonical_json(self.identity()))

    def to_json(self) -> dict:
        return self.identity()
