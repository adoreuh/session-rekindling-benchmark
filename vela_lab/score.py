"""Dependent-variable extraction, stratified into four layers.

Contract §11 forbids answering "did the model forget?" with a single number.
The layers are:

**L1 Temporal sensitivity** — did the decay function actually read elapsed time?
Read from the recorded temporal weights.  No model call involved.

**L2 Accessibility** — did elapsed time change which evidence was reachable?
Read from the recorded ranks and the delivered set.  No model call involved.

**L3 Behavioural recall** — did the model's *answer* change?  Requires a real
model and, for anything beyond lexical matching, a judge.

**L4 Attribution** — where did a failure occur?  This layer is the point of the
whole design, and it is where the previous version went wrong: it treated
"gold was in the top-k" and "gold was in the request" as the same fact.  They
are not, and the exposure chain records them separately:

    no_retrieval          the agent never asked
    not_selected          asked, but gold was not in the returned set
    selected_not_delivered selected, but absent from the recorded request
    delivered_not_used    in the request, and the answer still disagrees

The last of those is a statement about behaviour.  It is *not* a claim about the
model's internal reasoning — "it saw the evidence and chose to ignore it" is not
observable from outside the model.  Report it as exposure without use.
"""

from __future__ import annotations

from typing import Any, Sequence


ATTRIBUTION_LABELS = (
    "no_retrieval",
    "not_selected",
    "selected_not_delivered",
    "delivered_not_used",
    "delivered_and_used",
    "not_applicable",
)


def layer_l1(events: Sequence[dict]) -> dict:
    """Temporal sensitivity: what weight did the retriever apply?"""
    series = []
    for event in events:
        if event["type"] != "memory.retrieval":
            continue
        candidates = event["data"].get("candidates", [])
        weights = [c.get("temporal_weight") for c in candidates]
        series.append(
            {
                "seq": event["seq"],
                "clock": event["data"].get("clock_instant"),
                "min_weight": min(weights) if weights else None,
                "max_weight": max(weights) if weights else None,
                "gold_weight": _gold_field(event, "temporal_weight"),
            }
        )
    return {
        "layer": "L1_temporal_sensitivity",
        "retrievals": series,
        "responds_to_elapsed_time": _varies([row["gold_weight"] for row in series]),
    }


def layer_l2(events: Sequence[dict]) -> dict:
    """Accessibility: did elapsed time change what was reachable?

    Retrievals are split by who initiated them, and that split matters:

    ``auto``  the host-driven retrieval performed once per user turn.  This is the
              one the experimental design governs, so this is where the
              accessibility manipulation is realized and where the primary L2
              numbers come from.
    ``probe`` a search the agent chose to make.  It obeys the same visibility
              policy, but *which query it uses* is behaviour, not design.  A model
              that searches with a better query can find what the host-driven
              retrieval missed, and reporting that as "the retriever worked" would
              credit the mechanism for the agent's effort.

    Reporting a single ``l2[-1]`` would therefore mix a design variable with a
    behavioural one.  Both are recorded; only ``auto`` enters the headline.
    """
    series = []
    for event in events:
        if event["type"] != "memory.retrieval":
            continue
        data = event["data"]
        hits = data.get("hits", [])
        gold = [h for h in hits if h.get("is_gold")]
        series.append(
            {
                "seq": event["seq"],
                "mode": data.get("mode", ""),
                "clock": data.get("clock_instant"),
                "query": data.get("query", ""),
                "eligible_count": len(data.get("eligible_ids", [])),
                "returned": [h["doc_id"] for h in hits],
                "gold_rank": gold[0]["rank"] if gold else None,
                "gold_returned": bool(gold),
                "result_count": len(hits),
            }
        )
    host_driven = [row for row in series if row["mode"] == "auto"]
    agent_initiated = [row for row in series if row["mode"] == "probe"]
    return {
        "layer": "L2_accessibility",
        "retrievals": series,
        "host_driven": host_driven,
        "agent_initiated": agent_initiated,
        "host_driven_gold_returned": (
            host_driven[-1]["gold_returned"] if host_driven else None
        ),
        "host_driven_gold_rank": host_driven[-1]["gold_rank"] if host_driven else None,
        "agent_searched_again": len(agent_initiated) > 0,
        "agent_recovered_gold": (
            any(row["gold_returned"] for row in agent_initiated)
            if agent_initiated else False
        ),
        "gold_access_changes": _varies([row["gold_returned"] for row in series]),
        "gold_rank_changes": _varies([row["gold_rank"] for row in series]),
    }


def date_forms(iso_date: str) -> tuple:
    """Rendering variants of an ISO date, as visible text might state it.

    Needed because the "did the answer reflect the evidence?" check has to match
    *content*, and the content of an evidence claim is the date it introduces.
    The document id cannot serve: it is harness vocabulary that no natural answer
    contains, so testing for it made ``delivered_and_used`` unreachable and
    labelled every usable answer ``delivered_not_used`` — the reverse of what the
    answer text showed.

    Deliberately a small, explicit set of conventions rather than a date parser,
    so what counts as a match is inspectable.  A parser would accept formats the
    evidence never used, which would make the signal unfalsifiable.
    """
    import re

    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", (iso_date or "").strip())
    if not match:
        return (iso_date.strip(),) if (iso_date or "").strip() else ()
    year, month, day = match.groups()
    m, d = int(month), int(day)
    return tuple(dict.fromkeys((
        f"{year}-{month}-{day}",
        f"{m} 月 {d} 日",
        f"{m}月{d}日",
        f"{year} 年 {m} 月 {d} 日",
        f"{year}年{m}月{d}日",
    )))


def layer_l4(events: Sequence[dict], *, gold_marker: str,
             answer_text: str, decisive_terms: Sequence[str] = ()) -> dict:
    """Attribution: walk the exposure chain for one gold document.

    ``decisive_terms`` are the rendered forms of what the delivered evidence
    asserts — for this study, the operative date.  The answer counts as
    reflecting the evidence when any of them appears.  When none are supplied the
    check falls back to the document id, which is kept only so an earlier run's
    record stays readable; it is not a usable signal for a real answer.
    """
    retrievals = [e for e in events if e["type"] == "memory.retrieval"]
    requests = [e for e in events if e["type"] == "model.request"]

    if not retrievals:
        return {
            "layer": "L4_attribution",
            "gold_marker": gold_marker,
            "label": "no_retrieval",
            "chain": {
                "retrieval_issued": False,
                "gold_eligible": False,
                "gold_selected": False,
                "gold_delivered": False,
            },
        }

    eligible = False
    selected = False
    delivered = False
    delivered_texts = []
    for retrieval in retrievals:
        for candidate in retrieval["data"].get("candidates", []):
            if candidate.get("doc_id") == gold_marker:
                eligible = True
        for hit in retrieval["data"].get("hits", []):
            if hit.get("doc_id") == gold_marker:
                selected = True
                if hit.get("visible_text"):
                    delivered_texts.append(hit["visible_text"])
    for request in requests:
        for message in request["data"].get("messages", []):
            content = message.get("content")
            if isinstance(content, str) and any(
                text in content for text in delivered_texts
            ):
                delivered = True

    signals = tuple(decisive_terms) or (gold_marker,)
    answer_uses = bool(answer_text) and any(
        signal and signal in answer_text for signal in signals
    )
    if not eligible:
        label = "not_applicable"
    elif not selected:
        label = "not_selected"
    elif not delivered:
        label = "selected_not_delivered"
    elif answer_uses:
        label = "delivered_and_used"
    else:
        label = "delivered_not_used"

    return {
        "layer": "L4_attribution",
        "gold_marker": gold_marker,
        "label": label,
        "chain": {
            "retrieval_issued": True,
            "gold_eligible": eligible,
            "gold_selected": selected,
            "gold_delivered": delivered,
        },
        "caveat": (
            "'delivered_not_used' means the evidence reached the final request and "
            "the answer did not reflect it. It does not establish what the model "
            "attended to internally."
        ),
    }


def _gold_field(event: dict, field: str) -> Any:
    for hit in event["data"].get("hits", []):
        if hit.get("is_gold"):
            return hit.get(field)
    return None


def _varies(values: Sequence[Any]) -> bool:
    distinct = {v for v in values}
    return len(distinct) > 1


def score_run(events: Sequence[dict], *, gold_marker: str,
              answer_text: str, decisive_terms: Sequence[str] = ()) -> dict:
    """Assemble the four-layer report for one run.

    ``decisive_terms`` is what the evidence asserts, in the renderings an answer
    could plausibly use; see ``date_forms``.  Passing it is what makes the L4
    "used / not used" split mean something.
    """
    return {
        "L1": layer_l1(events),
        "L2": layer_l2(events),
        "L3": {
            "layer": "L3_behavioural_recall",
            "answer_length": len(answer_text),
            "note": (
                "L3 requires a real model. Lexical signals are recorded here; "
                "semantic correctness needs a blind judge and is reported "
                "separately."
            ),
        },
        "L4": layer_l4(
            events, gold_marker=gold_marker, answer_text=answer_text,
            decisive_terms=decisive_terms,
        ),
    }
