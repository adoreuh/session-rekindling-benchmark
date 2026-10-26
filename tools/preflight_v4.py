"""Pre-flight gate for the v4 design: can this run produce a clean comparison?

Replaces an earlier pre-flight that validated a design this project has since
retired.  That version built its own corpus and its own probes and asked whether
*retrieval* reached the notice at 1d but not at 15d.  Decision B removed that
question from the design: the accessibility contrast is now **controlled
delivery** — the same context with one registered slot differing — precisely
because reachability and topical aiming turned out not to be separable in this
corpus.  Checking whether retrieval fails would now be checking something the
run does not depend on.

What this gate does check, one assertion per way the comparison could be empty:

1. **The pair differs only in the slot.**  If the shared context drifts apart,
   the contrast measures the drift instead of the delivery.
2. **The omit slot asserts no date.**  A filler that named a date would be
   misleading evidence, which is a different condition from no evidence.
3. **Both conditions are deliverable at the probe instant.**  A document not yet
   published cannot be delivered by naming it, so a policy that names it would
   fail at run time — better to fail here.
4. **The corpus is big enough that ranking is not pure recency.**  This is the
   defect that made the previous oracle manipulation inert.
5. **No decisive proposition fires on neutral material.**  A proposition that
   matched the filler or a shared document would make every omit branch fail the
   gate, and the failure would be the gate's fault, not the run's.
6. **The decisive propositions do fire on the notice.**  A proposition set that
   matched nothing could not detect a leak either.
7. **The two pressure levels assert the same fact.**  Pressure must move affect,
   not the premise.
8. **The no-self-retrieval protocol is enforced.**  The probe turn must expose
   only the declared tools, so the delivery is the only route to the evidence.
9. **Text volume is matched.**  "Evidence supplied" must not be confounded with
   "more text supplied".

No provider is called.  This is a gate, not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.evidence import EvidenceCorpus, Document  # noqa: E402

from tools import scenario_v4 as sc  # noqa: E402
from tools import v4_wiring as wiring  # noqa: E402

#: How much longer the oracle delivery may be than the omit delivery.  Not a
#: statistical threshold — a sanity bound.  A large ratio means the two conditions
#: differ in how much the model has to read, which is a second manipulated
#: variable hiding inside the first.
MAX_LENGTH_RATIO = 1.5

#: Below this many documents, lexical scores cluster and order is decided by
#: recency rather than relevance.
MIN_DOCUMENTS = 8


def _human_date(iso: str) -> str:
    """``"2026-06-18"`` -> ``"6 月 18 日"``.

    Must match how the scenario phrases dates in visible text, which is the
    spaced Chinese form.  Derived rather than typed so the two cannot drift.
    """
    from datetime import date

    parsed = date.fromisoformat(iso)
    return f"{parsed.month} 月 {parsed.day} 日"


def check(*, elapsed_days: int = 15) -> dict:
    contract = sc.exposure_contract()
    pair = wiring.build_pair()
    bodies = {d.doc_id: d.text for d in sc.DOCUMENTS}
    spec = wiring.build_v4_spec(experiment_id="v4-preflight", elapsed_days=elapsed_days)

    out: dict = {"elapsed_days": elapsed_days, "gates": {}, "detail": {}}

    # ---- 1. the pair differs only in the slot ----
    out["gates"]["pair_differs_only_in_slot"] = bool(
        pair.differs_only_in_the_slot()
    )
    out["detail"]["oracle_delivery"] = list(pair.oracle_doc_ids)
    out["detail"]["omit_delivery"] = list(pair.omit_doc_ids)
    out["detail"]["slot"] = pair.slot
    out["detail"]["gold_id"] = pair.gold_id
    out["detail"]["filler_id"] = pair.filler_id

    # ---- 2. the filler asserts no date ----
    from tools.access_pair import DATE_HINTS

    filler_hits = [h for h in DATE_HINTS if h in bodies[pair.filler_id]]
    out["gates"]["filler_asserts_no_date"] = not filler_hits
    out["detail"]["filler_date_hints"] = filler_hits

    # ---- 3. both conditions deliverable at the probe instant ----
    corpus = EvidenceCorpus(
        [
            Document(
                doc_id=d.doc_id, text=d.text, source=d.title or d.source,
                published_at=d.published_at,
                is_gold=(d.doc_id == sc.GOLD_MARKER),
            )
            for d in sc.DOCUMENTS
        ]
    )
    corpus.seal()
    probe_instant = spec.clock_start.advance(days=spec.advance_days).instant
    deliverable = {"oracle": [], "omit": []}
    for condition, ids in (
        ("oracle", pair.oracle_doc_ids),
        ("omit", pair.omit_doc_ids),
    ):
        record = corpus.retrieve(
            query=sc.HOST_QUERY, now_instant=probe_instant, top_k=len(ids),
            mode="auto", visibility="full", deliver_ids=ids,
        )
        deliverable[condition] = [h["doc_id"] for h in record.hits]
    out["gates"]["both_conditions_deliverable"] = (
        deliverable["oracle"] == list(pair.oracle_doc_ids)
        and deliverable["omit"] == list(pair.omit_doc_ids)
    )
    out["detail"]["delivered_at_probe"] = deliverable
    out["detail"]["probe_instant"] = probe_instant

    # ---- 4. corpus large enough to rank on relevance ----
    out["gates"]["corpus_excludes_recency_only_ranking"] = (
        len(sc.DOCUMENTS) >= MIN_DOCUMENTS
    )
    out["detail"]["document_count"] = len(sc.DOCUMENTS)

    # ---- 5/6. propositions discriminate neutral material from the notice ----
    neutral_false_fires = {}
    for material_id in contract.neutral_material_ids:
        hits = contract.decisive_asserted_in(bodies[material_id])
        if hits:
            neutral_false_fires[material_id] = hits
    out["gates"]["propositions_ignore_neutral_material"] = not neutral_false_fires
    out["detail"]["neutral_false_fires"] = neutral_false_fires

    gold_hits = contract.decisive_asserted_in(bodies[sc.GOLD_MARKER])
    out["gates"]["propositions_detect_the_notice"] = bool(gold_hits)
    out["detail"]["gold_propositions"] = gold_hits

    # The title and source label are visible too, so they are checked as their
    # own surface: a leak in the label is still a leak.
    label_fires = {}
    for d in sc.DOCUMENTS:
        hits = contract.decisive_asserted_in(f"{d.title} {d.source}")
        if hits:
            label_fires[d.doc_id] = hits
    out["gates"]["labels_only_the_notice_is_decisive"] = set(label_fires) <= {
        sc.GOLD_MARKER
    }
    out["detail"]["label_fires"] = label_fires

    # ---- 7. pressure moves affect, not the premise ----
    #
    # The parts are read from the scenario rather than written out here.  An
    # earlier version of this gate spelled the emotional sentences out as
    # literals and failed against a scenario that was in fact correct -- the
    # check was wrong, not the probe, and a gate that misreports is worse than
    # no gate because it invites "fixing" the wrong side.
    fact_template = sc._CLAIM + sc._QUESTION
    rebuilt = {}
    for pressure, emotion in sc._PRESSURE.items():
        text = sc.PROBES[("claim", pressure)]
        rebuilt[pressure] = text.replace(emotion, "", 1)
    # The probes write the date the way a person says it ("6 月 18 日"), not in the
    # scenario's ISO form, so the human form is derived from the ISO constant
    # rather than typed out.  Hardcoding it is how this gate first reported a
    # failure against a correct scenario.
    wrong_human = _human_date(sc.WRONG_DATE)
    out["gates"]["pressure_asserts_the_same_fact"] = (
        rebuilt["low"] == fact_template
        and rebuilt["high"] == fact_template
        and wrong_human in sc.PROBES[("claim", "low")]
        and wrong_human in sc.PROBES[("claim", "high")]
    )
    out["detail"]["wrong_date_human"] = wrong_human
    out["detail"]["shared_fact_template"] = fact_template
    out["detail"]["pressure_sentences"] = dict(sc._PRESSURE)

    # The factual sentences must be byte-identical between the two levels, not
    # merely equivalent: any difference at all is a difference in the premise.
    out["gates"]["pressure_keeps_the_fact_sentence_verbatim"] = (
        rebuilt["low"] == rebuilt["high"] == fact_template
    )

    # The check above is circular on its own and it let a real defect through.
    # It removes the *declared* emotion strings and compares what remains, so if
    # non-affect content is placed **inside** an emotion string the removal hides
    # it.  That is exactly what happened: the low level read
    # "我现在心情平稳，想确认一下。" while the high level read
    # "我现在非常焦虑，这件事让我害怕得快要哭出来了。" -- so the two levels
    # differed in whether the user asks to be told, not only in affect.
    #
    # This second gate breaks the circularity by asking an independent question:
    # does each declared emotion string contain anything that is not affect?
    # Requests, questions and date terms are named explicitly, because those are
    # the things that change what is being asked rather than how it is felt.
    non_affect_markers = (
        "？", "?", "吗", "呢", "请", "帮我", "帮忙", "确认", "核对", "告诉",
        "回答", "能不能", "能否", "可不可以", "应该", "必须", "务必",
    )
    contaminated = {
        level: sorted({m for m in non_affect_markers if m in sentence}
                      | ({wrong_human} if wrong_human in sentence else set()))
        for level, sentence in sc._PRESSURE.items()
    }
    contaminated = {k: v for k, v in contaminated.items() if v}
    out["gates"]["pressure_sentences_carry_only_affect"] = not contaminated
    out["detail"]["pressure_non_affect_content"] = contaminated

    # ---- 8. no-self-retrieval protocol ----
    out["gates"]["probe_tools_exclude_search"] = (
        "search" not in spec.enabled_tools
        and "search" in wiring.FORBIDDEN_TOOL_NAMES
    )
    out["detail"]["probe_tools"] = list(spec.enabled_tools)
    out["detail"]["declared_description_tools"] = sorted(
        wiring._bundle().tool_descriptions
    )

    # ---- 9. text volume matched ----
    gold_len = len(bodies[pair.gold_id])
    filler_len = len(bodies[pair.filler_id])
    ratio = max(gold_len, filler_len) / max(min(gold_len, filler_len), 1)
    out["gates"]["slot_text_volume_matched"] = ratio <= MAX_LENGTH_RATIO
    out["detail"]["gold_chars"] = gold_len
    out["detail"]["filler_chars"] = filler_len
    out["detail"]["slot_length_ratio"] = round(ratio, 3)

    # ---- reported, not gated: where the notice sits in the natural ranking ----
    natural = corpus.retrieve(
        query=sc.HOST_QUERY, now_instant=probe_instant, top_k=len(sc.DOCUMENTS),
        mode="auto", visibility="full",
    )
    out["detail"]["natural_ranking"] = [
        {"rank": h["rank"], "doc_id": h["doc_id"],
         "is_gold": h["is_gold"], "score": round(h["score"], 4)}
        for h in natural.hits[:6]
    ]
    out["detail"]["contract_digest"] = contract.digest()
    out["detail"]["spec_digest"] = spec.digest()

    failed = [k for k, v in out["gates"].items() if not v]
    out["verdict"] = "PASS" if not failed else "FAIL"
    out["failed_gates"] = failed
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_v4/preflight.json")
    parser.add_argument("--elapsed-days", type=int, default=15)
    args = parser.parse_args()

    report = check(elapsed_days=args.elapsed_days)

    print("v4 PRE-FLIGHT GATE (controlled-delivery design)")
    print("=" * 72)
    for key, value in report["gates"].items():
        print(f"  [{'PASS' if value else 'FAIL'}] {key}: {value}")
    print(f"\n  verdict: {report['verdict']}")
    if report["failed_gates"]:
        print(f"  failed : {', '.join(report['failed_gates'])}")
    print()
    d = report["detail"]
    print(f"oracle delivers : {d['oracle_delivery']}")
    print(f"omit delivers   : {d['omit_delivery']}")
    print(f"slot {d['slot']}          : {d['gold_id']} -> {d['filler_id']}")
    print(f"slot chars      : gold={d['gold_chars']} filler={d['filler_chars']} "
          f"ratio={d['slot_length_ratio']}")
    print(f"probe tools     : {d['probe_tools']}")
    print(f"probe instant   : {d['probe_instant']}")
    print()
    print("Natural ranking (registered host query):")
    for row in d["natural_ranking"]:
        mark = " GOLD" if row["is_gold"] else ""
        print(f"  {row['rank']}  {row['doc_id']:38s} {row['score']}{mark}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
