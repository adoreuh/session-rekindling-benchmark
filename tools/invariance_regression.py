"""Invariance regression: the ranking must NOT change with elapsed time.

Why this is a regression test rather than an experiment.

The design originally treated elapsed time as a manipulation: advance the clock
1d / 15d / 61d and see whether the accessible evidence changes.  It cannot.  For
multiplicative scoring with one decay rate per item and a fixed candidate set,
advancing time by ``Δ`` multiplies **every** score by the same positive factor::

    S_i(t + Δ) = L_i · e^{-(t + Δ - s_i)/T} = e^{-Δ/T} · S_i(t)

So the ranking and the top-k set are exactly invariant.  That is an identity, not
an empirical regularity, which means the correct test asserts invariance and
**fails when the ranking moves**.  A test that asserted "the ranking changes"
would be asserting arithmetic is broken.

The consequence for the study: a paid run across time points would spend tokens to
observe the identity.  It was cancelled and replaced by this check.  Time is
retained as an implementation contract and a stated research boundary, not as a
manipulation.

Two things this guards against, both of which look like a finding and are not:

* **Float underflow.** At extreme durations every score underflows to zero, so the
  ranking collapses to the tie-break rule.  That looks like "time destroyed
  accessibility" and is pure numerics.  Checked in the log domain.
* **A silently changed retriever.** If someone later edits the scoring so the
  ranking does respond to time, this test fails, and the failure is the intended
  signal: the invariance conditions no longer hold, so the research framing has to
  be restated rather than the test relaxed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.decay import PROFILES, resolve_profile  # noqa: E402
from vela_lab.evidence import Document, EvidenceCorpus  # noqa: E402

# A corpus with deliberately mixed ages, so a time-varying cause would have room to
# show.  If all items shared one age, invariance would be trivially true and the
# test would be vacuous.  Built in _build_documents below.
SOURCE_DAY = "2026-03-01T09:00:00+08:00"
SPREAD_DAYS = (0, 2, 5, 9, 20, 40)

# Rebuild with proper per-document timestamps (clearer than the comprehension above).
def _build_documents() -> tuple:
    from datetime import datetime, timedelta

    base = datetime.fromisoformat("2026-03-01T09:00:00+08:00")
    docs = []
    for d in SPREAD_DAYS:
        # doc_00 is the newest (published on the base day); later indices are older
        published = base - timedelta(days=d)
        docs.append(
            Document(
                doc_id=f"doc_{d:02d}",
                text=("预约变更通知：当前有效安排已确认。"
                      if d == 0 else
                      f"历史记录（{d} 天前）：预约安排的确认与提醒内容。"),
                authority="high" if d == 0 else "low",
                source="registry" if d == 0 else "notes",
                published_at=published.isoformat(),
                stance="supports_current" if d == 0 else "neutral",
                is_gold=(d == 0),
            )
        )
    return tuple(docs)


DOCUMENTS = _build_documents()
QUERY = "预约安排 变更通知"
TOP_K = 3

# Durations to sweep.  The extremes exist for the underflow check, not to find an
# effect; a day count that overflows the exponent is the interesting case.
DURATIONS = (0, 1, 15, 61, 365, 3650, 36500, 365000)


def _ranked_ids(corpus: EvidenceCorpus, *, now: str, decay, top_k: int) -> tuple:
    record = corpus.retrieve(
        query=QUERY, now_instant=now, top_k=top_k, mode="probe",
        temporal_decay=decay,
    )
    return tuple(h["doc_id"] for h in record.hits)


def _log_scores(corpus: EvidenceCorpus, *, now: str, decay) -> list:
    """Scores in the log domain, computed analytically.

    Comparing raw scores across durations is what produces the underflow illusion:
    at 365 000 days every score is 0.0 and the ranking looks destroyed.

    The subtle part is that taking ``log()`` of the *already-underflowed* float
    does not recover anything — ``exp(-a/T)`` is exactly 0.0 in float64 long before
    ``a/T`` stops being meaningful, so ``log(0.0)`` is undefined and the ordering
    information really is gone from the float.

    So the log weight is computed from its exponent instead::

        log f(a) = -a / T

    which stays finite for any duration.  If the ordering is still recoverable
    this way, then the raw-zero ranking is a numerics artifact, not a memory
    effect — which is exactly the distinction that has to be made before anyone
    reads "all scores are zero" as forgetting.
    """
    from vela_lab.clock import parse_instant

    rows = []
    now_dt = parse_instant(now)
    for doc in corpus.all_documents().values():
        lexical = float(corpus.retriever.score(QUERY, doc))
        log_lex = math.log(lexical) if lexical > 0 else None
        log_w = _log_weight(decay, observed_at=doc.published_at, now_dt=now_dt)
        total = (log_lex + log_w) if (log_lex is not None and log_w is not None) else None
        rows.append((doc.doc_id, log_lex, log_w, total))
    return rows


def _log_weight(decay, *, observed_at: str, now_dt) -> float | None:
    """``log`` of the decay weight, from the exponent rather than the float.

    Each shipped profile is exponential in the item's age, so the log weight is a
    linear function of age.  Computing it this way keeps it finite across the whole
    duration sweep instead of collapsing to ``-inf``.
    """
    from vela_lab.clock import parse_instant

    age_days = (now_dt - parse_instant(observed_at)).total_seconds() / 86400.0
    name = getattr(decay, "name", "")
    params = getattr(decay, "params", {}) or {}
    if name == "m00_none":
        return 0.0
    if name in ("m01_memorybank", "m01_memorybank_repo"):
        base = float(params.get("base_days", 5.0))
        return -age_days / base
    if name == "m02_generative_agents":
        per_hour = float(params.get("factor_per_hour", 0.995))
        return math.log(per_hour) * age_days * 24.0
    return None


def _instant_at(base_iso: str, days: int) -> str:
    from datetime import datetime, timedelta

    return (datetime.fromisoformat(base_iso) + timedelta(days=days)).isoformat()


def check(*, durations=DURATIONS, decays=None) -> dict:
    corpus = EvidenceCorpus(list(DOCUMENTS))
    corpus.seal()
    decays = decays or sorted(PROFILES)

    results = []
    for decay_name in decays:
        decay = resolve_profile(decay_name)
        base_rank = None
        entries = []
        for days in durations:
            now = _instant_at(SOURCE_DAY, days)
            ranked = _ranked_ids(corpus, now=now, decay=decay, top_k=TOP_K)

            # Gold reachability is the property the study cares about, so it is
            # checked directly rather than inferred from the ranking.
            record = corpus.retrieve(
                query=QUERY, now_instant=now, top_k=TOP_K, mode="probe",
                temporal_decay=decay,
            )
            gold_rank = record.gold_rank()

            if base_rank is None:
                base_rank = ranked
            log_rows = _log_scores(corpus, now=now, decay=decay)
            finite = [r[3] for r in log_rows if r[3] is not None]

            entries.append(
                {
                    "days": days,
                    "ranking": list(ranked),
                    "gold_rank": gold_rank,
                    "ranking_matches_baseline": ranked == base_rank,
                    # Underflow check: are all raw scores zero while the log-domain
                    # ordering still exists?
                    "all_raw_scores_zero": _all_raw_zero(corpus, now=now, decay=decay),
                    "log_domain_ordering_present": len(finite) == len(log_rows),
                }
            )
        results.append(
            {
                "decay": decay_name,
                "entries": entries,
                "ranking_invariant": all(e["ranking_matches_baseline"] for e in entries),
                "gold_rank_invariant": len({e["gold_rank"] for e in entries}) == 1,
                "underflow_observed": any(e["all_raw_scores_zero"] for e in entries),
                "underflow_kept_ordering": all(
                    e["log_domain_ordering_present"] for e in entries
                ),
            }
        )

    passed = all(
        r["ranking_invariant"] and r["gold_rank_invariant"] for r in results
    )
    return {
        "query": QUERY,
        "top_k": TOP_K,
        "documents": [d.doc_id for d in DOCUMENTS],
        "durations_days": list(durations),
        "invariance_holds": passed,
        "by_decay": results,
    }


def _all_raw_zero(corpus: EvidenceCorpus, *, now: str, decay) -> bool:
    scores = []
    for doc in corpus.all_documents().values():
        lexical = float(corpus.retriever.score(QUERY, doc))
        weight = float(decay(observed_at=doc.published_at, now=now))
        scores.append(lexical * weight)
    return bool(scores) and all(s == 0.0 for s in scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_v4/invariance.json")
    args = parser.parse_args()

    report = check()
    print("INVARIANCE REGRESSION — ranking must not change with elapsed time")
    print("=" * 74)
    for row in report["by_decay"]:
        mark = "PASS" if row["ranking_invariant"] and row["gold_rank_invariant"] else "FAIL"
        print(f"  [{mark}] {row['decay']:24s} "
              f"ranking_invariant={row['ranking_invariant']} "
              f"gold_rank_invariant={row['gold_rank_invariant']}"
              + (f"  underflow_seen=True ordering_kept={row['underflow_kept_ordering']}"
                 if row["underflow_observed"] else ""))
    print()
    print(f"  overall: {'PASS' if report['invariance_holds'] else 'FAIL'}")
    print()
    print("Rankings by duration (m01_memorybank):")
    for row in report["by_decay"]:
        if row["decay"] != "m01_memorybank":
            continue
        for e in row["entries"]:
            print(f"    {e['days']:>7d}d gold_rank={str(e['gold_rank']):>4s} "
                  f"{'|'.join(e['ranking'])}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
    print(f"\nwrote {path}")
    return 0 if report["invariance_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
