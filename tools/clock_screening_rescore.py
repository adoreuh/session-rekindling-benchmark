"""Re-derive the clock-source screening labels from saved probe results.

Why offline re-derivation rather than a re-run.  The screening tool gained a
*consistency* measure after the first five routes were already screened: the
earlier version reported time precision alone, and precision turned out to be
misleading on its own.  A route answering ``2024-06-14 15:47`` looked "concrete"
while giving three mutually different dates across three probes — invention, not a
clock.  Re-running would replace the observations instead of correcting the
reading, so the labels are recomputed from the answers already on disk and both
versions are kept side by side.

This is the same discipline the main experiment uses: the log is the source of
truth, and a change in interpretation must not become a chance to resample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.clock_source_probe import summarise  # noqa: E402

#: The reading each consistency outcome supports, for the temporal question.
IMPLICATION = {
    "consistent_and_real": (
        "carries a runtime date signal equal to the real date: its elapsed-time "
        "results are uninterpretable, because a second clock is in the conversation"
    ),
    "consistent_but_wrong": (
        "repeats one wrong date: it is guessing from priors, not reading a clock, "
        "so it is a usable subject for a temporal study"
    ),
    "inconsistent": (
        "states several different wrong dates: invention with no source, so it is a "
        "usable subject, though its date statements carry no information"
    ),
    "no_dates_stated": (
        "declines to state a date: usable, and it gives the temporal manipulation "
        "nothing to conflict with"
    ),
}


def rescore(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for model, entry in data.items():
        results = entry.get("results") if isinstance(entry, dict) else None
        if results is None:
            rows.append({"model": model, "error": "no probe results in file"})
            continue
        recomputed = summarise(model, results)
        consistency = recomputed.get("date_consistency") or {}
        label = consistency.get("label", "")
        rows.append({
            "model": model,
            "recorded": entry.get("summary"),
            "recomputed": recomputed,
            "consistency": label,
            "dates_seen": consistency.get("dates", []),
            "states_the_real_date": recomputed.get("states_the_real_date"),
            "usable_for_temporal": not recomputed.get("states_the_real_date"),
            "implication": (
                IMPLICATION["consistent_and_real"]
                if recomputed.get("states_the_real_date")
                else IMPLICATION.get(label, "")
            ),
        })
    return {"source": str(path), "routes": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--out", default="results_v4/clock_screening_final.json")
    args = parser.parse_args()

    everything = {}
    for target in args.sources:
        path = Path(target)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        result = rescore(path)
        everything[path.name] = result
        for row in result["routes"]:
            if "error" in row:
                print(f"  {row['model']:26s} ERROR {row['error']}")
                continue
            flag = "USABLE  " if row["usable_for_temporal"] else "EXCLUDE "
            print(f"  {flag} {row['model']:26s} consistency={row['consistency']:20s} "
                  f"dates_seen={row['dates_seen']}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(everything, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path}")

    # A single consolidated view of which routes may be temporal subjects.
    print("\nTEMPORAL SUBJECTS")
    for name, result in everything.items():
        for row in result["routes"]:
            if "error" in row:
                continue
            verdict = "yes" if row["usable_for_temporal"] else "NO"
            print(f"  {verdict:4s} {row['model']:26s} {row['implication']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
