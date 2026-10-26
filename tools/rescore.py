"""Re-derive scores for a completed run from its recorded events.

Why this exists rather than "just re-run it".  The event log is the source of
truth precisely so that analysis can be redone without re-asking the model.  A
paid branch is a sample; re-running to get a different *label* for the same
behaviour replaces the observation instead of correcting the interpretation, and
a scorer change would silently become a chance to resample.

So: scores are recomputed from the events already on disk, the previous verdicts
are preserved alongside the new ones, and the difference is reported rather than
applied quietly.  Nothing about the model's behaviour is altered — if a label
changes here, the old label was wrong.

This is the mechanism the L4 correction needed.  Its scorer tested for the gold
*document id* in the answer, which no natural answer contains, so every usable
answer was labelled ``delivered_not_used``.  Re-deriving is the only way to fix
that without invalidating the runs that had already happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.score import score_run  # noqa: E402


def _load_events(branch_dir: Path) -> list:
    path = branch_dir / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def rescore_run(run_dir: Path, *, gold_marker: str,
                decisive_terms) -> dict:
    """Recompute L1-L4 for every branch in ``run_dir``.

    Returns a report that keeps the recorded label next to the recomputed one.
    A silent replacement would make it impossible to see that a scorer changed,
    which is exactly the kind of drift the event log is supposed to prevent.
    """
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    changes = []
    for branch in report["branches"]:
        branch_dir = run_dir / "branches" / branch["branch_id"]
        events = _load_events(branch_dir)
        if not events:
            changes.append({
                "branch_id": branch["branch_id"],
                "error": "no events recorded",
            })
            continue
        recomputed = score_run(
            events,
            gold_marker=gold_marker,
            answer_text=branch["outcome"]["assistant_text"],
            decisive_terms=decisive_terms,
        )
        recorded_l4 = (branch.get("score") or {}).get("L4", {}).get("label", "")
        new_l4 = recomputed["L4"]["label"]
        entry = {
            "branch_id": branch["branch_id"],
            "recorded_l4": recorded_l4,
            "recomputed_l4": new_l4,
            "changed": recorded_l4 != new_l4,
            "rescore": recomputed,
        }
        changes.append(entry)
        # Recorded, not overwritten: the earlier verdict stays readable.
        branch.setdefault("score_recorded_before_rescore", branch.get("score"))
        branch["score"] = recomputed

    return {
        "run_dir": str(run_dir),
        "gold_marker": gold_marker,
        "decisive_terms": list(decisive_terms),
        "branches": changes,
        "changed_count": sum(1 for c in changes if c.get("changed")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories to re-score")
    parser.add_argument("--gold-marker", required=True)
    parser.add_argument("--decisive-date", default="",
                        help="ISO date the gold evidence asserts; its rendering "
                             "variants become the 'answer used the evidence' signal")
    parser.add_argument("--write", action="store_true",
                        help="write rescored.json into each run directory")
    args = parser.parse_args()

    from vela_lab.score import date_forms

    terms = date_forms(args.decisive_date) if args.decisive_date else ()
    if not terms:
        print("warning: no --decisive-date given, so the L4 used/not-used split "
              "falls back to the document id and will read as 'not used'",
              file=sys.stderr)

    exit_code = 0
    for path in args.runs:
        run_dir = Path(path)
        result = rescore_run(
            run_dir, gold_marker=args.gold_marker, decisive_terms=terms
        )
        print(f"=== {run_dir} ===")
        for entry in result["branches"]:
            if "error" in entry:
                print(f"  ERROR {entry['branch_id']}: {entry['error']}")
                exit_code = 1
                continue
            flag = "CHANGED" if entry["changed"] else "same   "
            print(f"  {flag} {entry['branch_id']:22s} "
                  f"{entry['recorded_l4'] or '-':22s} -> {entry['recomputed_l4']}")
        print(f"  {result['changed_count']} of {len(result['branches'])} changed")
        if args.write:
            out = run_dir / "rescore.json"
            out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"  wrote {out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
