"""L1/L2 across the full elapsed-time x decay grid, with no model calls.

L1 (temporal sensitivity) and L2 (accessibility) are properties of the retrieval
mechanism.  A model call cannot change either of them: retrieval is host-driven
and happens before the request is compiled.  So the grid can be measured with the
deterministic mock provider, which costs nothing, runs offline, and makes the
measurement exactly reproducible.

This is the layer where the project's earlier version found its most transferable
result: a multiplicative decay applied uniformly to every candidate cancels out of
the ranking, so the temporal weight can fall by orders of magnitude while the
returned documents are identical.  Whether that still holds for this scenario is
what this script measures.

What it does *not* measure: whether any model behaves differently.  L3 and L4 need
real models and are handled elsewhere.  Running this grid and reporting it as if
it were a behavioural result would be the exact conflation the contract forbids.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.decay import PROFILES, resolve_profile  # noqa: E402
from vela_lab.provider import MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools import scenario_ewm  # noqa: E402


def run_grid(*, out_root: Path, elapsed_days: list, decay_names: list) -> list:
    rows = []
    for decay_name in decay_names:
        decay = resolve_profile(decay_name)
        for days in elapsed_days:
            run_dir = out_root / f"{days}d_{decay_name}"
            if (run_dir / "report.json").exists():
                report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            else:
                spec = scenario_ewm.build_spec(
                    experiment_id=f"L12-{days}d-{decay_name}",
                    elapsed=f"{days}d", decay=decay, elapsed_days=days,
                )
                report = ExperimentRunner(
                    spec, MockModel(model_name="l12-offline"),
                    out_dir=run_dir,
                ).run()

            for branch in report["branches"]:
                parts = scenario_ewm.all_conditions()
                info = next(
                    (c for c in parts if c.branch_id == branch["branch_id"]), None
                )
                score = branch.get("score") or {}
                l1 = (score.get("L1") or {}).get("retrievals") or []
                l2 = (score.get("L2") or {}).get("retrievals") or []
                probe_l1 = l1[-1] if l1 else {}
                # Report the host-driven retrieval, which is what the design
                # governs.  The last retrieval may be the agent's own search,
                # whose query is behaviour rather than a design variable.
                l2_host = (score.get("L2") or {}).get("host_driven") or []
                probe_l2 = (l2_host[-1] if l2_host else (l2[-1] if l2 else {}))
                agent_searches = len(
                    (score.get("L2") or {}).get("agent_initiated") or []
                )
                # All candidates in the probe retrieval, to show the ordering.
                rows.append(
                    {
                        "decay": decay_name,
                        "elapsed_days": days,
                        "notice_age_days": days - 4,
                        "branch_id": branch["branch_id"],
                        "assertion": info.assertion if info else "",
                        "accessibility": info.accessibility if info else "",
                        "valid": branch["run_valid"],
                        # L1
                        "gold_temporal_weight": probe_l1.get("gold_weight"),
                        "min_weight": probe_l1.get("min_weight"),
                        "max_weight": probe_l1.get("max_weight"),
                        # L2
                        "gold_returned": probe_l2.get("gold_returned"),
                        "gold_rank": probe_l2.get("gold_rank"),
                        "returned": "|".join(probe_l2.get("returned") or []),
                        "result_count": probe_l2.get("result_count"),
                        "agent_searches": agent_searches,
                        "agent_recovered_gold": (
                            score.get("L2") or {}
                        ).get("agent_recovered_gold"),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_l12")
    parser.add_argument("--results", default="results")
    parser.add_argument("--days", nargs="+", type=int, default=[5, 19, 65])
    parser.add_argument("--decay", nargs="+", default=sorted(PROFILES))
    args = parser.parse_args()

    rows = run_grid(out_root=Path(args.out), elapsed_days=args.days,
                    decay_names=args.decay)
    out_csv = Path(args.results) / "l12_grid.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows: {len(rows)}  -> {out_csv}\n")
    header = f"{'decay':22s} {'age':>4s} {'gold_w':>10s} {'returned':>5s} {'rank':>5s}  order"
    print(header)
    print("-" * len(header))
    seen = set()
    for row in rows:
        key = (row["decay"], row["elapsed_days"], row["accessibility"])
        if key in seen:
            continue
        seen.add(key)
        if row["accessibility"] != "natural":
            continue
        weight = row["gold_temporal_weight"]
        weight_text = f"{weight:.6g}" if isinstance(weight, (int, float)) else "n/a"
        print(
            f"{row['decay']:22s} {row['notice_age_days']:>3d}d "
            f"{weight_text:>10s} {str(row['gold_returned']):>5s} "
            f"{str(row['gold_rank']):>5s}  {row['returned']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
