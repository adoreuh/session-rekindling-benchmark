"""L1/L2 measurement and cross-run aggregation.

Two things happen here, and they are kept apart on purpose.

**L1 and L2 need no model.**  They read the recorded retrievals: the temporal
weight each candidate received (L1) and whether the gold document was reachable
(L2).  A result at these layers is a statement about the retrieval mechanism,
not about any model's behaviour, and it holds regardless of what the model
replied.

**L3 and L4 come from the judge.**  They are joined in by branch id.

The report is stratified rather than summarised, because the layers can and do
move in opposite directions: a decay profile can drive L1 down by orders of
magnitude while leaving L2 unchanged, since a uniform multiplicative factor
cancels out of a ranking.  Reporting a single "memory decayed" number would hide
that, and would license a behavioural claim the data does not support.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.facts import check as fact_check  # noqa: E402




def parse_branch(branch_id: str) -> dict:
    """``ask_high_oracle`` -> assertion / pressure / accessibility."""
    assertion = "neutral_query" if branch_id.startswith("ask") else "false_claim"
    rest = branch_id[len("ask_"):] if assertion == "neutral_query" else branch_id[len("claim_"):]
    pressure, _, accessibility = rest.partition("_")
    return {"assertion": assertion, "pressure": pressure,
            "accessibility": accessibility}


# Fields a report must carry for the analysis to be meaningful.  Checked before
# reading, because this analysis has twice read a derived field that looked like
# the quantity it wanted and was not: once reading the last retrieval when the
# design governs the host-driven one, and once reading a field that did not exist
# in that run's vintage and silently became a zero.  A missing field must raise,
# not default.
REQUIRED_BRANCH_FIELDS = ("branch_id", "label", "run_valid", "outcome")


def _check_report_vintage(report: dict, run_dir: Path) -> None:
    branches = report.get("branches") or []
    if not branches:
        raise ValueError(f"{run_dir}: report has no branches")
    for branch in branches:
        missing = [f for f in REQUIRED_BRANCH_FIELDS if f not in branch]
        if missing:
            raise ValueError(
                f"{run_dir}: branch {branch.get('branch_id')!r} is missing {missing}; "
                "refusing to analyse a report that predates the fields this code reads"
            )


def collect_run(run_dir: Path) -> list:
    """Extract one row per branch from a completed run."""
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _check_report_vintage(report, run_dir)

    judge = {}
    judge_path = run_dir / "judge.json"
    if judge_path.exists():
        payload = json.loads(judge_path.read_text(encoding="utf-8"))
        for row in payload.get("branches", []):
            judge[row["branch_id"]] = row

    retrieval_rows = []
    for branch in report["branches"]:
        branch_id = branch["branch_id"]
        parts = parse_branch(branch_id)
        score = branch.get("score") or {}
        l1 = score.get("L1") or {}
        l2_block = score.get("L2") or {}
        l2 = l2_block.get("retrievals") or []
        # Two different retrievals live in a branch and they must not be mixed:
        # the host-driven one, which the design governs, and any search the agent
        # chose to make, whose query is behaviour rather than a design variable.
        # The headline accessibility numbers come from the former.
        host_driven = l2_block.get("host_driven") or []
        agent_initiated = l2_block.get("agent_initiated") or []
        probe = host_driven[-1] if host_driven else (l2[-1] if l2 else {})
        weights = [r.get("gold_weight") for r in l1.get("retrievals", [])]
        weights = [w for w in weights if w is not None]
        judged = judge.get(branch_id, {})
        answer = (branch.get("outcome") or {}).get("assistant_text", "")
        fact = fact_check(answer) if answer else None
        retrieval_rows.append(
            {
                "run": run_dir.name,
                "model": report["_meta"]["model"] if "_meta" in report else run_dir.parent.name,
                "elapsed": report["_meta"]["elapsed"] if "_meta" in report else "",
                "elapsed_days": report["_meta"]["elapsed_days"] if "_meta" in report else 0,
                "decay": report["_meta"]["decay"] if "_meta" in report else "",
                "branch_id": branch_id,
                **parts,
                "valid": branch["run_valid"],
                "auto_outcome": (branch.get("auto_retrieval") or {}).get("outcome", ""),
                # L1
                "gold_temporal_weight": probe and _gold_weight(l1, probe),
                "l1_responds": l1.get("responds_to_elapsed_time"),
                # L2
                "result_count": probe.get("result_count"),
                "l2_query": probe.get("query", ""),
                # Counts only what happened during the branch's own turn; the
                # prefix runs its own loop and its retrievals are not the probe's.
                "agent_searches": len(agent_initiated),
                # Four separate levels.  `requested` is the model's willingness;
                # the other two say whether the attempt got through.  Reporting
                # only executions would make "never asked" and "asked and refused"
                # look the same.
                "search_requested": (branch.get("search_accounting") or {}).get(
                    "search_requested"),
                "search_executed": (branch.get("search_accounting") or {}).get(
                    "search_executed"),
                "search_evidence_delivered": (
                    branch.get("search_accounting") or {}
                ).get("search_evidence_delivered"),
                "agent_recovered_gold": l2_block.get("agent_recovered_gold"),
                "agent_searched_again": l2_block.get("agent_searched_again"),
                "gold_rank": probe.get("gold_rank"),
                "gold_returned": probe.get("gold_returned"),
                # L3/L4
                "judge_label": judged.get("label", ""),
                "is_target_failure": judged.get("is_target_failure"),
                "min_field_agreement": judged.get("min_field_agreement"),
                # Factual layer, computed in code rather than judged.  Reported
                # alongside the judge's own reading of the same field so that
                # agreement is measured instead of assumed.
                "fact_current_date_stated": fact.current_date_stated if fact else None,
                "fact_old_date_mentioned": fact.superseded_date_mentioned if fact else None,
                "fact_old_as_current": (
                    fact.superseded_asserted_as_current if fact else None
                ),
                "judge_said_current_date": (
                    judged.get("verdict", {}).get("current_date_stated", {}).get("value")
                    if judged else None
                ),
                "answer_chars": len(answer),
                "failure": ((branch.get("outcome") or {}).get("failure") or {}).get("reason", ""),
                # L4 exposure chain
                "l4_label": score.get("L4", {}).get("label", ""),
            }
        )
    return retrieval_rows


def _gold_weight(l1: dict, probe: dict):
    for row in l1.get("retrievals", []):
        if row.get("clock") == probe.get("clock"):
            return row.get("gold_weight")
    return None


def find_runs(root: Path) -> list:
    return sorted(p.parent for p in root.rglob("report.json"))


def write_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list) -> dict:
    """Stratified summary: rates per condition, per layer."""
    out: dict = {"by_model": {}, "by_layer": {}}

    models = sorted({r["model"] for r in rows})
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        by_cell = {}
        for row in subset:
            key = f"{row['assertion']}|{row['pressure']}|{row['accessibility']}"
            cell = by_cell.setdefault(
                key, {"n": 0, "gold_returned": 0, "target_failure": 0,
                      "judged": 0, "labels": {}}
            )
            cell["n"] += 1
            if row["gold_returned"]:
                cell["gold_returned"] += 1
            if row["is_target_failure"] is not None:
                cell["judged"] += 1
                if row["is_target_failure"]:
                    cell["target_failure"] += 1
            label = row["judge_label"] or "(none)"
            cell["labels"][label] = cell["labels"].get(label, 0) + 1
        out["by_model"][model] = by_cell

    # Adoption rate on the conflict probes, split by pressure and accessibility.
    adopt: dict = {}
    for row in rows:
        if row["assertion"] != "false_claim" or row["is_target_failure"] is None:
            continue
        pressure = row["pressure"]
        adopt.setdefault(pressure, {"n": 0, "fail": 0})
        adopt[pressure]["n"] += 1
        if row["is_target_failure"]:
            adopt[pressure]["fail"] += 1
    out["by_layer"]["L4_adoption_by_pressure"] = {
        k: {**v, "rate": round(v["fail"] / v["n"], 3) if v["n"] else None}
        for k, v in sorted(adopt.items())
    }

    # L1: does the recorded temporal weight track elapsed time?
    l1 = {}
    for row in rows:
        key = row["decay"] or "none"
        l1.setdefault(key, []).append(row["gold_temporal_weight"])
    out["by_layer"]["L1_gold_weight_by_decay"] = {
        k: {
            "n": len([w for w in v if w is not None]),
            "min": min([w for w in v if w is not None], default=None),
            "max": max([w for w in v if w is not None], default=None),
        }
        for k, v in sorted(l1.items())
    }

    # The key contrast: L1 falls while L2 does not.
    reach = {}
    for row in rows:
        key = row["decay"] or "none"
        reach.setdefault(key, {"n": 0, "returned": 0})
        if row["gold_returned"] is not None:
            reach[key]["n"] += 1
            if row["gold_returned"]:
                reach[key]["returned"] += 1
    out["by_layer"]["L2_gold_reachability_by_decay"] = {
        k: {
            **v,
            "rate": round(v["returned"] / v["n"], 3) if v["n"] else None,
        }
        for k, v in sorted(reach.items())
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        for run_dir in find_runs(Path(root)):
            rows.extend(collect_run(run_dir))

    if not rows:
        print("no completed runs found", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    write_csv(rows, out_root / "branches_long.csv")
    summary = summarise(rows)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"rows: {len(rows)}")
    print(f"models: {sorted({r['model'] for r in rows})}")
    print(f"wrote {out_root / 'branches_long.csv'}")
    print(f"wrote {out_root / 'summary.json'}")
    print()
    print("=== L4 adoption on conflict probes, by pressure ===")
    for key, value in summary["by_layer"]["L4_adoption_by_pressure"].items():
        print(f"  pressure={key:5s} n={value['n']:3d} rate={value['rate']}")
    print()
    print("=== L1 gold temporal weight by decay ===")
    for key, value in summary["by_layer"]["L1_gold_weight_by_decay"].items():
        print(f"  {key:24s} n={value['n']:3d} min={value['min']} max={value['max']}")
    print()
    print("=== L2 gold reachability by decay ===")
    for key, value in summary["by_layer"]["L2_gold_reachability_by_decay"].items():
        print(f"  {key:24s} n={value['n']:3d} rate={value['rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
