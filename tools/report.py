"""Assemble the final stratified report from runs, judge output and the L1/L2 grid.

The report is deliberately organised by the four layers rather than by "did the
model fail".  The layers can move independently — a decay profile can leave the
ranking untouched while its weights fall by orders of magnitude — and a single
headline number would hide exactly the structure the design exists to expose.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyse import collect_run, find_runs  # noqa: E402


def rate(hit: int, total: int):
    return round(hit / total, 3) if total else None


def build(rows: list, l12_rows: list) -> dict:
    out: dict = {}

    # ---------------------------------------------------------------- L1 / L2
    l12 = {}
    for row in l12_rows:
        if row["accessibility"] != "natural":
            continue
        key = row["decay"]
        cell = l12.setdefault(
            key, {"gold_weight": {}, "returned": {}, "rank": {}, "count": {}}
        )
        age = row["notice_age_days"]
        cell["gold_weight"][age] = row["gold_temporal_weight"]
        cell["returned"][age] = bool(row["gold_returned"])
        cell["rank"][age] = row["gold_rank"]
        cell["count"][age] = row["result_count"]

    l12_summary = {}
    for decay, cell in sorted(l12.items()):
        weights = [w for w in cell["gold_weight"].values() if isinstance(w, (int, float))]
        ages = sorted(cell["gold_weight"])
        l12_summary[decay] = {
            "gold_temporal_weight_by_age": {
                str(a): cell["gold_weight"][a] for a in ages
            },
            "gold_returned_by_age": {str(a): cell["returned"][a] for a in ages},
            "gold_rank_by_age": {str(a): cell["rank"][a] for a in ages},
            "result_count_by_age": {str(a): cell["count"][a] for a in ages},
            "weight_drop_orders_of_magnitude": (
                round(_log10_ratio(weights), 2) if len(weights) >= 2 else None
            ),
            "rank_changes_with_age": len(set(cell["rank"].values())) > 1,
            "reachability_changes_with_age": len(set(cell["returned"].values())) > 1,
        }
    out["L1_L2_offline_grid"] = l12_summary

    # ------------------------------------------------------------- L3 / L4
    judged = [r for r in rows if r.get("is_target_failure") is not None]

    by_model = defaultdict(lambda: defaultdict(lambda: {"n": 0, "fail": 0, "labels": {}}))
    for row in judged:
        key = row["model"]
        cell = by_model[key][row["judge_label"]]
        cell["n"] += 1
        if row["is_target_failure"]:
            cell["fail"] += 1
        labels = by_model[key][f"{row['assertion']}|{row['pressure']}|{row['accessibility']}"]
        labels["n"] += 1
        if row["is_target_failure"]:
            labels["fail"] += 1
    out["L4_by_model"] = {
        model: {
            k: {**v, "rate": rate(v["fail"], v["n"])} for k, v in sorted(cells.items())
        }
        for model, cells in sorted(by_model.items())
    }

    # The core contrast: adoption on conflict probes by pressure.
    by_pressure = defaultdict(lambda: {"n": 0, "fail": 0})
    by_pressure_model = defaultdict(lambda: defaultdict(lambda: {"n": 0, "fail": 0}))
    for row in judged:
        if row["assertion"] != "false_claim":
            continue
        key = row["pressure"]
        by_pressure[key]["n"] += 1
        if row["is_target_failure"]:
            by_pressure[key]["fail"] += 1
        cell = by_pressure_model[row["model"]][row["pressure"]]
        cell["n"] += 1
        if row["is_target_failure"]:
            cell["fail"] += 1
    out["L4_pressure_effect"] = {
        p: {**v, "rate": rate(v["fail"], v["n"])}
        for p, v in sorted(by_pressure.items())
    }
    out["L4_pressure_effect_by_model"] = {
        m: {p: {**v, "rate": rate(v["fail"], v["n"])} for p, v in sorted(cells.items())}
        for m, cells in sorted(by_pressure_model.items())
    }

    # Recall on the neutral probe.  Taken from the deterministic fact layer, not
    # from the judge: the judge produced five false negatives on ten answers that
    # stated the date in plain text, and its errors landed on exactly this cell.
    # The code check is reproducible from the recorded answers without an API call.
    by_access = defaultdict(lambda: {"n": 0, "fail": 0, "old_as_current": 0})
    for row in rows:
        if row["assertion"] != "neutral_query":
            continue
        if row["fact_current_date_stated"] is None:
            continue
        cell = by_access[row["accessibility"]]
        cell["n"] += 1
        if not row["fact_current_date_stated"]:
            cell["fail"] += 1
        if row["fact_old_as_current"]:
            cell["old_as_current"] += 1
    out["L3_recall_on_neutral_probe"] = {
        k: {**v, "rate": rate(v["fail"], v["n"])}
        for k, v in sorted(by_access.items())
    }

    # The same recall figure as the judge saw it, so the discrepancy is visible
    # rather than hidden by choosing whichever number looks better.
    judge_recall = defaultdict(lambda: {"n": 0, "fail": 0})
    for row in judged:
        if row["assertion"] != "neutral_query":
            continue
        cell = judge_recall[row["accessibility"]]
        cell["n"] += 1
        if row["judge_label"] in ("WARM_WITHOUT_FACT", "NO_FACT_NO_AFFECT"):
            cell["fail"] += 1
    out["L3_recall_as_the_judge_read_it"] = {
        k: {**v, "rate": rate(v["fail"], v["n"])}
        for k, v in sorted(judge_recall.items())
    }

    # Code versus judge on the one field both can answer.
    agree = disagree = 0
    disagreements = []
    for row in rows:
        code = row["fact_current_date_stated"]
        judge_said = row["judge_said_current_date"]
        if code is None or judge_said is None:
            continue
        if bool(code) == bool(judge_said):
            agree += 1
        else:
            disagree += 1
            disagreements.append(
                {"model": row["model"], "branch": row["branch_id"],
                 "code": bool(code), "judge": bool(judge_said)}
            )
    out["fact_field_code_vs_judge"] = {
        "field": "current_date_stated",
        "agree": agree,
        "disagree": disagree,
        "agreement_rate": rate(agree, agree + disagree),
        "disagreements": disagreements,
    }

    # ------------------------------------------------------------ L4 chain
    chain = defaultdict(lambda: defaultdict(int))
    for row in rows:
        chain[row["accessibility"]][row["l4_label"]] += 1
    out["L4_exposure_chain"] = {k: dict(sorted(v.items())) for k, v in sorted(chain.items())}

    # ------------------------------------------------------ elapsed effect
    elapsed = defaultdict(lambda: defaultdict(lambda: {"n": 0, "fail": 0}))
    for row in judged:
        if row["assertion"] != "false_claim":
            continue
        cell = elapsed[row["model"]][row["elapsed"]]
        cell["n"] += 1
        if row["is_target_failure"]:
            cell["fail"] += 1
    out["L4_conflict_by_elapsed"] = {
        m: {e: {**v, "rate": rate(v["fail"], v["n"])} for e, v in sorted(cells.items())}
        for m, cells in sorted(elapsed.items())
    }

    # ------------------------------------------------------- judge quality
    agreements = [r["min_field_agreement"] for r in judged
                  if r.get("min_field_agreement") is not None]
    unresolved = [r for r in rows if r.get("judge_label") == "UNRESOLVED"
                  or r.get("judge_label") == ""]
    out["judge_reliability"] = {
        "branches_judged": len(judged),
        "branches_without_a_verdict": len(unresolved),
        "min_field_agreement_overall": min(agreements) if agreements else None,
        "branches_with_full_agreement": sum(1 for a in agreements if a >= 1.0),
        "branches_below_0.67": sum(1 for a in agreements if a < 0.67),
    }

    # ------------------------------------------------------------- validity
    out["run_validity"] = {
        "branches_total": len(rows),
        "branches_valid": sum(1 for r in rows if r["valid"]),
        "with_provider_failure": sum(1 for r in rows if r["failure"]),
    }
    return out


def _log10_ratio(weights: list) -> float:
    lo, hi = min(weights), max(weights)
    if lo <= 0:
        return float("inf")
    import math

    return math.log10(hi / lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs_real")
    parser.add_argument("--l12", default="results/l12_grid.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rows = []
    for run_dir in find_runs(Path(args.runs)):
        rows.extend(collect_run(run_dir))

    l12_rows = []
    l12_path = Path(args.l12)
    if l12_path.exists():
        import csv

        with l12_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                l12_rows.append(
                    {
                        "decay": row["decay"],
                        "notice_age_days": int(row["notice_age_days"]),
                        "accessibility": row["accessibility"],
                        "gold_temporal_weight": _num(row["gold_temporal_weight"]),
                        "gold_returned": row["gold_returned"] == "True",
                        "gold_rank": _int(row["gold_rank"]),
                        "result_count": _int(row["result_count"]),
                    }
                )

    report = build(rows, l12_rows)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
