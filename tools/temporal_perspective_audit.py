"""Read-only audit of frozen VELA artifacts for the temporal-ability review.

Run from any directory. This script performs no model or judge API calls and
does not change existing artifacts. Counts are recomputed after branch.forked.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
OUT = HARN / "results_temporal_audit_20261002"
ROOTS = {
    "hy3": "runs_expiry_hy3",
    "Qwen3.5-397B-A17B": "runs_expiry_Qwen3_5-397B-A17B",
    "DeepSeek-V4-Flash": "runs_expiry_DeepSeek-V4-Flash",
    "DeepSeek-V4.1-Flash": "runs_expiry_DeepSeek-V4_1-Flash",
    "qwen3-vl-plus": "runs_expiry_qwen3-vl-plus",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def events(cell):
    path = cell / "branches" / "probe" / "events.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def branch_events(cell):
    es = events(cell)
    if es is None:
        return None
    forks = [i for i, e in enumerate(es) if e["type"] == "branch.forked"]
    if not forks:
        raise ValueError(f"Missing fork boundary: {cell}")
    return es[forks[-1] + 1:]


def fetch_count(es):
    if es is None:
        return None
    return sum(e["type"] == "tool.call" and e.get("data", {}).get("tool_name") == "world_time"
               and not e.get("data", {}).get("injected") for e in es)


def verdict(cell):
    p = cell / "judge.expiry.json"
    if not p.exists():
        return {}
    vs = read(p).get("branches", {}).get("probe", {}).get("verdict", {})
    return {k: v.get("value") for k, v in vs.items() if isinstance(v, dict)}


def answer(cell):
    es = branch_events(cell) or []
    texts = [e.get("data", {}).get("text", "") for e in es if e["type"] == "assistant.message"]
    return texts[-1] if texts else ""


def cell_for(row, vis):
    if vis in ("natural", "delivered"):
        return HARN / ROOTS[row["route"]] / "expiry" / row["route"] / f'{row["profile"]}_{row["arm"]}_{vis}'
    root = "runs_e1_forced/e1" if vis == "forced" else "runs_e2_relation/e2"
    return HARN / root / row["route"] / f'{row["profile"]}_{row["arm"]}_{vis}'


def bool_flip(counts):
    return len({n > 0 for n in counts if n is not None}) > 1


def main():
    OUT.mkdir(exist_ok=True)
    summary = {"scope": "local frozen artifacts only; no API calls"}
    e2 = read(HARN / "results_v4/e2_analysis.json")
    complete = [r for r in e2["rows"] if all(isinstance(r[v], bool)
                for v in ("natural", "delivered", "forced", "relation"))]
    summary["four_condition_common_sample"] = {
        "n": len(complete),
        "by_condition": {v: {"stale": sum(r[v] for r in complete), "n": len(complete)}
                         for v in ("natural", "delivered", "forced", "relation")},
        "by_route": {route: {v: {"stale": sum(r[v] for r in complete if r["route"] == route),
                                  "n": sum(r["route"] == route for r in complete)}
                            for v in ("natural", "delivered", "forced", "relation")}
                     for route in sorted({r["route"] for r in complete})},
    }
    e1 = read(HARN / "results_v4/e1_analysis.json")
    gains = []
    for r in e1["rows"]:
        if (r.get("natural") or {}).get("stale") is True and (r.get("forced") or {}).get("stale") is False:
            natural_cell = cell_for(r, "natural")
            gains.append({"route": r["route"], "profile": r["profile"], "arm": r["arm"],
                          "probe_fetches": fetch_count(branch_events(natural_cell))})
    summary["e1_improved_cells"] = {
        "n": len(gains), "natural_already_fetched": sum((r["probe_fetches"] or 0) > 0 for r in gains),
        "natural_did_not_fetch": sum(r["probe_fetches"] == 0 for r in gains), "rows": gains,
    }
    expiry = read(HARN / "results_v4/expiry_analysis_single.json")["rows"]
    joint = Counter()
    correct_time_stale = []
    correct_time_bad_action = []
    for r in expiry:
        if r["arm"] not in ("post", "late") or not r.get("has_verdict"):
            continue
        f = r.get("fields", {})
        joint[(f.get("current_state_assumed"), f.get("temporal_frame_correct"),
               f.get("support_action_appropriate"))] += 1
        item = {"route": r["route"], "cell": r["cell"], "visibility": r["visibility"]}
        if f.get("temporal_frame_correct") is True and f.get("current_state_assumed") is True:
            correct_time_stale.append(item)
        if f.get("temporal_frame_correct") is True and f.get("support_action_appropriate") is False:
            correct_time_bad_action.append(item)
    summary["expiry_joint_fields"] = [dict(csa=k[0], temporal_correct=k[1], action_correct=k[2], n=v)
                                       for k, v in joint.items()]
    summary["correct_time_but_stale"] = correct_time_stale
    summary["correct_time_but_bad_action"] = correct_time_bad_action
    relation_field_counts = {k: Counter() for k in ("current_state_assumed", "temporal_frame_correct", "support_action_appropriate")}
    for r in e2["rows"]:
        vs = verdict(cell_for(r, "relation"))
        for k in relation_field_counts:
            relation_field_counts[k][str(vs.get(k))] += 1
    summary["e2_relation_field_counts"] = {k: dict(v) for k, v in relation_field_counts.items()}

    # Restrict the reused E3 current-reference arm to the five main runs.
    # The legacy glob also picked up two hy3 pilot cells.
    e3_routes = {}
    for route, root in ROOTS.items():
        current_paths = sorted((HARN / root / "expiry" / route).glob("*_post_natural"))
        current_paths += sorted((HARN / root / "expiry" / route).glob("*_late_natural"))
        fixed_paths = sorted((HARN / "runs_e3_fixed_ref" / "e3" / route).glob("*_ref_*_natural"))
        current_fetches = [fetch_count(branch_events(p)) for p in current_paths]
        fixed_fetches = [fetch_count(branch_events(p)) for p in fixed_paths]
        if any(n is None for n in current_fetches + fixed_fetches):
            raise ValueError(f"Missing E3 branch log: {route}")
        current_n = len(current_paths)
        fixed_n = len(fixed_paths)
        current_f = sum(n > 0 for n in current_fetches)
        fixed_f = sum(n > 0 for n in fixed_fetches)
        resolved = [(p, verdict(p).get("current_state_assumed")) for p in fixed_paths]
        resolved = [(p, v) for p, v in resolved if isinstance(v, bool)]
        e3_routes[route] = {
            "current_reference_cells": current_n, "current_reference_fetches": current_f,
            "fixed_reference_cells": fixed_n, "fixed_reference_fetches": fixed_f,
            "stm": current_f / current_n - fixed_f / fixed_n,
            "fixed_reference_resolved": len(resolved),
            "fixed_reference_correct": sum(("_ref_before_" in p.name) == v for p, v in resolved),
        }
    summary["e3_main_runs_recount"] = e3_routes

    repeat_rows = []
    for old in read(HARN / "results_v4/repeat_stability.json")["rows"]:
        name = f'{old["kind"]}_{old["arm"]}_{old["visibility"]}'
        baseline = HARN / ROOTS[old["route"]] / "expiry" / old["route"] / name
        paths = [baseline] + sorted((HARN / "runs_repeat").glob(f'rep*/{old["route"]}/{name}'))
        runs = []
        for p in paths:
            be = branch_events(p)
            if be is None:
                continue
            runs.append({"source": "baseline" if p == baseline else p.parents[1].name,
                         "path": str(p.relative_to(HARN)), "probe_fetches": fetch_count(be),
                         "whole_log_fetches": fetch_count(events(p)),
                         "initial_payload_digest": next((e.get("data", {}).get("payload_digest")
                             for e in be if e["type"] == "model.request"), None),
                         "stale": verdict(p).get("current_state_assumed"),
                         "has_report": (p / "report.json").exists()})
        labels = [r["stale"] for r in runs if isinstance(r["stale"], bool)]
        repeated = [r for r in runs if r["source"] != "baseline"]
        repeat_rows.append({"route": old["route"], "cell": name, "n_logs": len(runs),
                            "n_resolved": len(labels), "probe_binary_flip": bool_flip([r["probe_fetches"] for r in runs]),
                            "probe_count_flip": len({r["probe_fetches"] for r in runs}) > 1,
                            "repeat_only_probe_binary_flip": bool_flip([r["probe_fetches"] for r in repeated]),
                            "initial_payload_identical": len({r["initial_payload_digest"] for r in runs}) == 1,
                            "old_count_flip": old["fetch_flip"],
                            "verdict_unanimous": len(set(labels)) <= 1 if labels else None,
                            "runs": runs})
    summary["repeat_raw_recount"] = {
        "cells": len(repeat_rows),
        "logs_per_cell": dict(Counter(r["n_logs"] for r in repeat_rows)),
        "resolved_per_cell": dict(Counter(r["n_resolved"] for r in repeat_rows)),
        "binary_flip_with_baseline": sum(r["probe_binary_flip"] for r in repeat_rows),
        "binary_flip_repeats_only": sum(r["repeat_only_probe_binary_flip"] for r in repeat_rows),
        "count_flip_with_baseline": sum(r["probe_count_flip"] for r in repeat_rows),
        "verdict_unanimous_with_baseline": sum(r["verdict_unanimous"] is True for r in repeat_rows),
        "initial_payload_identical_cells": sum(r["initial_payload_identical"] for r in repeat_rows),
        "by_route": {route: {
            "n": sum(r["route"] == route for r in repeat_rows),
            "binary_flip": sum(r["probe_binary_flip"] for r in repeat_rows if r["route"] == route),
            "count_flip": sum(r["probe_count_flip"] for r in repeat_rows if r["route"] == route),
            "verdict_unanimous": sum(r["verdict_unanimous"] is True for r in repeat_rows if r["route"] == route),
        } for route in sorted({r["route"] for r in repeat_rows})}, "rows": repeat_rows,
    }
    proofs = []
    for vis in ("natural", "forced", "relation"):
        row = {"route": "DeepSeek-V4-Flash", "profile": "visa_interview_i1", "arm": "post"}
        p = cell_for(row, vis)
        es = branch_events(p) or []
        reasoning = [e.get("data", {}).get("reasoning_content", "") for e in es if e["type"] == "model.response"]
        proofs.append({"condition": vis, "path": str(p.relative_to(HARN)), "verdict": verdict(p),
                       "probe_fetches": fetch_count(es), "answer": answer(p), "reasoning": reasoning})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "paired_case.json").write_text(json.dumps(proofs, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in summary.items():
        if k in ("correct_time_but_stale", "correct_time_but_bad_action"):
            print(k, "n", len(v))
        elif isinstance(v, dict):
            print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "rows"}, ensure_ascii=False))
        else:
            print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
