"""E3 analysis: selective temporal monitoring.

STM = P(F=1 | CN=1) - P(F=1 | CN=0), where CN=1 is the natural expiry arm
(model must fetch world_time itself) and CN=0 is the fixed-reference arm
(reference instant is in the probe, so the clock is useless).  F counts only
model-originated world_time calls in the probe branch (injected calls carry
``injected`` and are excluded by construction — E3 has none anyway).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]


def probe_fetches(cell_dir: Path) -> int:
    ev = cell_dir / "branches" / "probe" / "events.jsonl"
    if not ev.exists():
        return 0
    rows = [json.loads(l) for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()]
    fork = next((i for i, r in enumerate(rows)
                 if r.get("type") == "branch.forked"), 0)
    n = 0
    for r in rows[fork:]:
        d = r.get("data", {})
        if (r.get("type") == "tool.call" and d.get("tool_name") == "world_time"
                and not d.get("injected")):
            n += 1
    return n


def label(cell_dir: Path):
    jf = cell_dir / "judge.expiry.json"
    if not jf.exists():
        return None
    j = json.loads(jf.read_text(encoding="utf-8"))
    v = (j.get("branches") or {}).get("probe", {}).get("verdict", {})
    return (v.get("current_state_assumed") or {}).get("value")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", default=str(HARN / "runs_e3_fixed_ref"))
    ap.add_argument("--orig", default=str(HARN))
    ap.add_argument("--out", default=str(HARN / "results_v4" / "e3_analysis.json"))
    args = ap.parse_args()

    # CN=1: natural post/late cells from the original expiry runs.
    cn1 = {}
    for d in glob.glob(str(Path(args.orig) / "runs_expiry_*" / "expiry" / "*" / "*_post_natural")) + \
             glob.glob(str(Path(args.orig) / "runs_expiry_*" / "expiry" / "*" / "*_late_natural")):
        d = Path(d)
        route = d.parent.name
        profile = d.name.rsplit("_", 2)[0]
        cn1.setdefault(route, []).append({"profile": profile,
            "arm": "post" if "_post_" in d.name else "late",
            "fetch": probe_fetches(d) > 0, "label": label(d)})

    # CN=0: ref arms from the E3 run.
    cn0 = {}
    for d in glob.glob(str(Path(args.e3) / "e3" / "*" / "*_ref_*_natural")):
        d = Path(d)
        route = d.parent.name
        arm = "ref_before" if "_ref_before_" in d.name else "ref_after"
        profile = d.name.rsplit("_", 2)[0]
        cn0.setdefault(route, []).append({"profile": profile,
            "arm": arm, "fetch": probe_fetches(d) > 0, "label": label(d)})

    routes = sorted(set(cn1) | set(cn0))
    out = {"routes": {}}
    for route in routes:
        c1 = cn1.get(route, []); c0 = cn0.get(route, [])
        p1 = sum(x["fetch"] for x in c1) / len(c1) if c1 else None
        p0 = sum(x["fetch"] for x in c0) / len(c0) if c0 else None
        stm = (p1 - p0) if (p1 is not None and p0 is not None) else None
        # CN=0 accuracy: ref_before should be upcoming (csa True), ref_after past (csa False).
        acc = [x for x in c0 if x["label"] is not None]
        n_acc = sum((x["arm"] == "ref_before") == bool(x["label"]) for x in acc)
        out["routes"][route] = {
            "cn1_cells": len(c1), "cn1_fetch_rate": p1,
            "cn0_cells": len(c0), "cn0_fetch_rate": p0,
            "stm": stm,
            "cn0_accuracy": f"{n_acc}/{len(acc)}" if acc else None,
            "cn0_detail": c0,
        }
        print(f"{route:24s} CN1 fetch {p1 and round(p1,2)}  CN0 fetch {p0 and round(p0,2)}  "
              f"STM {stm and round(stm,2)}  CN0 acc {n_acc}/{len(acc)}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
