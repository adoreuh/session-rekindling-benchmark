"""Repeat-run verdict-level stability: do independent runs of the same cell
produce the same expiry verdict?  Per (route, kind, arm) group, report the
fetch-count spread, the stale-verdict unanimity, and the flip rate, plus a
bootstrap 95% CI for the flip rate.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
random.seed(20261009)


def _fetch(d: Path) -> int:
    ev = d / "branches" / "probe" / "events.jsonl"
    if not ev.exists():
        return 0
    n = 0
    for line in ev.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        dd = e.get("data", {})
        if (e.get("type") == "tool.call" and dd.get("tool_name") == "world_time"
                and not dd.get("injected")):
            n += 1
    return n


def _stale(d: Path):
    jf = d / "judge.expiry.json"
    if not jf.exists():
        return None
    j = json.loads(jf.read_text(encoding="utf-8"))
    v = (j.get("branches") or {}).get("probe", {}).get("verdict", {})
    return (v.get("current_state_assumed") or {}).get("value")


def main() -> int:
    root = HARN / "runs_repeat"
    groups = defaultdict(list)
    for rep_dir in sorted(p for p in root.glob("rep*") if p.is_dir()):
        rep = rep_dir.name
        for route_dir in sorted(rep_dir.iterdir()):
            if not route_dir.is_dir():
                continue
            for cell in sorted(route_dir.iterdir()):
                if not cell.is_dir() or not (cell / "report.json").exists():
                    continue
                name = cell.name
                for arm in ("post", "late"):
                    for vis in ("natural", "delivered"):
                        suffix = f"_{arm}_{vis}"
                        if name.endswith(suffix):
                            kind = name[: -len(suffix)]
                            groups[(route_dir.name, kind, arm, vis)].append({
                                "rep": rep, "fetch": _fetch(cell),
                                "stale": _stale(cell)})

    rows = []
    for (route, kind, arm, vis), runs in sorted(groups.items()):
        fetches = [r["fetch"] for r in runs]
        labels = [r["stale"] for r in runs if r["stale"] is not None]
        stale_n = sum(1 for x in labels if x)
        unanimous = len(set(labels)) <= 1 if labels else None
        rows.append({
            "route": route, "kind": kind, "arm": arm, "visibility": vis,
            "n": len(runs), "n_judged": len(labels),
            "fetch_counts": fetches,
            "fetch_unanimous": len(set(fetches)) <= 1,
            "stale_labels": labels, "stale_count": stale_n,
            "verdict_unanimous": unanimous,
            "fetch_flip": len(set(fetches)) > 1,
        })

    judged = [r for r in rows if r["n_judged"] >= 2]
    fetch_flip = sum(r["fetch_flip"] for r in judged) / len(judged) if judged else None
    verdict_unan = (sum(1 for r in judged if r["verdict_unanimous"]) / len(judged)
                    if judged else None)

    # bootstrap CI over cells for the fetch-flip rate
    def boot(vals, n=10000):
        if not vals:
            return None
        out = []
        for _ in range(n):
            s = [vals[random.randrange(len(vals))] for _ in vals]
            out.append(sum(s) / len(s))
        out.sort()
        return [out[int(0.025 * n)], out[int(0.975 * n)]]

    flips = [1 if r["fetch_flip"] else 0 for r in judged]
    ci = boot(flips)
    out = {
        "cells_observed": len(rows),
        "cells_judged_ge2": len(judged),
        "fetch_flip_rate": fetch_flip,
        "fetch_flip_rate_ci95": ci,
        "verdict_unanimity": verdict_unan,
        "by_route": {},
        "rows": rows,
    }
    for route in sorted({r["route"] for r in judged}):
        sub = [r for r in judged if r["route"] == route]
        f = [1 if r["fetch_flip"] else 0 for r in sub]
        out["by_route"][route] = {
            "cells": len(sub),
            "fetch_flip_rate": sum(f) / len(f),
            "fetch_flip_rate_ci95": boot(f),
            "verdict_unanimity": (sum(1 for r in sub if r["verdict_unanimous"]) / len(sub)),
            "stale_cells": sum(1 for r in sub if r["stale_count"] > 0),
        }
    (HARN / "results_v4" / "repeat_stability.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cells observed {len(rows)}, judged>=2 {len(judged)}")
    print(f"fetch flip rate {fetch_flip:.3f} CI {ci}")
    print(f"verdict unanimity {verdict_unan:.3f}")
    for route, v in out["by_route"].items():
        print(f"  {route:24s} cells={v['cells']:3d} flip={v['fetch_flip_rate']:.3f} "
              f"unan={v['verdict_unanimity']:.3f} stale_cells={v['stale_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
