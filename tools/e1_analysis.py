"""E1 analysis: natural vs forced-fetch vs delivered, paired on (route, profile, arm).

Reads the three expiry trees:
- natural/delivered: runs_expiry_<route>/expiry/<route>/<profile>_{arm}_{vis}/
- forced:           runs_e1_forced/e1/<route>/<profile>_{arm}_forced/

Pairing key = (route, profile, arm).  The stale label comes from
judge.expiry.json -> branches.probe.verdict.current_state_assumed; cells
without a judged value are dropped as unresolved, never imputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]


def _load_label(cell_dir: Path):
    jf = cell_dir / "judge.expiry.json"
    if not jf.exists():
        return None
    j = json.loads(jf.read_text(encoding="utf-8"))
    v = (j.get("branches") or {}).get("probe", {}).get("verdict", {})
    item = v.get("current_state_assumed") or {}
    return item.get("value")


def _fetches(cell_dir: Path) -> int:
    ev = cell_dir / "branches" / "probe" / "events.jsonl"
    if not ev.exists():
        return 0
    n = 0
    for line in ev.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        d = e.get("data", {})
        if (e.get("type") == "tool.call" and d.get("tool_name") == "world_time"
                and not d.get("injected")):
            n += 1
    return n


def cells(root: Path, arm: str, vis: str):
    for route_dir in sorted(root.glob("e1/*/")):
        for d in sorted(route_dir.glob(f"*_{arm}_{vis}")):
            profile = d.name.rsplit(f"_{arm}_{vis}", 1)[0]
            yield route_dir.name, profile, d
    for exp_root in sorted(root.glob("runs_expiry_*/expiry/*/")):
        for d in sorted(exp_root.glob(f"*_{arm}_{vis}")):
            profile = d.name.rsplit(f"_{arm}_{vis}", 1)[0]
            yield exp_root.name, profile, d


def mcnemar(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    n = b + c
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=str(HARN / "runs_e1_forced"))
    ap.add_argument("--orig", default=str(HARN))
    ap.add_argument("--out", default=str(HARN / "results_v4" / "e1_analysis.json"))
    args = ap.parse_args()

    table = {}
    for arm in ("post", "late"):
        for vis, root in (("natural", Path(args.orig)),
                          ("delivered", Path(args.orig)),
                          ("forced", Path(args.e1))):
            for route, profile, d in cells(root, arm, vis):
                key = (route, profile, arm)
                table.setdefault(key, {})[vis] = {
                    "stale": _load_label(d), "fetches": _fetches(d),
                    "dir": str(d)}

    rows = []
    for (route, profile, arm), conds in sorted(table.items()):
        row = {"route": route, "profile": profile, "arm": arm,
               "natural": conds.get("natural"), "delivered": conds.get("delivered"),
               "forced": conds.get("forced")}
        rows.append(row)

    def stale_of(rec, vis):
        v = (rec.get(vis) or {}).get("stale")
        return None if v is None else bool(v)

    out = {"cells": len(rows), "rows": rows, "comparisons": {}}
    for a, b in (("natural", "forced"), ("forced", "delivered")):
        b_ = c_ = both = neither = 0
        for r in rows:
            x, y = stale_of(r, a), stale_of(r, b)
            if x is None or y is None:
                continue
            if x and not y: b_ += 1
            elif y and not x: c_ += 1
            elif x and y: both += 1
            else: neither += 1
        n = b_ + c_ + both + neither
        out["comparisons"][f"{a}_vs_{b}"] = {
            "n_pairs": n, "stale_a_only": b_, "stale_b_only": c_,
            "both_stale": both, "both_correct": neither,
            "p_exact": mcnemar(b_, c_),
            "stale_rate_a": (b_ + both) / n if n else None,
            "stale_rate_b": (c_ + both) / n if n else None}

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(json.dumps(out["comparisons"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
