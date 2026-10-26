"""E2 analysis: does stating the temporal relation cure what injecting the clock did not?

Pairs cells on (route, profile, arm) across four ways current time is presented:
  natural           clock reachable only through a tool call the model must make
  forced            an authoritative world_time result is injected (E1)
  delivered         the current instant is rendered in the observation
  relation          delivered plus the original date and the stated relation (E2)

`relation` differs from `delivered` only in that the conflict is stated rather
than left to be inferred.  A drop from `delivered` to `relation` therefore
measures framing sensitivity; no drop means the residual failure is not a
presentation problem.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]


def _stale(cell: Path):
    jf = cell / "judge.expiry.json"
    if not jf.exists():
        return None
    j = json.loads(jf.read_text(encoding="utf-8"))
    v = (j.get("branches") or {}).get("probe", {}).get("verdict", {})
    return (v.get("current_state_assumed") or {}).get("value")


def _scan(root: Path, pattern: str, arm: str, vis: str):
    """Scan one experiment family; ``pattern`` must select the *route* dirs."""
    out = {}
    for route_dir in sorted(p for p in root.glob(pattern) if p.is_dir()):
        if not route_dir.is_dir():
            continue
        for cell in sorted(route_dir.glob(f"*_{arm}_{vis}")):
                profile = cell.name.rsplit(f"_{arm}_{vis}", 1)[0]
                out[(route_dir.name, profile, arm)] = _stale(cell)
    return out


def mcnemar(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HARN / "results_v4" / "e2_analysis.json"))
    args = ap.parse_args()

    tables = {}
    for arm in ("post", "late"):
        tables[("natural", arm)] = _scan(HARN, "runs_expiry_*/expiry/*", arm, "natural")
        tables[("delivered", arm)] = _scan(HARN, "runs_expiry_*/expiry/*", arm, "delivered")
        tables[("forced", arm)] = _scan(HARN, "runs_e1_forced/e1/*", arm, "forced")
        # E2's route dirs sit one level deeper than the other families'.
        tables[("relation", arm)] = _scan(HARN, "runs_e2_relation/e2/*", arm, "relation")

    def get(vis, arm, key):
        return tables.get((vis, arm), {}).get(key)

    keys = sorted({k for tbl in tables.values() for k in tbl})
    rows = []
    for key in keys:
        row = {"route": key[0], "profile": key[1], "arm": key[2]}
        for vis in ("natural", "delivered", "forced", "relation"):
            row[vis] = get(vis, key[2], key)
        rows.append(row)

    def compare(a, b):
        ba = ca = both = neither = 0
        for r in rows:
            x, y = r[a], r[b]
            if x is None or y is None:
                continue
            if x and not y:
                ba += 1
            elif y and not x:
                ca += 1
            elif x and y:
                both += 1
            else:
                neither += 1
        n = ba + ca + both + neither
        return {"n_pairs": n, "a_stale_only": ba, "b_stale_only": ca,
                "both_stale": both, "both_correct": neither,
                "p_exact": mcnemar(ba, ca),
                "stale_rate_a": (ba + both) / n if n else None,
                "stale_rate_b": (ca + both) / n if n else None}

    out = {
        "cells": len(rows),
        "comparisons": {
            "delivered_vs_relation": compare("delivered", "relation"),
            "forced_vs_relation": compare("forced", "relation"),
            "natural_vs_relation": compare("natural", "relation"),
        },
        "by_route": {},
        "rows": rows,
    }
    for route in sorted({r["route"] for r in rows}):
        sub = [r for r in rows if r["route"] == route]

        def rate(vis):
            vals = [r[vis] for r in sub if r[vis] is not None]
            return (sum(1 for v in vals if v) / len(vals), len(vals)) if vals else (None, 0)

        out["by_route"][route] = {
            vis: {"stale_rate": rate(vis)[0], "n": rate(vis)[1]}
            for vis in ("natural", "delivered", "forced", "relation")
        }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    for name, c in out["comparisons"].items():
        print(f"{name:26s} n={c['n_pairs']:3d}  "
              f"{c['stale_rate_a'] if c['stale_rate_a'] is None else round(c['stale_rate_a'],3)} -> "
              f"{c['stale_rate_b'] if c['stale_rate_b'] is None else round(c['stale_rate_b'],3)}  "
              f"improve={c['a_stale_only']:3d} worsen={c['b_stale_only']:3d}  p={c['p_exact']:.3g}")
    print()
    for route, v in out["by_route"].items():
        print(f"{route:24s} " + "  ".join(
            f"{vis}={None if v[vis]['stale_rate'] is None else round(v[vis]['stale_rate'],2)}"
            f"({v[vis]['n']})" for vis in ("natural", "delivered", "forced", "relation")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
