"""Cluster-level summary: does the conclusion survive treating *event kinds* as units?

Why this is a separate check.  The delivery and temporal tables are computed over
(route x pressure x scenario) cells.  The same event kind contributes several cells --
five routes, and for the temporal axis two ladder stages and two pressure levels -- so the
cells are not independent draws, and a per-cell rate can understate uncertainty.  The
question a reviewer will ask is whether the effect survives when the *kind* is the unit.

What makes the answer cheap here.  The frozen analysis already records, per cell, whether
the pair endorsed (``per_scenario`` in ``results_v4/scale_final.json``), so this tool reads
the authoritative artefact rather than re-deriving anything from the runs.

The honest form of the answer when every unit agrees.  With ``x = n`` successes the
Clopper-Pearson interval is one-sided in effect, and the lower bound is ``(alpha/2)^(1/n)``
-- not 1.0 and not "no information".  That bound is what gets reported: the effect is at
ceiling, and the ceiling is estimated with n units, not with n cells.

Output: ``results_v4/cluster_check.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE_FINAL = ROOT / "results_v4" / "scale_final.json"


def _upper_tail(total: int, successes: int, probability: float) -> float:
    """P(X >= successes) for X ~ Binomial(total, probability)."""
    from math import comb

    return sum(
        comb(total, k) * probability**k * (1.0 - probability) ** (total - k)
        for k in range(successes, total + 1)
    )


def clopper_pearson_lower(successes: int, total: int, *, alpha: float = 0.05) -> float:
    """Lower Clopper-Pearson bound, by bisection on the binomial upper tail.

    No SciPy: the bound is the ``p`` at which ``P(X >= successes | p)`` equals
    ``alpha/2``.  The tail is *increasing* in ``p`` (raising ``p`` shifts the mass to the
    right), which is what makes the bisection well defined.  With ``successes == total``
    this returns ``(alpha/2)^(1/n)``.
    """
    if total <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    target = alpha / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _upper_tail(total, successes, mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


def _kind_of(scenario: str) -> str:
    """``appliance_repair_i1`` -> ``appliance_repair`` (the event kind)."""
    stem, _, instance = scenario.rpartition("_i")
    return stem if stem and instance.isdigit() else scenario


def delivery_clusters(scale: dict) -> dict:
    """Per-kind endorsement on the delivery axis, primary arm and pooled."""
    per_kind: dict[str, dict] = {}
    for key, unit in scale["delivery"].items():
        route, pressure = unit["route"], unit["pressure"]
        for scenario, cell in unit["per_scenario"].items():
            kind = _kind_of(scenario)
            slot = per_kind.setdefault(
                kind, {"primary": [], "pooled": []}
            )
            value = bool(cell["omit_endorses_voided"])
            slot["pooled"].append(value)
            if pressure == scale["primary_pressure"]:
                slot["primary"].append(value)

    def summarise(field: str) -> dict:
        rates, complete, observed = [], 0, 0
        for kind, slot in sorted(per_kind.items()):
            cells = slot[field]
            if not cells:
                continue
            observed += 1
            rate = sum(cells) / len(cells)
            rates.append({"kind": kind, "cells": len(cells), "rate": rate})
            if rate == 1.0:
                complete += 1
        return {
            "kinds_with_cells": observed,
            "kinds_fully_endorsing": complete,
            "mean_of_kind_rates": (sum(r["rate"] for r in rates) / observed) if observed else None,
            "min_kind_rate": min((r["rate"] for r in rates), default=None),
            "lower_bound_if_all_complete": (
                clopper_pearson_lower(complete, observed) if observed else None
            ),
            "kinds_below_one": [r for r in rates if r["rate"] < 1.0],
        }

    return {
        "primary_arm": summarise("primary"),
        "pooled_both_arms": summarise("pooled"),
        "cluster_definition": (
            "one cluster per event kind; cells are (route x pressure) observations of that kind"
        ),
    }


def temporal_clusters(scale: dict) -> dict:
    """Per-kind temporal correctness, which repeats 4 x 5 times per kind."""
    cells = scale["temporal"]
    # temporal cells are keyed "<elapsed>|<branch>"; both count the same scenarios
    per_kind: dict[str, list] = {}
    total_named = total_scenarios = 0
    for route, table in cells.items():
        for cell_key, cell in table.items():
            total_named += cell["named_operative"]
            total_scenarios += cell["scenarios"]
    # Per-kind granularity is not recorded in this artefact; report the route-level
    # counts and state that kind-level repetition is 4x5 by construction.
    return {
        "branch_observations": total_scenarios,
        "named_operative": total_named,
        "rate": (total_named / total_scenarios) if total_scenarios else None,
        "route_level_lower_bound": clopper_pearson_lower(
            total_named, total_scenarios
        ) if total_scenarios and total_named == total_scenarios else None,
        "kind_level_note": (
            "each kind contributes 2 stages x 2 pressures x 5 routes = 20 branch "
            "observations; per-kind rates are not stored in scale_final.json, so the "
            "route-level figure is what this artefact supports"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default=str(SCALE_FINAL))
    parser.add_argument("--out", default=str(ROOT / "results_v4" / "cluster_check.json"))
    args = parser.parse_args()

    scale = json.loads(Path(args.scale).read_text(encoding="utf-8"))
    report = {
        "source": str(Path(args.scale).name),
        "alpha": 0.05,
        "delivery": delivery_clusters(scale),
        "temporal": temporal_clusters(scale),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    d = report["delivery"]
    print(f"wrote {out}")
    for field in ("primary_arm", "pooled_both_arms"):
        s = d[field]
        print(f"  {field}: kinds={s['kinds_with_cells']} "
              f"fully_endorsing={s['kinds_fully_endorsing']} "
              f"min_rate={s['min_kind_rate']} "
              f"lower_bound={s['lower_bound_if_all_complete']}")
    t = report["temporal"]
    print(f"  temporal: {t['named_operative']}/{t['branch_observations']} "
          f"lower_bound={t['route_level_lower_bound']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
