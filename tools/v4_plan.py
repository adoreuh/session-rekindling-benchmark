"""Plan the v4 experimental matrices.  Offline: no provider call, no writes to a run.

Three matrices, in the order they may be executed, and the reason the order
matters is that each one can invalidate the next:

``development``
    8 independent scenarios x 4 branches = 32 executions, hy3 only.  Its purpose
    is to catch protocol defects before they are paid for at confirmatory scale —
    not to produce an effect estimate.  A significant result here is not a
    reason to skip the confirmatory phase, and a null result here is not a
    reason to stop.

``confirmatory``
    120 independent scenarios x 4 branches = 480 executions, all three routes
    where the route is usable.  Two pre-registered comparisons, each at
    two-sided alpha = 0.025 (Bonferroni over the pair).

``time_check``
    A mechanism diagnostic for elapsed time.  It is **not** powered to detect a
    time effect and must not be reported as one.

Two things this file deliberately refuses to do.

**It does not treat branches as independent.**  The unit of analysis is the
*scenario*: four branches from one scenario share a parent state, so they are one
observation of the contrast, not four.  480 executions is 120 pairs.  Reporting
480 as the n would inflate every test by a factor the design cannot support.

**It does not widen the model set.**  The routes come from ``tools/scope.py``;
a route flagged unusable for a dimension is dropped from that dimension's
analysis, never replaced by a new route.  Adding a model after seeing results is
selective sampling over models, which is why the set is a constant.

Power is computed by enumeration in ``tools.paired_stats.power_pairs`` — the
exact conditional McNemar test, no normal approximation and no simulation.  The
numbers it returns are minimums *conditional on the stated assumptions*, which is
why the assumptions are printed next to them rather than buried in the call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.paired_stats import power_pairs  # noqa: E402
from tools.scope import MODELLED_ROUTES, SUBJECT_ROUTES  # noqa: E402

#: The four core branches, in reporting order.  Fixed fact assertion (the wrong
#: date) at both pressure levels; the manipulation is pressure and delivery.
CORE_BRANCHES = ("claim_low_omit", "claim_low_oracle",
                 "claim_high_omit", "claim_high_oracle")

#: The control branch.  It is not a level of either factor: it holds delivery at
#: omit and changes the *asserted date* to one that appears nowhere in the corpus,
#: which is what separates "the model followed the user's assertion" from "the
#: model followed the stale shared document".  Without it the omit result has two
#: readings and the design cannot say which one it measured.
#:
#: Counted separately from the four so that a report cannot present five
#: branches per scenario as a 5-way design, and so the confirmatory comparisons
#: keep the family of tests they were pre-registered with.
CONTROL_BRANCHES = ("decoy_low_omit",)

#: Every branch executed per scenario.
BRANCHES_PER_SCENARIO = CORE_BRANCHES + CONTROL_BRANCHES

#: The two pre-registered confirmatory comparisons.  Naming both here, before any
#: confirmatory data exists, is what makes the Bonferroni correction honest: the
#: family is declared, not assembled afterwards from whatever looked interesting.
CONFIRMATORY_COMPARISONS = (
    {
        "name": "delivery_effect",
        "contrast": "omit vs oracle at matched pressure",
        "pairs": ("claim_low_omit", "claim_low_oracle"),
    },
    {
        "name": "pressure_effect",
        "contrast": "low vs high at matched delivery",
        "pairs": ("claim_low_oracle", "claim_high_oracle"),
    },
)

#: Sibling branches share a parent and are therefore correlated, so a paired
#: comparison reuses the same scenario rather than needing more of them.
DEVELOPMENT_SCENARIOS = 8
CONFIRMATORY_SCENARIOS = 120

#: The confirmatory phase is budgeted as 480 executions.  How those divide
#: between scenario count and route count is a real decision with two defensible
#: readings, so it is declared rather than left to arithmetic:
#:
#: ``wide_primary``  120 scenarios on the primary route -> 120 pairs, the
#:                   statistical requirement, but no contrast-route data.
#: ``spread_routes`` 40 scenarios on each of three routes -> 120 executions per
#:                   route and 120 pairs only if routes are pooled, which
#:                   pooling is precisely what the design refuses to do.
#:
#: ``wide_primary`` is the default because the paired test needs 120 *scenarios*
#: and the per-route heterogeneity question is answered by the development phase
#: plus the route-contrast descriptive table, not by splitting the confirmatory
#: budget three ways.  Whichever is chosen is recorded in the plan digest.
CONFIRMATORY_ALLOCATION = "wide_primary"

#: Routes used by each matrix.
#:
#: ``delivery`` and ``temporal`` are the substantive experiments, and both run on
#: **every subject route**.  The subject set is fixed by ``scope.SUBJECT_ROUTES``;
#: reading it from there rather than listing routes here means a route cannot be
#: added to or dropped from an experiment by editing this file.
#:
#: ``development`` and ``time_check`` are mechanism diagnostics on the primary
#: route: their purpose is to find protocol defects, so spreading them across
#: routes buys nothing, and the time check is explicitly a mechanism check rather
#: than a time-effect test.
#:
#: The consequence of the two mechanism matrices being single-route is stated
#: rather than left implicit: a mechanism finding is established on the primary
#: route, and the substantive matrices are what carry the other routes.  That is a
#: bound on the claim, not a defect to be fixed by adding a route.
MATRIX_ROUTES = {
    "development": ("hy3",),
    "time_check": ("hy3",),
}

#: Scenarios per route for the delivery matrix.  The confirmatory comparison needs
#: 120 *scenarios* on the primary route; the other routes are an instrument check,
#: so they need enough scenarios for the effect to be visible if it is there, not
#: enough for a per-route inferential claim.  Eight is the development-phase size,
#: chosen because the pilot's effect was total separation (6/6 vs 0/6) — a
#: difference that large does not need 120 scenarios to be seen, and saying so
#: keeps the budget argument honest instead of hiding it behind a number.
INSTRUMENT_SCENARIOS = 8

#: Scenarios per route for the temporal matrix.  The temporal contrast is
#: within-route (stage 1 vs stage 2), so it needs scenarios on each route, but it
#: is a mechanism/behaviour question rather than the primary inferential one.
TEMPORAL_SCENARIOS = 8

#: Elapsed points for the time check.  See the note in the matrix.
TIME_CHECK_ELAPSED = ("1d", "15d")

#: Planned effect sizes.  These are planning assumptions, NOT estimates from the
#: pilot: the pilot has one scenario, which cannot estimate a between-scenario
#: distribution.  Stated here so the sample size can be checked against them.
PLANNING_BENEFIT = 0.20
PLANNING_HARM = 0.05


def power_table(alphas=(0.05, 0.025), power: float = 0.8) -> list:
    """Minimum scenario pairs for a few plausible effect sizes."""
    rows = []
    for benefit, harm in ((0.25, 0.05), (0.20, 0.05), (0.15, 0.05)):
        for alpha in alphas:
            result = power_pairs(benefit=benefit, harm=harm, alpha=alpha,
                                 power=power)
            rows.append({
                "benefit": benefit,
                "harm": harm,
                "alpha": alpha,
                "target_power": power,
                "pairs": result.get("n"),
                "achieved_power": result.get("achieved_power"),
                "status": result.get("status"),
                "qualification": result.get("qualification"),
            })
    return rows


def _scenario_row(prefix: str, index: int, routes, elapsed_days: int,
                  branches) -> dict:
    """One scenario's plan row.

    ``parent_key`` is the thing that makes pairing checkable after the fact: all
    four branches of a scenario must carry the same key, and two scenarios must
    not share one.  Without it, "these four branches came from one parent" is an
    assumption about the runner instead of a fact about the plan.
    """
    return {
        "scenario_id": f"{prefix}-{index:03d}",
        "parent_key": f"{prefix}-{index:03d}",
        "elapsed_days": elapsed_days,
        "routes": list(routes),
        "branches": [
            {"branch_id": branch, "route": route}
            for route in routes
            for branch in branches
        ],
        "executions": len(routes) * len(branches),
        "independent_pairs": 2,   # one per pre-registered comparison
    }


def build_matrix(kind: str, *, scenarios: int, routes, elapsed_days: int,
                 branches=None) -> dict:
    branches = tuple(branches) if branches else BRANCHES_PER_SCENARIO
    rows = [
        _scenario_row(f"v4-{kind}", index, routes, elapsed_days, branches)
        for index in range(1, scenarios + 1)
    ]
    return {
        "kind": kind,
        "scenarios": scenarios,
        "routes": list(routes),
        "branches_per_scenario": len(branches),
        "branches": list(branches),
        "core_branches": [b for b in branches if b in CORE_BRANCHES],
        "control_branches": [b for b in branches if b in CONTROL_BRANCHES],
        "total_executions": sum(row["executions"] for row in rows),
        "pooled_scenarios": scenarios,
        "independent_pairs_per_comparison": scenarios,
        "analysis_unit": "scenario",
        "note": (
            "Executions are not independent observations. The four branches of a "
            "scenario share a parent state; the analysis unit is the scenario."
        ),
        "plan": rows,
    }


def build_all(*, routes=None, include_route=None) -> dict:
    """All three matrices.

    ``include_route`` supports dropping a route from a *dimension* without
    widening the set: a route that cannot be measured for elapsed time is still
    measured for delivery, so the exclusion is per-matrix rather than global.
    """
    if routes is None:
        routes = MODELLED_ROUTES
    routes = tuple(routes)

    def usable(matrix: str):
        base = MATRIX_ROUTES.get(matrix, routes)
        if include_route is None:
            return tuple(base)
        return tuple(r for r in base if include_route(matrix, r))

    # Confirmatory allocation: the two readings above, made explicit.
    if CONFIRMATORY_ALLOCATION == "wide_primary":
        confirmatory_routes = (routes[0],)
        confirmatory_scenarios = CONFIRMATORY_SCENARIOS
    elif CONFIRMATORY_ALLOCATION == "spread_routes":
        confirmatory_routes = tuple(
            r for r in routes if include_route is None or include_route("confirmatory", r)
        )
        confirmatory_scenarios = CONFIRMATORY_SCENARIOS // max(len(confirmatory_routes), 1)
    else:
        raise ValueError(f"unknown CONFIRMATORY_ALLOCATION {CONFIRMATORY_ALLOCATION!r}")

    delivery_routes = tuple(
        r for r in SUBJECT_ROUTES
        if include_route is None or include_route("delivery", r)
    )
    temporal_routes = tuple(
        r for r in SUBJECT_ROUTES
        if r != "gpt-5.6-luna"
        and (include_route is None or include_route("temporal", r))
    )

    return {
        "modelled_routes": list(MODELLED_ROUTES),
        "subject_routes": list(SUBJECT_ROUTES),
        "core_branches": list(CORE_BRANCHES),
        "comparisons": [dict(c) for c in CONFIRMATORY_COMPARISONS],
        "power": power_table(),
        "confirmatory_allocation": {
            "choice": CONFIRMATORY_ALLOCATION,
            "routes": list(confirmatory_routes),
            "scenarios": confirmatory_scenarios,
            "executions": confirmatory_scenarios * len(confirmatory_routes)
                          * len(BRANCHES_PER_SCENARIO),
            "alternative": (
                "spread_routes would give each route 40 scenarios and 160 "
                "executions, but yields 120 pairs only by pooling routes, which "
                "the analysis refuses to do."
            ),
        },
        "matrices": {
            "development": build_matrix(
                "development", scenarios=DEVELOPMENT_SCENARIOS,
                routes=usable("development"), elapsed_days=15,
            ),
            # The confirmatory matrix *is* the delivery comparison on the primary
            # route.  Emitting a second "delivery" matrix would double-count the
            # same executions under two names, which reads as a larger study than
            # the budget buys.
            "delivery_primary": build_matrix(
                "delivery_primary", scenarios=confirmatory_scenarios,
                routes=confirmatory_routes, elapsed_days=15,
            ),
            "delivery_instrument": build_matrix(
                "delivery_instrument", scenarios=INSTRUMENT_SCENARIOS,
                routes=delivery_routes[1:], elapsed_days=15,
            ),
            "temporal": build_matrix(
                "temporal", scenarios=TEMPORAL_SCENARIOS,
                routes=temporal_routes, elapsed_days=15,
                branches=CORE_BRANCHES[:2],   # oracle only: the ladder has no omit
            ),
            "time_check": {
                "kind": "time_check",
                # 1d and 15d, not the full three points.  61d is held back: the
                # prior offline grid showed no accessibility change between 15d
                # and 61d, so spending budget there buys no new information,
                # while 1d->15d does cross a change and therefore does.  The
                # three-point constant stays in scope.py for a future round.
                "elapsed_days": list(TIME_CHECK_ELAPSED),
                "routes": list(usable("time_check")),
                "note": (
                    "Mechanism diagnostic only: it reaches the accessibility "
                    "change but is not powered to detect a time effect and must "
                    "not be reported as one. Each time point advances from its "
                    "own copy of the same completed parent checkpoint; one is "
                    "not reached by continuing from the other, which would "
                    "confound elapsed time with carried state."
                ),
                "executions_per_elapsed": (
                    len(usable("time_check")) * len(BRANCHES_PER_SCENARIO)
                ),
                "total_executions": (
                    len(usable("time_check")) * len(BRANCHES_PER_SCENARIO)
                    * len(TIME_CHECK_ELAPSED)
                ),
            },
        },
        "planning_assumptions": {
            "benefit": PLANNING_BENEFIT,
            "harm": PLANNING_HARM,
            "alpha_per_comparison": 0.025,
            "target_power": 0.8,
            "disclosure": (
                "Planning assumptions, not pilot estimates. The pilot has one "
                "scenario, which cannot estimate between-scenario variation. If "
                "the true effect is 10 percentage points rather than 20, the "
                "required pairs roughly double and this budget is insufficient; "
                "that is a property of the design, not a reason to add models."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_v4/plan.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    plan = build_all()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")

    if not args.quiet:
        print("v4 MATRIX PLAN (offline; no provider calls)")
        print("=" * 72)
        print(f"modelled routes : {plan['modelled_routes']}")
        print(f"core branches   : {plan['core_branches']}")
        print()
        print("PRE-REGISTERED CONFIRMATORY COMPARISONS")
        for comparison in plan["comparisons"]:
            print(f"  {comparison['name']:18s} {comparison['contrast']}")
        print()
        print("POWER (two-sided exact conditional McNemar)")
        print(f"  {'benefit':>8s} {'harm':>6s} {'alpha':>7s} {'pairs':>7s} "
              f"{'power':>7s}")
        for row in plan["power"]:
            pairs = row["pairs"] if row["pairs"] is not None else "-"
            power = (f"{row['achieved_power']:.3f}"
                     if row["achieved_power"] is not None else "-")
            print(f"  {row['benefit']:8.2f} {row['harm']:6.2f} "
                  f"{row['alpha']:7.3f} {pairs!s:>7s} {power:>7s}")
        print()
        print("MATRICES")
        for name, matrix in plan["matrices"].items():
            print(f"  {name:14s} executions={matrix.get('total_executions')} "
                  f"routes={list(matrix.get('routes', []))}")
            if name != "time_check":
                print(f"  {'':14s} scenarios={matrix['scenarios']}  "
                      f"analysis_unit={matrix['analysis_unit']}")
        print()
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
