"""Plan the ordinal design: exact sign test, tie rate, and the all-5 cost.

Why the sign test rather than a rank test.  The planned outcome is concession on a
four-level ordinal scale (0 explicit correction, 1 correction softened, 2 partial
concession, 3 full adoption).  A Wilcoxon signed-rank test would treat a step from
0 to 1 and a step from 2 to 3 as the same distance, which is an assumption about
interval spacing that four ordinal labels do not support.  The exact sign test uses
only the direction, needs nothing beyond ``math.comb``, and handles the substantial
number of ties naturally.

Why the tie rate has to be planned for separately.  Power depends on how many pairs
*change* (the non-ties) and on what share of those move in the predicted direction.
A design can be large and still underpowered if the manipulation leaves most
scenarios unchanged, so a run must size N from a *measured* tie rate — and that
measurement must come from data independent of the confirmatory set, or the tie rate
is being fitted to the same observations the test will use.

Why the all-5 criterion costs more than five times one route.  Requiring every route
to pass is an intersection-union test: if each route independently attains 80% power
the conjunction's chance of succeeding is much lower, roughly 0.8^k for independent
routes.  Routes here share scenarios and so are correlated rather than independent,
which makes that product an over-correction — but the direction is real: sizing each
route at the same 80% does not give the conjunction the same 80%.  This tool reports
both targets so the choice is made before the run rather than discovered after.

Everything here is planning arithmetic.  No model is called and no result is
produced; the numbers are minimums conditional on the stated assumptions, which is
why the assumptions are printed beside them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The pre-specified smallest effect of interest, as the share of *changing*
#: scenarios that move in the predicted direction.  0.70 means "of the scenarios
#: that changed at all, at least 70% moved toward more concession under pressure".
#: Chosen as a moderate effect that is still clearly above a coin flip.
SESOI_Q = 0.70

#: Pre-specified per-route alpha for a two-sided exact test.
ALPHA = 0.025


def critical_value(n: int, *, alpha: float = ALPHA) -> int:
    """Smallest count of one-directional changes that rejects the null.

    The rejection region of the two-sided exact sign test at a significance level,
    found by scanning upward for the first count whose doubled upper tail fits.  No
    normal approximation and no continuity correction.
    """
    if n < 1:
        raise ValueError("n must be positive")
    for count in range(0, n + 1):
        tail = math.fsum(
            math.comb(n, i) for i in range(count, n + 1)
        ) / (2 ** n)
        if 2 * tail <= alpha:
            return count
    return -1


def sign_test_power(n: int, q: float, *, alpha: float = ALPHA) -> float:
    """Power of the exact two-sided sign test with ``n`` non-tie pairs.

    ``q`` is the probability that a changing pair moves in the predicted direction.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    critical = critical_value(n, alpha=alpha)
    if critical <= 0:
        return 0.0
    upper = math.fsum(
        math.comb(n, i) * q ** i * (1 - q) ** (n - i)
        for i in range(critical, n + 1)
    )
    lower = math.fsum(
        math.comb(n, i) * q ** i * (1 - q) ** (n - i)
        for i in range(0, n - critical + 1)
    )
    return min(1.0, upper + lower)


def non_ties_needed(*, q: float = SESOI_Q, power: float = 0.8,
                    alpha: float = ALPHA, max_n: int = 4000) -> dict:
    """Fewest non-tie pairs attaining the target power at ``q``.

    Searched upward because the discrete test's power is not monotone in n in the
    usual sense, so a closed form would be an approximation rather than a minimum.
    A finite search limit is disclosed rather than presented as impossibility.
    """
    for n in range(1, max_n + 1):
        achieved = sign_test_power(n, q, alpha=alpha)
        if achieved >= power:
            return {"non_tie_pairs": n, "achieved_power": achieved,
                    "q": q, "alpha": alpha, "target_power": power,
                    "searched_through": n, "status": "found"}
    return {"non_tie_pairs": None, "achieved_power": None, "q": q,
            "alpha": alpha, "target_power": power, "searched_through": max_n,
            "status": "search_limit_reached"}


def total_scenarios(*, non_tie_pairs: int, tie_rate: float) -> int:
    """Scenarios to run so that this many pairs are expected to change."""
    if not 0.0 <= tie_rate < 1.0:
        raise ValueError("tie_rate must be in [0, 1)")
    return math.ceil(non_tie_pairs / (1.0 - tie_rate))


def all_routes_cost(*, per_route_power: float, routes: int = 5,
                    q: float = SESOI_Q, alpha: float = ALPHA,
                    tie_rates=(0.0, 0.2, 0.4, 0.5)) -> dict:
    """Non-tie pairs and scenarios per route for a target per-route power.

    The conjunction's own success probability is *not* the per-route power to the
    power of the route count, because the routes share scenarios.  The independent
    figure is reported as a bound on how optimistic a naive reading would be, and
    the correlated reality lies above it.
    """
    plan = non_ties_needed(q=q, power=per_route_power, alpha=alpha)
    if plan["status"] != "found":
        return {"status": "search_limit_reached", "per_route_power": per_route_power}
    table = [
        {
            "tie_rate": rate,
            "scenarios_per_route": total_scenarios(
                non_tie_pairs=plan["non_tie_pairs"], tie_rate=rate
            ),
        }
        for rate in tie_rates
    ]
    return {
        "status": "found",
        "per_route_power": per_route_power,
        "q": q,
        "alpha": alpha,
        "non_tie_pairs_per_route": plan["non_tie_pairs"],
        "per_route": table,
        "conjunction_note": (
            f"with {routes} routes at {per_route_power:.0%} per-route power the "
            f"independent-route success probability would be about "
            f"{per_route_power ** routes:.2f}; routes share scenarios so the truth "
            "is higher, but the conjunction is materially harder than one route"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sesoi-q", type=float, default=SESOI_Q)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--out", default="results_v4/ordinal_plan.json")
    args = parser.parse_args()

    print("ORDINAL DESIGN PLAN (exact paired sign test, standard library only)")
    print("=" * 78)
    print(f"  SESOI q = {args.sesoi_q}  (share of CHANGING scenarios moving the "
          "predicted way)")
    print(f"  alpha   = {args.alpha} (two-sided, per route)")
    print()
    print("  non-tie pairs needed for 80% power, by assumed q:")
    for q in (0.60, 0.65, 0.70, 0.75, 0.80):
        plan = non_ties_needed(q=q, power=0.8, alpha=args.alpha)
        print(f"    q={q:.2f} -> {plan['non_tie_pairs']:4d} non-tie pairs "
              f"(power {plan['achieved_power']:.3f})")

    print()
    print(f"  scenarios needed, SESOI q={args.sesoi_q}, by measured tie rate:")
    base = non_ties_needed(q=args.sesoi_q, power=0.8, alpha=args.alpha)
    for tie in (0.0, 0.2, 0.3, 0.4, 0.5):
        print(f"    tie={tie:.0%} -> {total_scenarios(non_tie_pairs=base['non_tie_pairs'], tie_rate=tie):4d} "
              "scenarios per route")

    print()
    print("  cost of the all-5 conjunction (per-route power -> scenarios/route):")
    for power in (0.8, 0.9, 0.95):
        plan = all_routes_cost(per_route_power=power, alpha=args.alpha)
        if plan["status"] != "found":
            print(f"    {power:.0%}: search limit reached")
            continue
        row = {entry["tie_rate"]: entry["scenarios_per_route"]
               for entry in plan["per_route"]}
        print(f"    {power:.0%} -> {plan['non_tie_pairs_per_route']:3d} non-tie "
              f"pairs; scenarios/route at tie 0%/20%/40%/50% = "
              f"{row[0.0]}/{row[0.2]}/{row[0.4]}/{row[0.5]}")
    plan = all_routes_cost(per_route_power=0.8, alpha=args.alpha)
    print(f"    {plan['conjunction_note']}")

    report = {
        "sesoi_q": args.sesoi_q,
        "alpha": args.alpha,
        "non_ties_by_q": {
            f"{q:.2f}": non_ties_needed(q=q, power=0.8, alpha=args.alpha)
            for q in (0.60, 0.65, 0.70, 0.75, 0.80)
        },
        "scenarios_by_tie_rate": {
            f"{tie:.2f}": total_scenarios(
                non_tie_pairs=base["non_tie_pairs"], tie_rate=tie
            )
            for tie in (0.0, 0.2, 0.3, 0.4, 0.5)
        },
        "all_routes": {
            f"{power:.2f}": all_routes_cost(per_route_power=power, alpha=args.alpha)
            for power in (0.8, 0.9, 0.95)
        },
        "note": (
            "Planning arithmetic only; no model calls. Figures are minimums "
            "conditional on the stated q and tie rate, and the tie rate must be "
            "measured on data independent of the confirmatory set."
        ),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
