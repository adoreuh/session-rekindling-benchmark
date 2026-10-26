"""Tests for the ordinal design planner.

The planner's value is that its numbers can be frozen before a run, so the tests
are about the arithmetic being exact rather than approximate, and about the
assumptions being visible in the output rather than implied by it.

The reference values come from an independently specified table (non-tie pairs
needed at 80% power, alpha 0.025 two-sided: 240/104/58/37/24 for q =
0.60/0.65/0.70/0.75/0.80).  Reproducing them is a check on the implementation, not
a restatement of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ordinal_plan import (  # noqa: E402
    all_routes_cost,
    critical_value,
    non_ties_needed,
    sign_test_power,
    total_scenarios,
)


def test_non_tie_requirements_match_the_reference_table():
    """Five independent values, each a separate check on the search."""
    reference = {0.60: 240, 0.65: 104, 0.70: 58, 0.75: 37, 0.80: 24}
    for q, expected in reference.items():
        plan = non_ties_needed(q=q, power=0.8, alpha=0.025)
        assert plan["status"] == "found", q
        assert plan["non_tie_pairs"] == expected, (q, plan)
        assert plan["achieved_power"] >= 0.8


def test_critical_value_is_the_smallest_rejecting_count():
    """The boundary must be the first count whose doubled tail fits alpha.

    Getting this wrong by one is invisible in a power figure and moves every sample
    size, so it is pinned directly.
    """
    import math

    for n in (24, 37, 58, 104):
        c = critical_value(n, alpha=0.025)
        tail_at_c = math.fsum(math.comb(n, i) for i in range(c, n + 1)) / 2 ** n
        tail_before = math.fsum(math.comb(n, i) for i in range(c - 1, n + 1)) / 2 ** n
        assert 2 * tail_at_c <= 0.025
        assert 2 * tail_before > 0.025


def test_power_is_zero_at_a_coin_flip_and_rises_with_q():
    """A test that cannot distinguish the null from the alternative is broken."""
    assert sign_test_power(58, 0.5) == pytest.approx(0.025, abs=0.01)
    assert sign_test_power(58, 0.6) < sign_test_power(58, 0.7)
    assert sign_test_power(58, 0.7) < sign_test_power(58, 0.8)


def test_power_increases_with_n():
    assert sign_test_power(20, 0.7) < sign_test_power(58, 0.7)
    assert sign_test_power(58, 0.7) < sign_test_power(200, 0.7)


def test_tie_rate_inflates_the_scenario_count():
    """Power depends on non-ties, so a high tie rate costs scenarios."""
    non_ties = 58
    assert total_scenarios(non_tie_pairs=non_ties, tie_rate=0.0) == 58
    assert total_scenarios(non_tie_pairs=non_ties, tie_rate=0.2) == 73
    assert total_scenarios(non_tie_pairs=non_ties, tie_rate=0.4) == 97
    assert total_scenarios(non_tie_pairs=non_ties, tie_rate=0.5) == 116
    # And it is monotone, so no tie rate is cheaper than a smaller one.
    counts = [total_scenarios(non_tie_pairs=non_ties, tie_rate=r)
              for r in (0.0, 0.1, 0.2, 0.3, 0.4)]
    assert counts == sorted(counts)


def test_a_perfect_tie_rate_is_rejected():
    """A design where nothing changes cannot be sized."""
    with pytest.raises(ValueError):
        total_scenarios(non_tie_pairs=10, tie_rate=1.0)


def test_the_all_routes_plan_discloses_that_the_conjunction_is_harder():
    """Per-route power is not conjunction power, and the output must say so."""
    plan = all_routes_cost(per_route_power=0.8, alpha=0.025)
    assert plan["status"] == "found"
    assert plan["non_tie_pairs_per_route"] == 58
    assert "conjunction is materially harder" in plan["conjunction_note"]
    # The higher per-route power must cost more, not the same.
    higher = all_routes_cost(per_route_power=0.95, alpha=0.025)
    assert higher["non_tie_pairs_per_route"] > plan["non_tie_pairs_per_route"]


def test_q_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        sign_test_power(50, 0.0)
    with pytest.raises(ValueError):
        sign_test_power(50, 1.0)
