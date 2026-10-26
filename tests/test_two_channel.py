"""Tests for the two-channel paired analysis.

The analysis exists because the delivery contrast's primary outcome sits at floor
and ceiling, leaving pressure no room to act on it — while the same runs *do* show a
large pressure effect on the affect channel.  So the tests are about the two things
that make that comparison trustworthy: the channels stay separate, and the pairing is
by scenario rather than by branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.two_channel import (  # noqa: E402
    AFFECT_CHANNEL,
    FACT_CHANNEL,
    analyse,
    paired_table,
    rates,
)


def _cells(rows):
    """Build the (route, scenario, arm, field) -> {pressure: bool} mapping."""
    cells = {}
    for route, scenario, arm, field, low, high in rows:
        cells[(route, scenario, arm, field)] = {"low": low, "high": high}
    return cells


def test_the_two_channels_are_disjoint():
    """A field may not be read as both channels, or the comparison is circular."""
    assert not set(FACT_CHANNEL) & set(AFFECT_CHANNEL)


def test_pairing_is_by_scenario_so_each_scenario_contributes_once():
    cells = _cells([
        ("r", "s1", "oracle", "affect_validated", False, True),
        ("r", "s2", "oracle", "affect_validated", False, True),
    ])
    row = paired_table(cells, arm="oracle", field="affect_validated")
    assert row["pairs"] == 2
    assert row["increased_under_pressure"] == 2
    assert row["decreased_under_pressure"] == 0


def test_only_the_requested_arm_and_field_are_counted():
    cells = _cells([
        ("r", "s1", "oracle", "affect_validated", False, True),
        ("r", "s1", "omit", "affect_validated", False, False),
        ("r", "s1", "oracle", "false_premise_endorsed", True, True),
    ])
    affect = paired_table(cells, arm="oracle", field="affect_validated")
    assert affect["pairs"] == 1
    fact = paired_table(cells, arm="oracle", field="false_premise_endorsed")
    assert fact["pairs"] == 1
    # The omit cell must not leak into the oracle table.
    assert affect["increased_under_pressure"] == 1


def test_a_scenario_missing_one_pressure_is_skipped_not_defaulted():
    """An unpaired scenario is absent, not a zero.

    Counting a missing pressure level as "did not change" would dilute the effect
    with pairs that contributed no observation.
    """
    cells = {
        ("r", "s1", "oracle", "affect_validated"): {"low": False, "high": True},
        ("r", "s2", "oracle", "affect_validated"): {"low": False},
    }
    row = paired_table(cells, arm="oracle", field="affect_validated")
    assert row["pairs"] == 1


def test_the_direction_is_reported_not_assumed():
    """Pressure moving an outcome down must be visible as a separate count."""
    cells = _cells([
        ("r", "s1", "oracle", "affect_validated", True, False),
        ("r", "s2", "oracle", "affect_validated", False, True),
    ])
    row = paired_table(cells, arm="oracle", field="affect_validated")
    assert row["increased_under_pressure"] == 1
    assert row["decreased_under_pressure"] == 1


def test_marginal_rates_are_per_arm_and_pressure():
    cells = _cells([
        ("r", "s1", "oracle", "affect_validated", False, True),
        ("r", "s2", "oracle", "affect_validated", False, False),
    ])
    table = rates(cells, field="affect_validated")
    assert table["oracle|low"]["yes"] == 0
    assert table["oracle|low"]["n"] == 2
    assert table["oracle|high"]["yes"] == 1


def test_analysis_separates_the_channels_and_gates_the_primary_one():
    cells = _cells([
        ("a", "s1", "oracle", "affect_validated", False, True),
        ("a", "s1", "oracle", "false_premise_endorsed", True, True),
    ])
    report = analyse(cells)
    affect = report["channels"]["affect"]["fields"]["affect_validated"]["oracle"]
    fact = report["channels"]["fact"]["fields"][
        "false_premise_endorsed"]["oracle"]
    # The affect channel moved; the fact channel did not.
    assert affect["increased_under_pressure"] == 1
    assert fact["increased_under_pressure"] == 0
