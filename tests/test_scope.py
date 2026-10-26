"""The frozen model set.

The set is fixed by the requester.  These tests exist because the temptation to
widen it arrives exactly when it is most damaging: after seeing that the result is
near a ceiling and wanting a route that shows something else.  A test that fails on
an unlisted route makes that temptation a deliberate code edit.
"""

from __future__ import annotations

import pytest

from tools.scope import (
    COMMON_ELAPSED,
    MODELLED_ROUTES,
    OutOfScopeRoute,
    PRIMARY_ROUTE_ELAPSED,
    assert_in_scope,
    in_scope,
)


def test_the_declared_set_is_the_requested_one():
    """Pinned so widening the set is a deliberate, visible edit.

    The set was widened on 2026-09-17 at the requester's instruction, which is why
    this test names the members rather than a count: adding a route without
    updating the permission record should fail here.
    """
    assert set(MODELLED_ROUTES) == {
        "hy3",
        "Qwen3.5-397B-A17B",
        "DeepSeek-V4.1-Flash",
        "DeepSeek-V4-Flash",
        "qwen3-vl-plus",
        "gpt-5.6-luna",
    }


def test_subjects_are_a_subset_of_permitted_routes():
    """Permission and subjects are separate sets, and the nesting is the point.

    What may be run at all is not the same as whose data a given experiment uses:
    gpt-5.6-luna stays permitted while being excluded from the temporal axis,
    because it carries a runtime date the harness does not send.
    """
    from tools.scope import SUBJECT_ROUTES

    assert set(SUBJECT_ROUTES) < set(MODELLED_ROUTES)
    assert "gpt-5.6-luna" not in SUBJECT_ROUTES
    assert assert_in_scope(SUBJECT_ROUTES) == SUBJECT_ROUTES


def test_in_scope_routes_pass():
    assert assert_in_scope(MODELLED_ROUTES) == MODELLED_ROUTES
    assert in_scope(["hy3"]) is True


@pytest.mark.parametrize(
    "route",
    ["gpt-6-astra", "claude-opus-5", "Kimi-K3", "DeepSeek-V4-Pro", "GLM-5.3",
     "gpt-5.6-sol", "MiniMax-M2.7"],
)
def test_advertised_but_out_of_scope_routes_are_refused(route):
    with pytest.raises(OutOfScopeRoute, match="outside the frozen scope"):
        assert_in_scope([route])


def test_a_mixed_list_is_refused_entirely():
    """Partial acceptance would silently produce a table with a stray route."""
    with pytest.raises(OutOfScopeRoute):
        assert_in_scope(["hy3", "gpt-6-astra"])


def test_empty_selection_is_refused():
    with pytest.raises(OutOfScopeRoute, match="no models"):
        assert_in_scope([])


def test_the_error_explains_why_scope_matters():
    with pytest.raises(OutOfScopeRoute, match="selective sampling over models"):
        assert_in_scope(["gpt-6-astra"])


def test_primary_route_has_three_time_points_and_the_rest_one():
    assert len(PRIMARY_ROUTE_ELAPSED) == 3
    assert COMMON_ELAPSED in PRIMARY_ROUTE_ELAPSED
