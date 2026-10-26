"""Frozen scope for this study, enforced in code rather than in prose.

The model set is fixed by the requester and must not be extended.  That is a
resource constraint, but it is also a methodological one, and the methodological
half is why it is enforced here instead of being written down as a note.

Adding a model **after** seeing results is selective sampling over models.  If the
adoption rate turns out to be near ceiling for the three routes in scope, then
choosing additional routes to display would be choosing routes until the picture
looks interesting.  The same logic the contract applies to distractors applies to
models: do not widen the set in response to the result.

So the set is a constant, the runner refuses anything outside it, and widening it
is an explicit edit to this file that shows up in the spec digest — not a flag
passed at the command line.

That rule was applied on 2026-09-17, when the requester authorised three
additional routes.  The edit is recorded here rather than left implicit, and the
separation it created is the point: ``MODELLED_ROUTES`` is what may be run at all,
while ``SUBJECT_ROUTES`` is who a given experiment's data comes from.  Widening
permission before a run is fine; changing the subjects after seeing results would
not be, and keeping the two sets distinct is what makes that checkable.
"""

from __future__ import annotations

#: Every route the requester has authorised.  Order is the reporting order.
#:
#: The set was widened on 2026-09-17 at the requester's instruction (three new
#: routes), so this is a permitted superset rather than the subject set of any one
#: experiment.  Keeping permission and subjects separate matters: a run must draw
#: its subjects from a set fixed *before* the data, and widening the set after
#: seeing results is selective sampling over models.
MODELLED_ROUTES = (
    "hy3",
    "Qwen3.5-397B-A17B",
    "DeepSeek-V4.1-Flash",
    "DeepSeek-V4-Flash",
    "qwen3-vl-plus",
    "gpt-5.6-luna",
)

#: The subject set for the expanded experiments.
#:
#: The requester named three new routes plus "the previous two", read as hy3 and
#: Qwen3.5-397B-A17B: the two reported on in the most recent round, and the two
#: with no runtime clock signal, hence the clean subjects for a study that
#: manipulates world time.
#:
#: ``gpt-5.6-luna`` stays *permitted* — it is still a modelled route and its
#: delivery-axis behaviour is valid — but is not a subject of these runs, because
#: it carries a runtime calendar date the harness does not send
#: (``FINDING_clock_source_conclusion.md``), which makes its elapsed-time results
#: uninterpretable.  Stated explicitly so the exclusion is a recorded decision
#: rather than a silent omission.
SUBJECT_ROUTES = (
    "hy3",
    "Qwen3.5-397B-A17B",
    "DeepSeek-V4.1-Flash",
    "DeepSeek-V4-Flash",
    "qwen3-vl-plus",
)

# Roles, for reporting.
ROUTE_ROLES = {
    "hy3": "primary",
    "Qwen3.5-397B-A17B": "contrast",
    "DeepSeek-V4.1-Flash": "contrast",
    "DeepSeek-V4-Flash": "contrast",
    "qwen3-vl-plus": "contrast",
    "gpt-5.6-luna": "excluded_from_temporal_axis",
}

# Time points for the real-model runs.  15d is the common point all three routes
# share; 1d and 61d exist for hy3 only.
COMMON_ELAPSED = "15d"
PRIMARY_ROUTE_ELAPSED = ("1d", "15d", "61d")

# The gateway advertises more routes than this.  They are deliberately out of
# scope; see the module docstring for why that is not a limitation to be fixed.
OUT_OF_SCOPE_ROUTES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "gpt-6-astra",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.5",
    "Kimi-K3",
    "Kimi-K2.7-Code",
    "DeepSeek-V4-Pro",
    "qwen3-vl",
    "GLM-5.3",
    "MiniMax-M2.7",
)


class OutOfScopeRoute(Exception):
    """Raised when a route outside the frozen set is requested."""


def assert_in_scope(models) -> tuple:
    """Validate a requested model list against the frozen set.

    Fail-closed: an unknown route is refused rather than allowed through with a
    warning, because a run with an unlisted model would be indistinguishable in
    the results table from one with a listed model.
    """
    requested = tuple(models)
    if not requested:
        raise OutOfScopeRoute("no models requested")
    outside = [m for m in requested if m not in MODELLED_ROUTES]
    if outside:
        raise OutOfScopeRoute(
            f"route(s) outside the frozen scope: {outside}. The modelled set is "
            f"{list(MODELLED_ROUTES)}. Widening it is an edit to "
            f"tools/scope.py, not a command-line flag: adding a model after seeing "
            f"results is selective sampling over models."
        )
    return requested


def in_scope(models) -> bool:
    return all(m in MODELLED_ROUTES for m in models)
