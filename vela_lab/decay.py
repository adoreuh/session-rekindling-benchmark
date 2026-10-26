"""Temporal decay profiles.

Three named profiles, matching the published implementations this project
compares against.  Each is a pure function of elapsed world time, so retrieval
is reproducible offline and no model call is involved — L1 and L2 are measurable
without an API key.

**m00 / none** is the negative control.  **m01** is MemoryBank's Ebbinghaus
retention.  **m02** is the Generative Agents recency term.

A note carried over from the earlier version of this project, because it is easy
to get wrong in a way that invalidates a conclusion: under a *shared* decay rate
with time elapsing uniformly, multiplicative decay cancels out of the ranking.
If every candidate is multiplied by the same factor, the order is unchanged, so
the temporal weight can fall by orders of magnitude while retrieval returns
exactly the same documents.

That is a property of the arithmetic, not a failure.  It means elapsed time can
be observable in L1 and invisible in L2, and the two must be reported separately
rather than collapsed into one "memory decay" claim.  What *does* change the
ranking is a difference in item age, which is why the recency-competition design
varies the age of the target against a fixed-age competitor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from vela_lab.clock import parse_instant


def _hours_between(observed_at: str, now: str) -> float:
    delta = parse_instant(now) - parse_instant(observed_at)
    return max(0.0, delta.total_seconds() / 3600.0)


def _days_between(observed_at: str, now: str) -> float:
    return _hours_between(observed_at, now) / 24.0


@dataclass(frozen=True)
class DecayProfile:
    """A named decay function plus the parameters that define it."""

    name: str
    description: str
    _fn: Callable[..., float]
    params: dict

    def __call__(self, *, observed_at: str, now: str) -> float:
        return float(self._fn(observed_at=observed_at, now=now))

    def identity(self) -> dict:
        return {"name": self.name, "params": dict(sorted(self.params.items()))}

    def digest_input(self) -> dict:
        return self.identity()


def none_decay(*, observed_at: str, now: str) -> float:
    """Negative control: elapsed time has no effect on the score."""
    return 1.0


def memorybank_decay(*, observed_at: str, now: str, base_days: float = 5.0) -> float:
    """Ebbinghaus retention ``exp(-t / (base_days * strength))``.

    ``strength`` is fixed at 1.0 here.  MemoryBank's own implementation writes the
    retention term as ``math.exp(-t / 5 * S)``, in which the division binds
    before the multiplication, so ``S`` scales the score rather than the time
    constant.  That is reproduced deliberately as ``repo_formula`` in
    :data:`PROFILES` so the discrepancy is measurable rather than silently
    corrected in one direction.
    """
    days = _days_between(observed_at, now)
    return math.exp(-days / base_days)


def memorybank_repo_decay(*, observed_at: str, now: str,
                          strength: float = 1.0) -> float:
    """The upstream expression ``exp(-t/5*S)``, precedence bug preserved."""
    days = _days_between(observed_at, now)
    return math.exp(-days / 5.0 * strength)


def generative_agents_decay(*, observed_at: str, now: str,
                            factor_per_hour: float = 0.995) -> float:
    """Recency term from Park et al. 2023: ``factor ** hours``."""
    return factor_per_hour ** _hours_between(observed_at, now)


PROFILES: dict = {
    "m00_none": DecayProfile(
        name="m00_none",
        description="Negative control: no temporal weighting.",
        _fn=none_decay,
        params={},
    ),
    "m01_memorybank": DecayProfile(
        name="m01_memorybank",
        description="MemoryBank / Ebbinghaus retention, exp(-t / 5 days).",
        _fn=memorybank_decay,
        params={"base_days": 5.0},
    ),
    "m01_memorybank_repo": DecayProfile(
        name="m01_memorybank_repo",
        description="MemoryBank upstream expression exp(-t/5*S) with its "
                    "operator-precedence behaviour preserved.",
        _fn=memorybank_repo_decay,
        params={"strength": 1.0},
    ),
    "m02_generative_agents": DecayProfile(
        name="m02_generative_agents",
        description="Generative Agents recency term, 0.995 ** hours.",
        _fn=generative_agents_decay,
        params={"factor_per_hour": 0.995},
    ),
}


def resolve_profile(name: str) -> DecayProfile:
    if name not in PROFILES:
        raise KeyError(
            f"unknown decay profile {name!r}; available: {sorted(PROFILES)}"
        )
    return PROFILES[name]


def weight_table(profile: DecayProfile, *, observed_at: str,
                 instants: list) -> list:
    """Weights at several instants, for reporting L1 without any model call."""
    return [
        {"now": instant, "weight": profile(observed_at=observed_at, now=instant)}
        for instant in instants
    ]
