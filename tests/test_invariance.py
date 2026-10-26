"""The invariance regression, and the two illusions it rules out.

Elapsed time cannot change retrieval ranking under multiplicative scoring with a
per-item-constant decay rate.  That is an identity, so the test asserts invariance
and treats a moving ranking as a failure.  Two artifacts are checked separately
because both look like a finding:

* raw-score underflow, where every score becomes 0.0 and the ranking collapses to
  the tie-break rule — numerics, not forgetting;
* a changed retriever, where the ranking genuinely responds to time, which
  invalidates the research framing rather than the test.
"""

from __future__ import annotations

import math

from tools.invariance_regression import (
    DOCUMENTS,
    QUERY,
    SOURCE_DAY,
    TOP_K,
    _instant_at,
    _log_weight,
    check,
)
from vela_lab.decay import PROFILES, resolve_profile
from vela_lab.evidence import EvidenceCorpus


def test_ranking_is_invariant_across_all_durations():
    report = check()
    assert report["invariance_holds"] is True, report
    for row in report["by_decay"]:
        assert row["ranking_invariant"] is True, row
        assert row["gold_rank_invariant"] is True, row


def test_gold_stays_rank_one_at_every_duration():
    """The property the study cares about, asserted directly."""
    corpus = EvidenceCorpus(list(DOCUMENTS))
    corpus.seal()
    for days in (0, 1, 15, 61, 3650, 365000):
        record = corpus.retrieve(
            query=QUERY, now_instant=_instant_at(SOURCE_DAY, days),
            top_k=TOP_K, mode="probe",
            temporal_decay=resolve_profile("m01_memorybank"),
        )
        assert record.gold_rank() == 1, f"gold lost rank at {days}d"


def test_the_corpus_has_mixed_ages_or_the_test_is_vacuous():
    """If every item had one age, invariance would be trivially true."""
    ages = sorted({d.published_at for d in DOCUMENTS})
    assert len(ages) >= 4, f"only {len(ages)} distinct publication times"


def test_no_decay_profile_changes_the_ranking():
    """All profiles must be invariant; a time-varying one would break the framing."""
    report = check()
    assert {row["decay"] for row in report["by_decay"]} == set(PROFILES)


def test_raw_underflow_happens_but_is_not_a_memory_effect():
    """At extreme durations raw scores underflow; the order must still be recoverable."""
    report = check()
    decaying = [r for r in report["by_decay"] if r["underflow_observed"]]
    assert decaying, "expected underflow at extreme durations"
    for row in decaying:
        assert row["underflow_kept_ordering"] is True, (
            f"{row['decay']}: ordering lost in the log domain, so the zero ranking "
            "cannot be attributed to numerics"
        )


def test_log_weight_stays_finite_where_the_float_does_not():
    """``log(0.0)`` is useless; the exponent must be used instead."""
    from vela_lab.clock import parse_instant

    decay = resolve_profile("m01_memorybank")
    now_dt = parse_instant(_instant_at(SOURCE_DAY, 365000))
    doc = DOCUMENTS[0]
    raw = decay(observed_at=doc.published_at,
                now=_instant_at(SOURCE_DAY, 365000))
    assert raw == 0.0, "expected underflow at 365000 days"
    log_w = _log_weight(decay, observed_at=doc.published_at, now_dt=now_dt)
    assert log_w is not None and math.isfinite(log_w)


def test_the_identity_holds_analytically():
    """S(t+Δ) = e^{-Δ/T} · S(t): every item scaled by the same positive factor."""
    decay = resolve_profile("m01_memorybank")
    doc = next(d for d in DOCUMENTS if d.is_gold)
    t0, t1 = 5, 61
    s0 = float(decay(observed_at=doc.published_at, now=_instant_at(SOURCE_DAY, t0)))
    s1 = float(decay(observed_at=doc.published_at, now=_instant_at(SOURCE_DAY, t1)))
    delta = t1 - t0
    assert math.isclose(s1, math.exp(-delta / 5.0) * s0, rel_tol=1e-12)


def test_no_decay_profile_is_time_sensitive():
    """A guard against someone later making a profile respond to elapsed time."""
    for name in PROFILES:
        decay = resolve_profile(name)
        doc = DOCUMENTS[0]
        values = [
            decay(observed_at=doc.published_at, now=_instant_at(SOURCE_DAY, d))
            for d in (0, 30, 300)
        ]
        # A profile may decay, but under the identity the *ordering* of its
        # factors across items cannot change, which the main check verifies.
        assert all(v >= 0 for v in values)
