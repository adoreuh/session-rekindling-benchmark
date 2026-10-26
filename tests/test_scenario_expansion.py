"""Tests for the expanded scenario pool.

The pool exists to raise the number of independent experimental units without
resorting to re-dating one situation.  So the tests are about the two properties
that make the count meaningful: the kinds are genuinely different situations, and
the instances differ in their particulars while keeping the structure the gates
depend on.

The most important test here is the one that bounds the claim.  Forty instances of
one appointment would give a large N and support no generalisation, so the pool's
own report must state how many event kinds the count spans, and these tests pin that
the two layers stay distinct.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.scenario_expansion import (  # noqa: E402
    EVENT_KINDS,
    INSTANCE_VARIANTS,
    build_profiles,
    check_buildable,
)


def test_kinds_are_distinct_situations_not_respellings():
    """A pool of near-identical events would support no generalisation.

    Checked on the naming itself: distinct short names, distinct long names, and
    distinct offices.  Two kinds sharing all three would be one situation wearing
    two labels.
    """
    referents = [k.referent for k in EVENT_KINDS]
    assert len(set(referents)) == len(referents), "duplicate short names"
    longs = [k.referent_long for k in EVENT_KINDS]
    assert len(set(longs)) == len(longs), "duplicate long names"
    offices = [k.office for k in EVENT_KINDS]
    assert len(set(offices)) == len(offices), "duplicate issuing offices"


def test_the_pool_spans_many_kinds():
    """A lower bound, so shrinking the pool is a visible edit."""
    assert len(EVENT_KINDS) >= 40


def test_profiles_combine_kind_and_instance_without_collision():
    profiles = build_profiles(instances=4)
    assert len(profiles) == len(EVENT_KINDS) * 4
    ids = [p.profile_id for p in profiles]
    assert len(set(ids)) == len(ids)
    # Every kind contributes exactly the requested number of instances.
    from collections import Counter

    counts = Counter(p.profile_id.rsplit("_i", 1)[0] for p in profiles)
    assert set(counts.values()) == {4}


def test_dates_vary_by_kind_as_well_as_by_instance():
    """Otherwise the pool holds one calendar repeated, not many situations.

    The kind shifts the dates too.  Without that, forty kinds share four date
    triples and a reader could reasonably say the study never left one spring.
    """
    profiles = build_profiles(instances=4)
    triples = {(p.update_day, p.old_date, p.new_date) for p in profiles}
    assert len(triples) > len(INSTANCE_VARIANTS) * 2, (
        f"only {len(triples)} distinct date triples across {len(profiles)} "
        "profiles; the kind is not shifting the calendar"
    )


def test_the_new_date_is_always_later_than_the_old_one():
    """A backwards move is a different situation and is not what the pool encodes."""
    for profile in build_profiles(instances=4):
        assert profile.new_date > profile.old_date, profile.profile_id
        assert profile.update_day < profile.old_date, profile.profile_id


def test_every_instance_builds_and_keeps_the_propositions_clean():
    """The check that decides whether an instance is usable at all."""
    problems = check_buildable(
        build_profiles(instances=4), filler_id="routine_reminder_setup"
    )
    assert problems == [], problems


def test_the_long_name_is_usable_as_an_independent_referent():
    """The decisive propositions are built from both spellings, so both must work.

    A kind whose long name is unrelated to its short name is legitimate — the two
    are independent vocabulary — but the contract has to register both, which the
    following checks through the real contract.
    """
    from tools.scenario_factory import build_variant
    from tools.variant_wiring import _contract

    for profile in build_profiles(instances=1):
        variant = build_variant(
            scenario_id=f"t-{profile.profile_id}", profile=profile,
            filler_id="routine_reminder_setup",
        )
        contract = _contract(variant)
        assert profile.referent in contract.event_referents
        assert profile.referent_long in contract.event_referents


# ------------------------------------------------------- date plausibility

def test_no_instance_puts_an_appointment_on_a_weekend_or_fixed_holiday():
    """The confound this rule exists to remove, pinned by test.

    A delivery contrast asks whether withholding evidence changes whether the model
    adopts the user's date.  If that date is a Saturday or a public holiday, a model
    can answer "the office is closed, so going then will not work" — correct, and
    unrelated to evidence.  Seventeen of the first forty scenarios had such a date,
    and the one non-adopting branch out of 316 argued exactly that.
    """
    from tools.scenario_expansion import is_plausible_open_day

    problems = [
        (p.profile_id, p.old_date, p.new_date)
        for p in build_profiles(instances=4)
        if not is_plausible_open_day(p.old_date)
        or not is_plausible_open_day(p.new_date)
    ]
    assert problems == [], problems


def test_the_open_day_rule_is_conservative_and_deterministic():
    """It rules out only what is clearly closed, and does so repeatably."""
    from datetime import date as _date

    from tools.scenario_expansion import (
        is_plausible_open_day,
        next_open_day,
    )

    assert is_plausible_open_day(_date(2026, 4, 20)) is True     # Monday
    assert is_plausible_open_day(_date(2026, 4, 18)) is False    # Saturday
    assert is_plausible_open_day(_date(2026, 4, 19)) is False    # Sunday
    assert is_plausible_open_day(_date(2026, 5, 1)) is False     # Labour Day
    assert is_plausible_open_day(_date(2026, 10, 3)) is False    # National Day

    # Forward shift lands on the next open day, and is idempotent on one.
    assert next_open_day(_date(2026, 4, 18)) == _date(2026, 4, 20)
    assert next_open_day(_date(2026, 4, 20)) == _date(2026, 4, 20)
    # A holiday block is skipped entirely rather than landing inside it.
    assert next_open_day(_date(2026, 5, 1)) == _date(2026, 5, 6)


def test_the_gap_survives_the_open_day_shift():
    """Shifting both dates must not collapse them onto one another.

    A naive shift of only one date, or a fixed offset applied after shifting, could
    make the new date equal the old one — which would remove the world change the
    temporal contrast depends on.
    """
    for profile in build_profiles(instances=4):
        assert profile.new_date > profile.old_date, profile.profile_id
        assert profile.update_day < profile.old_date, profile.profile_id


def test_the_residual_is_named_in_the_module():
    """Movable holidays are not filtered, and the code says so.

    Spring Festival, Qingming, Dragon Boat and Mid-Autumn follow the lunar calendar
    and the State Council's annual announcement, so they cannot be derived
    deterministically here.  A silent gap would let a reader assume the filter is
    complete.
    """
    import inspect

    from tools import scenario_expansion

    source = inspect.getsource(scenario_expansion)
    assert "movable" in source.lower()
    assert "lunar" in source.lower()


# ------------------------------------------- reading facts instead of deriving

def test_operative_date_is_read_from_the_run_not_recomputed():
    """The regression that made the temporal rate collapse to 38%.

    The scenario generator changed after the runs were made: appointment dates now
    shift onto plausible open days.  An analysis that recomputed a date from today's
    generator therefore looked for a value the runs never used, and every answer
    scored as "did not name the operative date".

    The authoritative source is what the run delivered.  This test pins that the
    reader consults the run's own retrieval record, by checking it against real
    recorded evidence rather than against a recomputation.
    """
    import json
    from pathlib import Path as _Path

    from tools.expanded_analysis import _load_runs, _operative_date

    root = _Path("runs_scale_hy3")
    if not root.exists():
        import pytest

        pytest.skip("scale runs are not present in this checkout")

    runs = [r for r in _load_runs(root) if r["kind"] == "temporal"]
    assert runs, "no temporal runs found"
    for run in runs:
        operative = _operative_date(run)
        assert operative, f"{run['profile']}_{run['elapsed']}: no date recovered"

        # And it must equal the date the delivered notice states, read
        # independently from the same log.
        branch_dir = _Path(run["path"]).parent / "branches"
        stated = set()
        for events in branch_dir.glob("*/events.jsonl"):
            for line in events.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") != "memory.retrieval":
                    continue
                for hit in event["data"].get("hits", []):
                    text = hit.get("visible_text") or ""
                    marker = "现改为"
                    at = text.find(marker)
                    if at < 0:
                        continue
                    tail = text[at + len(marker):]
                    digits = []
                    number = ""
                    for char in tail:
                        if char.isdigit():
                            number += char
                        elif number:
                            digits.append(int(number))
                            number = ""
                        if len(digits) == 2:
                            break
                    if len(digits) == 2:
                        stated.add(f"2026-{digits[0]:02d}-{digits[1]:02d}")
        assert operative in stated, (
            f"{run['profile']}_{run['elapsed']}: analysis says {operative}, the "
            f"delivered notices state {sorted(stated)}"
        )
