"""Tests for the clock-source probe's classification helpers.

Both helpers here had a real bug that produced a wrong verdict, so they are
pinned by test rather than left to inspection.

The first bug: the precision classifier read ``2026-09-17T00:00Z`` as a concrete
midnight, and reported "the route has a live clock" — the exact opposite of what
the answer showed.  A model with only a date fills the time field to satisfy the
requested format.

The second bug: the summary counted *any* date as evidence of a date source, so a
route that computed a self-consistent arithmetic series from a wrong base date
was marked as having a clock.  Guessing the format is not reading a clock, so the
test is whether the date equals the real one.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.clock_source_probe import (  # noqa: E402
    classify_precision,
    states_the_real_date,
)

REAL = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- precision

def test_a_literal_placeholder_is_not_a_time():
    assert classify_precision("2026-09-17 --:-- CST") == "placeholder"
    assert classify_precision("2026-09-17T??:??Z") == "placeholder"


def test_midnight_placeholder_is_not_a_concrete_time():
    """The regression that mattered: this was read as a real midnight.

    A route answering ``00:00Z`` to "the time to the minute" is filling a field,
    not reporting a clock, and calling that precision would invert the finding.
    """
    assert classify_precision("2026-09-17T00:00Z") != "concrete"


def test_a_real_time_of_day_is_concrete():
    assert classify_precision("2026-09-17T14:37:00+08:00") == "concrete"
    assert classify_precision("17:05 CST") == "concrete"


def test_a_refusal_is_not_precision():
    assert classify_precision("Unable to determine the current UTC time.") == "none"
    assert classify_precision("") == "none"


# ------------------------------------------------------- real-date detection

def test_only_the_real_date_counts():
    """Guessing the format is not having a source.

    ``2024-06-14`` is a self-consistent base for a three-date series and is two
    years off.  Treating it as evidence would attribute a leak to a route that is
    merely completing the requested format.
    """
    assert states_the_real_date("2026-09-17", real=REAL)
    assert not states_the_real_date("2024-06-14 2024-06-17 2024-06-11", real=REAL)


def test_the_date_is_recognised_in_the_renderings_answers_actually_use():
    """Missing a rendering makes a route with a signal look clean."""
    for text in (
        "2026-09-17",
        "2026-09-17T00:00Z",
        "September 17, 2026",
        "Sep 17, 2026",
        "17 September 2026",
        "9 月 17 日",
        "2026年9月17日",
    ):
        assert states_the_real_date(text, real=REAL), text


def test_absences_and_other_dates_are_not_the_real_date():
    for text in ("Unknown", "NONE", "I cannot know.", "1999-01-01", "",
                 "September 16, 2026", "2026-09-18"):
        assert not states_the_real_date(text, real=REAL), text


def test_a_date_inside_a_longer_answer_is_found():
    body = ("I cannot determine the clock time, but the current date reads as "
            "**2026-09-17** in the context I was given.")
    assert states_the_real_date(body, real=REAL)
