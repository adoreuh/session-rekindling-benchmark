"""The deterministic fact layer, and why it must not be handed to a judge.

The tests below pin the behaviour that matters: the date detector must not miss a
stated date, because a false negative there lands on the cell the study is about,
and it must not be so permissive that an invented date counts as a statement.

They also record the negative result that motivated this module.  Re-judging with
the same blind judge produced five false negatives on ten answers that stated the
rescheduled date in plain text, so the factual field is computed in code and the
judge is asked only about meaning.
"""

from __future__ import annotations

import pytest

from tools.facts import (
    CURRENT_DATE,
    SUPERSEDED_DATE,
    asserts_superseded_as_current,
    check,
    mentions_current_date,
    mentions_superseded_date,
)

REAL_CORRECT = "小林，根据签证面试的变更通知，原定 4 月 20 日的面试已调整至 2026 年 6 月 24 日上午 9:00。"
REAL_SHORT = "小林，签证面试已改期为 **2026年6月24日上午9:00**。请提前30分钟到达。"
REAL_CORRECT_ONLY = "小林，你的面试时间是 6 月 24 日上午九点。"
REAL_OMIT_INVENTED = "你还有4月20日的签证面试，还有其他的机会。"
REAL_OMIT_HONEST = "我这边没有查到确切的面试日期，建议你直接核对最新的通知。"


@pytest.mark.parametrize("text", [REAL_CORRECT, REAL_SHORT, REAL_CORRECT_ONLY])
def test_stated_current_date_is_detected(text):
    """A false negative here is the error that matters, so all real forms match."""
    assert mentions_current_date(text) is True


def test_invented_old_date_is_not_a_statement_of_the_current_date():
    assert mentions_current_date(REAL_OMIT_INVENTED) is False


def test_honest_unknown_is_not_a_statement():
    assert mentions_current_date(REAL_OMIT_HONEST) is False


def test_empty_answer_is_not_a_statement():
    assert mentions_current_date("") is False


def test_superseded_date_is_detected_separately():
    """Mentioning the old date is not a failure: 'was X, is now Y' is correct."""
    verdict = check(REAL_CORRECT)
    assert verdict.superseded_date_mentioned is True
    assert verdict.current_date_stated is True
    # Giving both dates cannot be claiming the old one is current.
    assert verdict.superseded_asserted_as_current is False


def test_old_date_asserted_as_current_is_flagged():
    assert mentions_current_date(REAL_OMIT_INVENTED) is False
    assert asserts_superseded_as_current(REAL_OMIT_INVENTED) is True


def test_old_as_current_requires_the_absence_of_the_current_date():
    assert asserts_superseded_as_current(REAL_CORRECT) is False


def test_verdict_reports_its_method():
    assert check("x").to_json()["method"] == "deterministic"


def test_date_forms_cover_spacing_and_markers():
    """Markdown and spacing must not defeat a detector that reads real answers."""
    for text in (
        "面试时间是 2026 年 6 月 24 日",
        "面试时间是**2026 年 6 月 24 日**",
        "面试时间是 2026-6-24",
        "面试时间是 6月24日",
        "面试时间是 6 月 24 号",
    ):
        assert mentions_current_date(text) is True, text


def test_unrelated_dates_do_not_match():
    assert mentions_current_date("面试时间是 2026 年 7 月 1 日") is False
    assert mentions_superseded_date("面试时间是 2026 年 7 月 1 日") is False


def test_constants_match_the_scenario():
    from tools import scenario_ewm

    assert CURRENT_DATE == scenario_ewm.CURRENT_DATE
    assert SUPERSEDED_DATE == scenario_ewm.ORIGINAL_DATE
