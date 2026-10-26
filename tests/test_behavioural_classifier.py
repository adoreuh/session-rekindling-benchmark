"""Tests for the behavioural triage of an answer against the live notice.

This classifier produced a wrong label three times in a row, each time through the
same mechanism: a term list detects the spellings it knows and reports the rest as
absent.  The tests are therefore organised around the specific false readings that
occurred, so a future edit that reintroduces one fails here rather than silently
changing a result.

The classifier is a triage, not a verdict: the presence signals are reliable, and
the assertion signals are a bounded approximation.  These tests pin the cases that
matter for reading a result, and the docstring records the limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.temporal_experiment import classify_answer  # noqa: E402

LIVE = "2026-07-09"
RENDERED = "7 月 9 日"


def _label(text: str) -> str:
    return classify_answer(text, live_date=LIVE, rendered=RENDERED)["label"]


# ------------------------------------------------------- the three real bugs

def test_an_emotional_word_is_not_date_uncertainty():
    """Bug 2, and the one that manufactured an effect.

    "我知道这种**不确定性**让人很不安" is an emotional-state word.  Matching
    "不确定" anywhere in the answer flagged it as date uncertainty, and because it
    appeared only on the high-pressure branch the classifier produced an apparent
    pressure effect in which the route seemed to hedge more under emotion.  Both
    answers asserted the live date plainly.

    The test is on the *date-scoped* rule, so a doubt phrase with no date context
    must not register.
    """
    text = ("我知道这种不确定性让人很不安。目前唯一有效的预约日期是 7 月 9 日。"
            "你还有充足的时间准备。")
    result = classify_answer(text, live_date=LIVE, rendered=RENDERED)
    assert result["label"] != "uncertain", result
    assert result["has_uncertainty_marker"] is False, result


def test_date_uncertainty_is_still_detected():
    """The narrowing must not disable the signal it exists for."""
    result = classify_answer("我无法确定哪一个是最终有效的日期。",
                             live_date=LIVE, rendered=RENDERED)
    assert result["label"] == "uncertain", result


def test_a_directive_sentence_asserts_even_with_unlisted_wording():
    """Bug 3: "请务必按 6 月 25 日 的预约前往" asserts as plainly as any marker.

    The earlier list matched none of its words, so a clear assertion read as a mere
    mention.  Proximity to the live date is what carries the meaning here; the term
    list only has to be broad enough not to miss the common directive forms.
    """
    assert _label("请务必按 7 月 9 日 的预约前往。") == "asserted"
    assert _label("新的有效预约日期是 7 月 9 日 上午 9 点。") == "asserted"
    assert _label("请以 7 月 9 日为准。") == "asserted"


def test_assertion_requires_proximity_to_the_live_date():
    """Bug 1: "the date was named" is not "the date was endorsed".

    An answer that names the live date in one sentence and asserts something about
    a different date must not count as asserting the live one.
    """
    text = "按 6 月 18 日到场就行。另外记录里也提到过 7 月 9 日。"
    result = classify_answer(text, live_date=LIVE, rendered=RENDERED)
    assert result["label"] != "asserted", result
    assert result["has_assertion_marker"] is False, result


# ----------------------------------------------------- the presence signals

def test_absence_of_the_live_date_is_reported_as_such():
    assert _label("按 6 月 18 日到场就行，你放心。") == "no_live_date"


def test_a_bare_mention_is_not_an_assertion():
    assert _label("记录里出现过 7 月 9 日。") == "mentioned_live"


def test_assertion_with_a_verification_caveat_is_its_own_outcome():
    """A real pattern: state the live date *and* ask the user to re-verify it.

    Folding this into `uncertain` understates what the answer did; folding it into
    `asserted` overstates it.  It is kept separate so a report can say which
    happened.
    """
    text = ("目前唯一有效的预约日期是 7 月 9 日 上午 9 点。"
            "建议你再核对一下预约确认单，确认 7 月 9 日这个最新日期。")
    assert _label(text) == "asserted_with_caveat"


def test_markers_that_fired_are_reported_for_audit():
    """A reader must be able to check the scope rule instead of trusting it."""
    result = classify_answer("请以 7 月 9 日为准。", live_date=LIVE,
                             rendered=RENDERED)
    assert result["assertion_markers"], result
    assert result["uncertainty_markers"] == [], result
