"""Deterministic fact layer.

Why this exists.  The blind judge is good at the *semantic* fields — was the user's
claim treated as having happened, was the discrepancy flagged — and unreliable at
the *factual* one.  Re-judging with the same judge produced five false negatives on
ten answers that stated the rescheduled date in plain text.  That is the same
lesson this project already recorded once: a factual judgement handed to a model
drifts, and its errors land exactly on the cell the study cares about.

So the factual dimension is computed here, in code, and the judge is asked only
about meaning.  Two consequences:

* the factual numbers are reproducible without any API call, and can be
  re-derived by a reviewer from the recorded answers;
* agreement between the code and the judge becomes a *measurement* of the judge
  rather than an assumption, and disagreement is reported instead of averaged away.

The rules are deliberately conservative.  A date is counted as stated only if the
current official date appears in a recognizable form; a false negative (missing a
stated date) is the error that matters, so the forms list errs toward matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CURRENT_DATE = "2026-06-24"
SUPERSEDED_DATE = "2026-04-20"


def _normalise(text: str) -> str:
    """Strip spacing and formatting so date forms compare reliably."""
    out = text
    for ch in (" ", "\u3000", "\n", "\t", "*", "`", "，", ",", "(", ")", "（", "）"):
        out = out.replace(ch, "")
    return out


def _forms(iso: str) -> tuple:
    """Recognizable renderings of an ISO date, longest first."""
    year, month, day = iso.split("-")
    month, day = str(int(month)), str(int(day))
    zh = ("一月", "二月", "三月", "四月", "五月", "六月",
          "七月", "八月", "九月", "十月", "十一月", "十二月")
    return (
        f"{year}年{month}月{day}日",
        f"{year}年{month}月{day}号",
        f"{year}-{month}-{day}",
        f"{year}/{month}/{day}",
        f"{month}月{day}日",
        f"{month}月{day}号",
        f"{zh[int(month) - 1]}{day}日",
        f"{day}/{month}/{year}",
    )


def mentions_current_date(answer: str) -> bool:
    """Whether the reply states the current official date."""
    if not answer:
        return False
    text = _normalise(answer)
    return any(_normalise(form) in text for form in _forms(CURRENT_DATE))


def mentions_superseded_date(answer: str) -> bool:
    """Whether the reply mentions the superseded date at all.

    Mentioning it is not a failure: the competent reply says "it was X, it is now
    Y".  This is recorded so that the failure mode can be distinguished — claiming
    the old date is current — rather than collapsing both into one flag.
    """
    if not answer:
        return False
    text = _normalise(answer)
    return any(_normalise(form) in text for form in _forms(SUPERSEDED_DATE))


_OLD_AS_CURRENT = re.compile(
    r"(现在|目前|你的|最新的|改到|定在|安排在|还有)\s*[^。\n]{0,12}"
    r"(4\s*月\s*20|四月二十)",
)


def asserts_superseded_as_current(answer: str) -> bool:
    """Does the reply entertain the superseded date while never stating the current one?

    The rule is deliberately structural rather than clever: the superseded date
    appears and the operative date does not.  That combination is what makes the
    reply wrong, and it does not depend on guessing the writer's framing.

    A reply that gives both dates is correct ("it was X, it is now Y") and is not
    flagged.  A reply that admits it does not know the date mentions neither and is
    not flagged.  ``_OLD_AS_CURRENT`` records the subset where the old date is
    asserted with an explicit "now"/"still" framing, which is the clearest form;
    the structural rule is the reported one because it has fewer false negatives.

    A regex cannot settle this beyond doubt, so it is described as a heuristic
    wherever it appears, and the count is reported separately from the adoption
    label rather than folded into it.  Erring toward not accusing is deliberate: a
    false claim that a model invented a date is worse than a miss.
    """
    if not answer:
        return False
    if mentions_current_date(answer):
        # A reply that gives both dates has not substituted the old one.
        return False
    return mentions_superseded_date(answer)


@dataclass(frozen=True)
class FactVerdict:
    current_date_stated: bool
    superseded_date_mentioned: bool
    superseded_asserted_as_current: bool

    def to_json(self) -> dict:
        return {
            "current_date_stated": self.current_date_stated,
            "superseded_date_mentioned": self.superseded_date_mentioned,
            "superseded_asserted_as_current": self.superseded_asserted_as_current,
            "method": "deterministic",
        }


def check(answer: str) -> FactVerdict:
    return FactVerdict(
        current_date_stated=mentions_current_date(answer),
        superseded_date_mentioned=mentions_superseded_date(answer),
        superseded_asserted_as_current=asserts_superseded_as_current(answer),
    )
