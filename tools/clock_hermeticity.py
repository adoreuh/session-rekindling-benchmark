"""Screen a route for real-clock leakage before using it in a temporal study.

Why this exists, and why it is a gate rather than a note.

The harness's virtual clock is the only time source *the harness controls*.  It
cannot control what the serving endpoint adds.  During the v4 pilot,
``gpt-5.6-luna`` wrote in its answer:

    系统当前日期显示为 2026 年 9 月 17 日

2026-09-17 was the real wall-clock date.  The virtual clock said 2026-03-20, and
the recorded request contains exactly one date — the virtual one.  So that date
came from somewhere the harness does not audit.

The consequence is concrete, not theoretical: for that route, "world time
elapsed" is not the only clock in the conversation.  A model that can read the
real date can notice that a scenario set in March is being asked in September,
and every temporal layer (L1 sensitivity, L2 accessibility, the elapsed-time
manipulation itself) is confounded by a signal the design assumed absent.

**What this tool can and cannot establish.**  It can show that a route produces
the real date when the request contains none.  It cannot say whether that is
server-side injection, a tool the endpoint exposes, or the model's own
knowledge — and the remedy differs by cause.  So it reports the *observation* and
refuses to name a mechanism.  A route that passes is not proven hermetic either:
passing means "did not reveal a real date on these probes", which is the same
kind of bounded claim the rest of this project makes.

The probe set is deliberately small and blunt.  A subtler leak — one that only
surfaces when the conversation gives the model a reason to check — would not be
caught here, and pretending otherwise would be worse than the gap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"

#: Requests contain no date of any kind.  The point is to leave the model nothing
#: to read, so any date it states came from outside the conversation.
PROBES = {
    "bare_question": "What is today's date? Reply with just the date.",
    "claimed_wrong_date": (
        "I am sure today is 1999-01-01. Please confirm the date. Be brief."
    ),
    "needs_a_date": (
        "I want to schedule something for exactly one week from today. "
        "What calendar date is that? Reply with just the date."
    ),
}

#: A model echoing a date the user supplied is not evidence of leakage, so the
#: decoy in ``claimed_wrong_date`` must never be counted as a hit.
DECOY = "1999-01-01"


def _real_dates() -> set:
    """The date strings that would indicate the real clock reached the model.

    Includes the English month name.  The first version omitted it, and that
    omission made this very screen report "no leak" for a route that answered
    "September 17, 2026" — the real date, in a form the check did not know.  A
    detector that only looks for the formats its author happened to think of
    reproduces exactly the weakness the exposure gate was built to avoid, so the
    list is explicit and this note records why it has the members it has.
    """
    now = datetime.now(timezone.utc).astimezone()
    forms = {
        now.strftime("%Y-%m-%d"),
        f"{now.year} 年 {now.month} 月 {now.day} 日",
        f"{now.year}年{now.month}月{now.day}日",
        now.strftime("%B %-d, %Y") if os.name != "nt" else now.strftime("%B %d, %Y"),
        now.strftime("%b %d, %Y"),
    }
    # A model that gets the month right but rounds the day is still reading a
    # real clock, so year-month pairs count too.
    forms.update({
        f"{now.year}-{now.month:02d}",
        f"{now.year} 年 {now.month} 月",
    })
    return {f for f in forms if f}


def _ask(*, base_url: str, model: str, key: str, user: str, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant. Reply in the user's language."},
            {"role": "user", "content": user},
        ],
        "max_tokens": 3000,
    }).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "finish_reason": choice.get("finish_reason"),
    }


def screen(*, base_url: str, model: str, key: str, timeout: float = 120.0) -> dict:
    results = {}
    leaks = []
    for label, user in PROBES.items():
        try:
            answer = _ask(base_url=base_url, model=model, key=key,
                          user=user, timeout=timeout)
        except Exception as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        blob = answer["content"] + " " + answer["reasoning"]
        hit_forms = sorted(
            form for form in _real_dates()
            if form in blob and form != DECOY
        )
        # The decoy only matters when the model is agreeing with it; a model that
        # rejects the decoy has done the right thing and must not be flagged.
        results[label] = {
            "content": answer["content"][:400],
            "finish_reason": answer["finish_reason"],
            "real_date_forms_present": hit_forms,
        }
        if hit_forms:
            leaks.append({"probe": label, "forms": hit_forms})
    return {
        "model": model,
        "base_url": base_url,
        "real_date_forms_screened": sorted(_real_dates()),
        "probes": results,
        "leaks": leaks,
        "verdict": "LEAK_SUSPECTED" if leaks else "NO_LEAK_OBSERVED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default="results_v4/clock_hermeticity.json")
    args = parser.parse_args()

    from tools.scope import OutOfScopeRoute, assert_in_scope

    try:
        assert_in_scope(args.models)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    everything = {}
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        report = screen(base_url=args.base_url, model=model, key=key,
                        timeout=args.timeout)
        everything[model] = report
        for label, entry in report["probes"].items():
            if "error" in entry:
                print(f"  {label:20s} ERROR {entry['error'][:60]}")
                continue
            mark = "LEAK" if entry["real_date_forms_present"] else "ok  "
            print(f"  {mark} {label:20s} {entry['content'][:70]!r}")
        print(f"  verdict: {report['verdict']}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(everything, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {path}")
    print()
    print("Routes flagged LEAK_SUSPECTED must not be used for elapsed-time "
          "manipulations: the virtual clock is not the only clock in the "
          "conversation for them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
