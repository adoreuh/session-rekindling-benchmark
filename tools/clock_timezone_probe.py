"""Which timezone does the runtime date signal use?

Why this needs its own experiment.  The route carries a runtime calendar date
that the harness does not send.  ``FINDING_clock_source_conclusion.md``
establishes that the signal exists, is real-time (it equals the actual run date),
and has day granularity.  What it does not establish is **which clock** produces
it, and that is decidable from the client — but only inside a specific window.

The discriminator.  For most of the day UTC and UTC+8 agree on the date, so a
probe cannot tell them apart.  During the eight hours when Beijing has rolled over
to a new day and UTC has not, they disagree:

    北京 00:00-08:00  ->  UTC is still the previous date
    UTC 00:00-08:00   ->  Beijing is already the same date (no disagreement)

So the informative window is **Beijing 00:00 to 07:59**, and the reading is:

    answers the Beijing date -> the signal uses UTC+8, or a local/session zone
    answers the UTC date     -> the signal uses UTC

That distinction is not cosmetic.  A UTC date is what a *server-side* clock would
naturally emit, whereas a local date suggests the zone came from the session or the
request.  It narrows which layer could be responsible, which is exactly the
question the client cannot otherwise answer.

Two limits, stated because they bound the conclusion:

* the test only fires in an 8-hour window, so it is scheduled rather than run on
  demand — running it outside the window measures nothing and will say so;
* it cannot distinguish "UTC+8" from "whatever the session's local zone is" when
  those coincide.  It separates UTC from non-UTC, which is the layer-relevant
  split, and does not over-claim beyond that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"
BEIJING = timezone(timedelta(hours=8))

#: The question that yields a bare date.  Asking for a timestamp would invite a
#: placeholder, and a placeholder tells us nothing about the zone.
ASK = "What is today's date? Reply with just the date."


def window_state(now: datetime | None = None) -> dict:
    """Whether we are inside the informative window, and what dates to expect."""
    moment = now or datetime.now(timezone.utc)
    beijing = moment.astimezone(BEIJING)
    utc_date = moment.date()
    beijing_date = beijing.date()
    discriminating = utc_date != beijing_date
    return {
        "now_utc": moment.isoformat(timespec="seconds"),
        "now_beijing": beijing.isoformat(timespec="seconds"),
        "utc_date": utc_date.isoformat(),
        "beijing_date": beijing_date.isoformat(),
        "discriminating": discriminating,
        "next_window": None if discriminating else _next_window(moment),
    }


def _next_window(moment: datetime) -> dict:
    """When the next discriminating window opens and closes, in UTC and Beijing."""
    beijing = moment.astimezone(BEIJING)
    # The window is Beijing 00:00-08:00, i.e. UTC 16:00 (previous day) - 00:00.
    start_beijing = datetime.combine(
        beijing.date() + timedelta(days=1), datetime.min.time(), tzinfo=BEIJING
    )
    end_beijing = start_beijing + timedelta(hours=8)
    return {
        "opens_beijing": start_beijing.isoformat(timespec="seconds"),
        "closes_beijing": end_beijing.isoformat(timespec="seconds"),
        "opens_utc": start_beijing.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "closes_utc": end_beijing.astimezone(timezone.utc).isoformat(timespec="seconds"),
    }


def _ask(base_url: str, key: str, model: str, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": ASK}],
        "max_tokens": 2000,
    }).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    choice = (data.get("choices") or [{}])[0]
    return {
        "content": ((choice.get("message") or {}).get("content") or "").strip(),
        "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
    }


def read_zone(text: str, state: dict) -> str:
    """Which of the two candidate dates the answer names."""
    utc_date, beijing_date = state["utc_date"], state["beijing_date"]
    if not text:
        return "empty"
    for label, iso in (("utc", utc_date), ("beijing", beijing_date)):
        year, month, day = iso.split("-")
        for rendering in (
            iso,
            f"{int(month)} 月 {int(day)} 日",
            f"{int(month)}/{int(day)}",
            f"{int(month)}月{int(day)}日",
        ):
            if rendering in text:
                return label
    if text.strip().upper() in ("NONE", "UNKNOWN"):
        return "declined"
    return "other"


def run_once(*, base_url: str, key: str, models, timeout: float) -> dict:
    state = window_state()
    report = {"window": state, "models": {}}
    for model in models:
        try:
            answer = _ask(base_url, key, model, timeout)
            label = read_zone(answer["content"], state)
        except Exception as exc:
            report["models"][model] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        report["models"][model] = {
            "content": answer["content"][:200],
            "reading": label,
            "prompt_tokens": answer["prompt_tokens"],
        }
    # What the reading implies, stated only when the window was informative.
    if state["discriminating"]:
        for model, entry in report["models"].items():
            if "error" in entry:
                continue
            if entry["reading"] == "utc":
                entry["implication"] = (
                    "the signal's date matches UTC while Beijing has rolled over, "
                    "so the zone is UTC or a zone west of UTC+8"
                )
            elif entry["reading"] == "beijing":
                entry["implication"] = (
                    "the signal's date matches Beijing while UTC is a day behind, "
                    "so the zone is UTC+8 or a local/session zone"
                )
            elif entry["reading"] == "declined":
                entry["implication"] = (
                    "the route declined to state a date; the window is informative "
                    "but this route provides no reading"
                )
    else:
        report["note"] = (
            "UTC and Beijing agree on the date right now, so a correct answer "
            "cannot distinguish the two zones. Run inside Beijing 00:00-08:00."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--check-only", action="store_true",
                        help="report the window state without calling any model")
    parser.add_argument("--out", default="results_v4/clock_timezone.json")
    args = parser.parse_args()

    state = window_state()
    print("TIMEZONE DISCRIMINATION WINDOW")
    print("=" * 72)
    print(f"  now (UTC)     : {state['now_utc']}")
    print(f"  now (Beijing) : {state['now_beijing']}")
    print(f"  UTC date      : {state['utc_date']}")
    print(f"  Beijing date  : {state['beijing_date']}")
    print(f"  discriminating: {state['discriminating']}")
    if state["next_window"]:
        print(f"  next window   : {state['next_window']['opens_beijing']} "
              f"-> {state['next_window']['closes_beijing']} (Beijing)")

    if args.check_only:
        return 0

    if not args.models:
        print("\n--models is required unless --check-only is given", file=sys.stderr)
        return 2
    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"\nenvironment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    if not state["discriminating"]:
        print("\nNOT IN WINDOW: UTC and Beijing agree on the date, so this run "
              "cannot distinguish the zones. The call is still made and recorded "
              "so the reading can be compared with an in-window reading later.")

    report = run_once(base_url=args.base_url, key=key, models=args.models,
                      timeout=args.timeout)
    print()
    for model, entry in report["models"].items():
        if "error" in entry:
            print(f"  {model:24s} ERROR {entry['error'][:70]}")
            continue
        print(f"  {model:24s} -> {entry['reading']:9s} {entry['content'][:60]!r}")
        if entry.get("implication"):
            print(f"  {'':24s}    {entry['implication']}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    if isinstance(existing, dict):
        existing = [existing]
    existing.append(report)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path} ({len(existing)} reading(s) so far)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
