"""E4 baseline battery: do these models just fail date arithmetic?

What this adds, and what it deliberately is not.

The expiry runs show models treating a month-old record as current.  Before that
behaviour can be read as a *temporal-reasoning* failure — the clock crossed the
event date and the record was not retired — a cheaper explanation has to be
cleared: maybe the routes cannot do the date arithmetic at all, or cannot map
"a date that passed" onto "the scheduled thing is no longer upcoming".

E4 is the deterministic-gold baseline for that question.  It needs **no harness**:
no corpus, no gate, no event log.  Each cell is one bare user message with two
dates stated in the prompt, a question about their relation, and a gold answer
computed in this file rather than judged.  A route that fails here cannot pass
the harder test upstairs, so E4 scores bound what the expiry result can mean;
a route that passes here and still answers the probe as if D were future has
failed the *recognition*, not the subtraction.

The three tasks, per (scenario kind, route):

* **A — date arithmetic.**  ``事件日期是 {D}。参考日期是 {T}。两个日期相差多少天？``
  Gold = ``(T - D).days`` with ``T = D + 30``.  One prompt.
* **B — boundary comparison.**  ``某事项定于 {D} 举行。参考日期是 {T}。该事项的
  预定日期已经过去了吗？回答"是"或"否"。``  Gold is ``是`` for ``T = D + 30``
  and ``否`` for ``T = D - 15``.  Two prompts, so the gold flips and a route
  that always says 是 cannot score 100% by accident.
* **C — as-of querying.**  ``截至 {T}，{D} 的这个已排定事项还应该被当作"将要
  发生"来对待吗？回答"是"或"否"。``  Gold is ``否`` for ``T > D`` and ``是``
  for ``T < D``.  Two prompts, both sides emitted.

That is A(1) + B(2) + C(2) = 5 prompts per (kind, route) cell; 14 pure-time
kinds x 5 subject routes = 350 cells.  ``D`` is each profile's ``old_date`` —
the same operative date the expiry experiment's fact store holds — so a route's
E4 behaviour is directly comparable to its expiry cells on the same calendar.

Dates are rendered ISO (``2026-04-20``), not the corpus's spaced Chinese, so the
question tests date arithmetic rather than a format the model happens to see in
the scenario texts.

Modes:

* ``--emit``   writes ``<out>/e4_prompts.json`` — the cell list with gold.  Free.
* ``--run``    calls each route once per prompt (bare user message, no tools;
               temperature is the adapter's 0.0 default) and saves the raw
               ``ModelResponse.to_json()`` plus the cell to
               ``<out>/e4_runs/<route>/<cell_id>.json``.  Paid.
* ``--score``  re-reads the run files, parses answers — digits for A (date
               tokens stripped first, so "2026-04-20 到 2026-05-20 相差30天"
               scores on the 30; multiple distinct numbers or none is
               ``unparsed``) and 是/否 for B/C (negation-aware: "不是" and
               "否" both map to 否; both markers present is ``unparsed``) —
               and writes ``<out>/e4_report.json`` with per-route x task
               accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import OpenAICompatibleModel  # noqa: E402

from tools.expiry_experiment import DEFAULT_BASE_URL  # noqa: E402
from tools.scenario_expansion import build_profiles  # noqa: E402
from tools.scope import OutOfScopeRoute, SUBJECT_ROUTES, assert_in_scope  # noqa: E402

#: The pure-time scenario kinds, fixed a priori.  Instance 1 of each — the same
#: instances the expiry grid ran — so D here is the same D the expiry cells use.
PURE_TIME_KINDS = (
    "broadband_install_i1",
    "car_maintenance_i1",
    "employment_checkup_i1",
    "equipment_check_i1",
    "furniture_measure_i1",
    "homestay_checkin_i1",
    "ielts_registration_i1",
    "kindergarten_interview_i1",
    "lease_signing_i1",
    "loan_interview_i1",
    "specialist_clinic_i1",
    "tax_interview_i1",
    "thesis_defense_i1",
    "visa_interview_i1",
)

#: Days the reference date sits on each side of D.  +30 mirrors the expiry
#: experiment's `post` arm; -15 is the still-upcoming control.
T_AFTER_DAYS = 30
T_BEFORE_DAYS = 15


def _profiles_for(kinds) -> dict:
    """profile_id -> profile, for the fixed kind set, via the real builder."""
    wanted = set(kinds)
    pool = build_profiles(instances=4)
    found = {p.profile_id: p for p in pool if p.profile_id in wanted}
    missing = [k for k in kinds if k not in found]
    if missing:
        raise RuntimeError(
            f"kind(s) {missing} not produced by build_profiles; the fixed list "
            "is stale rather than the pool being short"
        )
    return {k: found[k] for k in kinds}


def build_cells(kinds=PURE_TIME_KINDS, routes=SUBJECT_ROUTES) -> list:
    """The full 350-cell battery, deterministic gold included."""
    profiles = _profiles_for(kinds)
    cells = []
    for kind in kinds:
        profile = profiles[kind]
        d = profile.old_date
        t_after = d + timedelta(days=T_AFTER_DAYS)
        t_before = d - timedelta(days=T_BEFORE_DAYS)
        for route in routes:
            tasks = (
                (
                    "A",
                    t_after,
                    f"事件日期是 {d.isoformat()}。参考日期是 {t_after.isoformat()}。"
                    "两个日期相差多少天？",
                    str((t_after - d).days),
                ),
                (
                    "B_past",
                    t_after,
                    f"某事项定于 {d.isoformat()} 举行。参考日期是 {t_after.isoformat()}。"
                    '该事项的预定日期已经过去了吗？回答"是"或"否"。',
                    "是",
                ),
                (
                    "B_future",
                    t_before,
                    f"某事项定于 {d.isoformat()} 举行。参考日期是 {t_before.isoformat()}。"
                    '该事项的预定日期已经过去了吗？回答"是"或"否"。',
                    "否",
                ),
                (
                    "C_asof_after",
                    t_after,
                    f"截至 {t_after.isoformat()}，{d.isoformat()} 的这个已排定事项"
                    '还应该被当作"将要发生"来对待吗？回答"是"或"否"。',
                    "否",
                ),
                (
                    "C_asof_before",
                    t_before,
                    f"截至 {t_before.isoformat()}，{d.isoformat()} 的这个已排定事项"
                    '还应该被当作"将要发生"来对待吗？回答"是"或"否"。',
                    "是",
                ),
            )
            for task, ref_date, prompt, gold in tasks:
                cells.append({
                    "cell_id": f"{kind}::{task}",
                    "route": route,
                    "kind": kind,
                    "task": task,
                    "ref_date": ref_date.isoformat(),
                    "event_date": d.isoformat(),
                    "prompt": prompt,
                    "gold": gold,
                })
    return cells


# --------------------------------------------------------------------- scoring

#: Date-shaped tokens are removed before digits are counted, so an answer that
#: echoes the two input dates does not get scored on "2026" instead of the gap.
_DATE_RE = re.compile(
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"          # 2026年4月20日
    r"|\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}"          # 2026-04-20 / 2026/4/20
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日"                       # 4月20日
    r"|\d{4}\s*年"                                        # bare 2026年
)
_NUM_RE = re.compile(r"-?\d+")


_BOLD_NUM = re.compile(r"\*\*(\d+)\s*天?\*\*")
_LAST_NUM = re.compile(r"(\d+)\s*天")


def parse_task_a(text: str) -> str:
    """The day count from a free-form answer; "unparsed" when ambiguous.

    Working figures (10, 20) share the text with the answer, so the count is
    taken from the strongest signal: a bolded ``**N天**``/``**N**`` first, then
    the last ``N天`` (Chinese arithmetic answers close on the total), then the
    only distinct number.  Anything else is unparsed rather than guessed.
    """
    text = text or ""
    bold = _BOLD_NUM.findall(text)
    if len(set(bold)) == 1:
        return str(int(bold[0]))
    last = _LAST_NUM.findall(text)
    if last:
        return str(int(last[-1]))
    stripped = _DATE_RE.sub(" ", text)
    numbers = _NUM_RE.findall(stripped)
    distinct = sorted(set(numbers))
    if len(distinct) == 1:
        return str(int(distinct[0]))
    return "unparsed"


_NEG_WORDS = ("否", "不是", "没有", "尚未", "未")
_POS_WORDS = ("是", "yes", "Yes", "YES")
_NEG_EN = ("no", "No", "NO")


def parse_yes_no(text: str) -> str:
    """是/否 with negation handled; "unparsed" when the text is ambiguous.

    "不是" and "否" count for 否 *before* a bare "是" can count for 是, because
    "不是" contains the character 是.  A response that asserts both sides —
    e.g. "是，也否" — is marked unparsed rather than resolved by order.
    """
    blob = re.sub(r"[，。！？、；：·…\s\"'\"'*_`]", "", text or "")
    # "是否" means "whether" — it contains 否 but asserts no negation, and a
    # reply that echoes the question word must not score as 否.
    blob = blob.replace("是否", "")
    hits = set()
    for word in _NEG_WORDS:
        if word in blob:
            hits.add("否")
    for word in _NEG_EN:
        if re.search(rf"(?<![A-Za-z]){word}(?![A-Za-z])", blob):
            hits.add("否")
    for word in _POS_WORDS:
        if word in ("yes", "Yes", "YES"):
            if re.search(r"(?<![A-Za-z])yes(?![A-Za-z])", blob, re.IGNORECASE):
                hits.add("是")
        elif re.search(rf"(?<!不){word}", blob):
            hits.add("是")
    if len(hits) == 1:
        return hits.pop()
    return "unparsed"


def parse_answer(task: str, content: str) -> str:
    if task == "A":
        return parse_task_a(content)
    return parse_yes_no(content)


# ---------------------------------------------------------------------- modes


def emit(*, out: Path, kinds, routes) -> int:
    cells = build_cells(kinds=kinds, routes=routes)
    path = out / "e4_prompts.json"
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"E4 EMIT: {len(cells)} cells "
          f"({len(kinds)} kinds x {len(routes)} routes x 5 tasks)")
    print(f"  wrote {path}")
    print("\n  sample prompts:")
    for cell in (cells[0], cells[1]):
        print(f"    [{cell['cell_id']} | {cell['route']}] gold={cell['gold']}")
        print(f"      {cell['prompt']}")
    return 0


def _load_cells(out: Path) -> list:
    path = out / "e4_prompts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run --emit first so the gold comes from the "
            "emitted file rather than a rebuilt list"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run(*, out: Path, routes, cells, base_url: str, api_key_env: str,
        timeout: float, max_tokens, overwrite: bool) -> int:
    selected = set(routes)
    todo = [c for c in cells if c["route"] in selected]
    if not todo:
        print("no cells for the selected routes", file=sys.stderr)
        return 2
    print(f"E4 RUN: {len(todo)} paid calls "
          f"({len(selected)} route(s) x {len(todo) // max(1, len(selected))} cells)")
    models = {}
    done = failed = skipped = 0
    for cell in todo:
        route = cell["route"]
        cell_dir = out / "e4_runs" / route
        cell_dir.mkdir(parents=True, exist_ok=True)
        target = cell_dir / f"{cell['cell_id'].replace('::', '_')}.json"
        if target.exists() and not overwrite:
            skipped += 1
            continue
        if route not in models:
            models[route] = OpenAICompatibleModel(
                base_url=base_url, model=route, api_key_env=api_key_env,
                timeout=timeout, max_transport_retries=1,
                temperature=0.0, max_tokens=max_tokens,
            )
        model = models[route]
        try:
            response = model.complete(
                {"messages": [{"role": "user", "content": cell["prompt"]}]}
            )
        except Exception as exc:
            target.write_text(json.dumps({
                "cell": cell, "error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            failed += 1
            print(f"  FAIL {route} {cell['cell_id']}: {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        target.write_text(json.dumps({
            "cell": cell, "response": response.to_json(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        print(f"  ok {route} {cell['cell_id']} -> {target.name} "
              f"({response.latency_s}s)", flush=True)
    print(f"\nE4 RUN done: {done} written, {skipped} skipped, {failed} failed")
    return 0 if not failed else 1


def score(*, out: Path, cells) -> int:
    """Accuracy per route x task, over the run files that exist."""
    per_cell = {}
    for cell in cells:
        route = cell["route"]
        target = out / "e4_runs" / route / f"{cell['cell_id'].replace('::', '_')}.json"
        entry = {"cell_id": cell["cell_id"], "route": route, "task": cell["task"],
                 "gold": cell["gold"]}
        if not target.exists():
            entry.update(parsed="missing", correct=False)
        else:
            record = json.loads(target.read_text(encoding="utf-8"))
            if "error" in record:
                entry.update(parsed="error", correct=False,
                             error=record["error"])
            else:
                content = record.get("response", {}).get("content", "")
                parsed = parse_answer(cell["task"], content)
                entry.update(parsed=parsed,
                             correct=(parsed == cell["gold"]),
                             content=content[:400])
        per_cell[f"{route}::{cell['cell_id']}"] = entry

    accuracy = {}
    for route in sorted({c["route"] for c in cells}):
        accuracy[route] = {}
        for task in ("A", "B_past", "B_future", "C_asof_after", "C_asof_before"):
            rows = [e for e in per_cell.values()
                    if e["route"] == route and e["task"] == task]
            if not rows:
                continue
            accuracy[route][task] = {
                "n": len(rows),
                "correct": sum(1 for r in rows if r["correct"]),
                "unparsed": sum(1 for r in rows if r["parsed"] == "unparsed"),
                "accuracy": round(
                    sum(1 for r in rows if r["correct"]) / len(rows), 4),
            }

    report = {"tasks": {"A": "date arithmetic", "B_*": "boundary comparison",
                        "C_*": "as-of querying"},
              "accuracy": accuracy, "cells": per_cell}
    path = out / "e4_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"E4 SCORE: wrote {path}")
    for route, tasks in accuracy.items():
        parts = "  ".join(
            f"{task}={entry['correct']}/{entry['n']}"
            + (f"({entry['unparsed']}unp)" if entry["unparsed"] else "")
            for task, entry in tasks.items()
        )
        print(f"  {route:24s} {parts}")
    return 0


# ------------------------------------------------------------------------ cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true",
                      help="write the prompt list with deterministic gold")
    mode.add_argument("--run", action="store_true",
                      help="call the routes, one paid call per prompt")
    mode.add_argument("--score", action="store_true",
                      help="parse saved responses into the accuracy report")
    parser.add_argument("--routes", nargs="+", default=None,
                        help="defaults to scope.SUBJECT_ROUTES")
    parser.add_argument("--out", default="results_v4")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default None: reasoning routes may spend a lot "
                             "before the visible answer; cap only knowingly")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run cells whose response file already exists")
    args = parser.parse_args()

    routes = tuple(args.routes) if args.routes else SUBJECT_ROUTES
    try:
        assert_in_scope(routes)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    out = Path(args.out)
    if args.emit:
        return emit(out=out, kinds=PURE_TIME_KINDS, routes=routes)

    cells = _load_cells(out)
    if args.run:
        import os
        if not os.environ.get(args.api_key_env):
            print(f"environment variable {args.api_key_env} is not set",
                  file=sys.stderr)
            return 2
        return run(out=out, routes=routes, cells=cells, base_url=args.base_url,
                   api_key_env=args.api_key_env, timeout=args.timeout,
                   max_tokens=args.max_tokens, overwrite=args.overwrite)
    return score(out=out, cells=cells)


if __name__ == "__main__":
    raise SystemExit(main())
