"""Run the temporal ladder against real routes.

What this measures, in the order the layers are defined:

**L2 accessibility** — at each ladder position, which notices were queryable.  A
function of elapsed time, by construction of the ladder.

**L3 behaviour** — did the model's answer follow the *live* notice?  The signal is
whether the answer names the operative date of the stage it was asked in.  An
answer naming the earlier stage's date, with that later notice already delivered,
is the interesting case: the model had the update and did not use it.

**L4 attribution** — from the event log, whether the gold notice reached the
request.

Why this is a separate runner from ``run_matrix``: the ladder is re-derived per
elapsed point (the live stage changes, and with it the gold document), so the
"one spec, many branches" shape of the matrix runner does not fit.  Each ladder
position is its own scenario, and the comparison between positions is the
temporal contrast.

The date-matching used for the behavioural signal is deliberately mechanical and
narrow.  It answers "which date did the answer name", which is checkable; whether
the answer *should* have named it is the judge's question and is not claimed here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.clock import parse_instant  # noqa: E402
from vela_lab.provider import OpenAICompatibleModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    PROFILES,
    build_ladder_variant,
)
from tools.scope import OutOfScopeRoute, assert_in_scope  # noqa: E402
from tools.variant_wiring import build_ladder_spec  # noqa: E402

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"

#: Ladder positions to run, as days after the world update.  Chosen around the
#: place the ladder changes stage so both regimes are represented; the exact
#: boundary is a property of the ladder, not of this list.
DEFAULT_POSITIONS = (5.0, 13.0, 15.0, 31.0)

#: Sub-day positions.  Now expressible, and a clean probe: a date-only clock
#: signal cannot tell these apart, so a difference here would mean the model is
#: reading something finer than a date — while no difference is the expected
#: result and is reported as such rather than as a failure.
SUB_DAY_POSITIONS = (0.0, 0.5)


#: Phrases that mark the answer as *asserting* a date is operative.  Distinct from
#: merely naming one, which is why this exists: an answer can list the live date and
#: still decline to endorse it.
#: Affirmative and directive phrases, tested **only within a sentence that also
#: names the live date**.  The proximity requirement is what carries the meaning;
#: the list itself is bounded and known to be incomplete.  Members were added as
#: real answers exposed gaps: "新的有效预约**日期是**" and "请务必**按**…前往" both
#: assert the live date and neither matched an earlier list.
ASSERTION_MARKERS = (
    "为准", "以此为准", "请以此", "唯一有效", "当前有效", "有效预约日期",
    "有效日期", "日期是", "是当前", "请于", "应以", "应以", "务必按",
    "按", "前往", "到场", "以 {date} 为准", "改为", "更改为",
)

#: Phrases that express unresolved conflict **about the date**.  A route that reads
#: the update and then declines to settle the question has not used the evidence,
#: which is a different behaviour from following it.
#:
#: Sentence-scoped, and the scope matters.  An earlier version tested these
#: markers anywhere in the answer, and the word "不确定" matched inside
#: "我知道这种**不确定性**让人很不安" — an emotional-state word.  That misfired on
#: the *high-pressure* branch only, manufacturing an apparent pressure effect in
#: which the route seemed to hedge more under emotion.  Reading the two answers
#: side by side showed both asserting the live date plainly.
#:
#: So a marker counts only when it shares a sentence with a date reference.  This
#: is the same lesson as the exposure gate's negation handling: a bare term list
#: detects the term wherever it appears, including where it means something else.
UNCERTAINTY_MARKERS = (
    "无法确定", "无法判断", "不能确定", "无法确认", "尚不确定",
    "不一致", "相互矛盾", "存在冲突", "信息冲突", "矛盾",
    "需要核实", "请核实", "建议核实", "尽快核实", "建议你再核对",
)

#: A sentence counts as being about the date question if it mentions a date or a
#: date-question word.  Without this the scope is "anywhere in the answer", which
#: is what produced the false pressure effect.
DATE_CONTEXT_MARKERS = (
    "月", "日", "日期", "预约", "面试", "确认单", "通知",
)

SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+")


def _sentences(text: str) -> list:
    return [part for part in SENTENCE_SPLIT.split(text or "") if part.strip()]


def _strip_spaces(text: str) -> str:
    """Remove all whitespace, for comparing rendered dates across renderings.

    Both the spaced and unspaced conventions are legitimate ways to write the same
    date, and a comparison that accepted only one would report a correct answer as
    a failure.
    """
    return "".join((text or "").split())


def _date_scoped_matches(text: str, markers) -> list:
    """Markers that share a sentence with date context."""
    hits = []
    for sentence in _sentences(text):
        if not any(marker in sentence for marker in DATE_CONTEXT_MARKERS):
            continue
        for marker in markers:
            if marker in sentence:
                hits.append(marker)
    return hits


def classify_answer(text: str, *, live_date: str, rendered: str) -> dict:
    """A bounded triage of how the answer related to the live notice.

    Read this as a **triage, not a verdict**, and the reason is empirical: three
    successive versions of this classifier each produced a wrong label in a way the
    next one fixed, and the failures were all the same shape — a term list detects
    the spellings it knows and reports the rest as absent.  In order:

    1. a mention test scored "应以 7 月 9 日为准" and "无法确定哪一个是最终有效的日期"
       as the same behaviour;
    2. an uncertainty test matched "不确定" inside "这种**不确定性**让人很不安",
       an emotional-state word, manufacturing an apparent pressure effect on the
       high-pressure branch only;
    3. an assertion test missed "请务必按 6 月 25 日 的预约前往", which asserts the
       live date as plainly as any marker on the list.

    So the classifier is deliberately narrowed to what it can actually do, and its
    output carries an explicit unclassified bucket rather than guessing:

    ``asserted``               an affirmative/directive term shares a sentence with
                               the live date, and no date-scoped doubt
    ``asserted_with_caveat``   both, which is a real pattern here
    ``uncertain``              date-scoped doubt, no affirmative term
    ``mentioned_live``         the live date appears with neither
    ``no_live_date``           the live date is absent

    The robust signals are the ones about **presence** — whether the live date and
    the superseded date appear at all.  Whether an answer *settles* the question is
    a semantic judgement, and for that the project already has a blind judge; this
    function exists to triage which answers need it, not to replace it.
    """
    body = text or ""
    # Whitespace is normalised before matching a rendered date, because the same
    # answer form appears both ways: the notices write "6 月 9 日" and answers
    # commonly write "6月9日".  Comparing raw text missed every answer of the second
    # form, which understated the temporal rates — the fifth time in this project
    # that a matching rule detected only the spellings its author happened to think
    # of.  Normalising is applied to both sides so neither form is privileged.
    flat_body = _strip_spaces(body)
    flat_rendered = _strip_spaces(rendered)
    names_live = (live_date in body
                  or (rendered and rendered in body)
                  or (flat_rendered and flat_rendered in flat_body))
    sentences = _sentences(body)
    live_sentences = [
        sentence for sentence in sentences
        if live_date in sentence or (rendered and rendered in sentence)
    ]
    # Assertion is tested by proximity: an affirmative term in the *same sentence*
    # as the live date.  Testing the whole answer conflates "the date was named" with
    # "the date was endorsed", which is the distinction being measured.
    assertion_hits = sorted({
        marker for sentence in live_sentences for marker in ASSERTION_MARKERS
        if marker in sentence
    })
    uncertain_hits = sorted(set(_date_scoped_matches(body, UNCERTAINTY_MARKERS)))
    uncertain = bool(uncertain_hits)
    assertive = bool(assertion_hits)
    # Four outcomes rather than three, because "asserted with a verification
    # suggestion" is a real and distinct pattern here.  The two routes' answers
    # showed it: one states the live date plainly, the other states it *and* asks
    # the user to re-verify it.  A binary test that lets uncertainty override
    # assertion would file the second as a failure to settle, which understates
    # what it did; a test that ignores the caveat would file it as identical to
    # the first, which overstates.
    if names_live and assertive and uncertain:
        label = "asserted_with_caveat"
    elif names_live and assertive:
        label = "asserted"
    elif uncertain:
        label = "uncertain"
    elif names_live:
        label = "mentioned_live"
    else:
        label = "no_live_date"
    return {
        "label": label,
        "names_live_date": names_live,
        "has_assertion_marker": assertive,
        "has_uncertainty_marker": uncertain,
        # Which markers fired, so a reader can audit the scope rule instead of
        # trusting it.
        "assertion_markers": assertion_hits,
        "uncertainty_markers": sorted(set(uncertain_hits)),
    }


def _dates_in(text: str, year: int) -> set:
    """Every calendar date the text names, in ISO form."""
    found = set()
    for month, day in re.findall(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", text or ""):
        found.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for month, day in re.findall(r"(\d{1,2})/(\d{1,2})", text or ""):
        found.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for y, month, day in re.findall(r"(\d{4})-(\d{2})-(\d{2})", text or ""):
        found.add(f"{y}-{month}-{day}")
    return found


def run_position(*, profile, days: float, model_name: str, out_root: Path,
                 base_url: str, api_key_env: str, timeout: float,
                 branch_ids=None) -> dict:
    update = parse_instant(profile.update_day.isoformat() + "T09:00:00+08:00")
    instant = (update + timedelta(days=days)).isoformat()
    ladder = build_ladder_variant(
        profile=profile, filler_id=FILLER_ID, instant=instant
    )
    label = f"{days:g}d" if days >= 1 else f"{int(days * 1440)}min"
    experiment_id = f"temporal-{model_name}-{profile.profile_id}-{label}"
    spec = build_ladder_spec(
        ladder=ladder, experiment_id=experiment_id, elapsed_days=days
    )
    if branch_ids:
        spec.branches = tuple(b for b in spec.branches
                              if b.branch_id in set(branch_ids))

    model = OpenAICompatibleModel(
        base_url=base_url, model=model_name, api_key_env=api_key_env,
        timeout=timeout, max_transport_retries=1,
    )
    out = out_root / model_name / label
    if out.exists():
        shutil.rmtree(out)
    started = time.monotonic()
    report = ExperimentRunner(spec, model, out_dir=out).run()
    wall = round(time.monotonic() - started, 1)

    # Behavioural signal: which dates each answer named, and whether the answer
    # followed the live notice.
    rows = []
    for branch in report["branches"]:
        answer = branch["outcome"].get("assistant_text", "")
        named = _dates_in(answer, profile.old_date.year)
        live_iso = ladder.correct_date.isoformat()
        rendered_live = (f"{ladder.correct_date.month} 月 "
                         f"{ladder.correct_date.day} 日")
        classification = classify_answer(answer, live_date=live_iso,
                                         rendered=rendered_live)
        rows.append({
            "branch_id": branch["branch_id"],
            "valid": branch["run_valid"],
            "invalid_reason": branch.get("invalid_reason", ""),
            "gate": [v["status"] for v in branch["exposure_verdicts"]],
            "delivered": branch["auto_retrieval"].get("hits"),
            "l4": (branch.get("score") or {}).get("L4", {}).get("label", ""),
            "dates_named": sorted(named),
            "followed_live_stage": classification["label"] in (
                "asserted", "asserted_with_caveat"
            ),
            "live_stage_relation": classification,
            "named_superseded_stage": any(
                s.isoformat() in named for s in
                (st.voids for st in ladder.stages if st.stage_id != ladder.stage.stage_id)
            ),
            "answer": answer,
            "answer_chars": len(answer),
        })

    return {
        "model": model_name,
        "profile": profile.profile_id,
        "elapsed_label": label,
        "elapsed_days": days,
        "probe_instant": instant,
        "live_stage": ladder.stage.stage_id,
        "correct_date": ladder.correct_date.isoformat(),
        "stages": [s.stage_id for s in ladder.stages],
        "wall_seconds": wall,
        "branches": rows,
        "parent_equality": report.get("parent_equality", {}),
    }


def reclassify(summary_path: Path) -> dict:
    """Re-derive the behavioural labels from saved answers, without re-running.

    Needed because the classifier improved after the runs: the first version tested
    only whether the live date was *mentioned*, which scored "应以 7 月 9 日为准"
    and "无法确定哪一个是最终有效的日期" as the same behaviour.  Re-running to get
    better labels would replace the observations instead of correcting the reading,
    so the labels are recomputed from the text already on disk.

    The previous labels are preserved beside the new ones, so a reader can see that
    the classifier changed rather than the model.
    """
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    changes = []
    for result in data:
        live = result["correct_date"]
        year, month, day = live.split("-")
        rendered = f"{int(month)} 月 {int(day)} 日"
        for branch in result["branches"]:
            before = branch.get("live_stage_relation", {}).get("label", "")
            after = classify_answer(branch["answer"], live_date=live,
                                    rendered=rendered)
            branch["live_stage_relation_recorded_before_reclassify"] = (
                branch.get("live_stage_relation")
            )
            branch["live_stage_relation"] = after
            # `followed_live_stage` must track the corrected label, not the old one.
            branch["followed_live_stage"] = after["label"] in (
                "asserted", "asserted_with_caveat"
            )
            changes.append({
                "model": result["model"],
                "elapsed": result["elapsed_label"],
                "stage": result["live_stage"],
                "branch_id": branch["branch_id"],
                "recorded": before,
                "recomputed": after["label"],
                "changed": before != after["label"],
            })
    return {"summary": str(summary_path), "changes": changes,
            "changed_count": sum(1 for c in changes if c["changed"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None,
                        help="required unless --reclassify or --dry-run is used")
    parser.add_argument("--profile", default="visa_interview")
    parser.add_argument("--positions", nargs="+", type=float,
                        default=list(DEFAULT_POSITIONS))
    parser.add_argument("--sub-day", action="store_true",
                        help="also run the sub-day positions")
    parser.add_argument("--branches", nargs="+", default=None,
                        help="restrict to these branch ids (fewer executions)")
    parser.add_argument("--out", default="runs_temporal")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reclassify", nargs="*", default=None,
                        help="re-derive behavioural labels from saved summaries "
                             "instead of calling any model")
    args = parser.parse_args()

    if args.reclassify:
        for target in args.reclassify:
            path = Path(target)
            if path.is_dir():
                path = path / "temporal_summary.json"
            if not path.exists():
                print(f"missing summary: {path}", file=sys.stderr)
                continue
            result = reclassify(path)
            print(f"=== {result['summary']} ===")
            for change in result["changes"]:
                flag = "CHANGED" if change["changed"] else "same   "
                print(f"  {flag} {change['model'][:20]:21s} "
                      f"{change['elapsed']:>7s} {change['branch_id']:20s} "
                      f"{change['recorded'] or '-':12s} -> {change['recomputed']}")
            print(f"  {result['changed_count']} of {len(result['changes'])} changed")
            path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")),
                                       ensure_ascii=False, indent=2, sort_keys=True),
                            encoding="utf-8")
        return 0

    try:
        assert_in_scope(args.models)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    profile = next((p for p in PROFILES if p.profile_id == args.profile), None)
    if profile is None:
        print(f"unknown profile {args.profile!r}", file=sys.stderr)
        return 2

    if not args.models:
        print("--models is required unless --reclassify is given", file=sys.stderr)
        return 2

    positions = list(args.positions)
    if args.sub_day:
        positions = list(SUB_DAY_POSITIONS) + positions

    if args.dry_run:
        for days in positions:
            update = parse_instant(profile.update_day.isoformat() + "T09:00:00+08:00")
            instant = (update + timedelta(days=days)).isoformat()
            ladder = build_ladder_variant(
                profile=profile, filler_id=FILLER_ID, instant=instant
            )
            branches = len(args.branches) if args.branches else "all"
            print(f"  {days:>7g}d  instant={instant:32s} stage="
                  f"{ladder.stage.stage_id[-9:]:10s} correct={ladder.correct_date} "
                  f"branches={branches}")
        print(f"\ntotal positions: {len(positions)} x models: {len(args.models)}")
        # The ladder spec is oracle-only, so the realistic count is two
        # branches per position, not five.
        per_position = len(args.branches) if args.branches else 2
        print(f"executions: "
              f"{len(positions) * len(args.models) * per_position}")
        return 0

    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    everything = []
    for model_name in args.models:
        for days in positions:
            update = parse_instant(profile.update_day.isoformat() + "T09:00:00+08:00")
            instant = (update + timedelta(days=days)).isoformat()
            ladder = build_ladder_variant(
                profile=profile, filler_id=FILLER_ID, instant=instant
            )
            label = f"{days:g}d" if days >= 1 else f"{int(days * 1440)}min"
            print(f"\n=== {model_name} | {label} | stage={ladder.stage.stage_id} "
                  f"correct={ladder.correct_date} ===", flush=True)
            try:
                result = run_position(
                    profile=profile, days=days, model_name=model_name,
                    out_root=out_root, base_url=args.base_url,
                    api_key_env=args.api_key_env, timeout=args.timeout,
                    branch_ids=args.branches,
                )
            except Exception as exc:
                print(f"  RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            everything.append(result)
            for row in result["branches"]:
                flag = "ok " if row["valid"] else "BAD"
                print(f"  {flag} {row['branch_id']:18s} "
                      f"relation={row['live_stage_relation']['label']:11s} "
                      f"named={row['dates_named']} l4={row['l4']}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "temporal_summary.json"
    summary.write_text(
        json.dumps(everything, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {summary}")

    # The temporal contrast: does the answer track the live stage?
    print()
    print("BEHAVIOURAL SUMMARY (denominator = valid branches)")
    for model_name in args.models:
        rows = [r for r in everything if r["model"] == model_name]
        print(f"  {model_name}")
        for result in rows:
            valid = [b for b in result["branches"] if b["valid"]]
            counts = {}
            for branch in valid:
                key = branch["live_stage_relation"]["label"]
                counts[key] = counts.get(key, 0) + 1
            summary_text = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"    {result['elapsed_label']:>7s} "
                  f"stage={result['live_stage'][-9:]:10s} "
                  f"correct={result['correct_date']}  {summary_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
