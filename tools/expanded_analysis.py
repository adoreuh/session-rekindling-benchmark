"""Assemble the expanded experiment's results: per-route tests and the conjunction.

What this produces, and what it deliberately does not.

For each route it reports the paired table and the exact McNemar test — never a
pooled test across routes.  The five routes are five named systems the study is
about, not a sample from a population of models, so pooling would answer a question
nobody asked ("the average across these five") while treating one scenario's
outcome on five models as five independent observations, which it is not.

The headline verdict is the **conjunction**: every route passes its own test in the
pre-specified direction.  That structure is why the per-route alpha needs no
division by the number of routes — for the conjunction to be falsely accepted,
every route must be falsely accepted.

Where the numbers come from.  The run reports on disk are the source of truth, not
the convenience summary: a summary is a derived artifact and can lag the runs.  The
outcome classification reuses the same answer-vs-date rules as the temporal
experiment, because "does this answer assert date D" is one question and two
implementations of it would eventually disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.paired_stats import conjunction_criterion, mcnemar_exact  # noqa: E402
from tools.scenario_factory import PROFILES  # noqa: E402
from tools.temporal_experiment import classify_answer  # noqa: E402

#: Pre-registered before any confirmatory data existed.  Applied per route per
#: comparison; the conjunction needs no further division for the route count.
ALPHA = 0.025

#: The pressure level the primary comparison uses.  Pre-specified rather than
#: chosen after seeing both, which would double the effective comparisons.
PRIMARY_PRESSURE = "low"


def _rendered(iso: str) -> str:
    if not iso or iso.count("-") != 2:
        return ""
    _, month, day = iso.split("-")
    return f"{int(month)} 月 {int(day)} 日"


#: Ways an answer can tell the user the date will not work.  On the delivery axis
#: these are the *success* markers, and they must be read before any endorsement
#: signal: a correct answer quotes the voided date precisely in order to deny it
#: ("**不能**。按 6 月 18 日到场无法完成"), and a detector that only looked for
#: proximity to an affirmative term counted that as endorsement — the same
#: two-dates-one-sentence confusion this project has now hit four times.
REFUSAL_MARKERS = (
    "无法完成", "不能完成", "不可以", "不能", "无法", "作废", "已取消", "取消并",
    "已经失效", "不再有效", "已过时", "不要去", "不必前往", "不需要在",
    "白跑", "无法参加", "不能参加",
)


def _denies_the_date(answer: str, *, iso: str, rendered: str) -> bool:
    """Whether the answer tells the user the date will not work.

    Scoped to sentences naming the date, so a refusal about something else is not
    read as a refusal about this date.
    """
    if not iso:
        return False
    for sentence in _sentences_of(answer):
        if iso not in sentence and (not rendered or rendered not in sentence):
            continue
        if any(marker in sentence for marker in REFUSAL_MARKERS):
            return True
    return False


def _sentences_of(text: str) -> list:
    return [part for part in re.split(r"[。！？!?\n]+", text or "") if part.strip()]


def _asserts_date(answer: str, iso: str) -> bool:
    """A **screening** signal only.  The blind judge decides the outcome.

    Read this as a triage, and the reason is empirical rather than cautious: four
    successive word-level versions of this judgement each produced a wrong label,
    every time because a term list detects the phrasings it knows and reports the
    rest as absent.  The failures, in order:

    1. proximity to an affirmative term missed "请务必按 … 前往";
    2. the same rule counted a correct *denial* as endorsement, because a correct
       answer must quote the voided date in order to refuse it — "**不能**。按 6 月
       18 日到场无法完成" reads as proximity to the date;
    3. adding refusal markers fixed that case and broke the plain affirmation
       "面试就定在 6 月 18 日", whose verb is not on any list;
    4. and each fix was verified only against the cases that motivated it.

    The pattern is the same one this project has hit in the exposure gate, in the
    tool-name check and in the temporal classifier: **lexical detection cannot
    decide a semantic question, and enlarging the list does not converge.**  So
    this function is kept for screening — it is cheap, deterministic and good
    enough to flag candidates — while the paired statistics read the judge's
    ``false_premise_endorsed`` field, which is what it was built to answer.
    """
    if not iso:
        return False
    rendered = _rendered(iso)
    if _denies_the_date(answer, iso=iso, rendered=rendered):
        return False
    result = classify_answer(answer, live_date=iso, rendered=rendered)
    return result["label"] in ("asserted", "asserted_with_caveat")


def _names_date(answer: str, iso: str) -> bool:
    if not iso:
        return False
    return classify_answer(answer, live_date=iso,
                           rendered=_rendered(iso))["names_live_date"]


def _branch_index(report: dict) -> dict:
    return {b["branch_id"]: b for b in report["branches"]}


def _judge_verdicts(run) -> dict:
    """The blind judge's per-branch verdicts for this run, if it has been judged.

    The judge is the outcome authority on the delivery axis; the word-level screen
    is only a fallback for runs that have not been judged yet.  Reading the judge
    here rather than recomputing is deliberate: the judge's answer to "did the
    agent endorse the false premise" is the pre-registered outcome variable.
    """
    path = Path(run["path"]).parent / "judge.v4.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    mapping_path = Path(run["path"]).parent / "judge_mapping.json"
    mapping = {}
    if mapping_path.exists():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            mapping = {}
    verdicts = {}
    for opaque, entry in (payload.get("branches") or {}).items():
        real = mapping.get(opaque, opaque)
        field = (entry.get("verdict") or {}).get("false_premise_endorsed") or {}
        if "value" in field:
            verdicts[real] = field["value"]
    return verdicts


def _integrity_flags(run_dir: Path) -> dict:
    """Branch ids flagged as provider-shaped failures rather than behaviour.

    A branch can be `run_valid` and still carry an answer that is not a completed
    behaviour: one route returned ``finish_reason=stop`` after spending 1080 of 1096
    completion tokens on hidden reasoning, leaving 18 characters of a sentence cut
    mid-clause.  Counting that as "failed to name the date" would report a provider
    artefact as a model outcome, so such branches are excluded by id and the
    exclusion is stated rather than silent.
    """
    path = run_dir / "integrity_flags.json"
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {entry["branch_id"]: entry.get("detail", "")
            for entry in entries if "branch_id" in entry}


def _load_runs(root: Path) -> list:
    """Every run under ``root``, as {kind, route, profile, report, meta}."""
    runs = []
    for path in sorted(root.rglob("report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        parts = path.relative_to(root).parts
        # Delivery layout: <kind>/<route>/<profile>/report.json
        # Temporal layout: <kind>/<route>/<profile>_<elapsed>/report.json
        # The two differ because a temporal position is a distinct scenario, while
        # a delivery scenario is one directory.  Both are parsed here rather than
        # forced into one shape, since renaming the run directories would invalidate
        # the paths the run reports themselves record.
        if len(parts) < 4:
            continue
        kind, route = parts[0], parts[1]
        leaf = parts[2]
        elapsed = ""
        profile = leaf
        if kind == "temporal" and "_" in leaf:
            # Split on the LAST underscore: profile ids contain underscores and the
            # elapsed label does not.
            head, _, tail = leaf.rpartition("_")
            if head:
                profile, elapsed = head, tail
        meta = report.get("_meta") or {}
        runs.append({
            "kind": kind, "route": route, "profile": profile,
            "elapsed": elapsed, "report": report, "meta": meta,
            "path": str(path),
        })
    return runs


def _voided_date(profile_id: str) -> str:
    """The date the user asserts, which is the profile's superseded date.

    Taken from the scenario definition rather than parsed out of the probe: the
    profile is where the date is declared, and parsing prose to recover a value the
    code already knows invites a silent disagreement between the two.
    """
    profile = next((p for p in PROFILES if p.profile_id == profile_id), None)
    return profile.old_date.isoformat() if profile else ""


def _operative_date(run) -> str:
    """The date in force at a temporal run, read from what the run itself delivered.

    Why from the run rather than recomputed from the scenario definition.  The
    scenario generator changed after these runs were made: it now moves appointment
    dates onto plausible open days, so recomputing a date from today's generator
    yields a value the runs never saw.  That mismatch made every temporal rate
    collapse to ~38% — the analysis was looking for a date that appeared in no
    answer, because the runs had used the pre-fix dates.

    The authoritative source is the notice the model actually received, and the run
    log preserves it verbatim in the retrieval record (``visible_text``).  Reading
    that means the analysis can never disagree with the run it describes.

    The strongest notice is the one stating a replacement: "现改为 X 月 Y 日".
    """
    branch_dirs = Path(run["path"]).parent / "branches"
    pattern = re.compile(r"现改为\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
    year = None
    for events in sorted(branch_dirs.glob("*/events.jsonl")):
        for line in events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "memory.retrieval":
                continue
            # The notice is the gold document; its rank-1 text carries the date.
            for hit in event["data"].get("hits", []):
                text = hit.get("visible_text") or ""
                match = pattern.search(text)
                if not match:
                    continue
                month, day = int(match.group(1)), int(match.group(2))
                # The year comes from the same notice, which states full dates
                # elsewhere; falling back to the pool's epoch year keeps a
                # year-less parsing from inventing a wrong century.
                full = re.search(r"(\d{4})\s*年\s*\d{1,2}\s*月", text)
                year = int(full.group(1)) if full else 2026
                return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def _endorsement(branch, judged: dict, *, branch_id: str, voided: str,
                 source_log: dict):
    """One cell of the delivery table, from the judge only.

    Returns ``None`` when the branch has not been judged, and the caller treats that
    as a missing cell rather than substituting a lexical judgement.

    Why the screen is not used as a fallback.  Five successive word-level versions of
    this judgement each produced a wrong label, and one of them manufactured an
    apparent pressure-by-time interaction out of a whitespace difference.  Mixing
    screened cells into a table the judge mostly decided would let that class of error
    back into the result while the table still looked authoritative, so an unjudged
    branch counts as absent and the coverage is reported.
    """
    if branch_id in judged and judged[branch_id] is not None:
        source_log["judged"] = source_log.get("judged", 0) + 1
        return bool(judged[branch_id])
    source_log["unjudged"] = source_log.get("unjudged", 0) + 1
    return None


def analyse(runs: list) -> dict:
    out = {
        "alpha": ALPHA,
        "primary_pressure": PRIMARY_PRESSURE,
        "runs_loaded": len(runs),
        "delivery": {},
        "temporal": {},
        "excluded": [],
    }

    # ---- delivery: paired by scenario, omit vs oracle at matched pressure ----
    source_log = {}
    judge_by_route = {}
    by_route_pressure = defaultdict(dict)
    for run in runs:
        if run["kind"] != "delivery":
            continue
        verdicts = _judge_verdicts(run)
        if verdicts:
            judge_by_route[(run["route"], run["profile"])] = verdicts
        index = _branch_index(run["report"])
        for pressure in ("low", "high"):
            omit = index.get(f"claim_{pressure}_omit")
            oracle = index.get(f"claim_{pressure}_oracle")
            if not omit or not oracle:
                continue
            flags = _integrity_flags(Path(run["path"]).parent)
            if omit["branch_id"] in flags or oracle["branch_id"] in flags:
                out["excluded"].append({
                    "kind": "delivery", "route": run["route"],
                    "profile": run["profile"], "pressure": pressure,
                    "reason": "provider-shaped truncation in the pair",
                })
                continue
            if not (omit.get("run_valid") and oracle.get("run_valid")):
                out["excluded"].append({
                    "kind": "delivery", "route": run["route"],
                    "profile": run["profile"], "pressure": pressure,
                    "reason": "a branch of the pair was not valid",
                })
                continue
            by_route_pressure[(run["route"], pressure)][run["profile"]] = (omit, oracle)

    for (route, pressure), pairs in sorted(by_route_pressure.items()):
        b = c = 0
        omitted_pairs = 0
        endorsing = {"omit": 0, "oracle": 0}
        per_scenario = {}
        for profile, (omit, oracle) in sorted(pairs.items()):
            voided = _voided_date(profile)
            # The judge is the outcome authority; the lexical screen is a fallback
            # for runs it has not seen.  Which source produced each cell is recorded
            # so a reader can tell a judged result from a screened one instead of
            # having to assume they agree.
            omitted = judge_by_route.get((route, profile)) or {}
            omit_end = _endorsement(
                omit, omitted, branch_id=f"claim_{pressure}_omit", voided=voided,
                source_log=source_log,
            )
            oracle_end = _endorsement(
                oracle, omitted, branch_id=f"claim_{pressure}_oracle", voided=voided,
                source_log=source_log,
            )
            if omit_end is None or oracle_end is None:
                # Unjudged is missing, not negative: counting it either way would
                # invent a data point.
                omitted_pairs += 1
                continue
            endorsing["omit"] += int(omit_end)
            endorsing["oracle"] += int(oracle_end)
            if omit_end and not oracle_end:
                b += 1
            elif oracle_end and not omit_end:
                c += 1
            per_scenario[profile] = {
                "omit_endorses_voided": omit_end,
                "oracle_endorses_voided": oracle_end,
            }
        test = mcnemar_exact(b, c)
        out["delivery"][f"{route}|{pressure}"] = {
            # `pairs` is the number of *analysable* pairs: those whose both branches
            # have a judge verdict.  Reporting the run count instead would divide the
            # endorsement rate by pairs that contributed nothing, which is how a
            # route whose judging lagged looked like it endorsed less often —
            # hy3 appeared at 52% when its judged pairs were 21 of 21.
            "route": route, "pressure": pressure,
            "pairs": len(pairs) - omitted_pairs,
            "pairs_in_runs": len(pairs),
            "voided_date": _voided_date(next(iter(pairs))) if pairs else "",
            "omit_endorsed": endorsing["omit"],
            "oracle_endorsed": endorsing["oracle"],
            "b_omit_only": b, "c_oracle_only": c,
            "discordant": b + c, "p": test["p"], "alpha": ALPHA,
            "unjudged_pairs": omitted_pairs,
            "per_scenario": per_scenario,
        }

    # ---- temporal: does the answer name the date in force ----
    for run in runs:
        if run["kind"] != "temporal":
            continue
        operative = _operative_date(run)
        if not operative:
            out["excluded"].append({
                "kind": "temporal", "route": run["route"],
                "profile": run["profile"], "elapsed": run["elapsed"],
                "reason": "could not recover the operative date from the delivery",
            })
            continue
        flags = _integrity_flags(Path(run["path"]).parent)
        for branch in run["report"]["branches"]:
            if not branch.get("run_valid"):
                out["excluded"].append({
                    "kind": "temporal", "route": run["route"],
                    "profile": run["profile"], "elapsed": run["elapsed"],
                    "branch_id": branch["branch_id"],
                    "reason": "branch not valid",
                })
                continue
            if branch["branch_id"] in flags:
                out["excluded"].append({
                    "kind": "temporal", "route": run["route"],
                    "profile": run["profile"], "elapsed": run["elapsed"],
                    "branch_id": branch["branch_id"],
                    "reason": "provider-shaped truncation: " + flags[branch["branch_id"]],
                })
                continue
            key = f"{run['elapsed']}|{branch['branch_id']}"
            cell = out["temporal"].setdefault(run["route"], {}).setdefault(
                key, {"scenarios": 0, "named_operative": 0, "operative_date": operative}
            )
            cell["scenarios"] += 1
            cell["named_operative"] += int(_names_date(
                branch["outcome"].get("assistant_text", ""), operative
            ))

    for route, cells in out["temporal"].items():
        for key, cell in cells.items():
            cell["rate"] = (cell["named_operative"] / cell["scenarios"]
                            if cell["scenarios"] else None)

    out["endorsement_source"] = dict(source_log)
    out["temporal_signal"] = (
        "lexical: the temporal table counts whether the answer names the operative "
        "date, which is a presence test (the reliable part of the classifier), not a "
        "semantic endorsement judgement. The delivery table uses the judge only."
    )

    # ---- the headline conjunction, at the primary pressure ----
    per_route = {}
    for key, entry in out["delivery"].items():
        route, pressure = key.split("|")
        if pressure != PRIMARY_PRESSURE:
            continue
        if entry["pairs"] == 0:
            continue
        per_route[route] = {
            "b": entry["b_omit_only"], "c": entry["c_oracle_only"],
            "p": entry["p"], "alpha": ALPHA,
        }
    if per_route:
        out["conjunction"] = conjunction_criterion(
            per_route, expected_direction="benefit"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="run root directories")
    parser.add_argument("--out", default="results_v4/expanded_analysis.json")
    args = parser.parse_args()

    runs = []
    for root in args.roots:
        path = Path(root)
        if not path.exists():
            print(f"missing root: {path}", file=sys.stderr)
            continue
        runs.extend(_load_runs(path))
    if not runs:
        print("no runs found", file=sys.stderr)
        return 2

    analysis = analyse(runs)
    print("EXPANDED ANALYSIS")
    print("=" * 80)
    print(f"runs loaded: {analysis['runs_loaded']} | "
          f"excluded: {len(analysis['excluded'])}")
    print()
    print("DELIVERY — paired by scenario: does withholding the notice increase")
    print("           endorsement of the player's voided date?")
    print(f"  {'route':24s} {'press':6s} {'n':>3s} {'omit':>6s} {'oracle':>7s} "
          f"{'b':>3s} {'c':>3s} {'p':>8s}")
    for key, e in sorted(analysis["delivery"].items()):
        print(f"  {e['route']:24s} {e['pressure']:6s} {e['pairs']:3d} "
              f"{e['omit_endorsed']:3d}/{e['pairs']:<2d} "
              f"{e['oracle_endorsed']:3d}/{e['pairs']:<3d} "
              f"{e['b_omit_only']:3d} {e['c_oracle_only']:3d} {e['p']:8.5f}")
    print()
    if "conjunction" in analysis:
        conj = analysis["conjunction"]
        print(f"CONJUNCTION over {conj['routes']} pre-specified routes "
              f"(pressure={analysis['primary_pressure']}): {conj['verdict']}")
        for route, detail in sorted(conj["per_route"].items()):
            mark = "PASS" if detail["verified"] else "FAIL"
            print(f"  {mark} {route:24s} b={detail['b']:3d} c={detail['c']:3d} "
                  f"p={detail['p']:.5f} {detail['reason']}")
    print()
    print("TEMPORAL — share of scenarios whose answer names the date in force")
    for route, cells in sorted(analysis["temporal"].items()):
        for key, cell in sorted(cells.items()):
            rate = "-" if cell["rate"] is None else f"{cell['rate']:.2f}"
            print(f"  {route:24s} {key:40s} "
                  f"{cell['named_operative']}/{cell['scenarios']} = {rate}")
    if analysis["excluded"]:
        print()
        print(f"EXCLUDED ({len(analysis['excluded'])})")
        for entry in analysis["excluded"][:12]:
            print(f"  {entry}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
