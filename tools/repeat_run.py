"""Repeat-run subset: same expiry cell, run again, does the answer move?

Why this exists.

The expiry grid is one run per cell.  With observe_time=False the three natural
arms send **byte-identical requests** (the dry run prints the payload digests),
so any difference between repetitions of one cell is pure provider-side
non-determinism — or a difference in whether the model spontaneously calls
``world_time``.  Neither can be read off a single sample.  This tool re-runs a
fixed subset of cells N more times so the headline numbers can be reported with
a stability statement rather than as single draws.

The fixed subset, chosen a priori (recorded here so it cannot drift into
"the interesting cells"):

* **kinds** — 8 of the 14 pure-time kinds, picked to span the rigidity axis:
  4 high-rigid (missing the date forfeits the event: ``visa_interview_i1``,
  ``specialist_clinic_i1``, ``kindergarten_interview_i1``, ``lease_signing_i1``),
  3 mid (reschedulable at a cost: ``employment_checkup_i1``,
  ``homestay_checkin_i1``, ``broadband_install_i1``), and 1 low
  (``car_maintenance_i1``: the appointment is a routine and slippage is cheap);
* **routes** — ``qwen3-vl-plus``, ``Qwen3.5-397B-A17B``, ``DeepSeek-V4.1-Flash``;
* **conditions** — ``post_natural`` (D+30, no rendered clock) and
  ``late_natural`` (D+120, no rendered clock): the cells where the only route
  to the clock is the ``world_time`` tool and the payload is identical;
* **reps** — ``--reps`` (default 4) additional runs, so with the original
  ``runs_expiry_*`` cell there are 5 draws per cell.

Every rep reuses the real builders: ``build_expiry_variant`` +
``assert_no_supersession`` + ``build_conditions`` + ``assert_corpus_identical``
+ ``assert_world_static``, then the matching ``Condition`` is run with
``run_condition``.  The ``experiment_id`` is identical to the original cell's —
that is the point (same spec) — and the evidence stays separable because the
output root is ``runs_repeat/rep<k>/<route>/<profile>_<arm>_<visibility>/``.

Modes:

* default (paid) — OpenAICompatibleModel against the gateway; requires the API
  key env var.  ``--resume`` skips a cell that already has ``report.json``.
* ``--dry-run``  — MockModel, offline, evidence under ``_tmp/repeat-dryrun/``.
  Validates the build and the run path for free.
* ``--analyze``  — walks ``--src`` for ``<cell>/branches/probe/events.jsonl`` +
  ``report.json`` and reports, per cell: n runs, ``world_time`` fetch counts
  (``tool.call`` records **excluding** ones carrying the ``injected`` flag —
  an injected fetch is the harness's do(U=1), not the model's choice),
  final-answer digest uniqueness (the last non-empty assistant content,
  hashed), the proportion of cells whose reps were unanimous, and the fetch
  flip rate (fraction of cells where reps disagree on fetched/not-fetched).
  The walker accepts both ``runs_repeat/rep*/route/cell/`` layouts and flat
  ``runs_expiry_*/expiry/route/cell/`` layouts, where each condition dir of a
  cell counts as a rep — that is what makes ``--src runs_expiry_hy3`` a valid
  smoke input.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.jsonutil import digest  # noqa: E402
from vela_lab.provider import MockModel, OpenAICompatibleModel  # noqa: E402

from tools.expiry_experiment import (  # noqa: E402
    ARMS,
    DEFAULT_BASE_URL,
    assert_corpus_identical,
    assert_no_supersession,
    assert_world_static,
    build_conditions,
    build_expiry_variant,
    run_condition,
)
from tools.scenario_expansion import build_profiles  # noqa: E402
from tools.scope import OutOfScopeRoute, assert_in_scope  # noqa: E402

#: The fixed kind subset.  Order is the reporting order; the rigid-to-cheap
#: grouping is documented in the module docstring.
REPEAT_KINDS = (
    "visa_interview_i1",          # high-rigid
    "specialist_clinic_i1",       # high-rigid
    "kindergarten_interview_i1",  # high-rigid
    "lease_signing_i1",           # high-rigid
    "employment_checkup_i1",      # mid
    "homestay_checkin_i1",        # mid
    "broadband_install_i1",       # mid
    "car_maintenance_i1",         # low
)

#: Three routes: the vision route, the large MoE, and one DeepSeek-Flash line.
REPEAT_ROUTES = ("qwen3-vl-plus", "Qwen3.5-397B-A17B", "DeepSeek-V4.1-Flash")

#: The two natural-clock expired arms.  ``pre`` has no expiry to disagree on,
#: and ``delivered`` renders the clock so its payload differs from the natural
#: cells — the question here is spontaneous fetch behaviour.
REPEAT_ARMS = tuple((name, days) for name, days in ARMS if name in ("post", "late"))
REPEAT_VISIBILITIES = (("natural", False),)

#: A run directory is a rep cell when it contains this file.
_REPORT = "report.json"
#: Leaf-dir suffixes that make two conditions of one profile distinct reps
#: rather than distinct cells, in flat layouts like runs_expiry_*.
_COND_SUFFIX_RE = re.compile(r"_(pre|post|late)_(natural|delivered|forced)$")


def _profiles(kinds) -> dict:
    wanted = set(kinds)
    pool = build_profiles(instances=4)
    found = {p.profile_id: p for p in pool if p.profile_id in wanted}
    missing = [k for k in kinds if k not in found]
    if missing:
        raise RuntimeError(
            f"kind(s) {missing} missing from build_profiles; the fixed repeat "
            "list is stale rather than the pool being short"
        )
    return {k: found[k] for k in kinds}


def _repeat_conditions(*, profile, route):
    """The two post/late natural conditions for one (route, profile)."""
    variant = build_expiry_variant(profile=profile)
    assert_no_supersession(variant)
    conditions = build_conditions(
        variant=variant, route=route,
        arms=REPEAT_ARMS, visibilities=REPEAT_VISIBILITIES,
        extra_contract_tags={"experiment": "repeat-run"},
    )
    assert_corpus_identical(conditions)
    assert_world_static(conditions[0].spec, profile)
    return conditions


def _model(route, *, base_url, api_key_env, timeout):
    return OpenAICompatibleModel(
        base_url=base_url, model=route, api_key_env=api_key_env,
        timeout=timeout, max_transport_retries=1,
    )


def run_grid(*, routes, profiles, reps: int, out_root: Path, model_for,
             resume: bool) -> int:
    """reps x routes x kinds x 2 conditions; each rep is a full cell re-run."""
    cells = []
    for route in routes:
        for kind in REPEAT_KINDS:
            if kind not in profiles:
                continue
            cells.append((route, profiles[kind]))
    n_runs = len(cells) * reps * len(REPEAT_ARMS)
    print(f"REPEAT RUN: {len(cells)} (route, kind) pairs x {reps} rep(s) x "
          f"{len(REPEAT_ARMS)} conditions = {n_runs} runs")

    summary = []
    for k in range(1, reps + 1):
        for route, profile in cells:
            conditions = _repeat_conditions(profile=profile, route=route)
            for condition in conditions:
                out = (out_root / f"rep{k}" / route /
                       f"{profile.profile_id}_{condition.name}")
                if out.exists() and resume and (out / _REPORT).exists():
                    print(f"skip rep{k} {route} {profile.profile_id} "
                          f"{condition.name} (complete)")
                    continue
                if out.exists():
                    shutil.rmtree(out)
                try:
                    result = run_condition(
                        condition=condition, out_dir=out,
                        model=model_for(route),
                    )
                except Exception as exc:
                    traceback.print_exc()
                    summary.append({"rep": k, "route": route,
                                    "profile": profile.profile_id,
                                    "cell": condition.name,
                                    "error": f"{type(exc).__name__}: {exc}"})
                    continue
                branch = result["branches"][0]
                summary.append({"rep": k, "route": route,
                                "profile": profile.profile_id,
                                "cell": condition.name, **result})
                print(f"rep{k} {route} {profile.profile_id} {condition.name}: "
                      f"gate={branch['gate']} gold={branch['gold_delivered']}",
                      flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "repeat_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"wrote {path}; runs={len(summary)} "
          f"errors={sum(1 for r in summary if 'error' in r)}")
    return 0


# ------------------------------------------------------------------- analyze


def _iter_run_dirs(src: Path):
    """Every directory under src that holds a finished run."""
    for report in sorted(src.rglob(_REPORT)):
        yield report.parent


def _cell_key(src: Path, run_dir: Path):
    """(route, cell_group, rep_id) for one run directory.

    Two layouts are recognised.  ``runs_repeat/rep<k>/<route>/<cell>/`` gives
    rep_id=rep<k>, route=<route>, cell=<cell>.  A flat ``.../<route>/<cell>/``
    (e.g. ``runs_expiry_hy3/expiry/hy3/<profile>_<arm>_<vis>``) has no rep
    level, so each condition directory is treated as one rep of the
    (route, profile) cell — which is exactly how the expiry grid's six
    conditions double as a smoke input for this analyser.
    """
    rel = run_dir.relative_to(src)
    parts = rel.parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"rep\d+", part) and index + 2 == len(parts) - 1:
            return parts[index + 1], parts[index + 2], part
    route = parts[-2] if len(parts) >= 2 else "?"
    cell = parts[-1]
    group = _COND_SUFFIX_RE.sub("", cell)
    return route, group, cell


def _fetches(events_path: Path) -> int:
    """world_time tool.call records, excluding harness-injected ones.

    An ``injected`` record is the do(U=1) intervention from the forced-fetch
    experiment, not the model's choice; counting it would make every forced
    cell look like a fetch cell.
    """
    count = 0
    if not events_path.exists():
        return count
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        data = event.get("data", {})
        if (event.get("type") == "tool.call"
                and data.get("tool_name") == "world_time"
                and not data.get("injected")):
            count += 1
    return count


def _final_digest(run_dir: Path, events_path: Path) -> str:
    """Digest of the last non-empty assistant content in the run."""
    last = ""
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            data = event.get("data", {})
            if (event.get("type") == "assistant.message"
                    and (data.get("text") or "").strip()):
                last = data["text"]
    if not last:
        report_path = run_dir / _REPORT
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for branch in report.get("branches", []):
                text = (branch.get("outcome") or {}).get("assistant_text") or ""
                if text.strip():
                    last = text
    return digest(last) if last else ""


def analyze(*, src: Path, report_path: Path) -> int:
    cells: dict = {}
    for run_dir in _iter_run_dirs(src):
        route, group, rep_id = _cell_key(src, run_dir)
        events = run_dir / "branches" / "probe" / "events.jsonl"
        entry = cells.setdefault((route, group), {"route": route,
                                                  "cell": group, "reps": []})
        entry["reps"].append({
            "rep": rep_id,
            "dir": str(run_dir.relative_to(src)),
            "fetches": _fetches(events),
            "answer_digest": _final_digest(run_dir, events),
        })

    rows = []
    unanimous = flipped = 0
    for (route, group), entry in sorted(cells.items()):
        reps = entry["reps"]
        digests = {r["answer_digest"] for r in reps if r["answer_digest"]}
        fetched = [r["fetches"] > 0 for r in reps]
        cell_unanimous = len(digests) <= 1
        cell_flip = any(fetched) and not all(fetched)
        unanimous += cell_unanimous
        flipped += cell_flip
        rows.append({
            "route": route, "cell": group, "n_runs": len(reps),
            "fetches": [r["fetches"] for r in reps],
            "distinct_answers": len(digests),
            "unanimous": cell_unanimous,
            "fetch_flip": cell_flip,
            "reps": reps,
        })
        print(f"  {route:22s} {group:26s} n={len(reps)} "
              f"fetches={[r['fetches'] for r in reps]} "
              f"distinct_answers={len(digests)} "
              f"unanimous={cell_unanimous} flip={cell_flip}")

    n = len(rows)
    report = {
        "src": str(src),
        "cells": rows,
        "n_cells": n,
        "unanimous_cell_proportion": round(unanimous / n, 4) if n else None,
        "fetch_flip_rate": round(flipped / n, 4) if n else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nREPEAT ANALYZE: {n} cells | unanimous {unanimous}/{n} "
          f"({report['unanimous_cell_proportion']}) | fetch flips "
          f"{flipped}/{n} ({report['fetch_flip_rate']})")
    print(f"  wrote {report_path}")
    return 0


# ------------------------------------------------------------------------ cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="MockModel, offline; evidence under "
                           "_tmp/repeat-dryrun/")
    mode.add_argument("--analyze", action="store_true",
                      help="walk --src and report per-cell stability")
    parser.add_argument("--routes", nargs="+", default=None,
                        help=f"defaults to {list(REPEAT_ROUTES)}")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="subset of the fixed REPEAT_KINDS list")
    parser.add_argument("--reps", type=int, default=4,
                        help="additional repetitions per cell (default 4)")
    parser.add_argument("--out", default="runs_repeat")
    parser.add_argument("--src", default="runs_repeat",
                        help="root to walk in --analyze mode")
    parser.add_argument("--report", default="results_v4/repeat_analysis.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.analyze:
        return analyze(src=Path(args.src), report_path=Path(args.report))

    routes = tuple(args.routes) if args.routes else REPEAT_ROUTES
    try:
        assert_in_scope(routes)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    profiles = _profiles(REPEAT_KINDS)
    if args.profiles:
        wanted = set(args.profiles)
        unknown = wanted - set(REPEAT_KINDS)
        if unknown:
            print(f"--profiles outside the fixed repeat set: {sorted(unknown)}",
                  file=sys.stderr)
            return 2
        profiles = {k: p for k, p in profiles.items() if k in wanted}
    if not profiles:
        print("no scenarios selected", file=sys.stderr)
        return 2

    if args.dry_run:
        out_root = Path("_tmp") / "repeat-dryrun"
        return run_grid(routes=routes, profiles=profiles, reps=args.reps,
                        out_root=out_root, model_for=lambda route: MockModel(),
                        resume=False)

    import os
    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set",
              file=sys.stderr)
        return 2
    models = {}

    def model_for(route):
        if route not in models:
            models[route] = _model(route, base_url=args.base_url,
                                   api_key_env=args.api_key_env,
                                   timeout=args.timeout)
        return models[route]

    return run_grid(routes=routes, profiles=profiles, reps=args.reps,
                    out_root=Path(args.out), model_for=model_for,
                    resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
