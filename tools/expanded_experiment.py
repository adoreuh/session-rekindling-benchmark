"""The expanded experiment: delivery and temporal contrasts across all subjects.

Two contrasts, run in one pass because they share the scenario corpus, the gates
and the model adapters:

**delivery** — for each (route, scenario): the five branches of the controlled
pair.  ``claim_{low,high}_{omit,oracle}`` carry the manipulation; ``decoy_low_omit``
is the control that separates "followed the user's assertion" from "followed the
stale shared document", because the shared context contains the old confirmation
which asserts the same date the user does.

**temporal** — for each (route, scenario, ladder stage): the oracle branches only.
The ladder's chain-complete record cannot supply an omission condition, since a
reschedule notice asserts the user's date is void — the same content the omit
condition must lack — so the temporal contrast lives inside the oracle condition.

Why one runner rather than two.  Both contrasts need the same per-scenario setup,
the same exposure gate, and the same event log treatment.  Two runners would mean
two places where "what the model was sent" is assembled, and the project's whole
premise is that there is exactly one such place.

Execution hygiene:

* every run directory is fresh, because a run directory is evidence and is never
  overwritten;
* a failed call is recorded, not retried into a success — but a transport failure
  that the adapter itself retried is marked indeterminate rather than silently
  counted, since the first attempt may have reached the provider;
* the scope gate runs before any paid call.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import OpenAICompatibleModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    PROFILES,
    build_ladder_variant,
    build_variant,
)
from tools.scope import (  # noqa: E402
    OutOfScopeRoute,
    SUBJECT_ROUTES,
    assert_in_scope,
)
from tools.variant_wiring import (  # noqa: E402
    build_ladder_spec,
    build_variant_spec,
)

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"

#: Ladder positions for the temporal contrast, as days after the world update.
#: Chosen one either side of the place the ladder changes stage, so both regimes
#: appear in every scenario; the boundary itself is a property of the ladder.
TEMPORAL_POSITIONS = (5.0, 15.0)


def _model(model_name: str, *, base_url: str, api_key_env: str,
           timeout: float) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        base_url=base_url, model=model_name, api_key_env=api_key_env,
        timeout=timeout, max_transport_retries=1,
    )


def _summary_of(report: dict) -> dict:
    rows = []
    for branch in report["branches"]:
        rows.append({
            "branch_id": branch["branch_id"],
            "valid": branch["run_valid"],
            "invalid_reason": branch.get("invalid_reason", ""),
            "gate": [v["status"] for v in branch["exposure_verdicts"]],
            "delivered": branch["auto_retrieval"].get("hits"),
            "gold_delivered": branch["auto_retrieval"].get("gold_delivered"),
            "l4": (branch.get("score") or {}).get("L4", {}).get("label", ""),
            "answer": branch["outcome"].get("assistant_text", ""),
            "failure": (branch["outcome"].get("failure") or {}).get("reason", ""),
        })
    return {"branches": rows,
            "parent_equality": report.get("parent_equality", {})}


def run_delivery(*, route: str, profile, out_root: Path, base_url: str,
                 api_key_env: str, timeout: float) -> dict:
    variant = build_variant(
        scenario_id=f"delivery-{route}-{profile.profile_id}",
        profile=profile, filler_id=FILLER_ID,
    )
    spec = build_variant_spec(
        variant=variant,
        experiment_id=f"delivery-{route}-{profile.profile_id}",
        elapsed_days=15,
    )
    out = out_root / "delivery" / route / profile.profile_id
    if out.exists():
        shutil.rmtree(out)
    started = time.monotonic()
    report = ExperimentRunner(
        spec, _model(route, base_url=base_url, api_key_env=api_key_env,
                     timeout=timeout), out_dir=out,
    ).run()
    return {
        "contrast": "delivery",
        "route": route,
        "profile": profile.profile_id,
        "wall_seconds": round(time.monotonic() - started, 1),
        **_summary_of(report),
    }


def run_temporal(*, route: str, profile, days: float, out_root: Path,
                 base_url: str, api_key_env: str, timeout: float) -> dict:
    from datetime import timedelta

    from vela_lab.clock import parse_instant

    update = parse_instant(profile.update_day.isoformat() + "T09:00:00+08:00")
    instant = (update + timedelta(days=days)).isoformat()
    ladder = build_ladder_variant(
        profile=profile, filler_id=FILLER_ID, instant=instant
    )
    label = f"{days:g}d"
    spec = build_ladder_spec(
        ladder=ladder, experiment_id=f"temporal-{route}-{profile.profile_id}-{label}",
        elapsed_days=days,
    )
    out = out_root / "temporal" / route / f"{profile.profile_id}_{label}"
    if out.exists():
        shutil.rmtree(out)
    started = time.monotonic()
    report = ExperimentRunner(
        spec, _model(route, base_url=base_url, api_key_env=api_key_env,
                     timeout=timeout), out_dir=out,
    ).run()
    return {
        "contrast": "temporal",
        "route": route,
        "profile": profile.profile_id,
        "elapsed_label": label,
        "elapsed_days": days,
        "live_stage": ladder.stage.stage_id,
        "correct_date": ladder.correct_date.isoformat(),
        "wall_seconds": round(time.monotonic() - started, 1),
        **_summary_of(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", nargs="+", default=None,
                        help="defaults to scope.SUBJECT_ROUTES")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="scenario profile ids; defaults to all")
    parser.add_argument("--source", choices=("factory", "expansion"),
                        default="factory",
                        help="factory = the pilot's hand-written pool; expansion = "
                             "the multi-kind pool")
    parser.add_argument("--per-kind", type=int, default=0,
                        help="with --source expansion, take this many instances per "
                             "event kind; 0 means all")
    parser.add_argument("--instance-variants", type=int, default=4,
                        help="instances per kind to build before --per-kind filters")
    parser.add_argument("--contrast", choices=("delivery", "temporal", "both"),
                        default="both")
    parser.add_argument("--out", default="runs_expanded")
    parser.add_argument("--temporal-positions", nargs="+", type=float, default=None,
                        help="ladder positions in days after the world update; defaults to "
                             "TEMPORAL_POSITIONS, whose pair straddles the stage boundary. "
                             "A pair inside one stage is the pure elapsed-time contrast.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    routes = tuple(args.routes) if args.routes else SUBJECT_ROUTES
    try:
        assert_in_scope(routes)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    # The scenario source.  The hand-written pool is the pilot's; the expansion
    # pool is what the larger runs use, because eight scenarios cannot support a
    # claim about situations in general — and because per-kind instance counts let
    # a run trade breadth against depth explicitly rather than by accident.
    if args.source == "expansion":
        from tools.scenario_expansion import build_profiles

        pool = build_profiles(instances=args.instance_variants)
        if args.per_kind > 0:
            seen = {}
            selected = []
            for profile in pool:
                kind = profile.profile_id.rsplit("_i", 1)[0]
                seen.setdefault(kind, 0)
                if seen[kind] < args.per_kind:
                    seen[kind] += 1
                    selected.append(profile)
            pool = selected
        profiles = pool
    else:
        profiles = [p for p in PROFILES
                    if args.profiles is None or p.profile_id in set(args.profiles)]
    if args.profiles:
        wanted = set(args.profiles)
        profiles = [p for p in profiles if p.profile_id in wanted]
    if not profiles:
        print("no scenarios selected", file=sys.stderr)
        return 2

    if args.dry_run:
        delivery_calls = len(routes) * len(profiles) * 5
        temporal_calls = len(routes) * len(profiles) * len(TEMPORAL_POSITIONS) * 2
        print("EXPANDED EXPERIMENT PLAN")
        print(f"  source   : {args.source}")
        print(f"  routes   : {len(routes)} -> {list(routes)}")
        kinds = {p.profile_id.rsplit("_i", 1)[0] for p in profiles}
        print(f"  scenarios: {len(profiles)} across {len(kinds)} event kinds")
        print(f"  delivery : {len(routes)}x{len(profiles)}x5 = {delivery_calls} calls")
        print(f"  temporal : {len(routes)}x{len(profiles)}x"
              f"{len(TEMPORAL_POSITIONS)}x2 = {temporal_calls} calls")
        print(f"  total    : {delivery_calls + temporal_calls} calls")
        print(f"  per route: {(delivery_calls + temporal_calls) // max(len(routes), 1)}")
        return 0

    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    everything = []
    jobs = []
    if args.contrast in ("delivery", "both"):
        jobs.append(("delivery", None))
    if args.contrast in ("temporal", "both"):
        for days in (args.temporal_positions or TEMPORAL_POSITIONS):
            jobs.append(("temporal", days))

    for route in routes:
        for profile in profiles:
            for kind, days in jobs:
                key = f"{kind} | {route} | {profile.profile_id}" + (
                    f" | {days:g}d" if days is not None else ""
                )
                print(f"\n=== {key} ===", flush=True)
                try:
                    if kind == "delivery":
                        result = run_delivery(
                            route=route, profile=profile, out_root=out_root,
                            base_url=args.base_url, api_key_env=args.api_key_env,
                            timeout=args.timeout,
                        )
                    else:
                        result = run_temporal(
                            route=route, profile=profile, days=days,
                            out_root=out_root, base_url=args.base_url,
                            api_key_env=args.api_key_env, timeout=args.timeout,
                        )
                except Exception as exc:
                    print(f"  RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    everything.append({
                        "contrast": kind, "route": route,
                        "profile": profile.profile_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                everything.append(result)
                valid = sum(1 for b in result["branches"] if b["valid"])
                print(f"  wall {result['wall_seconds']}s | "
                      f"valid {valid}/{len(result['branches'])}", flush=True)
                for row in result["branches"]:
                    flag = "ok " if row["valid"] else "BAD"
                    print(f"    {flag} {row['branch_id']:18s} "
                          f"l4={row['l4']:20s} "
                          f"ans={len(row['answer']):5d} "
                          f"{row['invalid_reason'] or row['failure'][:40]}",
                          flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "expanded_summary.json"
    summary.write_text(
        json.dumps(everything, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {summary}")
    print(f"runs: {len(everything)}; "
          f"errors: {sum(1 for r in everything if 'error' in r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
