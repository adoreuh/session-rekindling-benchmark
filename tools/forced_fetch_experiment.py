"""E1 forced-fetch: do(U=1) — authoritative time injected as a tool result.

Design: the probe and all context are byte-identical to the `natural` expiry
arm, but before the model answers, the harness executes `world_time` and
inserts the result into model-visible history as a real tool exchange.  The
user question is NOT modified — this is `do(U=1)`, not "ask the model to
check".  Comparisons vs `natural` (self-selected fetch) and `delivered`
(rendered time) are paired McNemar on the same (route, profile, arm) cells.

Arms: post (D+30), late (D+120) — pre has no expiry to fail on.
Visibilities: `forced` only; natural and delivered already exist on disk.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import MockModel, OpenAICompatibleModel  # noqa: E402
from vela_lab.errors import ValidationError  # noqa: E402
from tools.expiry_experiment import (
    INTERVENTION_VISIBILITIES,  # noqa: E402
    ARMS,
    build_conditions,
    build_expiry_variant,
    assert_no_supersession,
    assert_corpus_identical,
    assert_world_static,
    run_condition,
    proposition_map,
    _print_audit,
    DEFAULT_BASE_URL,
)
from tools.scenario_factory import PROFILES as FACTORY_PROFILES  # noqa: E402
from tools.scenario_expansion import build_profiles as build_expansion  # noqa: E402
from tools.scope import OutOfScopeRoute, SUBJECT_ROUTES, assert_in_scope  # noqa: E402

#: post/late only — the expired regime E1 intervenes on.
FORCED_ARMS = tuple((name, days) for name, days in ARMS if name != "pre")
#: one visibility: the tool injection IS the condition.
FORCED_VISIBILITIES = (INTERVENTION_VISIBILITIES[0],)  # declared by the experiment,)
FORCED_TOOLS = ("world_time",)
CONTRACT_TAG = {"experiment": "E1-forced-fetch", "forced_tools": list(FORCED_TOOLS)}


def build_forced_conditions(*, variant, route):
    conditions = build_conditions(
        variant=variant, route=route,
        arms=FORCED_ARMS, visibilities=FORCED_VISIBILITIES,
        forced_tools=FORCED_TOOLS, extra_contract_tags=CONTRACT_TAG,
    )
    # The two arms must differ only in the clock and ids — probes, branch
    # specs, corpus are one object family.
    identity = assert_corpus_identical(conditions)
    assert identity["unexpected"] == []
    return conditions


def dry_run(*, routes, profiles, json_path="") -> int:
    print("E1 FORCED-FETCH DRY RUN (do(U=1), MockModel, offline)")
    rows, failures = [], []
    scratch_root = Path(tempfile.gettempdir()) / "e1-dryrun-work"
    # Windows temp cleanup can be ACL-denied under sandboxed shells; park the
    # scratch inside a workspace-adjacent temp dir instead of the default.
    try:
        import tempfile as _t
        probe = _t.TemporaryDirectory(dir=str(scratch_root.parent.parent))
        probe.cleanup()
    except Exception:
        pass
    scratch_root.mkdir(parents=True, exist_ok=True)
    import os as _os
    scratch = str(scratch_root)
    if True:
        for route in routes:
            for profile in profiles:
                variant = build_expiry_variant(profile=profile)
                assert_no_supersession(variant)
                conditions = build_forced_conditions(variant=variant, route=route)
                for condition in conditions:
                    out_dir = Path(scratch) / f"{route}-{profile.profile_id}-{condition.name}"
                    try:
                        summary = run_condition(
                            condition=condition, out_dir=out_dir, model=MockModel(),
                        )
                    except Exception as exc:
                        failures.append(f"{route}/{profile.profile_id}/{condition.name}: {type(exc).__name__}: {exc}")
                        continue
                    # The injection must be visible: tool.call with injected flag.
                    events = []
                    ev = out_dir / "branches" / "probe" / "events.jsonl"
                    if ev.exists():
                        events = [json.loads(l) for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()]
                    injected = [e for e in events if e.get("type") == "tool.call"
                                and e.get("data", {}).get("injected")]
                    got_time = any("current_time" in json.dumps(e.get("data", {}), ensure_ascii=False)
                                   for e in events if e.get("type") == "tool.visible_result")
                    ok = len(injected) >= 1 and got_time
                    if not ok:
                        failures.append(f"{route}/{profile.profile_id}/{condition.name}: injection not visible")
                    rows.append({"route": route, "profile": profile.profile_id,
                                 "cell": condition.name, "injected": len(injected),
                                 "time_visible": got_time})
                    print(f"  {'ok' if ok else 'BAD'} {route:22s} {profile.profile_id:24s} {condition.name:14s} injected={len(injected)} time={got_time}")
    if json_path:
        Path(json_path).write_text(json.dumps({"rows": rows, "failures": failures},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nE1 DRY RUN: {len(rows)} cells, {len(failures)} failures")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


def _model(route, *, base_url, api_key_env, timeout):
    return OpenAICompatibleModel(base_url=base_url, model=route,
        api_key_env=api_key_env, timeout=timeout, max_transport_retries=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", nargs="+", default=None)
    p.add_argument("--profiles", nargs="+", default=None)
    p.add_argument("--out", default="runs_e1_forced")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key-env", default="VELA_API_KEY")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-json", default="")
    args = p.parse_args()

    routes = tuple(args.routes) if args.routes else SUBJECT_ROUTES
    try:
        assert_in_scope(routes)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr); return 3

    pool = list(FACTORY_PROFILES) + list(build_expansion(instances=4))
    profiles = [pr for pr in pool if args.profiles is None or pr.profile_id in set(args.profiles)]
    if args.profiles:
        profiles = [pr for pr in profiles if pr.profile_id in set(args.profiles)]
    if not profiles:
        print("no scenarios selected", file=sys.stderr); return 2

    if args.dry_run:
        return dry_run(routes=routes, profiles=profiles, json_path=args.dry_run_json)

    import os
    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr); return 2

    planned = []
    for route in routes:
        for profile in profiles:
            variant = build_expiry_variant(profile=profile)
            assert_no_supersession(variant)
            conditions = build_forced_conditions(variant=variant, route=route)
            planned.append((route, profile, conditions))
    calls = sum(len(c) for _, _, c in planned)
    print(f"E1 FORCED-FETCH: {len(planned)} pairs, {calls} paid calls")

    out_root = Path(args.out)
    everything = []
    for route, profile, conditions in planned:
        for condition in conditions:
            key = f"E1 | {route} | {profile.profile_id} | {condition.name}"
            out = out_root / "e1" / route / f"{profile.profile_id}_{condition.name}"
            if out.exists() and args.resume and (out / "report.json").exists():
                print(f"skip {key}"); continue
            if out.exists():
                shutil.rmtree(out)
            try:
                result = run_condition(condition=condition, out_dir=out,
                    model=_model(route, base_url=args.base_url,
                                 api_key_env=args.api_key_env, timeout=args.timeout))
            except Exception as exc:
                # A cell's gate rejection or indeterminate provider error is
                # that cell's outcome, not the batch's: record it and keep
                # going so one bad parent does not strand the rest of the grid.
                traceback.print_exc()
                everything.append({"route": route, "profile": profile.profile_id,
                    "cell": condition.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "skipped": True})
                continue
            everything.append({"route": route, "profile": profile.profile_id, **result})
            print(f"{key}: wall {result['wall_seconds']}s gate={result['branches'][0]['gate']}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "e1_summary.json").write_text(
        json.dumps(everything, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_root / 'e1_summary.json'}; runs={len(everything)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
