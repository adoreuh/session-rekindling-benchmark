"""E2 rendered-relation: do(D) — the temporal relation is rendered, not computed.

Design: same corpus and probe as `natural`/`forced`, but the model-visible
context additionally carries a neutral, structured statement of the temporal
relation between the current instant and the event date:

    Current world time: 2026-06-05
    Event date:         2026-05-06
    Temporal relation:  event date is in the past

It does NOT say "therefore the appointment is invalid" — that would answer
Z2 for the model.  E2 intervenes on D (detection): if rendered-relation still
fails, the break is at deactivation, not at computing T > tau.

Arms: post, late.  Visibilities: `relation` only.
Injection: three assistant-visible lines via a synthetic `world_time`-style
tool result is wrong (it would look like the model asked); instead the runner
injects a `time_context` observation naming the relation — the same channel
`delivered` uses, extended with the explicit relation line.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows temp ACL under the sandbox can refuse TemporaryDirectory cleanup;
# park scratch inside the workspace before tempfile is first used.
_tmp = Path(__file__).resolve().parents[1] / "_tmp"
_tmp.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_tmp)
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)

from vela_lab.provider import MockModel, OpenAICompatibleModel  # noqa: E402
from vela_lab.errors import ValidationError  # noqa: E402
from vela_lab.jsonutil import digest  # noqa: E402
from vela_lab.runner import ExperimentSpec  # noqa: E402
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
    target_instant,
    parse_instant,
    DEFAULT_BASE_URL,
    Condition,
)
from tools.scenario_factory import PROFILES as FACTORY_PROFILES  # noqa: E402
from tools.scenario_expansion import build_profiles as build_expansion  # noqa: E402
from tools.scope import OutOfScopeRoute, SUBJECT_ROUTES, assert_in_scope  # noqa: E402

E2_ARMS = tuple((name, days) for name, days in ARMS if name != "pre")
#: visibility name is "relation": the rendered relation is the condition.
E2_VISIBILITIES = (INTERVENTION_VISIBILITIES[1],)  # declared by the experiment,)


def _relation_text(variant, arm: str) -> str:
    """The neutral rendered relation — facts of position, never the verdict."""
    p = variant.profile
    target = parse_instant(target_instant(variant, dict(E2_ARMS)[arm]))
    d = p.old_date
    days = (target.date() - d).days
    rel = "已经过去" if days > 0 else ("尚未到来" if days < 0 else "今天")
    return (
        f"当前世界时间: {target.date().isoformat()}\n"
        f"事项原定日期: {d.isoformat()}\n"
        f"时间关系: 事项日期{rel}（相隔 {abs(days)} 天）"
    )


class _RelationSession:
    """Wrap the branch session: inject the rendered relation as time_context."""

    def __init__(self, session, text):
        self._s = session
        self._text = text

    def __getattr__(self, name):
        return getattr(self._s, name)

    def add_time_context(self):
        self._s.observations.time_context(
            self._text, event_seq=self._s.head_seq() + 1,
            digest_=digest(self._text))
        self._s.append("observation.compiled", {
            "kind": "time_context", "text": self._text,
            "injected": "relation",
        })


def build_relation_conditions(*, variant, route):
    """post/late × relation; observe_time False, relation injected manually."""
    # Both arms' renders are registered: the gate must permit the relation text
    # it is about to see, and neither render asserts a registered proposition.
    registered = tuple(_relation_text(variant, name) for name, _ in E2_ARMS)
    conditions = build_conditions(
        variant=variant, route=route,
        arms=E2_ARMS, visibilities=E2_VISIBILITIES,
        extra_registered_text=registered,
        extra_contract_tags={"experiment": "E2-rendered-relation"},
    )
    # observe_time=False so the spec's own time_context never fires; the
    # relation text is delivered by wrapping add_time_context at run time.
    assert_corpus_identical(conditions)
    return conditions


def run_condition_relation(*, condition: Condition, variant, out_dir: Path,
                           model) -> dict:
    """run_condition with the rendered relation injected as time_context."""
    import vela_lab.runner as _runner
    orig = getattr(_runner, "_RelationSession", None)
    text = _relation_text(variant, condition.arm)

    # Patch Session.add_time_context for this run only: the relation text
    # replaces the bare clock render.  observe_time=False means the runner
    # never calls add_time_context on its own, so we inject by subclassing the
    # Session object used inside _run_branch — simplest is a spec-level flag
    # the runner does not know; instead inject post-fork via a shim that wraps
    # the returned child session.  The clean hook: patch ExperimentRunner's
    # branch session after fork is not exposed, so wrap run_condition by
    # monkeypatching Session.add_time_context on the class for the duration.
    import vela_lab.session as _sess
    orig_atc = _sess.Session.add_time_context

    def _patched(self):
        self.observations.time_context(
            text, event_seq=self.head_seq() + 1, digest_=digest(text))
        self.append("observation.compiled", {
            "kind": "time_context", "text": text, "injected": "relation",
        })

    _sess.Session.add_time_context = _patched
    try:
        # observe_time is False in the spec, so call the patched method by
        # flipping a copy of the spec?  Simpler: temporarily pretend
        # observe_time True so the runner invokes our patched method.
        import dataclasses
        spec2 = dataclasses.replace(condition.spec, observe_time=True)
        cond2 = Condition(arm=condition.arm, visibility=condition.visibility,
                          target=condition.target, spec=spec2)
        return run_condition(condition=cond2, out_dir=out_dir, model=model)
    finally:
        _sess.Session.add_time_context = orig_atc


def dry_run(*, routes, profiles, json_path="") -> int:
    print("E2 RENDERED-RELATION DRY RUN (MockModel, offline)")
    rows, failures = [], []
    scratch = Path(os.environ["TEMP"]) / "e2-dryrun"
    scratch.mkdir(parents=True, exist_ok=True)
    for route in routes:
        for profile in profiles:
            variant = build_expiry_variant(profile=profile)
            assert_no_supersession(variant)
            conditions = build_relation_conditions(variant=variant, route=route)
            for condition in conditions:
                out_dir = scratch / f"{route}-{profile.profile_id}-{condition.name}"
                try:
                    summary = run_condition_relation(
                        condition=condition, variant=variant,
                        out_dir=out_dir, model=MockModel())
                except Exception as exc:
                    failures.append(f"{route}/{profile.profile_id}/{condition.name}: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    continue
                ev = out_dir / "branches" / "probe" / "events.jsonl"
                texts = ev.read_text(encoding="utf-8") if ev.exists() else ""
                ok = "时间关系" in texts and "injected" in texts
                if not ok:
                    failures.append(f"{route}/{profile.profile_id}/{condition.name}: relation not visible")
                rows.append({"route": route, "profile": profile.profile_id,
                             "cell": condition.name, "ok": ok})
                print(f"  {'ok' if ok else 'BAD'} {route:22s} {profile.profile_id:24s} {condition.name}")
    if json_path:
        Path(json_path).write_text(json.dumps({"rows": rows, "failures": failures},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nE2 DRY RUN: {len(rows)} cells, {len(failures)} failures")
    return 1 if failures else 0


def _model(route, *, base_url, api_key_env, timeout):
    return OpenAICompatibleModel(base_url=base_url, model=route,
        api_key_env=api_key_env, timeout=timeout, max_transport_retries=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", nargs="+", default=None)
    p.add_argument("--profiles", nargs="+", default=None)
    p.add_argument("--out", default="runs_e2_relation")
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
    if not profiles:
        print("no scenarios selected", file=sys.stderr); return 2

    if args.dry_run:
        return dry_run(routes=routes, profiles=profiles, json_path=args.dry_run_json)

    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr); return 2

    planned = []
    for route in routes:
        for profile in profiles:
            variant = build_expiry_variant(profile=profile)
            assert_no_supersession(variant)
            conditions = build_relation_conditions(variant=variant, route=route)
            planned.append((route, profile, variant, conditions))
    calls = sum(len(c) for _, _, _, c in planned)
    print(f"E2 RENDERED-RELATION: {len(planned)} pairs, {calls} paid calls")

    out_root = Path(args.out)
    everything = []
    for route, profile, variant, conditions in planned:
        for condition in conditions:
            key = f"E2 | {route} | {profile.profile_id} | {condition.name}"
            out = out_root / "e2" / route / f"{profile.profile_id}_{condition.name}"
            if out.exists() and args.resume and (out / "report.json").exists():
                print(f"skip {key}"); continue
            if out.exists():
                shutil.rmtree(out)
            try:
                result = run_condition_relation(
                    condition=condition, variant=variant, out_dir=out,
                    model=_model(route, base_url=args.base_url,
                                 api_key_env=args.api_key_env, timeout=args.timeout))
            except Exception as exc:
                traceback.print_exc()
                everything.append({"route": route, "profile": profile.profile_id,
                    "cell": condition.name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            everything.append({"route": route, "profile": profile.profile_id, **result})
            print(f"{key}: wall {result['wall_seconds']}s")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "e2_summary.json").write_text(
        json.dumps(everything, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_root / 'e2_summary.json'}; runs={len(everything)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
