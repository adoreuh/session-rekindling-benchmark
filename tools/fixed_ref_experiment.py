"""E3 fixed-reference (CN=0) controls: the probe states its own reference instant.

Design: the expiry corpus, delivery, contract and gates are reused verbatim --
the world is static, the record places the event on ``D``, and the question is
the same state-neutral "how should I arrange tonight?".  What changes is the
probe: each arm pins a *reference date* ``t_ref``, so the gold answer flips on
``D`` without the model ever needing the wall clock:

* ``ref_before`` -- "answer as of D-15": the event is still upcoming under the
  stated reference, so a correct answer treats the record as current;
* ``ref_after``  -- "answer as of D+30": the event is already past under the
  stated reference, so a correct answer treats the record as historical.

Because the reference instant is fixed in the probe text, acquiring
``world_time`` cannot change the answer: the tool stays enabled (an identical
tool surface) but fetching it is uninformative.  ``F = P(model fetches)`` under
this CN=0 regime therefore measures *indiscriminate* checking rather than
need-driven monitoring, and the selective-monitoring contrast is
``STM = P(F | CN=1) - P(F | CN=0)`` against the ``natural`` expiry cells, where
the clock is needed.

The world clock still advances to the stated reference instant -- the same
``target_instant`` / ``_advance_from`` machinery, fed the arm's offset -- so a
fetch would return exactly the date the probe already named: consistent, and
provably uninformative rather than conflicting.

Only ``natural`` visibility is run: ``observe_time=False``, nothing renders the
clock.  2 arms x 1 visibility = 2 cells per (route, profile).

Provenance note: the gate requires every authored string on screen to be
registered as dialogue.  ``expiry_experiment.build_conditions`` registers the
base ``PROBE`` and the frozen background but not per-arm probes, so E3 probes
-- which carry a date ``PROBE`` does not -- are registered here by wrapping
``make_request_gate``: every probe this module mints is appended to the
registered dialogue before the gate checks a request.  Following the v4
wiring's rule, the *whole* probe set is registered, not just the arm running.
"""
from __future__ import annotations

import os
import tempfile as _tempfile
from pathlib import Path

# Workspace-local scratch for the offline dry run: under the Windows sandbox,
# tempfile in the platform default temp hits ACL denials, so TEMP/TMP are
# pointed at a workspace directory *before* tempfile is imported.  Forced, not
# setdefault: the shell already exports TEMP, which would leave the default
# path in effect exactly where the denial lives.
_tmp = Path(__file__).resolve().parents[1] / "_tmp"
_tmp.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(_tmp)
os.environ["TMP"] = str(_tmp)

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import MockModel, OpenAICompatibleModel  # noqa: E402
from vela_lab.errors import ValidationError  # noqa: E402
import tools.expiry_experiment as _expiry  # noqa: E402
from tools.expiry_experiment import (  # noqa: E402
    ALLOWED_FIELD_DIFFERENCES,
    DEFAULT_BASE_URL,
    PROBE,
    assert_no_supersession,
    assert_world_static,
    build_conditions,
    build_expiry_variant,
    corpus_fingerprint,
    differing_fields,
    run_condition,
)
from vela_lab.jsonutil import digest  # noqa: E402
from tools.scenario_factory import _md, _md_full  # noqa: E402
from tools.scenario_expansion import build_profiles as build_expansion  # noqa: E402
from tools.scope import OutOfScopeRoute, SUBJECT_ROUTES, assert_in_scope  # noqa: E402


#: The 14 pure-time profiles: the first instance of each sampled kind in the
#: expansion pool.  Frozen as a constant for the same reason the route set is
#: frozen in tools/scope.py -- the subject set is fixed before the data, and
#: widening it after seeing results would be selective sampling.
E3_PROFILE_IDS = (
    "visa_interview_i1",
    "thesis_defense_i1",
    "equipment_check_i1",
    "employment_checkup_i1",
    "broadband_install_i1",
    "car_maintenance_i1",
    "furniture_measure_i1",
    "homestay_checkin_i1",
    "ielts_registration_i1",
    "kindergarten_interview_i1",
    "lease_signing_i1",
    "loan_interview_i1",
    "specialist_clinic_i1",
    "tax_interview_i1",
)

#: The two reference-date arms, as whole days relative to the event date D.
#: The same offset drives both the probe's stated reference and the world
#: clock's landing instant, so a world_time fetch would only repeat what the
#: probe already asserted.
E3_ARMS = (("ref_before", -15), ("ref_after", 30))
REF_OFFSETS = dict(E3_ARMS)

#: One visibility: ``natural`` (observe_time=False).  The clock is reachable
#: only through the elective world_time tool -- which is the behaviour under
#: measurement.
E3_VISIBILITIES = (("natural", False),)

CONTRACT_TAG = {"experiment": "E3-fixed-ref"}

#: Reason codes that would mean the fixed reference leaked into a decisive
#: assertion or the delivery broke; any one of them fails the cell.
BAD_GATE_CODES = frozenset({
    "decisive_leak", "unregistered_content", "insufficient_evidence",
    "slot_mismatch", "parent_contaminated", "undeclared_material_present",
    "declared_material_missing", "forbidden_tool_exposed",
    "undeclared_tool_exposed",
})


# ------------------------------------------------------------------ probes


#: Every probe this module has minted.  Registration (below) reads the whole
#: set, mirroring the v4 rule that registering all probes -- not only the
#: running branch's -- keeps registration a property of the scenario.
_SEEN_PROBES: list = []


def ref_date_for_arm(arm: str, variant) -> object:
    """``D + arm_offset``: the date the probe instructs the model to stand on."""
    if arm not in REF_OFFSETS:
        raise ValidationError(f"E3: unknown arm {arm!r}")
    return variant.profile.old_date + timedelta(days=REF_OFFSETS[arm])


def probe_for_arm(arm: str, variant) -> str:
    """The probe for one arm: the base probe prefixed by its reference date.

    The reference date is derived from the profile's own ``old_date`` -- never
    hardcoded -- so each of the 14 profiles states its own D-15 / D+30.  The
    suffix is the expiry probe verbatim: same question, same neutrality, and
    answerable on both sides of the stated reference.
    """
    ref = ref_date_for_arm(arm, variant)
    text = f"请以 {_md_full(ref)}为基准回答：{PROBE}"
    _SEEN_PROBES.append(text)
    return text


def _probe_rendering_leaks(probe: str, profile) -> list:
    """Renderings of the profile's own dates found inside the probe.

    The probe's ref date is a *different* date than D by construction, so none
    of ``old_date``/``new_date``'s renderings -- spaced, unspaced or ISO -- may
    appear.  A hit would mean the probe carries a term a sufficiency group or
    a non-decisive proposition could match, which is exactly the leak this
    check exists to surface.
    """
    leaks = []
    for label, day in (("old_date", profile.old_date),
                       ("new_date", profile.new_date)):
        forms = (
            day.isoformat(),
            _md(day),
            _md_full(day),
            f"{day.year}年{day.month}月{day.day}日",
            f"{day.month}月{day.day}日",
        )
        for form in forms:
            if form and form in probe:
                leaks.append(f"{label}:{form}")
    return sorted(set(leaks))


# Register every minted probe as authored dialogue.  ``gate_factory`` inside
# ``build_conditions`` resolves ``make_request_gate`` from the module globals
# at run time -- after all probes have been minted -- so wrapping it here is
# the only in-process edit needed; the expiry module itself is untouched.
# Each probe is registered as a singleton *group* so the residue check strips
# it as a whole literal rather than character-by-character.
_make_request_gate_orig = _expiry.make_request_gate


def _registering_make_request_gate(contract, manifest, allowed_surfaces,
                                   registered_dialogue, **kwargs):
    probes = tuple(dict.fromkeys(_SEEN_PROBES))
    # The probes must be *prepended*.  Inside ``trace_block`` a flat string in
    # ``registered_dialogue`` is treated as a group of *characters* and strips
    # every occurrence of each character globally, so the base ``PROBE``
    # registration would eat the probe's suffix before a whole-probe entry
    # placed after it could match.  Singleton groups first means the probe is
    # stripped as one literal; the character-level entries then clean up the
    # same way they always did.
    return _make_request_gate_orig(
        contract, manifest, allowed_surfaces,
        tuple((p,) for p in probes) + tuple(registered_dialogue),
        **kwargs,
    )


_expiry.make_request_gate = _registering_make_request_gate


# ------------------------------------------------------------------ cells


def assert_corpus_identical_e3(conditions) -> dict:
    """The expiry identity check, with ``branches`` excised from the corpus.

    Identical in intent to ``assert_corpus_identical``: a whole-field diff
    against the allow-list, then a fingerprint over the corpus fields, so the
    two cells can differ only in the clock, the ids and the probe text the
    branch carries.  It exists separately because the patched shared version
    raises ``KeyError`` on ``extra_allowed`` fields when building its ``parts``
    audit -- the keys are popped from the fingerprint dicts before ``parts``
    reads them.  Excising ``branches`` here is the design's claim, not a
    workaround for its own sake: the probe *is* the manipulated variable.
    """
    specs = [condition.spec for condition in conditions]
    differing = differing_fields(specs)
    allowed = ALLOWED_FIELD_DIFFERENCES | {"branches"}
    unexpected = sorted(differing - allowed)
    if unexpected:
        raise ValidationError(
            "the E3 conditions differ outside the allowed fields "
            f"{sorted(allowed)}: {unexpected}"
        )
    canonical = []
    for spec in specs:
        fingerprint = corpus_fingerprint(spec)
        fingerprint.pop("branches", None)
        canonical.append(fingerprint)
    root = digest(canonical[0])
    for spec, other in zip(specs[1:], canonical[1:]):
        if digest(other) != root:
            changed = sorted(
                name for name in canonical[0]
                if canonical[0][name] != other[name]
            )
            raise ValidationError(
                f"E3 {spec.experiment_id}: the corpus differs from the other "
                f"cell in {changed}"
            )
    return {
        "fingerprint": root,
        "differing_fields": sorted(differing),
        "unexpected": unexpected,
        "allowed": sorted(ALLOWED_FIELD_DIFFERENCES),
        "parts": {
            name: digest([entry[name] for entry in canonical])
            for name in ("documents", "world_events", "prefix_turns",
                         "stable_material", "host_query", "prompt")
        },
    }


def build_e3_conditions(*, variant, route):
    """The two cells for one (route, profile), built on the expiry machinery."""
    conditions = build_conditions(
        variant=variant, route=route,
        arms=E3_ARMS, visibilities=E3_VISIBILITIES,
        probe_for_arm=probe_for_arm,
        extra_contract_tags=CONTRACT_TAG,
    )
    identity = assert_corpus_identical_e3(conditions)
    if identity["unexpected"]:
        raise ValidationError(
            f"E3 corpus drifted in unexpected fields: {identity['unexpected']}"
        )
    # Judge-facing verdict fields: the reference the probe stated and the
    # state the event is in under that reference.
    for condition in conditions:
        ref = ref_date_for_arm(condition.arm, variant)
        expiry = condition.spec.delivery["expiry"]
        expiry["ref_instant"] = condition.target
        expiry["ref_date"] = ref.isoformat()
        expiry["ref_offset_days"] = REF_OFFSETS[condition.arm]
        expiry["gold_state"] = (
            "upcoming" if ref < variant.profile.old_date else "past"
        )
    return conditions


# ------------------------------------------------------------------ dry run


def dry_run(*, routes, profiles, json_path: str = "") -> int:
    """Build and run both cells offline, checking what must hold.

    Offline means ``MockModel``: no provider call, no API key read.  The gate
    is the real gate on the real compiled request.  Scratch lives under the
    workspace ``_tmp`` (see the module head) rather than the platform temp.
    """
    print("E3 FIXED-REFERENCE DRY RUN (CN=0 control, MockModel, offline)")
    print("  arms: ref_before = D-15, ref_after = D+30; visibility: natural "
          "(observe_time=False)")
    print(f"  probe: '请以 <t_ref>为基准回答：{PROBE}'")
    scratch_root = Path(tempfile.gettempdir()) / "e3-dryrun-work"
    scratch_root.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for route in routes:
        for profile in profiles:
            variant = build_expiry_variant(profile=profile)
            audit = assert_no_supersession(variant)
            print(f"\n  === route {route} | profile {profile.profile_id} "
                  f"| D={profile.old_date.isoformat()} ===")
            print(f"    ref_before={ref_date_for_arm('ref_before', variant)} "
                  f"ref_after={ref_date_for_arm('ref_after', variant)}")
            try:
                conditions = build_e3_conditions(variant=variant, route=route)
            except Exception as exc:
                failures.append(
                    f"{route}/{profile.profile_id}: build failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"    BUILD FAILED: {type(exc).__name__}: {exc}")
                continue
            world = assert_world_static(conditions[0].spec, profile)
            print(f"    world static: PASS (kinds={world['kinds']}) | "
                  f"supersession audit: {'PASS' if audit['clean'] else 'FAIL'}")
            for condition in conditions:
                spec = condition.spec
                expiry = spec.delivery["expiry"]
                probe = spec.branches[0].user_text
                leaks = _probe_rendering_leaks(probe, profile)
                out_dir = (scratch_root
                           / f"{route}-{profile.profile_id}-{condition.name}")
                # The dry run's scratch is disposable evidence, not a run dir:
                # clear it so a re-run of the offline check does not trip the
                # runner's refuse-to-overwrite guard on leftovers.
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                try:
                    summary = run_condition(
                        condition=condition, out_dir=out_dir, model=MockModel(),
                    )
                except Exception as exc:
                    failures.append(
                        f"{route}/{profile.profile_id}/{condition.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"      BAD {condition.name:20s} RUN FAILED: "
                          f"{type(exc).__name__}: {exc}")
                    continue
                branch = summary["branches"][0]
                parent = summary["parent_check"]
                reasons = sorted({code for codes in branch["gate_reasons"]
                                  for code in codes})
                bad = sorted(set(reasons) & BAD_GATE_CODES)
                clean = (
                    branch["valid"]
                    and set(branch["gate"]) == {"PASS"}
                    and parent.get("status") == "PASS"
                    and not bad
                    and not leaks
                )
                mark = "ok " if clean else "BAD"
                if not clean:
                    failures.append(
                        f"{route}/{profile.profile_id}/{condition.name}: "
                        f"valid={branch['valid']} gate={branch['gate']} "
                        f"parent={parent.get('status')} reasons={reasons} "
                        f"leaks={leaks} "
                        f"parent_reasons={parent.get('reason_codes')}"
                    )
                print(f"      {mark} {condition.name:20s} "
                      f"target={condition.target} "
                      f"ref={expiry['ref_date']} ({expiry['gold_state']}) "
                      f"gate={'/'.join(branch['gate'])} "
                      f"parent={parent.get('status')} "
                      f"gold={branch['gold_delivered']} "
                      f"leaks={leaks or 'none'}")
                rows.append({
                    "route": route, "profile": profile.profile_id,
                    "cell": condition.name,
                    "target_instant": condition.target,
                    "ref_instant": expiry["ref_instant"],
                    "ref_date": expiry["ref_date"],
                    "gold_state": expiry["gold_state"],
                    "probe": probe,
                    "probe_leaks": leaks,
                    "observe_time": spec.observe_time,
                    "gate": branch["gate"],
                    "gate_reasons": reasons,
                    "parent": parent.get("status"),
                    "gold_delivered": branch["gold_delivered"],
                    "valid": branch["valid"],
                })
    if json_path:
        Path(json_path).write_text(
            json.dumps({"rows": rows, "failures": failures},
                       ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\n  wrote {json_path}")
    print(f"\nE3 DRY RUN: {len(rows)} cells, {len(failures)} failures")
    for failure in failures:
        print("  -", failure)
    return 1 if failures else 0


# ------------------------------------------------------------------ paid run


def _model(route, *, base_url, api_key_env, timeout):
    return OpenAICompatibleModel(
        base_url=base_url, model=route,
        api_key_env=api_key_env, timeout=timeout, max_transport_retries=1,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", nargs="+", default=None)
    p.add_argument("--profiles", nargs="+", default=None,
                   help="defaults to the frozen E3_PROFILE_IDS set")
    p.add_argument("--out", default="runs_e3_fixed_ref")
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
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    pool = {pr.profile_id: pr for pr in build_expansion(instances=4)}
    wanted = list(args.profiles) if args.profiles else list(E3_PROFILE_IDS)
    missing = [pid for pid in wanted if pid not in pool]
    if missing:
        print(f"unknown profile id(s): {missing}", file=sys.stderr)
        return 2
    profiles = [pool[pid] for pid in wanted]
    if not profiles:
        print("no scenarios selected", file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(routes=routes, profiles=profiles,
                       json_path=args.dry_run_json)

    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set",
              file=sys.stderr)
        return 2

    planned = []
    for route in routes:
        for profile in profiles:
            variant = build_expiry_variant(profile=profile)
            assert_no_supersession(variant)
            conditions = build_e3_conditions(variant=variant, route=route)
            assert_world_static(conditions[0].spec, profile)
            planned.append((route, profile, conditions))
    calls = sum(len(c) for _, _, c in planned)
    print(f"E3 FIXED-REF: {len(planned)} pairs, {calls} paid calls")

    out_root = Path(args.out)
    everything = []
    for route, profile, conditions in planned:
        for condition in conditions:
            key = f"E3 | {route} | {profile.profile_id} | {condition.name}"
            out = (out_root / "e3" / route
                   / f"{profile.profile_id}_{condition.name}")
            if out.exists() and args.resume and (out / "report.json").exists():
                print(f"skip {key}")
                continue
            if out.exists():
                shutil.rmtree(out)
            try:
                result = run_condition(
                    condition=condition, out_dir=out,
                    model=_model(route, base_url=args.base_url,
                                 api_key_env=args.api_key_env,
                                 timeout=args.timeout),
                )
            except Exception as exc:
                traceback.print_exc()
                everything.append({
                    "route": route, "profile": profile.profile_id,
                    "cell": condition.name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            everything.append({
                "route": route, "profile": profile.profile_id, **result,
            })
            print(f"{key}: wall {result['wall_seconds']}s "
                  f"gate={result['branches'][0]['gate']}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "e3_summary.json").write_text(
        json.dumps(everything, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_root / 'e3_summary.json'}; runs={len(everything)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
