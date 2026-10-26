"""Retry the transport-failed units of a large run, then report what is missing.

Why a separate step, and why the retry is bounded.

A big run over five routes meets two failure shapes that are not the model's doing:
the endpoint's rate limit (HTTP 429) when one route is driven hard, and plain
transport errors.  The design's rule — stated before any of this ran — is that a
transport failure allows the **same unit** to be retried and never allows a new
scenario to be substituted for it.  A retry that swapped in a fresh scenario would
be resampling after seeing a failure, which is exactly what the stopping rule
forbids.

So this script:

* finds units with a ``report.json`` absent, or present but with no valid branch;
* re-runs only those, with the same scenario and the same conditions;
* reports anything still missing rather than papering over it, because a scenario
  that cannot be run is a fact about the study, not an inconvenience.

Bounded by ``--rounds`` so a persistently failing unit cannot become an unbounded
loop that keeps spending calls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.scenario_expansion import build_profiles  # noqa: E402


def _has_usable_report(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    branches = report.get("branches") or []
    return any(branch.get("run_valid") for branch in branches)


def scan(root: Path, route: str, expected_profiles) -> dict:
    """What exists and what is missing under one route's run directory."""
    route_dir = root / "delivery" / route
    present, absent = [], []
    for profile in expected_profiles:
        path = route_dir / profile.profile_id / "report.json"
        (present if _has_usable_report(path) else absent).append(profile.profile_id)

    temporal_present, temporal_absent = [], []
    temporal_dir = root / "temporal" / route
    for profile in expected_profiles:
        for days in ("5d", "15d"):
            path = temporal_dir / f"{profile.profile_id}_{days}" / "report.json"
            (temporal_present if _has_usable_report(path)
             else temporal_absent).append(f"{profile.profile_id}_{days}")
    return {
        "route": route, "root": str(root),
        "delivery_present": len(present), "delivery_absent": absent,
        "temporal_present": len(temporal_present),
        "temporal_absent": temporal_absent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--root-prefix", default="runs_scale_")
    parser.add_argument("--per-kind", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--sleep-between", type=float, default=30.0,
                        help="pause before each route's retry, to let a rate "
                             "limit reset")
    parser.add_argument("--report", default="results_v4/scale_coverage.json")
    args = parser.parse_args()

    expected = build_profiles(instances=4, per_kind=None) if False else build_profiles(
        instances=args.per_kind
    )
    from tools.scenario_expansion import EVENT_KINDS

    # Rebuild exactly what the runs were given: one instance per kind.
    from tools.scenario_expansion import INSTANCE_VARIANTS

    expected = build_profiles(kinds=EVENT_KINDS, instances=1,
                              instance_variants=INSTANCE_VARIANTS)
    print(f"expected profiles: {len(expected)}")

    coverage = {}
    for route in args.routes:
        safe = route.replace(".", "_")
        root = Path(f"{args.root_prefix}{safe}")
        if not root.exists():
            print(f"missing root for {route}: {root}")
            continue
        coverage[route] = scan(root, route, expected)

    # The retry pass: only the units the scan found absent.
    for round_index in range(args.rounds):
        pending = [(route, row) for route, row in coverage.items()
                   if row["delivery_absent"] or row["temporal_absent"]]
        if not pending:
            break
        print()
        print(f"--- retry round {round_index + 1} ---")
        for route, row in pending:
            profiles = sorted({entry.rsplit("_", 1)[0] for entry in
                               row["temporal_absent"]})
            profiles += [p for p in row["delivery_absent"]
                         if p not in profiles]
            if not profiles:
                continue
            safe = route.replace(".", "_")
            print(f"  {route}: retrying {len(profiles)} scenarios")
            time.sleep(args.sleep_between)
            command = [
                sys.executable, "-m", "tools.expanded_experiment",
                "--source", "expansion", "--per-kind", "1",
                "--routes", route, "--profiles", *profiles,
                "--contrast", "both", "--out", f"{args.root_prefix}{safe}",
                "--timeout", str(args.timeout),
            ]
            # Output is streamed to this process's stdout rather than captured:
            # a retry pass takes many minutes and silence during it is
            # indistinguishable from a hang, which makes it impossible to tell
            # whether the retry is working or the endpoint is refusing everything.
            subprocess.run(command, timeout=7200)
            coverage[route] = scan(root, route, expected)
            print(f"    -> now delivery {coverage[route]['delivery_present']}/40 "
                  f"temporal {coverage[route]['temporal_present']}/80", flush=True)

    print()
    print("COVERAGE")
    print("=" * 78)
    for route, row in sorted(coverage.items()):
        d_ok = row["delivery_present"]
        t_ok = row["temporal_present"]
        print(f"  {route:24s} delivery {d_ok}/40  temporal {t_ok}/80")
        if row["delivery_absent"]:
            print(f"      delivery missing: {row['delivery_absent'][:6]}"
                  f"{' ...' if len(row['delivery_absent']) > 6 else ''}")
        if row["temporal_absent"]:
            print(f"      temporal missing: {row['temporal_absent'][:6]}"
                  f"{' ...' if len(row['temporal_absent']) > 6 else ''}")

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path}")

    total_ok = sum(row["delivery_present"] + row["temporal_present"]
                   for row in coverage.values())
    total_expected = len(coverage) * 120
    print(f"usable runs: {total_ok}/{total_expected}")
    return 0 if total_ok == total_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
