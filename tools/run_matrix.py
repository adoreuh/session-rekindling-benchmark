"""Run the experiment matrix against real model routes.

Design decisions worth stating.

**The run directory is evidence.**  Each run writes its own log, checkpoint and
frozen policy, and a run directory is never overwritten: a second run with the
same name is refused.  If a run must be redone, it gets a new directory so the
first one remains available for comparison.

**Failures are kept.**  A branch whose provider call fails is recorded with its
failure, not dropped.  Whether a condition fails more often than another is
itself a result — the contract requires the full allocation to be reported, not
just the completed branches.

**The key never enters an artifact.**  The adapter is given the *name* of an
environment variable; only that name is recorded.  A post-run scan asserts this.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.decay import resolve_profile  # noqa: E402
from vela_lab.provider import OpenAICompatibleModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools import scenario_ewm  # noqa: E402
from tools.scope import MODELLED_ROUTES, OutOfScopeRoute, assert_in_scope  # noqa: E402

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"

#: Design versions this runner can execute.  They are **not interchangeable**:
#: v3 measures the combined effect of a conflict probe on retrieval and on the
#: answer, whereas v4 removes self-retrieval and controls delivery directly.  A
#: pooled table would mix two different measurements under one label, so the
#: version is recorded on every report and the analyser refuses to pool them.
DESIGNS = ("v3", "v4")


def build_model(model_name: str, *, base_url: str, api_key_env: str,
                temperature: float | None, timeout: float):
    return OpenAICompatibleModel(
        base_url=base_url,
        model=model_name,
        api_key_env=api_key_env,
        timeout=timeout,
        temperature=temperature,
        max_transport_retries=1,
    )


def build_spec_for(*, design: str, model_name: str, elapsed: str,
                   decay_name: str, elapsed_days: int | None):
    """Build the spec for one cell, dispatching on the design version."""
    if design == "v4":
        from tools.v4_wiring import build_v4_spec

        # v4 takes a day count directly.  Its ``elapsed`` label still comes from
        # the scenario naming, so the two designs' labels mean the same thing
        # (days after the world update) even though their mechanics differ.
        days = elapsed_days if elapsed_days is not None else _days_from_label(elapsed)
        return build_v4_spec(
            experiment_id=f"ewm4-{model_name}-{elapsed}",
            elapsed_days=days,
        )
    decay = resolve_profile(decay_name)
    return scenario_ewm.build_spec(
        experiment_id=f"ewm-{model_name}-{elapsed}-{decay_name}",
        elapsed=elapsed, decay=decay, elapsed_days=elapsed_days,
    )


def _days_from_label(label: str) -> int:
    """``"15d"`` -> 15.  Kept explicit rather than parsed loosely: a label that
    cannot be read is an error, not a default."""
    text = str(label).strip().lower().rstrip("d")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"elapsed label {label!r} is not '<n>d'; pass --elapsed-days for an "
            "explicit day count"
        ) from exc


def run_one(*, model_name: str, out_root: Path, elapsed: str, decay_name: str,
            base_url: str, api_key_env: str, temperature: float | None,
            timeout: float, elapsed_days: int | None = None,
            design: str = "v3") -> dict:
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; expected one of {DESIGNS}")
    run_dir = out_root / design / model_name / f"{elapsed}_{decay_name}"
    spec = build_spec_for(
        design=design, model_name=model_name, elapsed=elapsed,
        decay_name=decay_name, elapsed_days=elapsed_days,
    )
    model = build_model(
        model_name, base_url=base_url, api_key_env=api_key_env,
        temperature=temperature, timeout=timeout,
    )
    started = time.monotonic()
    report = ExperimentRunner(spec, model, out_dir=run_dir).run()
    report["_meta"] = {
        "design": design,
        "model": model_name,
        "elapsed": elapsed,
        "decay": decay_name,
        "elapsed_days": spec.advance_days,
        "wall_seconds": round(time.monotonic() - started, 1),
        "transport_attempts": getattr(model, "attempt_log", []),
    }
    path = run_dir / "report.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def summarise(report: dict) -> dict:
    """Per-branch outcome at a glance, before the judge runs."""
    rows = []
    for branch in report["branches"]:
        outcome = branch["outcome"]
        auto = branch.get("auto_retrieval", {})
        score = branch.get("score") or {}
        l2 = (score.get("L2") or {}).get("retrievals") or [{}]
        verdicts = branch.get("exposure_verdicts") or []
        rows.append(
            {
                "branch_id": branch["branch_id"],
                "label": branch["label"],
                "valid": branch["run_valid"],
                "invalid_reason": branch.get("invalid_reason", ""),
                "audit_findings": len(branch["audit_findings"]),
                "failure": (outcome.get("failure") or {}).get("reason", ""),
                "requests": outcome["request_count"],
                "tool_calls": outcome["tool_calls"],
                "auto_outcome": auto.get("outcome", ""),
                "gold_delivered": auto.get("gold_delivered"),
                "delivered": auto.get("hits"),
                "result_count": l2[-1].get("result_count"),
                "gold_rank": l2[-1].get("gold_rank"),
                "gate": verdicts[-1].get("status") if verdicts else "",
                "l4": (score.get("L4") or {}).get("label", ""),
                "answer_chars": len(outcome.get("assistant_text", "")),
            }
        )
    return {
        "rows": rows,
        "parent": report.get("parent_equality", {}),
        "exposure": report.get("exposure", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--design", choices=DESIGNS, default="v3",
                        help="which scenario design to run; the two are not "
                             "poolable, so the choice is recorded per run")
    parser.add_argument("--elapsed", nargs="+", default=["15d"])
    parser.add_argument("--decay", nargs="+", default=["m01_memorybank"])
    parser.add_argument("--elapsed-days", type=int, default=None,
                        help="override the elapsed value with an explicit day count")
    parser.add_argument("--out", default="runs_real")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="build the spec and report the matrix without calling")
    args = parser.parse_args()

    if args.dry_run:
        try:
            assert_in_scope(args.models)
        except OutOfScopeRoute as exc:
            print(f"scope violation: {exc}", file=sys.stderr)
            return 3
        for model_name in args.models:
            for elapsed in args.elapsed:
                spec = build_spec_for(
                    design=args.design, model_name=model_name, elapsed=elapsed,
                    decay_name=args.decay[0], elapsed_days=args.elapsed_days,
                )
                print(f"=== {args.design} {model_name} {elapsed} ===")
                print(f"branches: {len(spec.branches)}")
                for branch in spec.branches:
                    print(f"  {branch.branch_id:28s} {branch.user_text[:34]}")
                print(f"tools: {list(spec.enabled_tools)}")
                print(f"advance_days: {spec.advance_days}")
                print(f"spec digest: {spec.digest()[:16]}")
        print(f"total runs: {len(args.models) * len(args.elapsed) * len(args.decay)}")
        print(f"frozen scope: {list(MODELLED_ROUTES)}")
        print(f"requested:    {args.models}")
        return 0

    # The model set is frozen.  Checked before any provider call so a mistaken
    # `--models` argument fails immediately instead of producing a run whose
    # results table would look like every other run's.
    try:
        assert_in_scope(args.models)
    except OutOfScopeRoute as exc:
        print(f"scope violation: {exc}", file=sys.stderr)
        return 3

    if not os.environ.get(args.api_key_env):
        print(f"environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    everything = {}
    for model_name in args.models:
        for elapsed in args.elapsed:
            for decay_name in args.decay:
                key = f"{args.design}|{model_name}|{elapsed}|{decay_name}"
                print(f"\n=== {key} ===", flush=True)
                try:
                    report = run_one(
                        model_name=model_name, out_root=out_root, elapsed=elapsed,
                        decay_name=decay_name, base_url=args.base_url,
                        api_key_env=args.api_key_env,
                        temperature=args.temperature, timeout=args.timeout,
                        elapsed_days=args.elapsed_days, design=args.design,
                    )
                except Exception as exc:
                    print(f"  RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
                    everything[key] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                summary = summarise(report)
                everything[key] = summary
                wall = report["_meta"]["wall_seconds"]
                print(f"  wall {wall}s | parent identical: "
                      f"{summary['parent']['identical_parent']}", flush=True)
                # The gate is reported per branch, because a branch that was
                # rejected never reached the model: its absence from the answer
                # counts is a measurement fact, not a model behaviour.
                for row in summary["rows"]:
                    flag = "ok " if row["valid"] else "BAD"
                    print(
                        f"  {flag} {row['branch_id']:26s} "
                        f"gate={row['gate'] or '-':12s} "
                        f"gold={str(row['gold_delivered']):5s} "
                        f"n={str(row['result_count']):4s} "
                        f"req={row['requests']} ans={row['answer_chars']:5d} "
                        f"{row['l4'] or row['invalid_reason'] or row['failure'][:28]}",
                        flush=True,
                    )

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "matrix_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(everything, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nwrote {out_root / 'matrix_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
