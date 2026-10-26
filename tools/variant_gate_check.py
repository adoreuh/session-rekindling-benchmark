"""Run every generated scenario variant through the full offline gate path.

This is the check that decides whether the generator is usable at scale.  A
variant that builds is not a variant that measures: the gate has to pass on the
actual serialized request, the pair has to differ only in the slot, and the two
conditions have to be equal in delivered volume.

One instance here is one scenario, and the runner produces four branches from it.
Running all eight gives the 32-branch development matrix's offline half — no
provider call, so it can be re-run freely and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.scenario_factory import FILLER_ID, PROFILES, build_variant  # noqa: E402
from tools.variant_wiring import build_variant_spec  # noqa: E402


def check_variant(profile, *, out_root: Path, elapsed_days: int = 15) -> dict:
    variant = build_variant(
        scenario_id=f"v4-{profile.profile_id}-15d",
        profile=profile, filler_id=FILLER_ID,
    )
    spec = build_variant_spec(
        variant=variant, experiment_id=f"v4-{profile.profile_id}-15d",
        elapsed_days=elapsed_days,
    )
    out = out_root / profile.profile_id
    if out.exists():
        shutil.rmtree(out)
    # A fixed reply: the model's behaviour is irrelevant here.  What is under
    # test is whether the harness can build and check the request at all.
    report = ExperimentRunner(
        spec, MockModel(scripted=["我理解你的担心。"]), out_dir=out
    ).run()

    problems = []
    for branch in report["branches"]:
        verdicts = branch["exposure_verdicts"]
        if not verdicts:
            problems.append(f"{branch['branch_id']}: never checked")
            continue
        if not branch["run_valid"]:
            problems.append(
                f"{branch['branch_id']}: invalid ({branch['invalid_reason']}) "
                f"{branch.get('gate_rejection')}"
            )
        statuses = {v["status"] for v in verdicts}
        if statuses != {"PASS"}:
            problems.append(f"{branch['branch_id']}: gate {sorted(statuses)}")

    # The pairing property, read from the recorded delivery rather than assumed.
    by_id = {b["branch_id"]: b for b in report["branches"]}
    pairing = {}
    for pressure in ("low", "high"):
        oracle = by_id[f"claim_{pressure}_oracle"]["auto_retrieval"]["hits"]
        omit = by_id[f"claim_{pressure}_omit"]["auto_retrieval"]["hits"]
        diff = [i for i, (a, b) in enumerate(zip(oracle, omit)) if a != b]
        pairing[pressure] = {
            "same_count": len(oracle) == len(omit),
            "differing_slots": diff,
            "slot_differs_only": len(oracle) == len(omit) and len(diff) == 1,
        }
        if not pairing[pressure]["slot_differs_only"]:
            problems.append(f"pair {pressure} does not differ only in the slot")

    bodies = {d[0]: d[2] for d in variant.documents}
    lengths = {}
    for pressure in ("low", "high"):
        o = sum(len(bodies[i]) for i in by_id[f"claim_{pressure}_oracle"]["auto_retrieval"]["hits"])
        m = sum(len(bodies[i]) for i in by_id[f"claim_{pressure}_omit"]["auto_retrieval"]["hits"])
        ratio = max(o, m) / max(min(o, m), 1)
        lengths[pressure] = {"oracle_chars": o, "omit_chars": m, "ratio": round(ratio, 3)}
        if ratio > 1.5:
            problems.append(f"pair {pressure} text volume ratio {ratio:.2f} > 1.5")

    parent = report.get("exposure", {}).get("parent_check") or {}
    if parent.get("status") != "PASS":
        problems.append(f"parent check {parent.get('status')}: {parent.get('reason_codes')}")

    return {
        "profile_id": profile.profile_id,
        "scenario_id": variant.scenario_id,
        "gold_id": variant.gold_id,
        "documents": len(variant.documents),
        "pairing": pairing,
        "lengths": lengths,
        "parent_check": parent.get("status"),
        "parent_equality": report.get("parent_equality", {}),
        "problems": problems,
        "ok": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_gate/v4_factory")
    parser.add_argument("--report", default="results_v4/factory_check.json")
    parser.add_argument("--elapsed-days", type=int, default=15)
    args = parser.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)

    results = []
    print("OFFLINE VARIANT CHECK (no provider call)")
    print("=" * 76)
    for profile in PROFILES:
        result = check_variant(profile, out_root=out_root,
                              elapsed_days=args.elapsed_days)
        results.append(result)
        mark = "ok  " if result["ok"] else "FAIL"
        low = result["lengths"]["low"]["ratio"]
        print(f"  [{mark}] {result['profile_id']:20s} docs={result['documents']:2d} "
              f"ratio={low} parent={result['parent_check']}")
        for problem in result["problems"]:
            print(f"         - {problem}")

    ok = sum(1 for r in results if r["ok"])
    failed = [r["profile_id"] for r in results if not r["ok"]]
    print()
    print(f"{ok}/{len(results)} scenarios clean")
    if failed:
        print(f"failed: {failed}")

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"results": results, "clean": ok, "total": len(results),
                    "failed": failed},
                   ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
