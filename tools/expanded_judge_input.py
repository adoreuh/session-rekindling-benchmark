"""Build blind-judge inputs for the expanded runs, per scenario and per contrast.

Why a new builder rather than extending the pilot's.  The pilot's version assumed a
single scenario module with two fixed dates.  The expanded runs span eight scenario
profiles, each with its own event name and dates, plus two contrasts with different
official facts — so the dates and the event vocabulary have to be derived from the
scenario definition the run was built from, not from a module-level constant.

Two redactions, and the reason for each:

**Condition and pressure are stripped from the branch id.**  The judge must not be
able to infer which condition it is reading, or it will grade the condition rather
than the answer.  A hash of the branch id gives a stable join key that carries no
information.

**The official facts are given as dates, in the renderings the answer might use.**
Not as a document: handing the judge the notice would turn the question into reading
comprehension of the notice, whereas the question is whether the *answer* treated
the operative date as operative.

The probe type is the one design fact the judge receives, because
``false_premise_endorsed`` has nothing to be about without it.  It is derived from
the scenario's own construction — every scenario here asserts a date the world has
superseded — rather than parsed from a label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.clock import parse_instant  # noqa: E402
from tools.scenario_factory import PROFILES, build_ladder  # noqa: E402

#: The pressure levels judged.  The decoy control is judged too: whether it
#: endorsed the decoy date is exactly the datapoint that separates
#: following-the-user from following-the-stale-document.
JUDGED_BRANCHES = ("claim_low_omit", "claim_low_oracle",
                   "claim_high_omit", "claim_high_oracle",
                   "decoy_low_omit")


def _opaque(branch_id: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}|{branch_id}".encode("utf-8")).hexdigest()
    return f"b{digest[:10]}"


def _render(value: str) -> str:
    """A date in the spaced Chinese form the answers use."""
    year, month, day = value.split("-")
    return f"{int(month)} 月 {int(day)} 日"


def _profile(profile_id: str):
    """Look the profile up in either pool.

    Both pools are consulted because the runs under one root may come from either:
    the pilot used the hand-written eight, the expansion used the generated forty
    kinds.  Checking only the hand-written pool left every generated profile's
    official facts empty, which the judge correctly rejected — a lookup gap that
    presented as a judge failure.
    """
    found = next((p for p in PROFILES if p.profile_id == profile_id), None)
    if found is not None:
        return found
    from tools.scenario_expansion import EVENT_KINDS, INSTANCE_VARIANTS, build_profiles

    pool = build_profiles(kinds=EVENT_KINDS, instances=1,
                          instance_variants=INSTANCE_VARIANTS)
    return next((p for p in pool if p.profile_id == profile_id), None)


def _delivery_facts(run) -> dict:
    """Official facts for a delivery run: the profile's superseded and new dates."""
    profile = _profile(run["profile"])
    if profile is None:
        return {}
    return {
        "current_date": _render(profile.new_date.isoformat()),
        "superseded_date": _render(profile.old_date.isoformat()),
    }


def _temporal_facts(run, elapsed: str) -> dict:
    """Official facts for a temporal position: the stage live at that instant.

    Derived from the ladder and the elapsed label, which is how the run itself was
    built; the alternative — reading the delivered notice — would put the evidence
    in front of the judge.
    """
    profile = _profile(run["profile"])
    if profile is None:
        return {}
    try:
        days = float(elapsed.rstrip("d")) if elapsed.rstrip("d") else 0.0
    except ValueError:
        return {}
    update = parse_instant(profile.update_day.isoformat() + "T09:00:00+08:00")
    instant = (update + timedelta(days=days)).isoformat()
    from tools.scenario_factory import authoritative_stage

    stages = build_ladder(profile)
    stage = authoritative_stage(stages, instant)
    return {
        "current_date": _render(stage.sets.isoformat()),
        "superseded_date": _render(stage.voids.isoformat()),
    }


#: The elapsed labels the temporal runs use.  Matched explicitly rather than by
#: splitting on an underscore, because profile ids also end in an underscore-plus-digit
#: segment ("appliance_repair_i1").  Splitting on the last underscore turned that into
#: the profile "appliance_repair_i" with the elapsed "1", which then produced empty
#: official facts and made the judge reject every branch — a failure that looked like
#: a judge problem and was a naming collision.
ELAPSED_SUFFIXES = ("5d", "15d", "1d", "30d", "61d")


def _split_leaf(leaf: str) -> tuple:
    """Split ``<profile>_<elapsed>`` into its parts, matching the elapsed label."""
    for suffix in ELAPSED_SUFFIXES:
        marker = f"_{suffix}"
        if leaf.endswith(marker):
            return leaf[: -len(marker)], suffix
    return leaf, ""


def build_for(kind: str, run_dir: Path, *, salt: str) -> dict:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if kind == "temporal":
        profile_id, elapsed = _split_leaf(run_dir.name)
        facts = _temporal_facts({"profile": profile_id}, elapsed)
    else:
        facts = _delivery_facts({"profile": run_dir.name})

    branches = []
    mapping = {}
    for branch in report["branches"]:
        if branch["branch_id"] not in JUDGED_BRANCHES:
            continue
        if not branch.get("run_valid"):
            continue
        opaque = _opaque(branch["branch_id"], salt)
        mapping[opaque] = branch["branch_id"]
        branches.append({
            "branch_id": opaque,
            "probe_type": "false_claim",
            "user_text": branch.get("user_text", ""),
            "answer": branch["outcome"].get("assistant_text", ""),
            "official_facts": facts,
        })
    return {"official_facts": facts, "branches": branches, "_mapping": mapping}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="a run root, e.g. runs_expanded_hy3")
    parser.add_argument("--salt", default="expanded-judge")
    parser.add_argument("--out-suffix", default="judge_inputs.jsonl")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"missing root: {root}", file=sys.stderr)
        return 2

    written = 0
    for report_path in sorted(root.rglob("report.json")):
        run_dir = report_path.parent
        parts = run_dir.relative_to(root).parts
        kind = parts[0] if parts else ""
        if kind not in ("delivery", "temporal"):
            continue
        payload = build_for(kind, run_dir, salt=args.salt)
        if not payload["branches"]:
            continue
        (run_dir / "responses.json").write_text(
            json.dumps({"official_facts": payload["official_facts"],
                        "branches": payload["branches"]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "judge_mapping.json").write_text(
            json.dumps(payload["_mapping"], ensure_ascii=False, indent=2,
                       sort_keys=True),
            encoding="utf-8",
        )
        written += 1

        # Blinding check, asserted rather than intended: the judge's input must not
        # name the condition.
        blob = json.dumps(payload["branches"], ensure_ascii=False).lower()
        leaks = [term for term in ("oracle", "omit", "decoy", "delivered_not_used")
                 if term in blob]
        if leaks:
            print(f"BLINDING LEAK in {run_dir}: {leaks}", file=sys.stderr)
            return 1

    print(f"wrote {written} judge inputs under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
