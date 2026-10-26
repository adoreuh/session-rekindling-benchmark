"""Build the blind judge's input from a completed run directory.

Kept as a separate step, and a separate file, because blinding is a property of
*what is handed to the judge*, not of the judge's good intentions.  Building the
input here means the redaction is auditable and re-runnable: a reader can check
that the condition label never entered the payload, rather than trusting that the
judge was asked not to look.

Three redactions, each for a stated reason.

**Condition and pressure are removed from the branch id.**  The judge sees an
opaque local id.  It has to be stable so results can be joined back, and it has
to leak nothing so the judge cannot infer which condition it is reading.  A
sequence of ids derived from a hash of the branch id gives both.

**Internal markers never enter.**  Document ids, the gold marker, the exposure
contract, and the delivered-document list are experiment vocabulary.  A judge
that has seen ``visa_reschedule_official_20260305`` knows which branch is which,
and a judge that has seen the delivered list knows the answer before reading it.

**The world truth is supplied as dates, not as a document.**  The judge is given
the operative date and the superseded date and asked whether the answer reflects
them.  Giving it the gold *document* would let it grade reading comprehension
instead of whether the answer used the evidence.

The probe type is the one piece of design information the judge receives, and it
is necessary: with no assertion there is nothing for ``false_premise_endorsed``
to be about.  It is derived from the scenario's own condition key, not from the
branch label, so a mislabelled branch cannot mislead the judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _opaque_id(branch_id: str, salt: str) -> str:
    """A stable, non-informative local id.

    A hash of the branch id rather than a position, so the judge cannot infer
    grouping from ordering — consecutive ids would pair the two branches of one
    comparison, which is exactly the contrast it must not know about.
    """
    digest = hashlib.sha256(f"{salt}|{branch_id}".encode("utf-8")).hexdigest()
    return f"b{digest[:10]}"


def _human_date(iso: str) -> str:
    from datetime import date

    parsed = date.fromisoformat(iso)
    return f"{parsed.year} 年 {parsed.month} 月 {parsed.day} 日"


def build(run_dir: Path, *, scenario_module: str = "tools.scenario_v4",
          salt: str = "v4-judge") -> dict:
    import importlib

    module = importlib.import_module(scenario_module)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    official_facts = {
        "current_date": _human_date(module.RIGHT_DATE),
        "superseded_date": _human_date(module.WRONG_DATE),
    }

    branches = []
    mapping = {}
    for branch in report["branches"]:
        branch_id = branch["branch_id"]
        # The probe type comes from the scenario's own condition key -- every v4
        # probe asserts a false date -- rather than from parsing the branch id.
        # Parsing a label would make the judge's framing depend on the naming
        # convention rather than on the design.
        probe_type = "false_claim"
        opaque = _opaque_id(branch_id, salt)
        mapping[opaque] = branch_id
        branches.append({
            "branch_id": opaque,
            "probe_type": probe_type,
            "user_text": branch["user_text"],
            "answer": branch["outcome"]["assistant_text"],
        })

    return {
        "official_facts": official_facts,
        "branches": branches,
        # Written to a sibling file by the caller, not into the judge's input.
        "_mapping": mapping,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--scenario-module", default="tools.scenario_v4")
    parser.add_argument("--salt", default="v4-judge")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    payload = build(run_dir, scenario_module=args.scenario_module, salt=args.salt)
    mapping = payload.pop("_mapping")

    out = Path(args.out) if args.out else run_dir / "responses.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "judge_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Blinding, asserted rather than intended.  The judge's input is scanned for
    # every term that would tell it which condition it is reading.
    forbidden = (
        "oracle", "omit", "natural", "gold", "visa_reschedule",
        "pressure", "low", "high", "condition", "branch",
    )
    blob = json.dumps(payload, ensure_ascii=False).lower()
    leaks = sorted({term for term in forbidden if term in blob})
    # "branch" appears in the documented key name, so it is expected; the rest
    # are not.  Checked explicitly so the tolerance is visible and narrow.
    leaks = [t for t in leaks if t not in ("branch",)]
    print(f"wrote {out}")
    print(f"wrote {run_dir / 'judge_mapping.json'}")
    print(f"branches handed to the judge: {len(payload['branches'])}")
    print(f"official facts: {payload['official_facts']}")
    if leaks:
        print(f"BLINDING LEAK: judge input contains {leaks}", file=sys.stderr)
        return 1
    print("blinding check: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
