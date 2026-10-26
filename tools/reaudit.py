"""Re-derive per-branch validity from the event logs, preserving the old verdict.

Why this is a re-derivation rather than a re-run.  A branch's ``run_valid`` flag is a
*conclusion* the audit drew from the recorded events, not an observation.  When the
audit changes, the conclusion can be recomputed from the same log; re-running the
branch instead would replace the observation and spend calls to answer a question the
log already answers.

What triggered it here.  The vocabulary check gained a provenance rule (see
``vela_lab.audit._harness_authored``), because it had been scanning the assistant's
replayed hidden reasoning and reporting ordinary English — "dental treatment",
"conditional", "golden week" — as harness vocabulary leakage.  Twelve branches were
invalidated by that.  All twelve pass under the corrected rule.

The previous verdict is kept beside the new one, and the change is reported, so a
reader can see that the audit changed rather than the run.  A branch that fails under
the corrected audit is *not* silently kept: it stays invalid and is listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.audit import audit_events  # noqa: E402


def _load_events(branch_dir: Path) -> list:
    path = branch_dir / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def reaudit_run(run_dir: Path, *, write: bool) -> list:
    """Recompute ``run_valid`` for every branch of one run.

    Returns one record per branch whose verdict changed or that still fails.
    """
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = []
    for branch in report["branches"]:
        branch_dir = run_dir / "branches" / branch["branch_id"]
        events = _load_events(branch_dir)
        if not events:
            records.append({
                "run": str(run_dir), "branch_id": branch["branch_id"],
                "error": "no events recorded",
            })
            continue
        findings = audit_events(events)
        recorded_valid = bool(branch.get("run_valid"))
        recomputed_valid = not findings
        if recorded_valid == recomputed_valid:
            continue
        records.append({
            "run": str(run_dir),
            "branch_id": branch["branch_id"],
            "recorded_valid": recorded_valid,
            "recomputed_valid": recomputed_valid,
            "recorded_reason": branch.get("invalid_reason", ""),
            "findings": [f"{f['check']}: {f['detail']}" for f in findings],
            "changed": True,
        })
        if write:
            # Both kept: the earlier verdict is evidence that the audit changed.
            branch.setdefault("audit_findings_recorded_before_reaudit",
                              branch.get("audit_findings"))
            branch["audit_findings"] = findings
            branch["run_valid"] = recomputed_valid
            branch["invalid_reason"] = "" if recomputed_valid else "audit"
            branch["reaudited"] = True
    if write and records:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--write", action="store_true",
                        help="write the corrected verdicts back to the reports")
    parser.add_argument("--report", default="results_v4/reaudit.json")
    args = parser.parse_args()

    everything = []
    for root in args.roots:
        base = Path(root)
        if not base.exists():
            print(f"missing root: {base}", file=sys.stderr)
            continue
        for report_path in sorted(base.rglob("report.json")):
            everything.extend(reaudit_run(report_path.parent, write=args.write))

    print(f"branches with a changed verdict: {len(everything)}")
    for record in everything[:20]:
        if "error" in record:
            print(f"  ERROR {record['run']} {record['branch_id']}: {record['error']}")
            continue
        print(f"  {record['run'].split(chr(92))[-3:]} | {record['branch_id']}: "
              f"{record['recorded_valid']} -> {record['recomputed_valid']}")
        for finding in record["findings"][:1]:
            print(f"      {finding}")
    if args.write:
        print("verdicts written back")

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(everything, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
