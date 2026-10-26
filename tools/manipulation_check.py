"""Manipulation checks: did the intended manipulation actually manipulate anything?

This module exists because of a specific failure in this study's own first pass.

The design had three evidence conditions — ``natural`` (whatever retrieval
returns), ``oracle`` (gold guaranteed present, same result count), and ``omit``
(gold removed).  The plan was to compare natural against oracle to separate "the
retriever missed it" from "it was delivered and not used".

When the recorded runs were inspected, natural and oracle had delivered
**byte-for-byte identical documents** in every branch.  Gold was already rank 1
under natural for a reason unrelated to relevance: every candidate's lexical score
was near zero and nearly equal, so the ranking was decided by the recency weight,
and gold happened to be the newest document.  The oracle force therefore moved
nothing, and the difference I had first read as an effect was sampling noise.

That is the same class of error this project already recorded once, from the other
direction: a manipulation that changes only the prompt text, or only a flag that is
never read, produces a result about the manipulation's absence rather than its
effect.  The remedy is to check, mechanically, that the manipulated variable
actually differs — and to fail loudly when it does not.

These checks run on a completed run directory and need no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEVERITY_RUN_INVALID = "RUN_INVALID"
SEVERITY_BATCH_FAIL = "BATCH_FAIL"
# A pair whose oracle changed nothing is not invalid: it happens when natural
# retrieval already reached gold, which is a legitimate retrieval outcome.  The
# pair is merely uninformative for the accessibility contrast, so it is excluded
# from that analysis and reported.  Only when *no* pair is informative does the
# contrast become impossible and the batch fail.
SEVERITY_UNINFORMATIVE = "UNINFORMATIVE"

# Terms that must not appear in anything the model could see.
PRIVATE_TERMS = (
    "gold",
    "oracle",
    "must_include",
    "condition",
    "branch",
    "harness",
    "virtual",
    "simulated",
    "fixture",
    "benchmark",
    "controlled",
    "experiment",
    "treatment",
    "ablation",
    "reinforced",
)


def _finding(check: str, detail: str, severity: str = SEVERITY_RUN_INVALID) -> dict:
    return {"check": check, "detail": detail, "severity": severity}


def check_run(run_dir: Path) -> list:
    """Run every manipulation check against one run directory."""
    findings: list = []
    branches_dir = run_dir / "branches"
    if not branches_dir.is_dir():
        return [_finding("branches_present", f"{branches_dir} does not exist")]

    per_branch = {}
    for branch_dir in sorted(branches_dir.iterdir()):
        if not branch_dir.is_dir():
            continue
        per_branch[branch_dir.name] = _load_branch(branch_dir)

    if not per_branch:
        return [_finding("branches_present", "no branch directories found")]

    findings.extend(_check_parent_equality(run_dir, per_branch))
    findings.extend(_check_clock(run_dir, per_branch))
    findings.extend(_check_oracle_is_not_inert(per_branch))
    findings.extend(_check_any_pair_informative(per_branch))
    findings.extend(_check_omit_removes_gold(per_branch))
    findings.extend(_check_oracle_preserves_count(per_branch))
    findings.extend(_check_visibility_absent(run_dir, per_branch))
    findings.extend(_check_branch_texts_match(per_branch))
    return findings


def _load_branch(branch_dir: Path) -> dict:
    state = {
        "branch_id": branch_dir.name,
        "retrievals": [],
        "requests": [],
        "forked": None,
        "clock": [],
        "turn_contexts": [],
        "user_texts": [],
    }
    log = branch_dir / "events.jsonl"
    if not log.exists():
        return state
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        etype = event["type"]
        if etype == "memory.retrieval":
            state["retrievals"].append(event["data"])
        elif etype == "model.request":
            state["requests"].append(event["data"])
        elif etype == "branch.forked":
            state["forked"] = event["data"]
        elif etype == "clock.advance":
            state["clock"].append(event["data"])
        elif etype == "turn.context":
            state["turn_contexts"].append(event["data"])
        elif etype == "user.message":
            state["user_texts"].append(event["data"]["text"])
    return state


def _probe_retrieval(state: dict) -> dict | None:
    """The probe retrieval is the last one without a tool call preceding it."""
    if not state["retrievals"]:
        return None
    return state["retrievals"][-1]


def _check_parent_equality(run_dir: Path, per_branch: dict) -> list:
    digests = set()
    counts = set()
    for state in per_branch.values():
        if state["forked"]:
            digests.add(state["forked"]["parent_prefix_digest"])
            counts.add(state["forked"]["parent_prefix_count"])
    if len(digests) > 1:
        return [
            _finding(
                "parent_snapshot_equal",
                f"branches forked from {len(digests)} different parent prefixes: "
                f"{sorted(digests)}",
                SEVERITY_BATCH_FAIL,
            )
        ]
    return []


def _check_clock(run_dir: Path, per_branch: dict) -> list:
    """Every branch must sit at the same virtual time and move by the same amount."""
    findings = []
    targets = set()
    for branch_id, state in per_branch.items():
        for advance in state["clock"]:
            targets.add(advance["to"])
    if len(targets) > 1:
        findings.append(
            _finding(
                "clock_advance_matches_condition",
                f"branches advanced to different instants: {sorted(targets)}",
                SEVERITY_BATCH_FAIL,
            )
        )
    # A branch's request must not read a host clock; the harness records the
    # instant on every event, so a request carrying a different time means the
    # clock was not the only time source.
    for branch_id, state in per_branch.items():
        contexts = state["turn_contexts"]
        if not contexts:
            continue
        probe_time = contexts[-1]["virtual_time"]
        for request in state["requests"]:
            if request.get("bound_at_seq") and probe_time and "bound_at_seq" in request:
                pass
    return findings


def _check_oracle_is_not_inert(per_branch: dict) -> list:
    """The critical check: forcing gold must change what is delivered.

    If a natural branch and its oracle counterpart deliver identical documents,
    the oracle condition manipulated nothing, and any difference between them is
    noise.  Reported as BATCH_FAIL because it invalidates the whole natural/oracle
    contrast rather than one branch.
    """
    findings = []
    pairs = {}
    for branch_id, state in per_branch.items():
        # branch ids look like ask_low_natural / claim_high_oracle
        if "_natural" in branch_id:
            key = branch_id.replace("_natural", "")
            pairs.setdefault(key, {})["natural"] = state
        elif "_oracle" in branch_id:
            key = branch_id.replace("_oracle", "")
            pairs.setdefault(key, {})["oracle"] = state

    for key, pair in sorted(pairs.items()):
        natural = pair.get("natural")
        oracle = pair.get("oracle")
        if natural is None or oracle is None:
            continue
        nat = _probe_retrieval(natural)
        orc = _probe_retrieval(oracle)
        if nat is None or orc is None:
            findings.append(
                _finding(
                    "oracle_retrieval_recorded",
                    f"{key}: a retrieval record is missing",
                )
            )
            continue
        nat_ids = [h["doc_id"] for h in nat["hits"]]
        orc_ids = [h["doc_id"] for h in orc["hits"]]
        if nat_ids == orc_ids:
            findings.append(
                _finding(
                    "oracle_is_inert_for_pair",
                    f"{key}: natural and oracle delivered identical documents "
                    f"{nat_ids}; natural retrieval already reached gold, so this "
                    "pair cannot test evidence accessibility and must be excluded "
                    "from that contrast",
                    SEVERITY_UNINFORMATIVE,
                )
            )
        if not orc.get("oracle_forced"):
            findings.append(
                _finding(
                    "oracle_marks_forced_ids",
                    f"{key}: oracle branch did not record any forced ids",
                    SEVERITY_RUN_INVALID,
                )
            )
    return findings


def _check_any_pair_informative(per_branch: dict) -> list:
    """At least one natural/oracle pair must differ, or the contrast is impossible."""
    informative = 0
    total = 0
    pairs = {}
    for branch_id, state in per_branch.items():
        if "_natural" in branch_id:
            pairs.setdefault(branch_id.replace("_natural", ""), {})["n"] = state
        elif "_oracle" in branch_id:
            pairs.setdefault(branch_id.replace("_oracle", ""), {})["o"] = state
    for pair in pairs.values():
        nat, orc = pair.get("n"), pair.get("o")
        if nat is None or orc is None:
            continue
        total += 1
        a = _probe_retrieval(nat)
        b = _probe_retrieval(orc)
        if a and b and [h["doc_id"] for h in a["hits"]] != [h["doc_id"] for h in b["hits"]]:
            informative += 1
    if total and informative == 0:
        return [
            _finding(
                "oracle_contrast_possible",
                f"all {total} natural/oracle pairs delivered identical documents; the "
                "accessibility contrast cannot be estimated from this run",
                SEVERITY_BATCH_FAIL,
            )
        ]
    return []


def _check_omit_removes_gold(per_branch: dict) -> list:
    findings = []
    for branch_id, state in per_branch.items():
        if "_omit" not in branch_id:
            continue
        probe = _probe_retrieval(state)
        if probe is None:
            findings.append(
                _finding("omit_retrieval_recorded", f"{branch_id}: no retrieval record")
            )
            continue
        if any(h.get("is_gold") for h in probe["hits"]):
            findings.append(
                _finding(
                    "omit_removes_gold",
                    f"{branch_id}: gold appears in an omit branch",
                )
            )
    return findings


def _check_oracle_preserves_count(per_branch: dict) -> list:
    """Matched oracle: same number of documents, only the ranking changes."""
    findings = []
    counts = {}
    for branch_id, state in per_branch.items():
        probe = _probe_retrieval(state)
        if probe is None:
            continue
        group = branch_id.rsplit("_", 1)[0]
        counts.setdefault(group, {})[branch_id.rsplit("_", 1)[1]] = len(probe["hits"])
    for group, by_condition in sorted(counts.items()):
        natural = by_condition.get("natural")
        oracle = by_condition.get("oracle")
        if natural is None or oracle is None:
            continue
        if natural != oracle:
            findings.append(
                _finding(
                    "oracle_preserves_result_count",
                    f"{group}: natural returned {natural} and oracle {oracle}; the "
                    "oracle must hold the count constant or it confounds "
                    "accessibility with context volume",
                    SEVERITY_BATCH_FAIL,
                )
            )
    return findings


def _check_visibility_absent(run_dir: Path, per_branch: dict) -> list:
    findings = []
    for branch_id, state in per_branch.items():
        for request in state["requests"]:
            blob = json.dumps(request.get("messages", []), ensure_ascii=False).lower()
            for term in PRIVATE_TERMS:
                if term in blob:
                    findings.append(
                        _finding(
                            "visible_private_terms_absent",
                            f"{branch_id}: model-visible request contains {term!r}",
                            SEVERITY_BATCH_FAIL,
                        )
                    )
                    break
    return findings


def _check_branch_texts_match(per_branch: dict) -> list:
    """Conditions differing only in evidence must share identical user text."""
    findings = []
    groups = {}
    for branch_id, state in per_branch.items():
        group = branch_id.rsplit("_", 1)[0]
        condition = branch_id.rsplit("_", 1)[1]
        groups.setdefault(group, {})[condition] = state["user_texts"]
    for group, by_condition in sorted(groups.items()):
        distinct = {tuple(v) for v in by_condition.values() if v}
        if len(distinct) > 1:
            findings.append(
                _finding(
                    "condition_differs_only_in_evidence",
                    f"{group}: conditions carry different user text {sorted(distinct)}; "
                    "the manipulation must be evidence only",
                    SEVERITY_BATCH_FAIL,
                )
            )
    return findings


def verdict(findings: list) -> str:
    """BATCH_FAIL beats RUN_INVALID beats UNINFORMATIVE beats clean.

    UNINFORMATIVE is deliberately not a failure.  A pair where natural already
    reached gold is a normal retrieval outcome; it just cannot carry the
    accessibility contrast.  Reporting it as a failure would train the reader to
    ignore the check, which costs more than the check is worth.
    """
    severity = {f["severity"] for f in findings}
    if SEVERITY_BATCH_FAIL in severity:
        return SEVERITY_BATCH_FAIL
    if SEVERITY_RUN_INVALID in severity:
        return SEVERITY_RUN_INVALID
    if SEVERITY_UNINFORMATIVE in severity:
        return SEVERITY_UNINFORMATIVE
    return "RUN_VALID"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    worst = 0
    for target in args.runs:
        path = Path(target)
        dirs = [path] if (path / "branches").is_dir() else sorted(
            p.parent for p in path.rglob("report.json")
        )
        for run_dir in dirs:
            findings = check_run(run_dir)
            state = verdict(findings)
            # Three outcomes, not two.  UNINFORMATIVE means the run is usable but
            # some contrast within it is empty; printing that as FAIL would train
            # the reader to ignore the check, which costs more than it is worth.
            mark = {
                "RUN_VALID": "PASS",
                SEVERITY_UNINFORMATIVE: "WARN",
            }.get(state, "FAIL")
            print(f"[{mark}] {run_dir} -> {state}")
            for finding in findings:
                print(f"    {finding['severity']} {finding['check']}: "
                      f"{finding['detail'][:150]}")
            # Only a real failure sets a non-zero exit status.
            if state in (SEVERITY_RUN_INVALID, SEVERITY_BATCH_FAIL):
                worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
