#!/usr/bin/env python3
"""Build dual-annotator packages from VELA expiry runs.

For each run cell this emits one record:

    sample_id           stable id (run dir name)
    subject_route       model under test (annotators do NOT see this)
    arm, visibility     condition (annotators do NOT see this)
    world_state         {current_time, event_date, local dates}
    user_text           probe question
    answer              assistant final response
    time_fetch          {fetched, n_calls, usable} — trace-derived, NOT judged
    glm_verdict         existing judge labels (annotators do NOT see this)

Annotators receive only: world_state summary, user_text, answer.
The GLM judge already has labels in judge.expiry.json (reused as Annotator A).
ChatGPT is Annotator B with the same rubric.

Usage:
    python tools/annotation_package.py --root harness --out harness/_consult/annotation \
        --pilot 40 --seed 20260928
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict


RUN_GLOB = "runs_expiry_*/expiry/*/*/judge.expiry.json"
FIELDS = ("current_state_assumed", "temporal_frame_correct",
          "support_action_appropriate", "affect_validated")
CELL_RE = re.compile(r"^(?P<profile>.+)_(?P<arm>pre|post|late)_(?P<vis>natural|delivered)$")


def load_events(run_dir: str) -> list:
    path = os.path.join(run_dir, "branches", "probe", "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def trace_fetch(rows: list) -> dict:
    """Did the run call world_time in the probe branch? Return count + usability."""
    calls = [r for r in rows if r.get("type") == "tool.call"
             and r.get("data", {}).get("tool_name") == "world_time"]
    # probe-branch tool calls: after branch.forked
    fork = next((i for i, r in enumerate(rows) if r.get("type") == "branch.forked"), 0)
    probe_calls = [r for r in rows[fork:] if r.get("type") == "tool.call"
                   and r.get("data", {}).get("tool_name") == "world_time"]
    usable = None
    for r in rows[fork:]:
        if r.get("type") == "tool.visible_result" and "current_time" in r.get("data", {}).get("result", {}):
            usable = r["data"]["result"]["current_time"]
    return {"n_calls_prefix": len(calls) - len(probe_calls),
            "n_calls_probe": len(probe_calls),
            "usable_result": usable}


def glm_verdict(j: dict) -> dict:
    v = (j.get("branches") or {}).get("probe", {}).get("verdict", {})
    out = {}
    for f in FIELDS:
        item = v.get(f) or {}
        out[f] = {"value": item.get("value"),
                  "unresolved_reason": item.get("unresolved_reason")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="harness")
    ap.add_argument("--out", default="harness/_consult/annotation")
    ap.add_argument("--pilot", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()

    records = []
    for jf in sorted(glob.glob(os.path.join(args.root, RUN_GLOB))):
        j = json.load(open(jf, encoding="utf-8"))
        cond = j.get("condition", {})
        run_dir = os.path.dirname(jf)
        name = os.path.basename(run_dir)
        m = CELL_RE.match(name)
        if not m:
            continue
        report_p = os.path.join(run_dir, "report.json")
        if not os.path.exists(report_p):
            continue
        report = json.load(open(report_p, encoding="utf-8"))
        branch = (report.get("branches") or [{}])[0]
        answer = (branch.get("outcome") or {}).get("assistant_text")
        if not isinstance(answer, str) or not answer.strip():
            continue
        rows = load_events(run_dir)
        fetch = trace_fetch(rows)
        records.append({
            "sample_id": name,
            "subject_route": cond.get("subject_route"),
            "profile_id": cond.get("profile_id"),
            "arm": cond.get("arm") or m.group("arm"),
            "visibility": cond.get("visibility") or m.group("vis"),
            "world_state": {
                "event_date": cond.get("event_date"),
                "current_instant": cond.get("current_instant"),
            },
            "user_text": branch.get("user_text") or "",
            "answer": answer,
            "time_fetch": fetch,
            "glm_verdict": glm_verdict(j),
        })

    # ---- stratified pilot sample -------------------------------------------
    # strata: arm × visibility × glm csa outcome × subject route
    rng = random.Random(args.seed)
    strata = defaultdict(list)
    for r in records:
        csa = r["glm_verdict"]["current_state_assumed"]["value"]
        key = (r["arm"], r["visibility"], str(csa), r["subject_route"])
        strata[key].append(r)
    per = max(1, args.pilot // len(strata)) if strata else 0
    pilot = []
    for key in sorted(strata):
        rng.shuffle(strata[key])
        pilot.extend(strata[key][:per])
    rng.shuffle(pilot)
    pilot = pilot[:args.pilot]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "records_all.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "pilot40.json"), "w", encoding="utf-8") as f:
        json.dump(pilot, f, ensure_ascii=False, indent=1)

    # annotator-facing view (no route/condition/verdict)
    def public_view(r):
        return {"sample_id": r["sample_id"],
                "world_state": r["world_state"],
                "user_text": r["user_text"],
                "answer": r["answer"]}
    with open(os.path.join(args.out, "pilot40_public.json"), "w", encoding="utf-8") as f:
        json.dump([public_view(r) for r in pilot], f, ensure_ascii=False, indent=1)

    print("records:", len(records))
    print("pilot:", len(pilot))
    print("pilot strata:", Counter((r["arm"], r["visibility"],
          str(r["glm_verdict"]["current_state_assumed"]["value"])) for r in pilot))
    print("written:", os.path.join(args.out, "pilot40.json"),
          os.path.join(args.out, "pilot40_public.json"), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
