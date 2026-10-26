"""The naturally-expired contrast: did the answer notice that the event is over?

What this tool reads, and why from the artefacts rather than from the design.  Each cell is
one (profile, arm, visibility) run: ``<root>/expiry/<route>/<profile>_<arm>_<visibility>/``.
The arm and the visibility are read from the directory name, which is what the runner wrote;
the four judged fields come from ``judge.expiry.json`` beside it; and whether the agent went
and looked at the clock is read from the branch log, not inferred -- with
``observe_time=False`` nothing renders the current instant, so a second ``model.request`` in
the log can only mean the model called ``world_time``.

The failure this contrast is looking for.  ``current_state_assumed`` means "the answer
organizes itself around the event as not yet having happened".  Before the boundary that is
*correct*; after it, it is the failure.  So the interesting number is
``current_state_assumed`` on the ``post`` and ``late`` arms, split by whether the clock was
delivered or had to be fetched.

Output: ``results_v4/expiry_analysis.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELDS = (
    "current_state_assumed",
    "temporal_frame_correct",
    "support_action_appropriate",
    "affect_validated",
)
ARMS = ("pre", "post", "late")
VISIBILITIES = ("delivered", "natural")
#: On these arms, treating the event as still ahead is the failure.
EXPIRED_ARMS = ("post", "late")


def _split_cell(name: str) -> tuple[str, str, str] | None:
    """``visa_interview_i1_post_natural`` -> (profile, arm, visibility)."""
    for visibility in VISIBILITIES:
        if not name.endswith("_" + visibility):
            continue
        stem = name[: -(len(visibility) + 1)]
        for arm in ARMS:
            if stem.endswith("_" + arm):
                return stem[: -(len(arm) + 1)], arm, visibility
    return None


def _probe_tool_calls(cell_dir: Path) -> dict:
    """Did the *probe* turn go and look at the clock?

    Why the last user message and not simply "how many requests".  The branch log holds two
    turns -- the prefix turn and the probe turn -- and in the ``natural`` arm the prefix turn
    calls ``world_time`` too (nothing renders the clock, so it fetches it).  Counting
    requests therefore cannot separate "the agent looked when it mattered" from "the agent
    looked earlier for its own reasons".  What matters is what happened after the probe text
    was delivered, so the tool counts ``tool.call`` events that follow the *last*
    ``user.message`` in the branch log.
    """
    log = cell_dir / "branches" / "probe" / "events.jsonl"
    out = {"probe_tool_calls": 0, "probe_looked_at_clock": False, "requests": 0}
    if not log.exists():
        return out
    events = []
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    last_user = -1
    for index, event in enumerate(events):
        if event.get("type") == "user.message":
            last_user = index
        elif event.get("type") == "model.request":
            out["requests"] += 1
    for event in events[last_user + 1:]:
        if event.get("type") != "tool.call":
            continue
        out["probe_tool_calls"] += 1
        if (event.get("data") or {}).get("tool_name") == "world_time":
            out["probe_looked_at_clock"] = True
    return out


def collect(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        for cell_dir in sorted(root.glob("expiry/*/*")):
            if not cell_dir.is_dir():
                continue
            split = _split_cell(cell_dir.name)
            if split is None:
                continue
            profile, arm, visibility = split
            route = cell_dir.parent.name
            row = {
                "root": root.name, "route": route, "profile": profile,
                "kind": profile.rsplit("_i", 1)[0], "arm": arm,
                "visibility": visibility, "cell": cell_dir.name,
            }
            report = cell_dir / "report.json"
            verdict = cell_dir / "judge.expiry.json"
            row["has_report"] = report.exists()
            row["has_verdict"] = verdict.exists()
            row.update(_probe_tool_calls(cell_dir))
            if verdict.exists():
                try:
                    payload = json.loads(verdict.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                    row["verdict_error"] = str(exc)
                    rows.append(row)
                    continue
                branch = (payload.get("branches") or {}).get("probe") or {}
                fields = branch.get("verdict") or {}
                row["fields"] = {f: (fields.get(f) or {}).get("value") for f in FIELDS}
                row["expected"] = (branch.get("derived") or {}).get("expected") or {}
            rows.append(row)
    return rows


def _tally(rows: list[dict], field: str) -> dict:
    counts = Counter()
    for row in rows:
        value = (row.get("fields") or {}).get(field, "absent")
        counts["absent" if value == "absent" else str(value)] += 1
    resolved = counts["True"] + counts["False"]
    return {
        "cells": len(rows), "true": counts["True"], "false": counts["False"],
        "unresolved": counts["None"], "absent": counts["absent"],
        "rate_true_over_resolved": (counts["True"] / resolved) if resolved else None,
    }


def summarise(rows: list[dict]) -> dict:
    judged = [r for r in rows if r.get("fields")]
    out: dict = {"cells": len(rows), "judged": len(judged)}

    by_arm: dict = {}
    for arm in ARMS:
        for visibility in VISIBILITIES:
            subset = [r for r in judged if r["arm"] == arm and r["visibility"] == visibility]
            if not subset:
                continue
            by_arm[f"{arm}|{visibility}"] = {
                "cells": len(subset),
                "clock_fetched": sum(1 for r in subset if r["probe_looked_at_clock"]),
                **{f: _tally(subset, f) for f in FIELDS},
            }
    out["by_arm_and_visibility"] = by_arm

    # The headline: after the boundary, does the answer still treat the event as ahead?
    expired = [r for r in judged if r["arm"] in EXPIRED_ARMS]
    out["expired_arms"] = {
        "cells": len(expired),
        "current_state_assumed": _tally(expired, "current_state_assumed"),
        "temporal_frame_correct": _tally(expired, "temporal_frame_correct"),
        "support_action_appropriate": _tally(expired, "support_action_appropriate"),
        "by_visibility": {
            v: _tally([r for r in expired if r["visibility"] == v], "current_state_assumed")
            for v in VISIBILITIES
        },
        "by_route": {
            route: _tally([r for r in expired if r["route"] == route], "current_state_assumed")
            for route in sorted({r["route"] for r in expired})
        },
        # The mechanism question: among the cells where the agent *did* fetch the clock,
        # how often did it still treat the event as ahead?  This separates "did not look"
        # from "looked and did not use it".
        "fetched_vs_outcome": {
            f"{visibility}|{'fetched' if fetched else 'did_not_fetch'}": _tally(
                [r for r in expired
                 if r["visibility"] == visibility and bool(r["probe_looked_at_clock"]) == fetched],
                "current_state_assumed",
            )
            for visibility in VISIBILITIES
            for fetched in (True, False)
        },
    }

    before = [r for r in judged if r["arm"] == "pre"]
    out["pre_arm"] = {
        "cells": len(before),
        "current_state_assumed": _tally(before, "current_state_assumed"),
        "temporal_frame_correct": _tally(before, "temporal_frame_correct"),
        "support_action_appropriate": _tally(before, "support_action_appropriate"),
    }

    natural = [r for r in judged if r["visibility"] == "natural"]
    out["clock_fetching"] = {
        "natural_cells": len(natural),
        "fetched_a_clock": sum(1 for r in natural if r["probe_looked_at_clock"]),
        "by_arm": {
            arm: {
                "cells": len([r for r in natural if r["arm"] == arm]),
                "fetched": sum(1 for r in natural if r["arm"] == arm and r["probe_looked_at_clock"]),
            }
            for arm in ARMS
        },
    }

    # Kind-level, so the unit of analysis is the event kind rather than the cell.
    kinds = defaultdict(list)
    for row in judged:
        kinds[row["kind"]].append(row)
    out["by_kind"] = {
        "kinds": len(kinds),
        "kinds_with_any_expired_failure": sum(
            1 for kind, rs in kinds.items()
            if any(r["arm"] in EXPIRED_ARMS
                   and (r.get("fields") or {}).get("current_state_assumed") is True
                   for r in rs)
        ),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", help="run roots (default: all runs_expiry_*)")
    parser.add_argument("--out", default=str(ROOT / "results_v4" / "expiry_analysis.json"))
    args = parser.parse_args()

    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        roots = sorted(p for p in ROOT.glob("runs_expiry_*") if p.is_dir())
    rows = collect(roots)
    report = summarise(rows)
    report["roots"] = [str(r) for r in roots]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": report, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    print(f"  cells={report['cells']} judged={report['judged']}")
    for arm in ARMS:
        for visibility in VISIBILITIES:
            key = f"{arm}|{visibility}"
            cell = report["by_arm_and_visibility"].get(key)
            if not cell:
                continue
            csa = cell["current_state_assumed"]
            print(f"  {key:22s} n={cell['cells']:3d} "
                  f"current_state_assumed true={csa['true']:3d} false={csa['false']:3d} "
                  f"unres={csa['unresolved']:3d} clock_fetched={cell['clock_fetched']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
