"""Is a clean pure-elapsed-time contrast still available in the ladder design?

The executed temporal contrast compares **5d with 15d**.  That pair straddles the ladder
boundary (the second notice is published at ``gap_days = 14``), so the two conditions differ
in two ways at once: how much world time has passed, and what the world says.  A difference
between them is therefore attributable to the *pair*, not to time alone.

A pure elapsed-time contrast needs two instants **inside one stage**: the world identical --
no notice published in between, same authoritative date, same authoritative document, same
shared material -- with only the clock advanced.  This tool builds both candidates through
the same code path the paid runs use and reports every field that differs, so the arm can be
judged before it is bought:

* the ``pure_dt`` pair (5d vs 13d) must differ in **the clock and nothing else**; any other
  difference is a defect in the design, not a finding;
* the ``stage`` pair (5d vs 15d) must differ in stage, authoritative date and corpus, which
  is the positive control that the ladder really does move the world.

No model calls are made.  Output: ``results_v4/pure_dt_prep.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.scenario_expansion import build_profiles  # noqa: E402
from tools.scenario_factory import (  # noqa: E402
    FILLER_ID,
    build_ladder,
    build_ladder_variant,
)
from vela_lab.clock import parse_instant  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NEAR_DAYS, FAR_DAYS, STAGE_DAYS = 5.0, 13.0, 15.0

#: Fields that must be byte-identical inside one stage.  ``time_context`` is
#: deliberately excluded: it is the manipulation.
COMPARED = (
    "stage_id",
    "stage_live",
    "authoritative_date",
    "authoritative_doc",
    "shared_ids",
    "preceding_ids",
    "documents",
    "prefix_turns",
    "stable_material",
    "elapsed_base_days",
)


def _instant(profile, days: float) -> str:
    update = parse_instant(profile.update_day.isoformat() + "T09:00:00+08:00")
    return (update + timedelta(days=days)).isoformat()


def observe(profile, days: float) -> dict:
    instant = _instant(profile, days)
    ladder = build_ladder_variant(profile=profile, filler_id=FILLER_ID, instant=instant)
    variant = ladder.variant
    return {
        "elapsed_days": days,
        "instant": instant,
        "time_context": f"Current date and time: {instant}",
        "stage_id": ladder.stage.stage_id,
        "stage_live": ladder.stage_live,
        "authoritative_date": variant.authoritative_date.isoformat(),
        "authoritative_doc": variant.gold_override,
        "shared_ids": list(variant.shared_ids),
        "preceding_ids": list(ladder.preceding_ids),
        "documents": [
            {"doc_id": d[0], "published_at": d[3], "text": d[2]}
            for d in variant.documents
        ],
        "prefix_turns": variant.prefix_turns,
        "stable_material": variant.stable_material,
        "elapsed_base_days": variant.elapsed_base_days,
    }


def _blob(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def diff(left: dict, right: dict) -> dict:
    out = {}
    for field in COMPARED:
        out[field] = "same" if _blob(left[field]) == _blob(right[field]) else "DIFFERS"
    out["time_context"] = "DIFFERS" if left["time_context"] != right["time_context"] else "same"
    return out


def publications_between(profile, lo_days: float, hi_days: float) -> list[str]:
    """Stages published in the half-open window (lo, hi]; non-empty means the world moves."""
    lo, hi = parse_instant(_instant(profile, lo_days)), parse_instant(_instant(profile, hi_days))
    return [s.stage_id for s in build_ladder(profile)
            if lo < parse_instant(s.published_at) <= hi]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-kind", type=int, default=1,
                        help="instances per event kind (the scale run used 1)")
    parser.add_argument("--out", default=str(ROOT / "results_v4" / "pure_dt_prep.json"))
    args = parser.parse_args()

    profiles = build_profiles(instances=max(1, args.per_kind))
    records, pure_defects, stage_defects = [], [], []

    for profile in profiles:
        near, far, stage = (observe(profile, d) for d in (NEAR_DAYS, FAR_DAYS, STAGE_DAYS))
        pure = diff(near, far)
        moved = diff(near, stage)
        window = publications_between(profile, NEAR_DAYS, FAR_DAYS)

        pure_ok = all(v == "same" for k, v in pure.items() if k != "time_context") \
            and pure["time_context"] == "DIFFERS" and not window
        stage_ok = moved["stage_id"] == "DIFFERS" and moved["authoritative_date"] == "DIFFERS" \
            and moved["authoritative_doc"] == "DIFFERS" and moved["preceding_ids"] == "DIFFERS"
        if not pure_ok:
            pure_defects.append({
                "profile": profile.profile_id,
                "non_clock_differences": {k: v for k, v in pure.items()
                                          if v == "DIFFERS" and k != "time_context"},
                "publications_in_window": window,
            })
        if not stage_ok:
            stage_defects.append({"profile": profile.profile_id, "stage_diff": moved})

        records.append({
            "profile": profile.profile_id,
            "kind": getattr(profile, "kind", None),
            "near": {k: near[k] for k in ("elapsed_days", "instant", "stage_id",
                                          "authoritative_date", "authoritative_doc")},
            "far": {k: far[k] for k in ("elapsed_days", "instant", "stage_id",
                                        "authoritative_date", "authoritative_doc")},
            "stage": {k: stage[k] for k in ("elapsed_days", "instant", "stage_id",
                                            "authoritative_date", "authoritative_doc")},
            "pure_pair_diff": pure,
            "stage_pair_diff": moved,
            "publications_in_pure_window": window,
        })

    payload = {
        "design": {
            "pure_dt_pair_days": [NEAR_DAYS, FAR_DAYS],
            "stage_pair_days": [NEAR_DAYS, STAGE_DAYS],
            "must_be_identical_in_pure_pair": list(COMPARED),
            "must_differ_in_pure_pair": ["time_context"],
            "stage_positive_control": ["stage_id", "authoritative_date",
                                       "authoritative_doc", "preceding_ids"],
        },
        "verdict": {
            "profiles": len(profiles),
            "pure_pair_clock_only": len(profiles) - len(pure_defects),
            "pure_pair_defects": pure_defects,
            "stage_pair_moves_world": len(profiles) - len(stage_defects),
            "stage_pair_defects": stage_defects,
        },
        "command": (
            "cd harness && VELA_API_KEY=... python tools/expanded_experiment.py "
            "--routes <route> --source expansion --per-kind 1 --contrast temporal "
            f"--temporal-positions {NEAR_DAYS:g} {FAR_DAYS:g} --out runs_pure_dt"
        ),
        "cost_shape": (
            "40 scenarios x 2 positions x 2 pressure branches = 160 branch executions "
            "(= 160 model calls) per route"
        ),
        "records": records,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(payload["verdict"], ensure_ascii=False)[:1200])
    print(payload["command"])
    print(payload["cost_shape"])
    return 0 if not (pure_defects or stage_defects) else 1


if __name__ == "__main__":
    raise SystemExit(main())
