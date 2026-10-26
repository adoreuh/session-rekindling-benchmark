"""What elapsed time actually changes, measured rather than assumed.

The problem this exists to catch.  The v4 design replaced time-dependent
retrieval with **controlled delivery** (an explicit ``deliver_ids`` list), which
is what made the accessibility contrast clean.  But ``deliver_ids`` states the
delivered set outright, so the temporal decay weight no longer decides anything:
every elapsed time delivers the same documents in the same order.

If that is so, then "world time elapsed" in this design changes exactly one
visible thing — the date string in the time context — and calling that a temporal
manipulation would be generous.  That is worth knowing before spending budget on
an elapsed-time comparison, and it is not visible from the configuration: the
delivered lists are equal, which looks like a well-matched design rather than a
dead knob.

So this tool measures, per elapsed point:

* the delivered set and its order (does time change it at all?);
* the eligibility set (which documents the clock has made available);
* which visible strings differ between two elapsed points.

The last one is the honest answer to "what is the temporal manipulation" for a
given design.  Reported as a diff, so a manipulation that amounts to one date
string is visible as one date string.

Two designs are compared:

``controlled``
    v4 as built: an explicit delivered list.  Expected to be time-invariant.
``temporal_ladder``
    a staged variant where the *world* changes over time — a second notice
    arrives partway through — so elapsed time changes the correct answer through
    the registered schedule rather than through a scoring weight.  This is the
    construction that keeps a temporal manipulation while staying decoupled from
    the probe's wording.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.evidence import EvidenceCorpus, Document  # noqa: E402

from tools.scenario_factory import FILLER_ID, PROFILES, build_variant  # noqa: E402
from tools.variant_wiring import build_variant_spec  # noqa: E402

#: Elapsed points spanning the scales the design cares about.  Sub-day is
#: included deliberately: a date-only clock signal cannot distinguish 0 from 30
#: minutes, so if the harness also cannot, the sub-day manipulation is empty for
#: *every* route rather than only for the one with the runtime date.
ELAPSED_DAYS = (0.0, 1 / 48, 1.0, 7.0, 15.0, 30.0, 61.0, 180.0)


def _label(days: float) -> str:
    if days < 1:
        return f"{int(round(days * 24 * 60))}min"
    return f"{int(days)}d"


def measure_controlled(profile, *, elapsed_days: float) -> dict:
    """What the v4 controlled-delivery design changes at one elapsed point."""
    variant = build_variant(
        scenario_id=f"t-{profile.profile_id}", profile=profile, filler_id=FILLER_ID
    )
    spec = build_variant_spec(
        variant=variant, experiment_id=f"t-{profile.profile_id}",
        elapsed_days=elapsed_days,
    )
    probe = spec.clock_start.advance(days=spec.advance_days)

    # The delivered set per condition, read from the branch policies.
    delivered = {}
    for branch in spec.branches:
        delivered[branch.branch_id] = list(branch.exposure.deliver_ids or ())

    # Eligibility: which documents the corpus would expose at this instant, with
    # no forcing.  This is where a decay profile would show up if it could.
    corpus = EvidenceCorpus([
        Document(doc_id=doc_id, text=text, source=title or source,
                 published_at=published_at, is_gold=(doc_id == variant.gold_id))
        for doc_id, title, text, published_at, source in variant.documents
    ])
    corpus.seal()
    record = corpus.retrieve(
        query=spec.host_query, now_instant=probe.instant,
        top_k=len(variant.documents), mode="auto", visibility="full",
    )

    return {
        "elapsed_days": elapsed_days,
        "elapsed_label": _label(elapsed_days),
        "probe_instant": probe.instant,
        "time_context": f"Current date and time: {probe.instant} ({probe.time_zone})",
        "delivered": delivered,
        "eligible_ids": sorted(record.eligible_ids),
        "natural_order": [h["doc_id"] for h in record.hits],
        "natural_scores": {h["doc_id"]: round(h["score"], 6) for h in record.hits},
    }


def diff_between(rows: list) -> dict:
    """Which fields actually differ across elapsed points."""
    if not rows:
        return {}
    keys = ("delivered", "eligible_ids", "natural_order", "time_context")
    out = {}
    for key in keys:
        values = [json.dumps(r[key], ensure_ascii=False, sort_keys=True)
                  for r in rows]
        out[key] = {
            "varies": len(set(values)) > 1,
            "distinct_values": len(set(values)),
        }
    # Score movement, to show whether the decay is doing anything at all even
    # when the ordering holds.
    first, last = rows[0]["natural_scores"], rows[-1]["natural_scores"]
    moved = {doc: (first.get(doc), last.get(doc)) for doc in first
             if doc in last and first[doc] != last[doc]}
    out["natural_scores"] = {
        "varies": bool(moved),
        "docs_whose_score_changed": len(moved),
        "example": dict(list(moved.items())[:3]),
    }
    return out


def measure_ladder(profile, *, elapsed_days: float) -> dict:
    """A staged world: a second notice arrives later, changing the answer.

    The point is that elapsed time changes what is *true*, through the registered
    schedule, rather than changing a ranking weight.  So the delivered set can
    stay fixed (keeping the clean contrast) while the correct answer moves —
    which is a temporal manipulation that survives controlled delivery.
    """
    variant = build_variant(
        scenario_id=f"t-{profile.profile_id}", profile=profile, filler_id=FILLER_ID
    )
    p = profile
    # The staged second notice: it supersedes the first reschedule.  Its content
    # is derived from the profile so no scenario can be tuned independently.
    from datetime import timedelta

    staged_id = f"{p.profile_id}_second_notice"
    staged_date = p.new_date + timedelta(days=21)
    staged_published = p.update_day + timedelta(days=14)
    staged_text = (
        f"{p.office}补充通知：此前改期的{p.referent_long}再次调整，"
        f"原定 {p.new_date.month} 月 {p.new_date.day} 日的安排已{p.cancel_word}，"
        f"现改为 {staged_date.month} 月 {staged_date.day} 日 "
        f"{p.booking_time}，地点不变。请以本通知为准。"
    )
    base = build_variant_spec(
        variant=variant, experiment_id=f"ladder-{p.profile_id}",
        elapsed_days=elapsed_days,
    )
    probe = base.clock_start.advance(days=base.advance_days)
    from vela_lab.clock import parse_instant

    staged_live = parse_instant(staged_published.isoformat() + "T09:00:00+08:00") \
        <= parse_instant(probe.instant)

    return {
        "elapsed_days": elapsed_days,
        "elapsed_label": _label(elapsed_days),
        "probe_instant": probe.instant,
        "staged_notice_live": staged_live,
        "correct_date_if_staged_live": staged_date.isoformat() if staged_live
                                      else p.new_date.isoformat(),
        "staged_notice": {
            "doc_id": staged_id,
            "published_at": f"{staged_published.isoformat()}T09:00:00+08:00",
            "text": staged_text,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_v4/temporal_study.json")
    parser.add_argument("--profile", default="visa_interview")
    args = parser.parse_args()

    profile = next((p for p in PROFILES if p.profile_id == args.profile), None)
    if profile is None:
        print(f"unknown profile {args.profile!r}", file=sys.stderr)
        return 2

    print("WHAT DOES ELAPSED TIME ACTUALLY CHANGE?")
    print("=" * 78)
    print(f"profile: {profile.profile_id}")
    print()

    controlled = [measure_controlled(profile, elapsed_days=d)
                  for d in ELAPSED_DAYS]
    print("-- controlled delivery (v4 as built) --")
    print(f"  {'elapsed':>8s}  {'probe_instant':32s}  delivered_set")
    for row in controlled:
        oracle = row["delivered"].get("claim_low_oracle", [])
        print(f"  {row['elapsed_label']:>8s}  {row['probe_instant']:32s}  "
              f"{len(oracle)} docs")
    verdict = diff_between(controlled)
    print()
    for key, entry in verdict.items():
        print(f"  {key:28s} varies={entry['varies']}"
              + (f" distinct={entry['distinct_values']}"
                 if 'distinct_values' in entry else "")
              + (f" changed_docs={entry.get('docs_whose_score_changed')}"
                 if 'docs_whose_score_changed' in entry else ""))

    ladder = [measure_ladder(profile, elapsed_days=d) for d in ELAPSED_DAYS]
    print()
    print("-- temporal ladder (a second notice arrives later) --")
    print(f"  {'elapsed':>8s}  {'probe_instant':32s}  staged_live  correct_date")
    for row in ladder:
        print(f"  {row['elapsed_label']:>8s}  {row['probe_instant']:32s}  "
              f"{str(row['staged_notice_live']):11s}  "
              f"{row['correct_date_if_staged_live']}")

    report = {
        "profile": profile.profile_id,
        "elapsed_days": list(ELAPSED_DAYS),
        "controlled": {"rows": controlled, "diff": verdict},
        "temporal_ladder": {"rows": ladder},
        "finding": {
            "controlled_delivery_is_time_invariant": not (
                verdict.get("delivered", {}).get("varies")
                or verdict.get("eligible_ids", {}).get("varies")
                or verdict.get("natural_order", {}).get("varies")
            ),
            "only_visible_change_under_controlled_delivery":
                "time_context" if not verdict.get("natural_order", {}).get("varies")
                else "retrieval ordering",
        },
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    print()
    print("FINDING")
    print(f"  controlled delivery is time-invariant: "
          f"{report['finding']['controlled_delivery_is_time_invariant']}")
    print(f"  only visible change: "
          f"{report['finding']['only_visible_change_under_controlled_delivery']}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
