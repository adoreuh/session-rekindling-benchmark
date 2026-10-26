#!/usr/bin/env python3
"""Reproduce every quantitative table in the paper from the released artefacts.

Paper: "Can Agents Keep Up with Time? An Empirical Analysis of Temporal
Adaptation After a Conversational Hiatus".

This script is deliberately offline and stdlib-only.  It reads

    data/responses.jsonl          one record per evaluated cell (2 940 cells)
    results_v4/e4_report.json     explicit-date baseline scoring (Table 4)

and prints Tables 3, 4, 5 and 6 exactly as they appear in the paper, each with a
PASS/FAIL line comparing the recomputed numbers against the published values.
Nothing is imputed: a cell whose judge label is unresolved is dropped from the
denominator rather than counted as a negative.

Usage::

    python reproduce_paper_tables.py            # all four tables
    python reproduce_paper_tables.py --table 3  # one table

Table 1 (research questions -> observables) and Table 2 are definitions, not
measurements, so they are not reproduced here.  Figure 4's pair counts come out
of the same matched-pair machinery used for Table 6 and are printed as a bonus.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "responses.jsonl")
E4 = os.path.join(HERE, "results_v4", "e4_report.json")

ROUTES = ["DeepSeek-V4-Flash", "Qwen3.5-397B-A17B", "hy3",
          "DeepSeek-V4.1-Flash", "qwen3-vl-plus"]
POST_DATE_ARMS = ("post", "late")
CONDITIONS = ("delivered", "forced", "relation")

#: Past-date wording rule R (paper section 3.2): the exposed reasoning states the
#: scheduled date is already behind us.  Deliberately lexical and conservative;
#: its false-positive rate on pre-date controls is reported in the paper.
R_PAST = re.compile(
    r"already (passed|past|happened|over|occurred|done)|has (already )?passed|"
    r"have (already )?passed|is (in the )?past|"
    r"(\d+|one|a|about|exactly one|over a) (day|days|week|weeks|month|months) ago|"
    r"days ago|month ago|weeks ago|is over|had passed|date has passed|"
    r"that\'s past|in the past|\(past\)|past\)|already took place|"
    r"已经过|已过|过去了", re.I)

# ---------------------------------------------------------------- published ---
# The camera-ready numbers, used only for the PASS/FAIL comparison.
PAPER = {
    "t3": {
        "DeepSeek-V4-Flash":   dict(clock=27, with_clock=(20, 24), no_clock=(1, 1),
                                    past_wording=(27, 27), accepted=(3, 23)),
        "Qwen3.5-397B-A17B":   dict(clock=25, with_clock=(19, 23), no_clock=(3, 3),
                                    past_wording=(24, 25), accepted=(3, 23)),
        "hy3":                 dict(clock=25, with_clock=(3, 24), no_clock=(2, 2),
                                    past_wording=(25, 25), accepted=(17, 22)),
        "DeepSeek-V4.1-Flash": dict(clock=28, with_clock=(0, 21), no_clock=(0, 0),
                                    past_wording=(28, 28), accepted=(18, 20)),
        "qwen3-vl-plus":       dict(clock=0, with_clock=(0, 0), no_clock=(25, 26),
                                    past_wording=(0, 0), accepted=(0, 0)),
    },
    "t4": {
        "DeepSeek-V4-Flash":   dict(Interval=(14, 14), Past=(14, 14), Future=(14, 14), Before=(14, 14), After=(14, 14)),
        "Qwen3.5-397B-A17B":   dict(Interval=(14, 14), Past=(14, 14), Future=(14, 14), Before=(14, 14), After=(14, 14)),
        "hy3":                 dict(Interval=(14, 14), Past=(14, 14), Future=(14, 14), Before=(14, 14), After=(14, 14)),
        "DeepSeek-V4.1-Flash": dict(Interval=(14, 14), Past=(14, 14), Future=(14, 14), Before=(14, 14), After=(13, 14)),
        "qwen3-vl-plus":       dict(Interval=(14, 14), Past=(1, 14), Future=(14, 14), Before=(14, 14), After=(14, 14)),
    },
    "t5": {
        "DeepSeek-V4-Flash":   dict(Natural=(21, 25), Delivered=(11, 25), Forced=(27, 38), Relation=(0, 37)),
        "Qwen3.5-397B-A17B":   dict(Natural=(22, 26), Delivered=(17, 27), Forced=(21, 38), Relation=(0, 38)),
        "hy3":                 dict(Natural=(5, 26), Delivered=(2, 26), Forced=(0, 37), Relation=(0, 38)),
        "DeepSeek-V4.1-Flash": dict(Natural=(0, 21), Delivered=(0, 26), Forced=(0, 37), Relation=(0, 38)),
        "qwen3-vl-plus":       dict(Natural=(25, 26), Delivered=(2, 28), Forced=(0, 38), Relation=(0, 38)),
        "All routes":          dict(Natural=(73, 124), Delivered=(32, 132), Forced=(48, 188), Relation=(0, 189)),
    },
    "t6": {
        "DeepSeek-V4-Flash":   dict(Delivered=(20, 11, 23), Forced=(21, 19, 25), Relation=(21, 0, 24)),
        "Qwen3.5-397B-A17B":   dict(Delivered=(21, 15, 25), Forced=(22, 13, 26), Relation=(22, 0, 26)),
        "hy3":                 dict(Delivered=(5, 1, 24), Forced=(5, 0, 26), Relation=(5, 0, 26)),
        "DeepSeek-V4.1-Flash": dict(Delivered=(0, 0, 20), Forced=(0, 0, 21), Relation=(0, 0, 21)),
        "qwen3-vl-plus":       dict(Delivered=(25, 1, 26), Forced=(25, 0, 26), Relation=(25, 0, 26)),
    },
    "fig4c": {"delivered": (118, 48, 5), "forced": (124, 44, 3), "relation": (123, 73, 0)},
}


def load_rows(path=DATA):
    if not os.path.exists(path):
        sys.exit("missing %s -- run this from the repository root" % path)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label(row, field):
    value = (row.get("labels") or {}).get(field)
    return value if isinstance(value, bool) else None


def post_date(rows, condition, route=None):
    """Post-date cells: arms ``post`` and ``late`` (the paper pools d+30 and d+120)."""
    out = []
    for r in rows:
        if r["arm"] not in POST_DATE_ARMS:
            continue
        if r["condition"] != condition:
            continue
        if route is not None and r["route"] != route:
            continue
        out.append(r)
    return out


def frac(pairs):
    if len(pairs) == 2:
        k, n = pairs
        return "%d/%d" % (k, n) if n else "--"
    return "%d->%d (%d)" % tuple(pairs)


def check(name, got, want):
    ok = tuple(got) == tuple(want)
    return "%-46s %-13s %-13s %s" % (name, frac(got), frac(want), "PASS" if ok else "FAIL"), ok


# ------------------------------------------------------------------ tables ---
def table3(rows):
    print("Table 3  Post-date Natural responses by route "
          "(clock invocations, upcoming framing, past-date wording, advice accepted)")
    header = ("%-22s %6s %12s %12s %14s %12s" %
              ("Route", "Clock", "Upcoming+clk", "Upcoming-clk", "Past wording", "Accepted"))
    print(header)
    print("-" * len(header))
    all_ok = True
    strict = R_PAST
    for route in ROUTES:
        cells = post_date(rows, "natural", route)
        clock = [r for r in cells if r.get("probe_clock_called")]
        with_clock = [r for r in clock if label(r, "current_state_assumed") is not None]
        no_clock = [r for r in cells
                    if not r.get("probe_clock_called")
                    and label(r, "current_state_assumed") is not None]
        reasoned = [r for r in clock if (r.get("reasoning") or "").strip()]
        both = [r for r in clock
                if label(r, "current_state_assumed") is not None
                and label(r, "support_action_appropriate") is not None]
        got = dict(
            clock=len(clock),
            with_clock=(sum(1 for r in with_clock if label(r, "current_state_assumed")), len(with_clock)),
            no_clock=(sum(1 for r in no_clock if label(r, "current_state_assumed")), len(no_clock)),
            past_wording=(sum(1 for r in reasoned if strict.search(r["reasoning"])), len(reasoned)),
            accepted=(sum(1 for r in both if label(r, "support_action_appropriate")), len(both)),
        )
        want = PAPER["t3"][route]
        print("%-22s %6d %12s %12s %14s %12s" % (
            route, got["clock"], frac(got["with_clock"]), frac(got["no_clock"]),
            frac(got["past_wording"]), frac(got["accepted"])))
        for key, label_text in (("with_clock", "upcoming framing w/ clock"),
                                ("no_clock", "upcoming framing, clock omitted"),
                                ("past_wording", "past-date wording R"),
                                ("accepted", "advice accepted P")):
            line, ok = check("    %-38s" % label_text, got[key], want[key])
            all_ok &= ok
            print(line)
        if got["clock"] != want["clock"]:
            print("    %-38s %-9d %-9d FAIL" % ("clock invocations", got["clock"], want["clock"]))
            all_ok = False
    print()
    return all_ok


def table4(rows):
    print("Table 4  Explicit-date baseline accuracy over the 14 core event types")
    if not os.path.exists(E4):
        print("  missing %s -- skipped" % E4)
        return True
    report = json.load(open(E4, encoding="utf-8"))
    accuracy = report.get("accuracy") or {}
    mapping = [("Interval", "A"), ("Past", "B_past"), ("Future", "B_future"),
               ("Before", "C_asof_before"), ("After", "C_asof_after")]
    print("%-22s %10s %10s %10s %10s %10s" %
          ("Route", "Interval", "Past", "Future", "Before", "After"))
    print("-" * 74)
    all_ok = True
    for route in ROUTES:
        got = {}
        per = accuracy.get(route) or {}
        for label_text, key in mapping:
            cell = per.get(key) or {}
            got[label_text] = (cell.get("correct", 0), cell.get("n", 0))
        print("%-22s %10s %10s %10s %10s %10s" % (
            route, *(frac(got[l]) for l, _ in mapping)))
        want = PAPER["t4"][route]
        for label_text, _ in mapping:
            line, ok = check("    %-38s" % label_text, got[label_text], want[label_text])
            all_ok &= ok
            if not ok:
                print(line)
    print()
    return all_ok


def table5(rows):
    print("Table 5  Post-date upcoming framing counts (S=1), d+30 and d+120 pooled")
    header = "%-22s %12s %12s %12s %12s" % ("Route", "Natural", "Delivered", "Forced", "Relation")
    print(header)
    print("-" * len(header))
    all_ok = True
    totals = {c: [0, 0] for c in ("natural",) + CONDITIONS}
    for route in ROUTES:
        got = {}
        for cond in ("natural",) + CONDITIONS:
            cells = [r for r in post_date(rows, cond, route)
                     if label(r, "current_state_assumed") is not None]
            got[cond] = (sum(1 for r in cells if label(r, "current_state_assumed")), len(cells))
            totals[cond][0] += got[cond][0]
            totals[cond][1] += got[cond][1]
        print("%-22s %12s %12s %12s %12s" % (
            route, frac(got["natural"]), frac(got["delivered"]),
            frac(got["forced"]), frac(got["relation"])))
        want = PAPER["t5"][route]
        for cond, name in (("natural", "Natural"), ("delivered", "Delivered"),
                           ("forced", "Forced"), ("relation", "Relation")):
            line, ok = check("    %-38s" % name, got[cond], want[name])
            all_ok &= ok
            if not ok:
                print(line)
    print("%-22s %12s %12s %12s %12s" % (
        "All routes", frac(tuple(totals["natural"])), frac(tuple(totals["delivered"])),
        frac(tuple(totals["forced"])), frac(tuple(totals["relation"]))))
    want_all = PAPER["t5"]["All routes"]
    for cond, name in (("natural", "Natural"), ("delivered", "Delivered"),
                       ("forced", "Forced"), ("relation", "Relation")):
        line, ok = check("    %-38s" % ("All routes " + name), tuple(totals[cond]), want_all[name])
        all_ok &= ok
        if not ok:
            print(line)
    print()
    return all_ok


def matched_pairs(rows):
    """(route, profile, arm) -> {condition: S} over post-date cells only."""
    index = defaultdict(dict)
    for r in rows:
        if r["arm"] not in POST_DATE_ARMS:
            continue
        index[(r["route"], r["profile"], r["arm"])][r["condition"]] = label(r, "current_state_assumed")
    return index


def table6(rows):
    print("Table 6  Matched post-date transitions Natural -> supplied-information condition")
    header = "%-22s %16s %16s %16s" % ("Route", "Delivered", "Forced", "Relation")
    print(header)
    print("-" * len(header))
    index = matched_pairs(rows)
    all_ok = True
    overall = defaultdict(lambda: [0, 0, 0])
    for route in ROUTES:
        got = {}
        for cond in CONDITIONS:
            base_n = cond_n = n = 0
            for (rt, _profile, _arm), cells in index.items():
                if rt != route:
                    continue
                a, b = cells.get("natural"), cells.get(cond)
                if a is None or b is None:
                    continue
                n += 1
                base_n += a
                cond_n += b
            got[cond] = (base_n, cond_n, n)
            overall[cond][0] += base_n
            overall[cond][1] += cond_n
            overall[cond][2] += n
        print("%-22s %16s %16s %16s" % (
            route,
            "%d->%d (%d)" % got["delivered"],
            "%d->%d (%d)" % got["forced"],
            "%d->%d (%d)" % got["relation"]))
        want = PAPER["t6"][route]
        for cond, name in (("delivered", "Delivered"), ("forced", "Forced"), ("relation", "Relation")):
            line, ok = check("    %-38s" % name, got[cond], want[name])
            all_ok &= ok
            if not ok:
                print(line)
    print()
    print("Figure 4(c) matched-pair transitions (natural -> condition, pairs, down, up)")
    for cond in CONDITIONS:
        down = up = 0
        base_n, cond_n, n = overall[cond]
        for cells in index.values():
            a, b = cells.get("natural"), cells.get(cond)
            if a is None or b is None:
                continue
            if a and not b:
                down += 1
            elif b and not a:
                up += 1
        got = (n, down, up)
        line, ok = check("    %-38s" % cond, got, PAPER["fig4c"][cond])
        all_ok &= ok
        print("%s  (%d -> %d upcoming)" % (line, base_n, cond_n))
    print()
    return all_ok


def reproduce(rows):
    fns = [("3", table3), ("4", table4), ("5", table5), ("6", table6)]
    results = {}
    for name, fn in fns:
        results[name] = fn(rows)
    print("=" * 74)
    for name, _ in fns:
        print("Table %s: %s" % (name, "reproduced" if results[name] else "MISMATCH"))
    print("=" * 74)
    return 0 if all(results.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=["3", "4", "5", "6"], help="reproduce one table only")
    ap.add_argument("--data", default=DATA, help="path to responses.jsonl")
    args = ap.parse_args()
    rows = load_rows(args.data)
    print("loaded %d cells from %s" % (len(rows), os.path.relpath(args.data, HERE)))
    print()
    if args.table:
        table = {"3": table3, "4": table4, "5": table5, "6": table6}[args.table]
        return 0 if table(rows) else 1
    return reproduce(rows)


if __name__ == "__main__":
    raise SystemExit(main())
