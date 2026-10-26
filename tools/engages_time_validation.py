"""Validate the text-level "engages time" detector.

The claim in the paper is that 46/48 stale *forced* cells never engage the
current instant they were handed.  That claim rests on a lexical detector, so
the detector itself has to be shown to fire when engagement must be present.
We run it on three populations:

* ``relation`` cells -- the relation is delivered in the context, so engagement
  should be near-universal (positive control);
* ``delivered`` cells that the judge scored correct -- the model used the
  rendered instant, so engagement should be high (positive control);
* ``forced`` cells that the judge scored stale -- the population of interest.
"""
import json, re
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
DATE_ARABIC = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日")
ELAPSED = ("已经过去", "已过去", "早已过去", "已经过", "过期", "上个月", "一个多月",
           "已失效", "已经结束", "已经过时", "来不及", "为时已晚")


def engages(answer: str) -> bool:
    return bool(DATE_ARABIC.search(answer)) or any(t in answer for t in ELAPSED)


def population(root: Path, vis_suffix: str, want_stale):
    """Yield (engages, route) for cells under ``root`` matching the suffix."""
    rows = []
    for rp in root.rglob("report.json"):
        if not rp.parent.name.endswith(vis_suffix):
            continue
        report = json.loads(rp.read_text(encoding="utf-8"))
        b = report["branches"][0]
        if not b.get("run_valid"):
            continue
        jf = rp.parent / "judge.expiry.json"
        if want_stale is not None:
            if not jf.exists():
                continue
            j = json.loads(jf.read_text(encoding="utf-8"))
            stale = ((j["branches"]["probe"]["verdict"].get("current_state_assumed")
                      or {}).get("value"))
            if stale is not want_stale:
                continue
        answer = (b.get("outcome") or {}).get("assistant_text") or ""
        rows.append((engages(answer), rp.parent.parent.name))
    return rows


checks = [
    ("relation cells (positive control: relation was delivered)",
     population(HARN / "runs_e2_relation" / "e2", "_relation", None)),
    ("delivered cells judged correct (positive control: model used the instant)",
     population(HARN / "runs_expiry_qwen3-vl-plus", "_delivered", False)),
    ("forced cells judged stale (population of interest)",
     population(HARN / "runs_e1_forced" / "e1", "_forced", True)),
]
for label, rows in checks:
    if not rows:
        print(f"{label}: no cells")
        continue
    hit = sum(1 for e, _ in rows if e)
    print(f"{label}\n    engages time {hit}/{len(rows)} = {hit/len(rows):.2f}")
    by_route = {}
    for e, r in rows:
        c = by_route.setdefault(r, [0, 0])
        c[1] += 1
        c[0] += bool(e)
    for r, (h, n) in sorted(by_route.items()):
        print(f"      {r:24s} {h}/{n}")
