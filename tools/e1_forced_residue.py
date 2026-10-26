"""Where does the break sit inside the forced arm?

The forced arm hands the model the current instant.  Two different failures are
still possible, and they need different fixes:

* ``D-not-formed``  -- the answer never engages the current date at all, so no
  temporal relation between "now" and the record was constructed;
* ``D-formed-then-kept`` -- the answer states the current date or the elapsed
  gap and still plans around the expired record, i.e. detection happened and
  activation was not withdrawn.

We read the second only from the answer text, so this is a text-level signal,
not a trace-level proof; it is reported as such.
"""
import json, re, glob
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
root = HARN / "runs_e1_forced" / "e1"
DATE_ARABIC = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
ELAPSED = ("已经过去", "已过去", "早已过去", "已经过", "过期", "上个月", "一个多月",
           "已失效", "已经结束", "已经过时")
rows = []
for rp in sorted(root.glob("*/*/report.json")):
    report = json.loads(rp.read_text(encoding="utf-8"))
    b = report["branches"][0]
    if not b.get("run_valid"):
        continue
    jf = rp.parent / "judge.expiry.json"
    if not jf.exists():
        continue
    j = json.loads(jf.read_text(encoding="utf-8"))
    stale = ((j["branches"]["probe"]["verdict"].get("current_state_assumed") or {})
             .get("value"))
    if stale is not True:
        continue
    answer = (b.get("outcome") or {}).get("assistant_text") or ""
    expiry = ((report.get("exposure") or {}).get("delivery") or {}).get("expiry") or {}
    instant = str(expiry.get("target_instant") or "")
    year, month, day = instant[:4], instant[5:7].lstrip("0") or "0", instant[8:10].lstrip("0")
    mentions_now = bool(
        re.search(rf"{year}\s*年\s*0?{month}\s*月\s*0?{day}\s*日", answer)
        or (year and year in answer and "今天" in answer))
    mentions_gap = any(t in answer for t in ELAPSED)
    rows.append({"route": rp.parent.parent.name, "cell": rp.parent.name,
                 "mentions_now": mentions_now, "mentions_gap": mentions_gap})

from collections import Counter
by_route = {}
for r in rows:
    c = by_route.setdefault(r["route"], Counter())
    c["stale_cells"] += 1
    if r["mentions_now"]:
        c["mentions_current_date"] += 1
    if r["mentions_gap"]:
        c["mentions_elapsed"] += 1
    if r["mentions_now"] or r["mentions_gap"]:
        c["engages_time"] += 1
print(f"forced cells that are stale and judged: {len(rows)}")
for route, c in sorted(by_route.items()):
    print(f"  {route:24s} stale={c['stale_cells']:2d}  "
          f"states current date={c['mentions_current_date']:2d}  "
          f"states elapsed={c['mentions_elapsed']:2d}  "
          f"engages time at all={c['engages_time']:2d}")
out_path = HARN / "results_v4" / "e1_forced_residue.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps({"rows": rows, "by_route": {k: dict(v) for k, v in by_route.items()}},
               ensure_ascii=False, indent=1), encoding="utf-8")
