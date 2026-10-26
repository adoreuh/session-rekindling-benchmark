"""E2 manipulation audit: the relation text really arrived, and it asserted a
position but never a verdict."""
import json, glob
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "runs_e2_relation" / "e2"
VERDICT_TERMS = ["无效", "作废", "失效", "取消", "过期", "不必", "无需", "应该重新", "已经不"]
need_markers = ["当前世界时间", "事项原定日期", "时间关系"]
present = verdict_hits = invalid = total = relation_ok = 0
missing = []
for rp in sorted(root.glob("*/*/report.json")):
    total += 1
    report = json.loads(rp.read_text(encoding="utf-8"))
    b = report["branches"][0]
    if not b.get("run_valid"):
        invalid += 1
    ev = rp.parent / "branches" / "probe" / "events.jsonl"
    text = ev.read_text(encoding="utf-8") if ev.exists() else ""
    # the observation.compiled record carrying the relation
    blob = ""
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        d = e.get("data", {})
        if e.get("type") == "observation.compiled" and d.get("injected") == "relation":
            blob = d.get("text", "")
    if all(m in blob for m in need_markers):
        relation_ok += 1
    else:
        missing.append(rp.parent.name)
    if any(t in blob for t in VERDICT_TERMS):
        verdict_hits += 1
print(f"cells={total} relation-observation-present={relation_ok} invalid_run={invalid}")
print(f"relation text asserting a verdict (must be 0)={verdict_hits}")
if missing:
    print("missing relation observation:", missing[:10], "..." if len(missing) > 10 else "")
