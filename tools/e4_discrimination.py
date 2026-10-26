"""Response-discrimination audit for E4.

Accuracy alone cannot tell a computed answer from an invariant one.  For each
route, compare the two directions of a task pair that differ ONLY in the
manipulated temporal relation (B: past vs future; C: as-of after vs before).
If a route emits the same answer in both directions, its accuracy on that pair
is produced by an invariant response, not by tracking the relation, and the
pair carries no evidence about competence.
"""
import json, glob, collections
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "results_v4" / "e4_runs"
ROUTES = ["hy3", "Qwen3.5-397B-A17B", "DeepSeek-V4-Flash",
          "DeepSeek-V4.1-Flash", "qwen3-vl-plus"]
PAIRS = [("B_past", "B_future"), ("C_asof_before", "C_asof_after")]
for task in ["A", "B_past", "B_future", "C_asof_after", "C_asof_before"]:
    pass

def load(route, task):
    out = {}
    for f in glob.glob(str(base / route / ("*_" + task + ".json"))):
        d = json.load(open(f, encoding="utf-8"))
        kind = d["cell"]["cell_id"].split("::")[0]
        out[kind] = ((d["response"].get("content") or "").strip()[:6],
                     d["cell"]["gold"])
    return out

print("per-task response distribution and gold (kind-level)")
for route in ROUTES:
    print("==", route)
    for task in ["A", "B_past", "B_future", "C_asof_after", "C_asof_before"]:
        rec = load(route, task)
        counter = collections.Counter(a for a, _ in rec.values())
        gold = collections.Counter(g for _, g in rec.values())
        print("   ", task.ljust(15), "ans=", dict(counter), " gold=", dict(gold))

print()
print("DISCRIMINATION AUDIT (same kind, two directions differing only in relation)")
for route in ROUTES:
    for a, b in PAIRS:
        ra, rb = load(route, a), load(route, b)
        kinds = sorted(set(ra) & set(rb))
        same = [k for k in kinds if ra[k][0] == rb[k][0]]
        # accuracy of each direction
        ca = sum(1 for k in kinds if ra[k][0] == ra[k][1])
        cb = sum(1 for k in kinds if rb[k][0] == rb[k][1])
        print(f"  {route:22s} {a:14s}/{b:14s} n={len(kinds):2d} "
              f"identical-answer cells={len(same):2d} acc {ca}/{len(kinds)} vs {cb}/{len(kinds)}"
              + ("   <== INVARIANT RESPONSE" if len(same) == len(kinds) else ""))
