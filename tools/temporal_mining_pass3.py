"""Pass 3: deictic anchoring of undated note expressions, E3 user-anchored frame, distance paired test."""
import json, re, math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results_temporal_mining"
rows = [json.loads(l) for l in (OUT / "rows.jsonl").open(encoding="utf-8")]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

def stale(r): return r["fields"].get("current_state_assumed")
def rate(xs):
    xs = list(xs); k = sum(bool(x) for x in xs)
    return {"k": k, "n": len(xs), "p": round(k / len(xs), 3) if xs else None}
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def now_pats(iso):
    dt = datetime.fromisoformat(iso); m, d = dt.month, dt.day
    return [f"{m}月{d}日", f"{m} 月 {d} 日", f"{dt:%Y-%m-%d}", f"{MONTHS[m-1]} {d}", f"{m}/{d}"]

# Misanchor: the note's undated "今天/today ... 准备/练习/prepar" placed on the current date
def misanchor(r):
    t = r["reasoning"]
    if not t or not r["current_instant"]: return False
    for p in now_pats(r["current_instant"]):
        for m in re.finditer(re.escape(p), t):
            w = t[max(0, m.start() - 25): m.end() + 40]
            if re.search(r"今天|today|Today", w) and re.search(r"准备|练习|prepar|practic|与助手|和助手|with (the )?assistant", w):
                return True
    if re.search(r"今晚[\"”']?就是(那个|当时|那天)|今天[\"”']?就是(用户和助手|那天|准备)|还在和助手一起准备|"
                 r"today.{0,20}(still )?(prepar|practic)", t): 
        return True
    return False

# Correct anchoring: the note's "tonight/today" assigned to an earlier time than now
ANCHOR_PAST = re.compile(r"(note|that|this|which|it)\s+(might|may|could|must|seems to|likely|probably)\s+(be|have been)\s+from|"
                         r"from (an )?earlier (time|date|conversation)|from before|before the (interview|exam|event|appointment|signing|test)|"
                         r"(eve|night) before|previous \"?tonight|that \"?tonight\"? (was|referred)|was (likely )?(on|from) (april|may|march)|"
                         r"notes might not all be current|(笔记|记录|这条|那条|这些)(信息|笔记|记录)?(可能)?是(之前|更早|当时)|"
                         r"那(个|时)的?[\"“]?今晚|当时的[\"“]?今晚|之前那晚|是之前的对话|发生在.{0,12}之前", re.I)

S = {}
main = [r for r in rows if r["exp"] in ("expiry", "e1", "e2") and r["arm"] in ("post", "late")]
for r in main:
    r["mis"] = misanchor(r); r["anc"] = bool(ANCHOR_PAST.search(r["reasoning"] or ""))

def cls(r):
    if r["mis"] and not r["anc"]: return "binds_note_to_now"
    if r["anc"]: return "anchors_note_to_past"
    return "no_explicit_anchor"

tab = {}
for vis in ("natural", "delivered", "forced", "relation"):
    g = [r for r in main if r["vis"] == vis and r["reasoning"].strip() and isinstance(stale(r), bool)]
    if vis == "natural": g = [r for r in g if r["probe_fetch"] > 0]
    c = defaultdict(lambda: [0, 0])
    for r in g:
        c[cls(r)][0] += bool(stale(r)); c[cls(r)][1] += 1
    tab[vis] = {k: {"stale": v[0], "n": v[1], "rate": round(v[0] / v[1], 3)} for k, v in c.items()}
S["note_anchor_vs_stale_cells_with_time"] = tab
pool = defaultdict(lambda: [0, 0])
for vis in ("natural", "delivered", "forced"):
    for k, v in tab[vis].items():
        pool[k][0] += v["stale"]; pool[k][1] += v["n"]
S["note_anchor_pooled_nat_del_forced"] = {k: {"stale": a, "n": b, "rate": round(a / b, 3)} for k, (a, b) in pool.items()}
S["note_anchor_by_route_forced"] = {}
for rt in sorted({r["route"] for r in main}):
    g = [r for r in main if r["vis"] == "forced" and r["route"] == rt and r["reasoning"].strip() and isinstance(stale(r), bool)]
    c = defaultdict(lambda: [0, 0])
    for r in g: c[cls(r)][0] += bool(stale(r)); c[cls(r)][1] += 1
    S["note_anchor_by_route_forced"][rt] = {k: f"{a}/{b}" for k, (a, b) in c.items()}

# quotes
q = []
for r in main:
    if r["vis"] in ("forced", "natural", "delivered") and stale(r) and r["mis"] and not r["anc"]:
        t = r["reasoning"]
        for p in now_pats(r["current_instant"]):
            i = t.find(p)
            if i >= 0:
                q.append(f"[{r['route']} {r['cell']}] ...{t[max(0,i-60):i+80]}..."); break
(OUT / "_misanchor_quotes.txt").write_text("\n".join(q), encoding="utf-8")

# E3: user-stated reference frame
e3 = [r for r in rows if r["exp"] == "e3"]
S["e3_by_arm_route"] = {}
for rt in sorted({r["route"] for r in e3}):
    d = {}
    for arm in ("ref_before", "ref_after"):
        g = [r for r in e3 if r["route"] == rt and r["arm"] == arm]
        d[arm] = {"stale_true": sum(stale(r) is True for r in g), "stale_false": sum(stale(r) is False for r in g),
                  "fetch": sum(r["probe_fetch"] > 0 for r in g), "n": len(g)}
    S["e3_by_arm_route"][rt] = d
gaps = Counter()
for r in e3:
    m = re.search(r"请以 ?(\d{4}) ?年 ?(\d+) ?月 ?(\d+) ?日", r["user_text"])
    if m and r["event_date"]:
        ref = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        gaps[f"{r['arm']}|{(ref - datetime.fromisoformat(r['event_date']).date()).days}"] += 1
S["e3_ref_gap_days"] = dict(gaps)

# Same-route comparison: frame stated by user (E3 ref_after) vs frame from tool (forced post)
S["frame_source_contrast"] = {}
for rt in sorted({r["route"] for r in e3}):
    a = [r for r in e3 if r["route"] == rt and r["arm"] == "ref_after" and isinstance(stale(r), bool)]
    f = [r for r in main if r["route"] == rt and r["vis"] == "forced" and isinstance(stale(r), bool)]
    d = [r for r in main if r["route"] == rt and r["vis"] == "delivered" and isinstance(stale(r), bool)]
    S["frame_source_contrast"][rt] = {"user_turn_reference_after": f"{sum(stale(r) for r in a)}/{len(a)}",
                                      "tool_injected_forced": f"{sum(stale(r) for r in f)}/{len(f)}",
                                      "system_line_delivered": f"{sum(stale(r) for r in d)}/{len(d)}"}

# Distance paired test pooled over delivered+forced
idx = {(r["route"], r["profile"], r["vis"], r["arm"]): stale(r) for r in main}
b = c = 0
for (rt, pf, v, arm), s in idx.items():
    if v in ("delivered", "forced") and arm == "post":
        l = idx.get((rt, pf, v, "late"))
        if isinstance(s, bool) and isinstance(l, bool):
            b += s and not l; c += l and not s
S["distance_pooled_del_forced"] = {"post_only": b, "late_only": c, "mcnemar_p": mcnemar(b, c)}
(OUT / "summary3.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(S, ensure_ascii=False, indent=1))