"""Pass 4: failure taxonomy, frame-source paired tests, unhedged completion, fetch need. Read-only."""
import json, re, math
from collections import Counter, defaultdict
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results_temporal_mining"
rows = [json.loads(l) for l in (OUT / "rows.jsonl").open(encoding="utf-8")]
st = lambda r: r["fields"].get("current_state_assumed")
PAST = re.compile(r"already (passed|past|happened|over|occurred|done)|has (already )?passed|have (already )?passed|is (in the )?past|"
                  r"ago\b|is over|had passed|in the past|\(past\)|already took place|已经过去|已经过了|已过|过去了|都过了|早就过|已经结束|已经完成", re.I)
MIS = re.compile(r"今天[（(]\s*(\d{4} ?年 ?)?\d+ ?月 ?\d+ ?日\s*[)）]|今天[（(]\d{4}-\d\d-\d\d[)）]|today \((\w+ \d+|\d{4}-\d\d-\d\d)\)", re.I)
A_PAST = re.compile(r"已经过去|已经过了|已过|过去了|都过了|已经结束|已经完成|早就|错过|过期")
HARD = re.compile(r"(已经|早就|早已)(结束|完成|举行|办完|考完|装好|做完|签完|量完|测完|顺利完成)")
HEDGE = re.compile(r"应该|可能|大概|也许|是否|吗|有没有|如果|要么|或者|或|还是|除非")
def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
S = {}
# taxonomy of stale cells that had the authoritative time (fetched natural, delivered, forced)
tax = defaultdict(Counter)
for r in rows:
    if r["exp"] not in ("expiry", "e1") or r["arm"] not in ("post", "late") or st(r) is not True or not r["reasoning"].strip():
        continue
    if r["vis"] == "natural" and r["probe_fetch"] == 0:
        continue
    p, m, a = bool(PAST.search(r["reasoning"])), bool(MIS.search(r["reasoning"])), bool(A_PAST.search(r["answer"]))
    k = ("computed_past" if p else "no_past_computed") + ("+note_today_bound_to_now" if m else "") + ("+answer_says_past" if a else "+answer_silent")
    tax[r["route"]][k] += 1; tax["ALL"][k] += 1
S["stale_with_time_taxonomy"] = tax
# frame source: same date given by user instruction (E3 ref_after) vs tool / system line / natural, post arm (30 d)
S["frame_source_paired"] = {}
for rt in ("DeepSeek-V4-Flash", "Qwen3.5-397B-A17B", "hy3", "qwen3-vl-plus", "DeepSeek-V4.1-Flash"):
    e3 = {r["profile"].replace("_ref", ""): st(r) for r in rows if r["exp"] == "e3" and r["route"] == rt and r["arm"] == "ref_after"}
    d = {}
    for exp, vis in (("e1", "forced"), ("expiry", "delivered"), ("expiry", "natural"), ("e2", "relation")):
        o = {r["profile"]: st(r) for r in rows if r["exp"] == exp and r["route"] == rt and r["vis"] == vis and r["arm"] == "post"}
        pr = [(e3[k], o[k]) for k in e3 if isinstance(e3.get(k), bool) and isinstance(o.get(k), bool)]
        b = sum(1 for x, y in pr if y and not x); c = sum(1 for x, y in pr if x and not y)
        d[vis] = {"n": len(pr), "user_ref_stale": sum(x for x, _ in pr), "other_stale": sum(y for _, y in pr), "p": round(mc(b, c), 5)}
    S["frame_source_paired"][rt] = d
# unhedged completion claims among non-stale answers
S["unhedged_completion"] = {}
for vis in ("natural", "delivered", "forced", "relation"):
    g = [r for r in rows if r["exp"] in ("expiry", "e1", "e2") and r["vis"] == vis and r["arm"] in ("post", "late") and st(r) is False]
    h = [r for r in g if any(HARD.search(s) and not HEDGE.search(s) for s in re.split(r"[。！\n；;]", r["answer"]))]
    S["unhedged_completion"][vis] = {"nonstale": len(g), "unhedged": len(h), "by_route": dict(Counter(r["route"] for r in h))}
# information-need: fetch when time absent (natural) vs present (delivered), all arms
ex = [r for r in rows if r["exp"] == "expiry"]
S["fetch_absent_vs_present"] = {rt: {"absent": f"{sum(r['probe_fetch']>0 for r in ex if r['route']==rt and r['vis']=='natural')}/{sum(r['route']==rt and r['vis']=='natural' for r in ex)}",
                                     "present": f"{sum(r['probe_fetch']>0 for r in ex if r['route']==rt and r['vis']=='delivered')}/{sum(r['route']==rt and r['vis']=='delivered' for r in ex)}"}
                                for rt in sorted({r["route"] for r in ex})}
S["notes_with_record_timestamp"] = f"{sum(bool(re.search(r'记录于|recorded at|\[\d{4}-', r['user_text'])) for r in ex)}/{len(ex)}"
(OUT / "summary4.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(S, ensure_ascii=False, indent=1))