"""Second pass: bilingual coders + paired tests. Read-only; writes results_temporal_mining/summary2.json."""
import json, re, math
from collections import Counter, defaultdict
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results_temporal_mining"
rows = [json.loads(l) for l in (OUT / "rows.jsonl").open(encoding="utf-8")]

PAST = re.compile(r"already (passed|past|happened|over|occurred|done)|has (already )?passed|have (already )?passed|"
                  r"is (in the )?past|ago\b|is over|had passed|in the past|\(past\)|already took place|"
                  r"已经过去|已经过了|已过|过去了|都过了|早就过|已经结束|已经完成", re.I)
TENSION = re.compile(r"discrepan|inconsisten|confus|\bodd\b|strange|weird|contradict|doesn't make sense|mismatch|"
                     r"矛盾|不一致|奇怪|不太对|有点怪|说不通|冲突", re.I)
# model re-anchors the user's present deictics ("tonight") to the note's past frame, or explicitly follows note plan
REANCHOR = re.compile(r"[\"“']?今晚[\"”']?就是(那个|当时|那天)|今天[\"”']?就是(用户和助手|那天|准备的那天)|"
                      r"基于笔记的语境|按笔记的语境|按照对话的设定|延续之前的对话语境|不需要纠结真实日期|"
                      r"假设的场景|假设性的|hypothetical|role-?play|scenario is set|as if it'?s still before|"
                      r"notes are contextually|the notes' (context|timeline)", re.I)
DEFER = re.compile(r"rely on the notes|based on the notes|follow the (notes|plan)|most (relevant|recent|applicable) note|"
                   r"the user (previously|already) (said|decided)|endorse that plan|respond based on|i should rely|"
                   r"根据用户(之前|最后|最近)的(决定|笔记|意愿)|尊重(用户|这个)(的)?决定|基于用户最近的意愿|按照(用户)?自己之前的决定|"
                   r"根据笔记\s*\d|我应该基于这些笔记|主要依据笔记", re.I)
NOTE_EARLIER = re.compile(r"from (an )?earlier|from before|before the (interview|exam|event|appointment|signing)|"
                          r"likely (was )?(from|on) the (eve|night)|note (might|may|seems to) be from|notes might not all be current|"
                          r"(笔记|记录|信息)(可能)?是之前|之前记录的|那是之前|发生在.{0,10}之前的对话", re.I)
COMPLETE_HARD = re.compile(r"(已经|早就|早已)(结束|完成|举行|办完|考完|装好|做完|签完|量完|测完|顺利完成)(了)?")
HEDGE = re.compile(r"应该|可能|大概|也许|是否|吗[？?]|有没有|如果")


def stale(r):
    return r["fields"].get("current_state_assumed")


def rate(xs):
    xs = list(xs); k = sum(bool(x) for x in xs)
    return {"k": k, "n": len(xs), "p": round(k / len(xs), 3) if xs else None}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


main = [r for r in rows if r["exp"] in ("expiry", "e1", "e2") and r["arm"] in ("post", "late")]
for r in main:
    t = r["reasoning"]
    a = r["answer"]
    sents = re.split(r"[。！\n]", a)
    hard = any(COMPLETE_HARD.search(s) and not HEDGE.search(s) for s in sents)
    r["c"] = {"cot": bool(t.strip()), "past": bool(PAST.search(t)), "tension": bool(TENSION.search(t)),
              "reanchor": bool(REANCHOR.search(t)), "defer": bool(DEFER.search(t)), "note_earlier": bool(NOTE_EARLIER.search(t)),
              "a_past": bool(re.search(r"已经过去|已经过了|已过|过去了|都过了|已经结束|已经完成|早就|错过|过期", a)),
              "a_hard_complete": hard,
              "a_clarify": bool(re.search(r"[？?]", a)) and bool(re.search(r"改期|延期|重新|是否|有没有|是不是|还没|新的|确认一下|实际情况|进展|结果|顺利", a))}

S = {}
# 1. stale & had the time: did reasoning compute expiry?
for vis in ("natural", "delivered", "forced"):
    g = [r for r in main if r["vis"] == vis and stale(r) is True and r["c"]["cot"]]
    if vis == "natural":
        g = [r for r in g if r["probe_fetch"] > 0]
    by = defaultdict(list)
    for r in g:
        by[r["route"]].append(r)
    S[f"stale_with_time_{vis}"] = {
        "pooled": {k: rate(r["c"][k] for r in g) for k in ("past", "tension", "reanchor", "defer", "note_earlier", "a_past")},
        "past_in_reasoning_but_not_in_answer": rate(r["c"]["past"] and not r["c"]["a_past"] for r in g),
        "by_route": {rt: {k: rate(x["c"][k] for x in v) for k in ("past", "tension", "reanchor", "defer", "a_past")} for rt, v in by.items()},
    }
# 2. contrast: non-stale cells with time
for vis in ("natural", "delivered", "forced", "relation"):
    g = [r for r in main if r["vis"] == vis and stale(r) is False and r["c"]["cot"]]
    S[f"nonstale_{vis}"] = {k: rate(r["c"][k] for r in g) for k in ("past", "tension", "reanchor", "defer", "note_earlier")}

# 3. unsupported completion assertions among non-stale answers
S["nonstale_answer_resolution"] = {}
for vis in ("natural", "delivered", "forced", "relation"):
    g = [r for r in main if r["vis"] == vis and stale(r) is False]
    S["nonstale_answer_resolution"][vis] = {
        "n": len(g), "hard_completion_assertion": rate(r["c"]["a_hard_complete"] for r in g),
        "clarifies_status": rate(r["c"]["a_clarify"] for r in g),
        "hard_assert_without_clarify": rate(r["c"]["a_hard_complete"] and not r["c"]["a_clarify"] for r in g),
        "by_route_hard": {rt: rate(r["c"]["a_hard_complete"] for r in g if r["route"] == rt) for rt in sorted({x["route"] for x in g})},
    }

# 4. elapsed distance: paired post vs late within (route, profile, vis)
idx = {(r["route"], r["profile"], r["vis"], r["arm"]): stale(r) for r in main}
S["distance_paired"] = {}
for vis in ("natural", "delivered", "forced"):
    b = c = both = neither = 0
    by = Counter()
    for (rt, pf, v, arm), s in idx.items():
        if v != vis or arm != "post":
            continue
        l = idx.get((rt, pf, v, "late"))
        if not isinstance(s, bool) or not isinstance(l, bool):
            continue
        if s and l: both += 1
        elif s and not l: b += 1
        elif l and not s: c += 1; by[rt] += 1
        else: neither += 1
    S["distance_paired"][vis] = {"post_only_stale": b, "late_only_stale": c, "both": both, "neither": neither,
                                 "mcnemar_p": round(mcnemar(b, c), 4), "late_only_by_route": dict(by)}
# reasoning of late-only stale under forced/delivered: what do they say?
lo = []
for (rt, pf, v, arm), s in idx.items():
    if v in ("forced", "delivered") and arm == "late" and s is True and idx.get((rt, pf, v, "post")) is False:
        lo.append((rt, pf, v))
S["late_only_examples"] = lo[:20]
lo_rows = [r for r in main if (r["route"], r["profile"], r["vis"]) in set(lo) and r["arm"] == "late"]
S["late_only_reasoning_codes"] = {k: rate(r["c"][k] for r in lo_rows if r["c"]["cot"]) for k in ("past", "tension", "reanchor", "defer", "note_earlier")}

# 5. fetch-need sensitivity: does the model fetch when the clock is already in context?
pre_all = [r for r in rows if r["exp"] == "expiry"]
S["fetch_when_time_absent_vs_present"] = {rt: {"natural(absent)": rate(r["probe_fetch"] > 0 for r in pre_all if r["route"] == rt and r["vis"] == "natural"),
                                              "delivered(present)": rate(r["probe_fetch"] > 0 for r in pre_all if r["route"] == rt and r["vis"] == "delivered")}
                                         for rt in sorted({r["route"] for r in pre_all})}
# 6. user turn: does the probe user text carry any explicit date? (history turn did)
S["probe_user_has_digit_date"] = rate(bool(re.search(r"结合你之前了解的情况.*\d+ ?月", r["user_text"].split("结合你之前了解的情况")[-1][:40])) for r in pre_all)

(OUT / "summary2.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")

# dump reanchor quotes
qs = []
for r in main:
    if r["vis"] in ("forced", "natural", "delivered") and stale(r) is True and r["c"]["reanchor"]:
        m = REANCHOR.search(r["reasoning"]); s = max(0, m.start() - 160)
        qs.append(f"[{r['route']} {r['cell']}] ...{r['reasoning'][s:m.end()+120]}...")
(OUT / "_reanchor_quotes.txt").write_text("\n\n".join(qs), encoding="utf-8")
hc = [f"[{r['route']} {r['cell']}] " + next(s for s in re.split(r'[。！\n]', r['answer']) if COMPLETE_HARD.search(s) and not HEDGE.search(s))
      for r in main if r["vis"] == "relation" and r["c"]["a_hard_complete"]]
(OUT / "_hard_complete_quotes.txt").write_text("\n".join(hc), encoding="utf-8")