"""Pass 5: candidate-explanation cues (failures vs successes), route concentration, gap effect by route,
coder false-positive proxy. Read-only; writes results_temporal_mining/summary5.json."""
import json, re, math
from collections import Counter, defaultdict
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results_temporal_mining"
rows = [json.loads(l) for l in (OUT / "rows.jsonl").open(encoding="utf-8")]
st = lambda r: r["fields"].get("current_state_assumed")
PAST = re.compile(r"already (passed|past|happened|over|occurred|done)|has (already )?passed|have (already )?passed|is (in the )?past|"
                  r"ago\b|is over|had passed|in the past|\(past\)|already took place|已经过去|已经过了|已过|过去了|都过了|早就过|已经结束|已经完成", re.I)
CUES = {
    "note_today_bound_to_now": re.compile(r"今天[（(]\s*(\d{4} ?年 ?)?\d+ ?月 ?\d+ ?日\s*[)）]|今天[（(]\d{4}-\d\d-\d\d[)）]|today \((\w+ \d+|\d{4}-\d\d-\d\d)\)|"
                                          r"[\"“']?今晚[\"”']?就是(那个|当时|那天)|今天[\"”']?就是(用户和助手|那天|准备)", re.I),
    "defers_to_user_decision": re.compile(r"用户(之前|已经|自己)?(已经)?(决定|说过|表示|明确)|user (previously |already )?(said|decided)|user's (own )?(plan|decision)|尊重|respect", re.I),
    "defers_to_notes": re.compile(r"rely on the notes|based on the notes|根据(用户)?(最后|最近)?(一条)?笔记|根据之前的对话|按照笔记|笔记的语境|the notes say|most (relevant|recent) note", re.I),
    "suspects_hypothetical": re.compile(r"hypothetical|假设|role-?play|scenario|测试|test(ing)? (whether|if)|trap|distractor|模拟", re.I),
    "suspects_date_error": re.compile(r"(日期|笔记|信息)(可能)?有误|搞错|混淆了日期|typo|mistake|wrong date|记错", re.I),
    "echoes_probe_wording": re.compile(r"之前了解的情况|结合之前|based on what (you|I) know|what you knew|from before|earlier knowledge|基于.{0,4}之前", re.I),
}
def had_time(r):
    return not (r["vis"] == "natural" and r["probe_fetch"] == 0)
base = [r for r in rows if r["exp"] in ("expiry", "e1") and r["arm"] in ("post", "late") and r["reasoning"].strip() and had_time(r)]
fail = [r for r in base if st(r) is True]
fail_past = [r for r in fail if PAST.search(r["reasoning"])]
ok_past = [r for r in base if st(r) is False and PAST.search(r["reasoning"])]
S = {"stale_with_time_and_reasoning": len(fail),
     "by_route_stale": dict(Counter(r["route"] for r in fail)),
     "by_route_stale_reasoning_past": dict(Counter(r["route"] for r in fail_past)),
     "cue_rates": {k: {"fail": f"{sum(bool(p.search(r['reasoning'])) for r in fail_past)}/{len(fail_past)}",
                       "success": f"{sum(bool(p.search(r['reasoning'])) for r in ok_past)}/{len(ok_past)}"} for k, p in CUES.items()}}
pre = [r for r in rows if r["exp"] == "expiry" and r["arm"] == "pre" and r["reasoning"].strip()]
S["past_coder_false_positive_on_pre"] = f"{sum(bool(PAST.search(r['reasoning'])) for r in pre)}/{len(pre)}"
main = [r for r in rows if r["exp"] in ("expiry", "e1") and r["arm"] in ("post", "late")]
idx = {(r["route"], r["profile"], r["vis"], r["arm"]): st(r) for r in main}
late_only = Counter(); post_only = Counter()
for (rt, pf, v, arm), s in idx.items():
    if v in ("delivered", "forced") and arm == "post":
        l = idx.get((rt, pf, v, "late"))
        if isinstance(s, bool) and isinstance(l, bool):
            if l and not s: late_only[rt] += 1
            if s and not l: post_only[rt] += 1
S["gap_effect_by_route_delivered_forced"] = {"late_only": dict(late_only), "post_only": dict(post_only)}
S["cue_rates_within_route"] = {}
for rt in ("DeepSeek-V4-Flash", "Qwen3.5-397B-A17B"):
    f_ = [r for r in fail_past if r["route"] == rt]; o_ = [r for r in ok_past if r["route"] == rt]
    S["cue_rates_within_route"][rt] = {k: {"fail": f"{sum(bool(p.search(r['reasoning'])) for r in f_)}/{len(f_)}",
                                           "success": f"{sum(bool(p.search(r['reasoning'])) for r in o_)}/{len(o_)}"} for k, p in CUES.items()}
(OUT / "summary5.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(S, ensure_ascii=False, indent=1))