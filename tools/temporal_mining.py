"""Read-only mining of frozen VELA artifacts from the model-temporal-ability angle.

No model or judge API calls. Writes only to results_temporal_mining/.
Probe-turn quantities are computed strictly after branch.forked; history-turn
(prefix) quantities strictly before it. Injected forced calls are excluded from
self-initiated fetch counts.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
OUT = HARN / "results_temporal_mining"

ROUTE_DIRS = {
    "hy3": "hy3",
    "Qwen3.5-397B-A17B": "Qwen3_5-397B-A17B",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    "DeepSeek-V4.1-Flash": "DeepSeek-V4_1-Flash",
    "qwen3-vl-plus": "qwen3-vl-plus",
}
VIS = ("natural", "delivered", "forced", "relation")


def cell_dirs():
    for route, rd in ROUTE_DIRS.items():
        for c in (HARN / f"runs_expiry_{rd}" / "expiry" / route).glob("*"):
            if c.is_dir():
                yield "expiry", route, c
    for exp, sub in (("e1", "runs_e1_forced/e1"), ("e2", "runs_e2_relation/e2"),
                     ("e3", "runs_e3_fixed_ref/e3")):
        for rdir in (HARN / sub).glob("*"):
            for c in rdir.glob("*"):
                if c.is_dir():
                    yield exp, rdir.name, c
    for rep in (HARN / "runs_repeat").glob("rep*"):
        for rdir in rep.glob("*"):
            for c in rdir.glob("*"):
                if c.is_dir():
                    yield "repeat:" + rep.name, rdir.name, c


def load_events(c):
    p = c / "branches" / "probe" / "events.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge(c):
    p = c / "judge.expiry.json"
    if not p.exists():
        return {}, {}
    j = json.loads(p.read_text(encoding="utf-8"))
    v = j.get("branches", {}).get("probe", {}).get("verdict", {})
    return j.get("condition", {}), {k: x.get("value") for k, x in v.items() if isinstance(x, dict)}


def parse_name(name):
    m = re.match(r"(.+?)_(pre|post|late|ref_before|ref_after)_(natural|delivered|forced|relation)$", name)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def is_injected(d):
    return bool(d.get("injected")) or str(d.get("call_id", "")).startswith("forced-")


def md_patterns(iso):
    if not iso:
        return []
    dt = datetime.fromisoformat(iso)
    m, d = dt.month, dt.day
    return [f"{m} 月 {d} 日", f"{m}月{d}日", f"{m} 月 {d}日", f"{m}月 {d} 日", f"{dt:%Y-%m-%d}"]


ANS_PAST = re.compile(r"已经过[去了]|已过|过去了|早已|已经结束|已经完成|已经举行|已经办|已经考|错过|已经到期|过期|上个月")
ANS_COMPLETE = re.compile(r"已经结束|已经完成|已经举行|已经办完|已经考完|已经装好|已经顺利|早就结束|已经(做|办|装|考|签|测|量)完|"
                          r"应该已经(结束|完成|办|做完|装|考)")
ANS_CLARIFY_TOPIC = re.compile(r"改期|延期|重新预约|是否已经|有没有(去|参加|完成|办|做)|是不是|还没|新的安排|新的(面试|预约)|"
                               r"确认一下|实际情况|进展|结果")
ANS_ECHO_NOTE = re.compile(r"早点睡|早点休息|早睡|明天再(试|练|完整|过)|明天精神|今晚(先)?不练|先不练")
R_PAST = re.compile(r"already (passed|past|happened|over|occurred|done)|has (already )?passed|have (already )?passed|"
                    r"is (in the )?past|(\d+|one|a|about|exactly one|over a) (day|days|week|weeks|month|months) ago|"
                    r"days ago|month ago|weeks ago|is over|had passed|date has passed|that's past|in the past|"
                    r"\(past\)|past\)|already took place|已经过|已过|过去了", re.I)
R_NOTE_STALE = re.compile(r"(note|notes|that|this|it|they|which) (might|may|could|must|seems to|likely|probably) "
                          r"(be|have been|refer|be referring) (from|to)|from (an )?earlier (time|date|conversation)|"
                          r"from before|before the (interview|exam|event|appointment)|previous (\"?tonight|day|night)|"
                          r"refers? to a (previous|past|different)|was written (before|earlier)|from a (previous|past|prior)|"
                          r"notes might not all be current|not (be )?current", re.I)
R_TENSION = re.compile(r"discrepan|inconsisten|confus|\bodd\b|strange|weird|contradict|doesn't make sense|"
                       r"makes no sense|mismatch|矛盾|不一致|hmm", re.I)
R_DEFER = re.compile(r"rely on the notes|based on the notes|follow the (notes|plan|user)|most (relevant|recent|applicable) note|"
                     r"stick (to|with)|endorse that plan|the user (previously|already) (said|decided)|"
                     r"user's (own )?plan|respond based on|based on (the )?(most recent|available|previous)|"
                     r"expect me to use the notes|i should rely", re.I)


def code(cond, answer, reasoning):
    now_p = md_patterns(cond.get("current_instant"))
    ev = cond.get("event_date")
    ev_p = md_patterns(ev + "T00:00:00") if ev else []
    a, r = answer or "", reasoning or ""
    return {
        "ans_now": any(p in a for p in now_p) or bool(re.search(r"今天是|现在是|今天已经|现在已经", a)),
        "ans_event_date": any(p in a for p in ev_p),
        "ans_past": bool(ANS_PAST.search(a)),
        "ans_complete_assert": bool(ANS_COMPLETE.search(a)),
        "ans_clarify": ("？" in a or "?" in a) and bool(ANS_CLARIFY_TOPIC.search(a)),
        "ans_echo_note": bool(ANS_ECHO_NOTE.search(a)),
        "r_has": bool(r.strip()),
        "r_past": bool(R_PAST.search(r)),
        "r_note_stale": bool(R_NOTE_STALE.search(r)),
        "r_tension": bool(R_TENSION.search(r)),
        "r_defer": bool(R_DEFER.search(r)),
    }


def extract(exp, route, c):
    es = load_events(c)
    if es is None:
        return None
    fork = [i for i, e in enumerate(es) if e["type"] == "branch.forked"]
    if not fork:
        return None
    f = fork[-1]
    pre, post = es[:f], es[f + 1:]
    profile, arm, vis = parse_name(c.name)
    cond, fields = judge(c)

    def fetches(seq):
        return [e for e in seq if e["type"] == "tool.call" and e["data"].get("tool_name") == "world_time"
                and not is_injected(e["data"])]

    def reasons(seq):
        return "\n".join(((e["data"].get("response") or {}).get("reasoning_content") or "")
                         for e in seq if e["type"] == "model.response")

    pf, hf = fetches(post), fetches(pre)
    answers = [e["data"].get("text", "") for e in post if e["type"] == "assistant.message"]
    pre_answers = [e["data"].get("text", "") for e in pre if e["type"] == "assistant.message"]
    first_req = next((e for e in post if e["type"] == "model.request"), None)
    user_text = ""
    if first_req:
        user_text = next((m.get("content", "") for m in first_req["data"].get("messages", [])
                          if m.get("role") == "user"), "")
    row = {
        "exp": exp, "route": route, "cell": c.name, "profile": profile, "arm": arm, "vis": vis,
        "event_date": cond.get("event_date"), "current_instant": cond.get("current_instant"),
        "fields": fields, "probe_fetch": len(pf), "history_fetch": len(hf),
        "answer": answers[-1] if answers else "", "reasoning": reasons(post),
        "prefix_answer": pre_answers[0] if pre_answers else "", "prefix_reasoning": reasons(pre),
        "payload_digest": first_req["data"].get("payload_digest") if first_req else None,
        "user_text": user_text,
    }
    row["codes"] = code(cond, row["answer"], row["reasoning"])
    row["prefix_binds_future"] = bool(re.search(r"还有|距离|剩下|倒计时|周时间|天时间|大约 ?\d+ ?(周|天)|还剩",
                                                row["prefix_answer"]))
    row["prefix_reason_computes_gap"] = bool(re.search(r"(\d+|about|roughly|around) ?(weeks?|days?|months?) "
                                                       r"(away|left|until|from now|remaining)|weeks away|days away",
                                                       row["prefix_reasoning"], re.I))
    return row


def rate(xs):
    xs = list(xs)
    k = sum(bool(x) for x in xs)
    return {"k": k, "n": len(xs), "p": round(k / len(xs), 3) if xs else None}


def stale(r):
    return r["fields"].get("current_state_assumed")


def main():
    OUT.mkdir(exist_ok=True)
    rows = [r for r in (extract(*t) for t in cell_dirs()) if r]
    with (OUT / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    S = {"n_rows": len(rows), "by_exp": Counter(r["exp"].split(":")[0] for r in rows)}
    main_rows = [r for r in rows if r["exp"] in ("expiry", "e1", "e2")]
    expired = [r for r in main_rows if r["arm"] in ("post", "late")]

    for scope in ("pooled", "by_route"):
        out = {}
        groups = [("all", list(ROUTE_DIRS))] if scope == "pooled" else [(k, [k]) for k in ROUTE_DIRS]
        for gname, rs in groups:
            for vis in VIS:
                g = [r for r in expired if r["vis"] == vis and r["route"] in rs and r["codes"]["r_has"]
                     and stale(r) is not None]
                if not g:
                    continue
                st = [r for r in g if stale(r)]
                ok = [r for r in g if stale(r) is False]
                out[f"{gname}|{vis}"] = {
                    "n_cot": len(g), "stale": len(st),
                    "stale_r_past": rate(r["codes"]["r_past"] for r in st),
                    "stale_r_past_ans_not_past": rate(r["codes"]["r_past"] and not r["codes"]["ans_past"] for r in st),
                    "stale_r_note_stale": rate(r["codes"]["r_note_stale"] for r in st),
                    "stale_r_tension": rate(r["codes"]["r_tension"] for r in st),
                    "stale_r_defer": rate(r["codes"]["r_defer"] for r in st),
                    "stale_ans_echo_note": rate(r["codes"]["ans_echo_note"] for r in st),
                    "ok_r_past": rate(r["codes"]["r_past"] for r in ok),
                    "ok_r_note_stale": rate(r["codes"]["r_note_stale"] for r in ok),
                    "ok_r_defer": rate(r["codes"]["r_defer"] for r in ok),
                    "ok_ans_echo_note": rate(r["codes"]["ans_echo_note"] for r in ok),
                }
        S[f"reasoning_vs_answer_{scope}"] = out

    S["answer_profile"] = {
        f"{vis}|stale={s_}": {k: rate(r["codes"][k] for r in expired if r["vis"] == vis and stale(r) is s_)
                              for k in ("ans_now", "ans_event_date", "ans_past", "ans_complete_assert",
                                        "ans_clarify", "ans_echo_note")}
        for vis in VIS for s_ in (True, False)}

    tax = {}
    for vis in VIS:
        g = [r for r in expired if r["vis"] == vis and stale(r) is False]
        c = Counter()
        for r in g:
            k = r["codes"]
            c[("complete" if k["ans_complete_assert"] else "no_complete") + "+" +
              ("clarify" if k["ans_clarify"] else "no_clarify")] += 1
        tax[vis] = {"n": len(g), **c}
    S["nonstale_resolution"] = tax

    hist = {}
    for route in ROUTE_DIRS:
        g = [r for r in main_rows if r["exp"] == "expiry" and r["route"] == route and r["vis"] == "natural"]
        hist[route] = {
            "history_fetch": rate(r["history_fetch"] > 0 for r in g),
            "history_answer_binds_interval": rate(r["prefix_binds_future"] for r in g),
            "history_reasoning_computes_gap": rate(r["prefix_reason_computes_gap"] for r in g),
            "probe_fetch_pre": rate(r["probe_fetch"] > 0 for r in g if r["arm"] == "pre"),
            "probe_fetch_expired": rate(r["probe_fetch"] > 0 for r in g if r["arm"] in ("post", "late")),
        }
    S["history_vs_probe"] = hist

    ff = {}
    for route in ROUTE_DIRS:
        g = [r for r in expired if r["vis"] == "natural" and r["route"] == route and stale(r) is not None]
        fe = [r for r in g if r["probe_fetch"] > 0]
        ff[route] = {"fetched": len(fe), "fetched_stale": sum(bool(stale(r)) for r in fe),
                     "fetched_stale_r_past": sum(bool(stale(r)) and r["codes"]["r_past"] for r in fe),
                     "fetched_stale_ans_now": sum(bool(stale(r)) and r["codes"]["ans_now"] for r in fe),
                     "not_fetched": len(g) - len(fe),
                     "not_fetched_stale": sum(bool(stale(r)) for r in g if r["probe_fetch"] == 0)}
    S["natural_fetch_decomposition"] = ff

    S["distance"] = {f"{vis}|{arm}": rate(stale(r) for r in expired if r["vis"] == vis and r["arm"] == arm
                                          and stale(r) is not None) for vis in VIS for arm in ("post", "late")}
    gaps = Counter()
    for r in main_rows:
        if r["current_instant"] and r["event_date"]:
            d = (datetime.fromisoformat(r["current_instant"]).date()
                 - datetime.fromisoformat(r["event_date"]).date()).days
            gaps[f"{r['arm']}|{d}"] += 1
    S["arm_gap_days"] = dict(sorted(gaps.items()))

    pre = [r for r in main_rows if r["arm"] == "pre"]
    S["pre_arm"] = {vis: {"fetch": rate(r["probe_fetch"] > 0 for r in pre if r["vis"] == vis),
                          "stale_field": dict(Counter(str(stale(r)) for r in pre if r["vis"] == vis)),
                          "echo_note": rate(r["codes"]["ans_echo_note"] for r in pre if r["vis"] == vis)}
                    for vis in ("natural", "delivered")}

    e3 = [r for r in rows if r["exp"] == "e3"]
    S["e3"] = {route: {
        "n": sum(r["route"] == route for r in e3),
        "fetch": sum(r["route"] == route and r["probe_fetch"] > 0 for r in e3),
        "fetch_r_mentions_mismatch_or_reference": sum(
            r["route"] == route and r["probe_fetch"] > 0 and bool(re.search(
                r"doesn't match|not match|differ|mismatch|actual (current )?date|real (current )?date|hypothetical|"
                r"baseline|reference", r["reasoning"], re.I)) for r in e3),
        "nofetch_r_says_unnecessary": sum(
            r["route"] == route and r["probe_fetch"] == 0 and bool(re.search(
                r"(not|no) need|isn't necessary|not necessary|don't need|already (given|provided)|baseline|redundant",
                r["reasoning"], re.I)) for r in e3),
    } for route in ROUTE_DIRS}

    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()