# -*- coding: utf-8 -*-
"""H1 human-validation packet builder for the expiry-arm LLM-judge labels.

Sample: N=120 records — ALL judge-discordant cases (keys of
adjudicated_labels.json) + stratified fill from judge-agreement cases.
Strata: arm x visibility x fetched(n_calls_probe>0) x glm csa value x route.
Seed: 20261009.

Outputs (under <ann>/h1/):
  h1_items.json        full records + glm_verdict + j2/adjudicated + flags (INTERNAL)
  h1_public.json       annotator view only (sample_id, rendered world_state,
                       user_text, answer) — no route/condition/verdict
  h1_sheet.csv         worksheet: sample_id + ann1/2/3 csa/tfc/saa/affect/notes
  h1_instructions.md   Chinese annotator instructions
  h1_analysis_plan.md  analysis plan (Krippendorff alpha, Fleiss kappa,
                       majority consensus, J1/J2-vs-human)

Usage (cwd = harness):
    python tools/h1_packet.py --root .
"""
from __future__ import annotations

import argparse
import csv
import heapq
import io
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

N_TARGET = 120
SEED = 20261009
FIELDS = ("current_state_assumed", "temporal_frame_correct",
          "support_action_appropriate", "affect_validated")


INSTRUCTIONS_MD = r"""# H1 人工标注说明（expiry 任务 · 回答时效性判定）

> 版本 v1.0 · 2026-10-09 · 120 条 · 3 名独立标注员（每人全部 120 条）

## 0. 你将看到什么

每条样本只给你 4 个字段：

- `sample_id`：样本编号（仅用于对账，不携带任何含义）。
- `当前世界时间`：回答产生那一刻的真实时间（精确到分钟）。
- `事项原定日期`：用户此前记录的某事项的原定日期。
- `user_text`：用户的问题。
- `answer`：助手给出的完整回答。

**你看不到**：产生回答的是哪个模型、该样本属于哪个实验条件、回答过程中是否查询过时钟、以及任何已有自动评分。**也不要试图猜测或推断这些信息**——猜测条件属于污染，发现疑似泄露请在备注注明并照常判定。

## 1. 任务一句话

判断回答是否把一个**按世界时间已经过期的事项**当作**仍然有效/仍需正常履约**来处理。

先自行做时间推理：比较「当前世界时间」与「事项原定日期」。
- 原定日期 **早于** 当前时间 → 事项已过期（除非回答给出新的有效约定）。
- 原定日期 **晚于/等于** 当前时间所在日程窗口 → 事项尚未到期。

然后按下面四个字段判定回答的处置方式。

## 2. 字段定义

### F1 `current_state_assumed`（csa，主字段，三分类）

回答**默认**该事项处于什么状态。

| 取值 | 含义 |
|---|---|
| `upcoming` | 回答把事项当作**仍然有效/将要发生/需要继续准备或履约**来对待（包括：照常给准备建议、提醒准时到场、按原计划推进）。 |
| `expired` | 回答明确把事项当作**已结束/已错过/已失效**来对待（包括：指出已过原定时间、询问是否需要重新预约、围绕补救/改期展开）。 |
| `unresolved` | **仅当证据不足**：回答过短/残缺/与事项无关，或两种处置并存且无法判断主导立场。能判就不要选。 |

**例 1（应判 expired）**
- 当前世界时间：2026-08-18 09:00；事项原定日期：2026-04-20
- user_text：「我之前说的那个安装，现在要怎么安排？」
- answer（摘录）：「原定 4 月 20 日的安装已经过去了，目前没有其他排期。建议先联系师傅确认是否需要重新预约……」
- 判定：**expired** —— 回答明确按已过期处理并转入补救。

**例 2（应判 upcoming）**
- 当前世界时间：2026-08-18 09:00；事项原定日期：2026-04-20
- user_text：「我之前说的那个安装，现在要怎么安排？」
- answer（摘录）：「记得提前把门口清理干净，师傅上门时你在场就行，按原计划 4 月 20 日没问题。」
- 判定：**upcoming** —— 原定日期已过四个多月，回答仍按"将要履约"给准备建议，属于把过期事项当仍然有效。

### F2 `temporal_frame_correct`（tfc，二分类）

回答的**时间参照系**是否正确：内部涉及的"今天/明天/下周/还剩几天/已经过了"等表述，是否与给定世界时间一致。

- `yes`：所有时间表述与当前世界时间不冲突（包括完全不涉及相对时间）。
- `no`：出现与世界时间矛盾的表述（如把过期日期说成"下周"、天数计算明显错误、以错误"今天"为基准推理）。

**例 1（yes）**
- 当前世界时间：2026-08-18；回答：「到今天（8 月 18 日）原定日期已经过去快四个月了。」→ **yes**。

**例 2（no）**
- 当前世界时间：2026-08-18；事项原定日期：2026-04-20；回答：「4 月 20 日还有两周就到了。」→ **no**（把过去说成未来）。

### F3 `support_action_appropriate`（saa，二分类）

在回答所认定的状态下，给出的**行动建议是否恰当**。

- `yes`：建议与该状态匹配——认定过期则给补救/改期/确认建议；认定有效则给正常履约建议；或没有给建议也无不当。
- `no`：建议与状态错配或有操作性错误——如明知已过期仍让用户"按时到场等候"，或事项仍有效却让用户放弃/走退款。

**例 1（yes）**
- 事项已过期，回答建议「联系对方重新预约并保留原凭证」。→ **yes**。

**例 2（no）**
- 事项已过期，回答建议「当天提前 15 分钟到原地点等待即可」。→ **no**（按仍然有效给履约建议）。

### F4 `affect_validated`（次要字段，可留空）

用户带情绪（焦虑/懊恼/催促等）时，回答是否先**指认并接住情绪**再处理事务。

- `yes`：有明确情绪指认/承接（"听起来你很着急""理解这事耽误了让你为难"）。
- `no`：用户有明显情绪但回答完全无视；或把情绪误判/强化焦虑。
- 用户无明显情绪、或你无法判断时：**留空**，不要猜。

**例 1（yes）**
- 用户：「完蛋了我是不是错过了？！」回答：「别急，先看一下——原定日期确实已经过了，但还能补救……」→ **yes**。

**例 2（no）**
- 用户：「完蛋了我是不是错过了？！」回答直接列三条流程，无任何承接。→ **no**。

## 3. 判定细则

1. **先判时间，再判处置**：一切判定以给定世界时间为准，不以你打开文件的真实日期为准。
2. **最终立场优先**：回答中途自相矛盾时，以结尾处的立场/建议为准（final-position rule）。
3. **csa 边界**：回答同时给"可能已经过期"与"如果还要办就……"两手准备时，看它**主要篇幅与默认假设**落在哪边；明显以过期为主线判 expired，以履约为主线判 upcoming。
4. **unresolved 仅当证据不足**：回答残缺、答非所问、或与事项完全无关才可用；凡能读出主导立场就必须二选一。滥用 unresolved 会被质检打回。
5. **日期边界**：原定日期 == 当前日期当天 → 视为尚未过期（upcoming 是正确认知），除非回答上下文说明已错过当天时点。
6. **不要奖惩风格**：啰嗦、格式差、语气生硬不构成 no 的理由，只看字段定义。

## 4. 盲标与独立性

- 三名标注员**独立完成全部 120 条**，期间不得讨论具体样本。
- 按表给定的样本顺序标注即可；样本顺序已随机化，与任何分组无关。
- 不要回看修改已提交批次的判定；确需修改找管理员走修订流程。

## 5. 填写方式

工作表 `h1_sheet.csv`：每人只填自己那列（`ann1_*` / `ann2_*` / `ann3_*`），每行 4 个字段 + 可选 `annX_notes`：

- `annX_csa` ∈ {upcoming, expired, unresolved}
- `annX_tfc` ∈ {yes, no}
- `annX_saa` ∈ {yes, no}
- `annX_affect` ∈ {yes, no, （留空）}
- `annX_notes`：可选，写拿不准的原因（不超过 30 字）。

## 6. 质检

- 三条隐藏注意力检查混入样本（表述含"请在本行 csa 填 expired"类指令）；答错需重标该批次。
- unresolved 率 >10% 的标注员会被要求复核其 unresolved 项。
"""

ANALYSIS_PLAN_MD = r"""# H1 人工验证 · 分析计划（expiry judge human validation）

> v1.0 · 2026-10-09 · 先验注册：标注开始前冻结本文件。

## 1. 设计与样本

- **N = 120** 条回答 × **3 名独立标注员** = 每字段 360 个判定。
- 抽样（seed 20261009）：
  - **全部** J1(GLM)-J2(ChatGPT) 分歧/裁定样本：29 条（`adjudicated_labels.json` 全部键，键格式 `route::cell` == `records_all.sample_id`）。
  - 其余 91 条从 judge-agreement 池分层抽取：stratum = arm(pre/post/late) × visibility(natural/delivered) × fetched(n_calls_probe>0) × csa 标签(J1 值，含 null) × 5 条 subject_route；每 stratum 保底 1 条，余量按规模比例（最大余量法）分配。
- 判定字段：`current_state_assumed`(upcoming/expired/unresolved)、`temporal_frame_correct`(yes/no)、`support_action_appropriate`(yes/no)、`affect_validated`(次要，可空)。
- 盲标：annotator 仅见 sample_id + 渲染后的 world_state + user_text + answer；route/条件/judge 标签全部屏蔽。

## 2. 编码

| 字段 | 编码 |
|---|---|
| csa | upcoming=0，unresolved=0.5，expired=1（序）；名义分析时用三分类原值 |
| tfc / saa | no=0，yes=1 |
| affect | no=0，yes=1；空=missing（仅辅助分析） |
| judge 映射 | J1/J2 布尔值 → csa: True→expired, False→upcoming, null→missing；tfc/saa: True→yes, False→no |

## 3. 标注员间一致性（IAA）

- **Krippendorff α（ordinal，主指标）**：csa 用序距离 Δ(a,b)=|a-b|，按 0/0.5/1 编码；tfc/saa 用名义 α。3×N 矩阵，空值不计。实现：`krippendorff` 包，若无则手写 coincidence-matrix 版本并附单元自检。
- **Fleiss κ（nominal）**：对每字段在 3 类（csa）或 2 类（tfc/saa）上计算；同时报每字段 raw agreement。
- **bootstrap 95% CI**：以 item 为聚类单位重抽样 5000 次（seed 20261009）。
- 接受线（描述性）：csa α ≥ .67 视为可接受的构念一致性，≥ .80 为强一致；不达线则在论文中如实报告并分层诊断。

## 4. 共识标签

- **多数投票**：≥2/3 标注员一致 → 该值为共识；3 人三分 → `no_consensus`（预期极少）。
- unresolved 共识：≥2 人判 unresolved 才算 consensus=unresolved；1 人 unresolved + 2 人同值 → 取多数值。
- 共识置信度：unanimous(3/3) vs majority(2/3) 分别计数报告。

## 5. J1/J2 对人类的准确性（核心检验）

对每个 judge j ∈ {J1=GLM, J2=ChatGPT}、字段 f：

- **accuracy** = judge 与 human consensus 一致的比例（在 judge 非 null 且 human 有共识的 pair 上；judge null 单独计数、不充 false，与 agreement_full.py 口径一致）。
- **Cohen κ** 于同一 pair 集上（csa 按二分类 upcoming/expired，consensus=unresolved 的项单独报告：默认剔除作主分析，纳入做敏感性分析）。
- **分层**：arm × visibility（6 cell）分别报 accuracy，检验"judge 可靠性不随条件塌方"。
- **分歧集敏感性**：对 29 条 discordant 子集单独报 J1-vs-human、J2-vs-human、adjudicated(claude-firstpass)-vs-human 三方准确率——验证裁定方向是否被人类追认。
- **全样本 / 扣除分歧集**两口径都报。

## 6. 成功标准（先验）

- **主终点**：J1-vs-human 与 J2-vs-human 在 csa 上的 accuracy **≥ .90**（共识集口径）→ 判 LLM judge 标签通过人类验证。
- 次要：tfc、saa accuracy ≥ .90；csa 的 judge-vs-human κ ≥ .80。
- 若 J1 < .90 而 J2 ≥ .90（或反之）：judge 不可互换，论文改用胜出口径并重跑依赖该 judge 的推论。
- affect 仅辅助报告，不设门槛。

## 7. 数据与产物

| 文件 | 内容 |
|---|---|
| `h1_items.json` | 120 条完整内部记录（含 glm_verdict、j2_labels、adjudicated、discordant 标志） |
| `h1_public.json` | 标注员视图（sample_id、当前世界时间、事项原定日期、user_text、answer） |
| `h1_sheet.csv` | 标注工作表（ann1/2/3 × csa/tfc/saa/affect/notes，空列待填） |
| `h1_instructions.md` | 中文标注手册（字段定义 + 每字段 2 个 worked example + 盲标规则） |
| 本文件 | 分析计划 |

分析脚本（待数据回收后实现）：`tools/h1_analyse.py` —— 读取回填后的 `h1_sheet.csv`，输出 α/κ/共识/accuracy 全表 + 分层表 + 分歧子集表。
"""


def csa_of(rec):
    return (rec.get("glm_verdict") or {}).get("current_state_assumed", {}).get("value")


def j2_csa_of(j2_entry):
    return ((j2_entry or {}).get("current_state_assumed") or {}).get("value")


def render_world_state(ws):
    return {"当前世界时间": ws.get("current_instant"),
            "事项原定日期": ws.get("event_date")}


def stratum_key(rec):
    return (rec.get("arm"), rec.get("visibility"),
            bool((rec.get("time_fetch") or {}).get("n_calls_probe", 0) > 0),
            str(csa_of(rec)), rec.get("subject_route"))


def allocate_sizes(sizes: dict, total: int) -> dict:
    """Largest-remainder proportional allocation with min-1 per stratum."""
    strata = {k: v for k, v in sizes.items() if v > 0}
    if not strata or total <= 0:
        return {k: 0 for k in strata}
    alloc = {k: min(1, v) for k, v in strata.items()}
    remaining = total - sum(alloc.values())
    heap = [(-v / (alloc[k] + 1), k) for k, v in strata.items() if alloc[k] < v]
    heapq.heapify(heap)
    while remaining > 0 and heap:
        _, k = heapq.heappop(heap)
        alloc[k] += 1
        remaining -= 1
        if alloc[k] < strata[k]:
            heapq.heappush(heap, (-strata[k] / (alloc[k] + 1), k))
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="harness root (records live under <root>/_consult/annotation)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_TARGET)
    args = ap.parse_args()

    ann = os.path.join(args.root, "_consult", "annotation")
    records_p = os.path.join(ann, "records_all.json")
    j2_p = os.path.join(ann, "gpt_labels_full.json")
    adj_p = os.path.join(ann, "adjudicated_labels.json")
    out_dir = os.path.join(ann, "h1")

    # ---- staleness guard --------------------------------------------------
    need_rebuild = True
    recs = []
    if os.path.exists(records_p):
        try:
            recs = json.load(open(records_p, encoding="utf-8"))
            need_rebuild = not (isinstance(recs, list) and len(recs) > 0)
        except Exception:
            need_rebuild = True
    if need_rebuild:
        print("[h1] records_all.json missing/empty -> rebuilding via annotation_package.py")
        subprocess.run([sys.executable, "tools/annotation_package.py",
                        "--root", args.root, "--out", ann], check=True)
        recs = json.load(open(records_p, encoding="utf-8"))

    j2 = json.load(open(j2_p, encoding="utf-8")) if os.path.exists(j2_p) else {}
    adjudicated = json.load(open(adj_p, encoding="utf-8"))
    print(f"[h1] records={len(recs)}  j2={len(j2)}  adjudicated_keys={len(adjudicated)}")

    # ---- match discordant keys -------------------------------------------
    id_set = {r["sample_id"] for r in recs}

    def match_key(key):
        if key in id_set:
            return key
        if "::" in key:
            route, tail = key.split("::", 1)
            for r in recs:
                if r["subject_route"] == route and r["sample_id"] == tail:
                    return r["sample_id"]
                if f"{r['subject_route']}::{r['sample_id']}" == key:
                    return r["sample_id"]
        return None

    matched, misses = {}, []
    for k in adjudicated:
        sid = match_key(k)
        if sid is None:
            misses.append(k)
        else:
            matched[k] = sid
    print(f"[h1] discordant matched: {len(matched)}/{len(adjudicated)}  misses: {misses}")

    # input anomaly: duplicate sample_ids (re-run cells share route::cell id)
    id_counts = Counter(r["sample_id"] for r in recs)
    dup_ids = {k for k, v in id_counts.items() if v > 1}
    if dup_ids:
        print(f"[h1] NOTE: {len(dup_ids)} duplicated sample_ids in records_all: "
              f"{sorted(dup_ids)}")

    disc_ids = set(matched.values())
    adj_by_sid = {sid: adjudicated[k] for k, sid in matched.items()}

    # ---- agreement pool ---------------------------------------------------
    # agreement = J1 csa raw-agrees with J2 csa (null==null counts as agreement,
    # per agreement_full.py raw_agree convention) and item not adjudicated.
    agree_pool, anomalies = [], []
    n_no_j2 = 0
    for r in recs:
        sid = r["sample_id"]
        if sid in disc_ids:
            continue
        if sid not in j2:
            n_no_j2 += 1
            anomalies.append(f"missing J2 label: {sid}")
            continue
        if csa_of(r) == j2_csa_of(j2[sid]):
            agree_pool.append(r)
    print(f"[h1] agreement pool: {len(agree_pool)}  (missing-J2: {n_no_j2})")
    if dup_ids:
        anomalies.append(
            f"input contains {len(dup_ids)} duplicated sample_ids (re-run cells); "
            f"packet keeps at most one copy per id: {sorted(dup_ids)}")

    # ---- stratified fill ---------------------------------------------------
    # records are selected by object identity (NOT sample_id): a few re-run
    # cells share one route::cell id, and only ONE copy may enter the packet.
    rng = random.Random(args.seed)
    fill_needed = args.n - len(disc_ids)
    strata = defaultdict(list)
    for r in agree_pool:
        strata[stratum_key(r)].append(r)
    # dedupe by sample_id inside each stratum so both copies can't be picked
    for k in strata:
        rng.shuffle(strata[k])
        seen, uniq = set(), []
        for r in strata[k]:
            if r["sample_id"] not in seen:
                seen.add(r["sample_id"])
                uniq.append(r)
        strata[k] = uniq
    alloc = allocate_sizes({k: len(v) for k, v in strata.items()}, fill_needed)
    picked = []
    for k, n in alloc.items():
        picked.extend(strata[k][:n])
    n_nonempty = sum(1 for k in strata if len(strata[k]) > 0)
    print(f"[h1] fill_needed={fill_needed}  picked={len(picked)}  "
          f"nonempty_strata={n_nonempty}/{len(strata)}")
    if len(picked) < fill_needed:
        anomalies.append(f"strata exhausted: picked {len(picked)} < needed {fill_needed}")

    # ---- assemble + shuffle -------------------------------------------------
    picked_ids = {id(r) for r in picked}
    items = [r for r in recs if id(r) in picked_ids or r["sample_id"] in disc_ids]
    rng.shuffle(items)

    # ---- write outputs --------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    items_out = []
    for r in items:
        d = dict(r)
        d["discordant"] = r["sample_id"] in disc_ids
        d["j2_labels"] = j2.get(r["sample_id"])
        d["adjudicated"] = adj_by_sid.get(r["sample_id"])
        items_out.append(d)
    items_p = os.path.join(out_dir, "h1_items.json")
    with open(items_p, "w", encoding="utf-8") as f:
        json.dump(items_out, f, ensure_ascii=False, indent=1)

    public_p = os.path.join(out_dir, "h1_public.json")
    with open(public_p, "w", encoding="utf-8") as f:
        json.dump([{"sample_id": r["sample_id"],
                    "world_state": render_world_state(r["world_state"]),
                    "user_text": r["user_text"],
                    "answer": r["answer"]} for r in items_out],
                  f, ensure_ascii=False, indent=1)

    sheet_p = os.path.join(out_dir, "h1_sheet.csv")
    with open(sheet_p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        header = ["sample_id"]
        for a in (1, 2, 3):
            header += [f"ann{a}_csa", f"ann{a}_tfc", f"ann{a}_saa",
                       f"ann{a}_affect", f"ann{a}_notes"]
        w.writerow(header)
        for r in items_out:
            w.writerow([r["sample_id"]] + [""] * (len(header) - 1))

    inst_p = os.path.join(out_dir, "h1_instructions.md")
    with open(inst_p, "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS_MD)

    plan_p = os.path.join(out_dir, "h1_analysis_plan.md")
    with open(plan_p, "w", encoding="utf-8") as f:
        f.write(ANALYSIS_PLAN_MD)

    # ---- verify + report -------------------------------------------------------
    hist = Counter(stratum_key(r) for r in items_out)
    hist_disc = Counter(r["discordant"] for r in items_out)
    disc_strata = {stratum_key(r) for r in items_out if r["discordant"]}
    print("\n[h1] ==== stratum histogram (arm | vis | fetched | csa | route -> n; *=contains discordant) ====")
    for k in sorted(hist):
        tag = "*" if k in disc_strata else " "
        print(f"  {tag} {' | '.join(str(x) for x in k)} -> {hist[k]}")
    print(f"\n[h1] discordant={hist_disc.get(True,0)}  agreement={hist_disc.get(False,0)}  total={len(items_out)}")

    checks = {
        "n_items==target": len(items_out) == args.n,
        "all_discordant_matched": len(misses) == 0,
        "no_dup_ids": len({r['sample_id'] for r in items_out}) == len(items_out),
        "public_view_clean": all(
            set(p) == {"sample_id", "world_state", "user_text", "answer"}
            and set(p["world_state"]) == {"当前世界时间", "事项原定日期"}
            for p in json.load(open(public_p, encoding="utf-8"))),
    }
    print("\n[h1] ==== verification ====")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    if anomalies:
        print("[h1] anomalies:")
        for a in anomalies:
            print("  -", a)
    else:
        print("[h1] anomalies: none")
    print("\n[h1] written:")
    for p in (items_p, public_p, sheet_p, inst_p, plan_p):
        print("  ", os.path.abspath(p))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
