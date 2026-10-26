"""Expand the scenario pool: many event kinds, each with several instances.

Why this exists.  The confirmatory design calls for 120 independent scenarios, and
the pilot has eight.  Adding scenarios by hand does not scale, and generating them
by changing only the dates would not be independent at all — a rescheduled interview
on 18 June and the same interview on 12 May are one situation, not two.

So the pool is built from two layers, and the distinction is kept because it bounds
what the study can claim:

``event kind``
    What is being rescheduled — a visa interview, a driving test, a device
    inspection.  Different kinds carry different vocabulary, different offices and
    a different reason the appointment matters, so they are genuinely different
    situations.

``instance``
    One kind realised with specific dates, an office naming variant and a booking
    time.  Instances of one kind share the *shape* of the situation and differ in
    its particulars.

The honest reading of the resulting count, which must travel with every result:
"N scenarios" means N independent experimental units spanning M event kinds, not N
distinct kinds.  A finding generalises across the kinds actually sampled and across
variation in dates and offices; it is not a random sample of all situations a
companion agent might meet, and no amount of instances per kind changes that.

Every instance is validated by the same checks as a hand-written scenario — an
unbuildable or proposition-firing instance fails here rather than as a gate failure
thousands of calls later.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.errors import ValidationError  # noqa: E402

from tools.scenario_factory import (  # noqa: E402
    EventProfile,
    build_variant,
)

#: The world's first day.  Every update date is derived from it so the instances
#: spread naturally over the spring rather than clustering on one tick.
EPOCH = date(2026, 3, 1)

#: Dates on which the office plausibly does not open, so a model can lawfully
#: refuse the appointment for a reason *unrelated* to the manipulation.
#:
#: Why this matters.  The delivery contrast asks whether withholding the notice
#: changes whether the model adopts the user's date.  If that date is a Saturday or
#: a public holiday, a model can answer "the office is closed, so going on the 2nd
#: will not work" — which is correct, and has nothing to do with evidence.  That
#: turns a lawful date check into noise on the manipulated variable, and it showed
#: up in the data: the single non-adopting branch out of 316 argued exactly that
#: ("5 月 2 日既是周末、又是法定假期期间…体检机构法定假期期间大概率是不开门的").
#:
#: Weekends are deterministic.  The fixed-date holidays below are stable in practice
#: (New Year, Labour Day, National Day).  The **movable** holidays are not covered —
#: Spring Festival, Qingming, Dragon Boat and Mid-Autumn follow the lunar calendar
#: and the State Council's annual announcement — so this is a partial filter and the
#: residual is recorded rather than papered over.
FIXED_HOLIDAYS = frozenset((
    (1, 1), (1, 2), (1, 3),                      # 元旦
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),      # 劳动节
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),   # 国庆
))


def is_plausible_open_day(value: date) -> bool:
    """Whether an office would plausibly be open on ``value``.

    Deliberately conservative: it only rules out cases where a model's refusal is
    clearly correct on calendar grounds alone.  It does not try to model the real
    opening hours of each institution, which is unknowable and would be tuning.
    """
    return value.weekday() < 5 and (value.month, value.day) not in FIXED_HOLIDAYS


def next_open_day(value: date) -> date:
    """The first plausible open day on or after ``value``."""
    shifted = value
    for _ in range(14):
        if is_plausible_open_day(shifted):
            return shifted
        shifted += timedelta(days=1)
    raise ValidationError(
        f"no plausible open day within two weeks of {value}; the calendar rule is "
        "probably wrong rather than the date"
    )


@dataclass(frozen=True)
class EventKind:
    """One kind of appointment that can be rescheduled.

    Only the naming and the vocabulary are here.  Dates come from the instance
    layer, so a kind cannot be tuned by giving it convenient dates.
    """

    kind_id: str
    referent: str
    referent_long: str
    office: str
    office_short: str
    #: Why the appointment matters to the user, used in the background material so
    #: the scenario reads as a real situation rather than a template.
    stake: str
    cancel_word: str = "取消"
    void_word: str = "作废"


#: Forty kinds.  Deliberately drawn from ordinary life rather than variations on one
#: theme: the point of a companion-agent study is that it meets mundane scheduling,
#: and a pool of forty near-identical interviews would support no generalisation at
#: all.  Each entry names a real queue a person waits in.
EVENT_KINDS = (
    EventKind("visa_interview", "签证面试", "赴德探亲签证面试",
              "签证预约中心", "预约中心", "这次面试决定探亲行程能否成行"),
    EventKind("driving_test", "路考", "机动车驾驶证路考",
              "车辆管理所考试中心", "考试中心", "路考通过才能拿到驾照"),
    EventKind("medical_exam", "体检", "出国前指定机构体检",
              "国际旅行卫生保健中心", "保健中心", "体检证明是签证材料的一部分"),
    EventKind("thesis_defense", "答辩", "学位论文答辩",
              "学院研究生教务办公室", "教务办公室", "答辩通过才能申请学位"),
    EventKind("house_viewing", "看房", "房屋现场看房预约",
              "中介预约服务台", "服务台", "看完这套房才能决定是否签约"),
    EventKind("language_test", "语言考试", "德语语言水平考试",
              "考试报名服务中心", "报名中心", "成绩单要赶在申请截止前拿到"),
    EventKind("equipment_check", "设备检验", "实验室设备年度检验",
              "设备管理处检验科", "检验科", "检验不合格设备就要停用"),
    EventKind("immigration_intake", "面签", "入境事务处现场面签",
              "入境事务处预约科", "预约科", "面签时间决定后续行程安排"),
    EventKind("residence_transfer", "户口迁移", "户口迁移办理",
              "户籍服务大厅", "服务大厅", "迁移办完才能办后续手续"),
    EventKind("employment_checkup", "入职体检", "入职前健康体检",
              "用人单位指定体检机构", "指定机构", "体检报告是入职材料之一"),
    EventKind("vehicle_inspection", "车辆年检", "机动车年检",
              "机动车检测站", "检测站", "年检过期就不能上路"),
    EventKind("skill_certification", "技能认证", "职业技能等级认证考试",
              "职业技能鉴定中心", "鉴定中心", "证书是岗位晋升的条件之一"),
    EventKind("loan_interview", "贷款面签", "住房贷款合同面签",
              "银行个人贷款中心", "贷款中心", "面签完成贷款才能放款"),
    EventKind("housing_fund", "公积金提取", "住房公积金提取办理",
              "住房公积金管理中心", "公积金中心", "提取到账才能垫付首付"),
    EventKind("pet_vaccination", "宠物接种", "宠物疫苗接种",
              "宠物医院预防接种门诊", "接种门诊", "接种记录要用于寄养登记"),
    EventKind("kindergarten_interview", "入园面谈", "幼儿园入园家长面谈",
              "幼儿园招生办公室", "招生办公室", "面谈结果决定入园名额"),
    EventKind("property_transfer", "房产过户", "房产过户登记",
              "不动产登记中心", "登记中心", "过户完成房子才算交付"),
    EventKind("notarization", "公证办理", "委托书公证办理",
              "公证处业务受理窗口", "受理窗口", "公证书是委托代办的凭据"),
    EventKind("specialist_clinic", "专科门诊", "医院专科门诊",
              "医院门诊预约平台", "预约平台", "这次门诊是复查的关键一环"),
    EventKind("license_renewal", "换证体检", "驾驶证换证体检",
              "驾驶人体检指定医院", "指定医院", "体检合格才能换发新证"),
    EventKind("postgrad_retest", "考研复试", "硕士研究生复试",
              "研究生招生办公室", "招生办公室", "复试成绩决定是否录取"),
    EventKind("ielts_registration", "雅思确认", "雅思考试报名确认",
              "考试报名确认中心", "确认中心", "确认后才算占住考位"),
    EventKind("child_checkup", "婴幼儿体检", "婴幼儿定期健康体检",
              "社区卫生服务中心", "卫生服务中心", "体检记录用于入园审核"),
    EventKind("marriage_registration", "婚姻登记", "结婚登记预约",
              "婚姻登记处", "登记处", "登记日期关系到后续安排"),
    EventKind("tax_interview", "税务面谈", "个人所得税汇算面谈",
              "税务局办税服务厅", "办税服务厅", "面谈结果影响退税进度"),
    EventKind("social_security_card", "社保卡领取", "社会保障卡领取",
              "社会保障卡服务中心", "服务中心", "领卡后才能办理报销"),
    EventKind("customs_declaration", "行李申报", "出境行李物品申报",
              "海关旅客申报服务点", "申报服务点", "申报手续影响登机时间"),
    EventKind("homestay_checkin", "民宿入住", "民宿入住登记",
              "民宿房东登记渠道", "登记渠道", "登记完成才能拿到门禁"),
    EventKind("teaching_review", "教学评估", "课程教学评估访谈",
              "学院教学评估办公室", "评估办公室", "访谈记录进入课程评审材料"),
    EventKind("project_review", "项目检查", "科研项目中期检查",
              "科研管理处项目管理科", "项目管理科", "检查结论影响后续拨款"),
    EventKind("broadband_install", "宽带装机", "家庭宽带上门装机",
              "宽带安装调度中心", "调度中心", "装好宽带才能开始远程办公"),
    EventKind("appliance_repair", "家电维修", "洗衣机上门维修",
              "家电维修派单中心", "派单中心", "修好才能正常使用"),
    EventKind("movers", "搬家上门", "搬家公司上门服务",
              "搬家公司调度台", "调度台", "搬家时间关系到退租日期"),
    EventKind("furniture_measure", "家具量尺", "定制家具上门量尺",
              "家具门店设计部", "设计部", "量尺数据决定生产排期"),
    EventKind("photo_session", "写真拍摄", "证件写真拍摄",
              "摄影工作室前台", "工作室前台", "照片要用于证件材料"),
    EventKind("final_interview", "终面", "数据分析岗终面",
              "招聘单位人事部", "人事部", "终面结果决定是否发 offer"),
    EventKind("lease_signing", "租房签约", "租房合同签约",
              "房屋租赁签约中心", "签约中心", "签约日期决定起租时间"),
    EventKind("dental_followup", "牙科复诊", "牙科治疗复诊",
              "口腔科门诊分诊台", "分诊台", "复诊是治疗流程的下一步"),
    EventKind("physiotherapy", "理疗预约", "康复理疗预约",
              "康复科预约处", "预约处", "理疗连续进行才有效果"),
    EventKind("car_maintenance", "保养预约", "车辆定期保养",
              "汽车服务中心", "服务中心", "保养提醒已经超期"),
)

#: Instance-level variation.  Each instance shifts the world update and the dates,
#: and rotates the office naming and booking time.
#:
#: These are particulars, not new kinds, and the count is reported accordingly.  A
#: spread of dates is worth having — it tests that the result does not depend on one
#: calendar — but it does not substitute for kind diversity.
INSTANCE_VARIANTS = (
    {"update_offset": 3, "lead_days": 45, "gap_days": 7, "booking_time": "上午 9 点"},
    {"update_offset": 6, "lead_days": 60, "gap_days": 14, "booking_time": "上午 10 点"},
    {"update_offset": 9, "lead_days": 75, "gap_days": 7, "booking_time": "下午 2 点"},
    {"update_offset": 12, "lead_days": 30, "gap_days": 21, "booking_time": "下午 3 点"},
)


def build_profiles(*, kinds=None, instances: int = 4,
                   instance_variants=None) -> list:
    """Expand kinds into concrete profiles.

    The same checks the hand-written pool passes are applied here, so a bad
    instance is caught at construction instead of as an exposure-gate failure after
    the calls are paid for.
    """
    kinds = tuple(kinds) if kinds else EVENT_KINDS
    variants = tuple(instance_variants or INSTANCE_VARIANTS)[:max(1, instances)]
    profiles = []
    for kind_index, kind in enumerate(kinds):
        for index, spec in enumerate(variants):
            # The kind also shifts the dates, not just the instance.
            #
            # Without this the pool holds one date triple per instance layer — forty
            # kinds sharing four calendars — and a reader could reasonably object
            # that the study never left the spring of 2026.  The shift is derived
            # from the kind's position, so it stays a property of the pool rather
            # than something a kind can be given.
            kind_shift = kind_index % 7
            update_day = EPOCH + timedelta(
                days=spec["update_offset"] + kind_shift
            )
            lead_days = spec["lead_days"] + ((kind_index * 3) % 15)
            # Both dates are moved onto plausible open days, so a model cannot reject
            # the appointment on calendar grounds.  Shifting forward keeps the
            # ordering the pool guarantees (notice before the appointment, new after
            # old), and the gap is preserved because both dates shift by the same
            # mechanism rather than by a fixed amount.
            old_date = next_open_day(update_day + timedelta(days=lead_days))
            new_date = next_open_day(old_date + timedelta(days=spec["gap_days"]))
            profile = EventProfile(
                profile_id=f"{kind.kind_id}_i{index + 1}",
                referent=kind.referent,
                referent_long=kind.referent_long,
                holder="用户",
                office=kind.office,
                office_short=kind.office_short,
                cancel_word=kind.cancel_word,
                void_word=kind.void_word,
                old_date=old_date,
                new_date=new_date,
                update_day=update_day,
                booking_time=spec["booking_time"],
                lead=kind.stake,
            )
            _check(profile, kind)
            profiles.append(profile)
    identifiers = [p.profile_id for p in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError("profile ids are not unique")
    # A repeated date triple is not an error — two different appointments can fall on
    # the same days — but the count is part of the record, so a reader can see how
    # much calendar variety the pool actually has.
    return profiles


def _check(profile: EventProfile, kind: EventKind) -> None:
    """Fail closed on an instance the gates could not check.

    These mirror the invariants the scenario builder depends on, so an instance that
    breaks one is reported here with its own name rather than surfacing as a gate
    failure on a run whose other branches succeeded.
    """
    if profile.old_date == profile.new_date:
        raise ValidationError(
            f"{profile.profile_id}: old and new dates coincide, so there is no change"
        )
    if profile.update_day >= profile.old_date:
        raise ValidationError(
            f"{profile.profile_id}: the notice arrives on or after the old date, so "
            "the user would already have missed it"
        )
    for label, value in (("old_date", profile.old_date),
                         ("new_date", profile.new_date)):
        if not is_plausible_open_day(value):
            raise ValidationError(
                f"{profile.profile_id}: {label} {value} is a weekend or a fixed "
                "holiday, so a model could refuse the appointment on calendar "
                "grounds rather than on the evidence"
            )
    if profile.new_date <= profile.old_date:
        raise ValidationError(
            f"{profile.profile_id}: the reschedule does not move the date later; a "
            "backwards move is a different situation and not what this pool encodes"
        )
    if not (kind.referent and kind.referent_long and kind.office):
        raise ValidationError(f"{profile.profile_id}: incomplete naming")
    # The long name must be usable as a referent on its own; the decisive
    # propositions are built from both spellings.
    if kind.referent not in kind.referent_long:
        # Not an error: "人脸识别" vs "面签" style pairs exist.  Recorded so a
        # reviewer knows the two spellings are independent vocabulary rather than
        # one being a substring of the other.
        pass


def check_buildable(profiles, *, filler_id: str) -> list:
    """Build each instance through the real scenario path and report failures.

    A profile that validates but cannot be assembled into a variant, or whose
    propositions fire on neutral material, is not usable; this is where that is
    discovered.
    """
    problems = []
    for profile in profiles:
        try:
            variant = build_variant(
                scenario_id=f"expand-{profile.profile_id}",
                profile=profile, filler_id=filler_id,
            )
            from tools.variant_wiring import _contract

            contract = _contract(variant)
            bodies = {d[0]: d[2] for d in variant.documents}
            for doc_id in variant.neutral_material_ids():
                hits = contract.decisive_asserted_in(bodies[doc_id])
                if hits:
                    problems.append(
                        f"{profile.profile_id}: neutral document {doc_id} asserts "
                        f"{hits}"
                    )
            if not contract.decisive_asserted_in(bodies[variant.gold_id]):
                problems.append(
                    f"{profile.profile_id}: the notice is not detected as decisive"
                )
        except Exception as exc:
            problems.append(f"{profile.profile_id}: {type(exc).__name__}: {exc}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--filler", default="routine_reminder_setup")
    parser.add_argument("--out", default="results_v4/scenario_pool.json")
    args = parser.parse_args()

    profiles = build_profiles(instances=args.instances)
    print("SCENARIO POOL")
    print("=" * 78)
    print(f"  event kinds : {len(EVENT_KINDS)}")
    print(f"  instances   : {args.instances} per kind")
    print(f"  profiles    : {len(profiles)}")
    print()
    problems = check_buildable(profiles, filler_id=args.filler)
    if problems:
        print(f"PROBLEMS ({len(problems)})")
        for problem in problems[:20]:
            print(f"  {problem}")
    else:
        print("all instances build and pass proposition checks")

    # Dates must not collapse onto one another, or the instances are not distinct.
    dates = {(p.update_day, p.old_date, p.new_date) for p in profiles}
    print()
    print(f"  distinct date triples: {len(dates)}/{len(profiles)}")
    for profile in profiles[:3] + profiles[-2:]:
        print(f"    {profile.profile_id:28s} upd={profile.update_day} "
              f"old={profile.old_date} new={profile.new_date}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "event_kinds": len(EVENT_KINDS),
        "instances_per_kind": args.instances,
        "profile_count": len(profiles),
        "distinct_date_triples": len(dates),
        "problems": problems,
        "profiles": [
            {"profile_id": p.profile_id, "kind": p.profile_id.rsplit("_i", 1)[0],
             "reference": p.referent_long, "update_day": p.update_day.isoformat(),
             "old_date": p.old_date.isoformat(), "new_date": p.new_date.isoformat()}
            for p in profiles
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
