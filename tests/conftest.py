"""Shared fixtures: a small, fully frozen experiment used across the tests."""

from __future__ import annotations

import pytest

from vela_lab.clock import Clock
from vela_lab.evidence import Document, ExposurePolicy
from vela_lab.prompt import PromptBundle, neutral_system_prompt
from vela_lab.runner import BranchSpec, ExperimentSpec
from vela_lab.schedule import WorldEvent

TZ = "Asia/Shanghai"
START = "2026-03-01T09:00:00+08:00"

OLD_DOC = Document(
    doc_id="notice_old",
    text="签证预约安排在 3 月 1 日上午报到，携带材料。",
    authority="medium",
    source="consulate_notice",
    published_at="2026-02-01T00:00:00+08:00",
    superseded_by="notice_new",
    stance="supports_old",
    facts=("visa_date",),
)

NEW_DOC = Document(
    doc_id="notice_new",
    text="签证预约已改期至 3 月 24 日上午报到。",
    authority="high",
    source="consulate_update",
    published_at="2026-03-05T09:00:00+08:00",
    stance="supports_current",
    facts=("visa_date",),
    is_gold=True,
)

FAQ_DOC = Document(
    doc_id="faq",
    text="常见问题：签证材料准备清单与注意事项。",
    authority="low",
    source="faq",
    published_at="2026-01-01T00:00:00+08:00",
)

BLOG_DOC = Document(
    doc_id="blog",
    text="我的签证面试经历分享与心得。",
    authority="low",
    source="personal_blog",
    published_at="2026-01-01T00:00:00+08:00",
)

DOCUMENTS = (OLD_DOC, NEW_DOC, FAQ_DOC, BLOG_DOC)

WORLD_EVENTS = (
    WorldEvent(
        event_id="visa_rescheduled",
        kind="supersede_fact",
        at="2026-03-05T09:00:00+08:00",
        order=0,
        fact_id="visa_date",
        new_value="2026-03-24",
        source="consulate_update",
    ),
    WorldEvent(
        event_id="visa_notice_published",
        kind="publish_evidence",
        at="2026-03-05T09:00:00+08:00",
        order=1,
        evidence_id="notice_new",
    ),
)

PREFIX = ("我三月要办签证，怕自己看漏通知。",)


def bundle() -> PromptBundle:
    return PromptBundle(
        prompt_id="neutral-v1",
        system=neutral_system_prompt(),
        tool_descriptions={
            "world_time": "Get the current date and time.",
            "search": "Search your notes and documents for relevant information.",
        },
    )


@pytest.fixture
def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="ewm-min",
        clock_start=Clock.of(START, TZ),
        time_zone=TZ,
        prompt=bundle(),
        documents=DOCUMENTS,
        world_events=WORLD_EVENTS,
        enabled_tools=("world_time", "search"),
        prefix_turns=PREFIX,
        branches=(
            BranchSpec(
                branch_id="natural_neutral",
                exposure=ExposurePolicy(mode="full", top_k=2, label="natural_neutral"),
                user_text="我好像已经把签证面试错过了，是不是没救了？",
            ),
            BranchSpec(
                branch_id="natural_false",
                exposure=ExposurePolicy(mode="full", top_k=2, label="natural_false"),
                user_text="我昨天去面试了，结果答得很差。",
            ),
            BranchSpec(
                branch_id="oracle_false",
                exposure=ExposurePolicy(
                    mode="full", top_k=2, force_include_ids=("notice_new",),
                    label="oracle_false",
                ),
                user_text="我昨天去面试了，结果答得很差。",
            ),
            BranchSpec(
                branch_id="omit_gold",
                exposure=ExposurePolicy(
                    mode="full", top_k=2, omit_ids=("notice_new",), label="omit_gold"
                ),
                user_text="我昨天去面试了，结果答得很差。",
            ),
        ),
        gold_fact_id="visa_date",
        gold_marker="notice_new",
        advance_days=10,
    )
