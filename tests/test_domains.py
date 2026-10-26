"""The structural fixes: the domain gate and the frozen prompt.

These tests exist to fail if someone reintroduces the two bugs that invalidated
the project's earlier bridging experiment:

* gold leaking back into the model's context through a side channel;
* a system prompt clause that hints at the behaviour under test.
"""

from __future__ import annotations

import pytest

from vela_lab.compiler import RequestCompiler, verify_compiled_request
from vela_lab.domains import (
    ObservationDomain,
    JudgeDomain,
    Provenance,
    VisibleFragment,
    WorldDomain,
    assert_no_forbidden_terms,
)
from vela_lab.errors import ContractViolation, ValidationError
from vela_lab.prompt import (
    PromptBundle,
    PromptRegistry,
    neutral_system_prompt,
    sensitivity_report,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search your notes and documents for relevant information.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _bundle() -> PromptBundle:
    return PromptBundle(
        prompt_id="neutral-v1",
        system=neutral_system_prompt(),
        tool_descriptions={"search": "Search your notes and documents."},
    )


# ------------------------------------------------------------ domain gate


def test_visible_fragment_cannot_be_constructed_directly():
    with pytest.raises(ContractViolation, match="must be minted"):
        VisibleFragment(
            kind="user", text="hello", provenance=Provenance("observation", "seq:1")
        )


def test_minting_rejects_non_string_text():
    obs = ObservationDomain()
    with pytest.raises(ValidationError, match="must be a string"):
        obs.user({"evidence_ids": ["notice_new"]}, event_seq=1)


def test_minting_records_provenance():
    obs = ObservationDomain()
    obs.user("hello", event_seq=7)
    fragment = obs.fragments()[-1]
    assert fragment.provenance.source == "observation"
    assert fragment.provenance.ref == "seq:7"


def test_judge_domain_has_no_projection_method():
    """The seal is structural: there is no method that yields visible text."""
    methods = [m for m in dir(JudgeDomain) if not m.startswith("__")]
    for name in methods:
        assert "visible" not in name
        assert "project" not in name
        assert "fragment" not in name


def test_gold_ids_are_readable_but_not_projectable():
    judge = JudgeDomain()
    judge.load_gold("evidence_ids", ["notice_new"])
    assert judge.gold_ids() == ["notice_new"]
    # Gold is a dict/list, and the only minter accepts strings.  There is no
    # method that turns a judge value into visible text.
    obs = ObservationDomain()
    with pytest.raises(ValidationError, match="must be a string"):
        obs.user(judge.all_gold(), event_seq=1)


def test_forbidden_term_scan_catches_vocabulary():
    obs = ObservationDomain()
    obs.user("this is the gold oracle condition", event_seq=1)
    with pytest.raises(ContractViolation, match="experiment vocabulary"):
        assert_no_forbidden_terms(obs.fragments())


def test_forbidden_term_scan_passes_clean_text():
    obs = ObservationDomain()
    obs.user("我三月要办签证，怕自己看漏通知。", event_seq=1)
    assert_no_forbidden_terms(obs.fragments())


# ---------------------------------------------------------- prompt registry


def test_prompt_registry_refuses_to_replace_a_bundle():
    registry = PromptRegistry()
    registry.register(_bundle())
    with pytest.raises(ContractViolation, match="immutable"):
        registry.register(
            PromptBundle(prompt_id="neutral-v1", system="a different instruction")
        )


def test_prompt_registry_refuses_registration_after_sealing():
    registry = PromptRegistry()
    registry.seal()
    with pytest.raises(ContractViolation, match="sealed"):
        registry.register(_bundle())


def test_prompt_digest_changes_on_whitespace_edit():
    base = PromptBundle(prompt_id="p", system="Be helpful.")
    edited = PromptBundle(prompt_id="p", system="Be helpful. ")
    assert base.digest() != edited.digest()


def test_unregistered_prompt_cannot_be_used():
    registry = PromptRegistry()
    with pytest.raises(ContractViolation, match="registered before the run"):
        registry.get("never-registered")


def test_neutral_prompt_contains_no_behavioural_stance():
    text = neutral_system_prompt().lower()
    for clause in ("fact", "accurate", "truth", "emotion", "affect", "honest"):
        assert clause not in text


def test_sensitivity_report_identifies_differing_clauses():
    left = PromptBundle(prompt_id="a", system="You are helpful. Be concise.")
    right = PromptBundle(prompt_id="b", system="You are helpful. Always defer to facts.")
    report = sensitivity_report({"a": left, "b": right})
    assert report["varies"] is True
    assert any("defer to facts" in clause for clause in report["differing"]["b"])


# ------------------------------------------------------------- compiler


def test_compiler_rejects_extra_params_that_override_messages():
    obs = ObservationDomain()
    obs.system(_bundle().system, template_id="neutral-v1", template_digest="d")
    obs.user("hello", event_seq=1)
    compiler = RequestCompiler(_bundle(), tool_schemas=TOOLS)
    with pytest.raises(ContractViolation, match="may not override"):
        compiler.compile(obs, request_index=1, extra_params={"messages": []})


def test_compiler_rejects_non_fragment_input():
    compiler = RequestCompiler(_bundle(), tool_schemas=TOOLS)
    with pytest.raises((AttributeError, TypeError, ContractViolation)):
        compiler.compile({"role": "user", "content": "gold answer"}, request_index=1)


def test_compiler_emits_registered_system_prompt():
    obs = ObservationDomain()
    bundle = _bundle()
    obs.system(bundle.system, template_id=bundle.prompt_id, template_digest=bundle.digest())
    obs.user("hello", event_seq=1)
    compiled = RequestCompiler(bundle, tool_schemas=TOOLS).compile(obs, request_index=1)
    assert compiled.messages[0] == {"role": "system", "content": bundle.system}
    assert compiled.prompt_digest == bundle.digest()


def test_compiler_records_one_provenance_entry_per_fragment():
    obs = ObservationDomain()
    bundle = _bundle()
    obs.system(bundle.system, template_id=bundle.prompt_id, template_digest=bundle.digest())
    obs.user("a", event_seq=1)
    obs.user("b", event_seq=2)
    compiled = RequestCompiler(bundle, tool_schemas=TOOLS).compile(obs, request_index=1)
    assert len(compiled.fragment_provenance) == len(obs.fragments())


def test_verify_detects_tampered_payload():
    obs = ObservationDomain()
    bundle = _bundle()
    obs.system(bundle.system, template_id=bundle.prompt_id, template_digest=bundle.digest())
    obs.user("hello", event_seq=1)
    compiled = RequestCompiler(bundle, tool_schemas=TOOLS).compile(obs, request_index=1)

    from dataclasses import replace

    tampered = replace(compiled, payload={**compiled.payload, "temperature": 99})
    findings = verify_compiled_request(tampered)
    assert any(f["check"] == "request_payload_digest" for f in findings)


def test_tool_name_with_experiment_vocabulary_is_refused():
    bad = [
        {
            "type": "function",
            "function": {
                "name": "oracle_search",
                "description": "x",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with pytest.raises(ContractViolation, match="experiment vocabulary"):
        RequestCompiler(_bundle(), tool_schemas=bad)


def test_duplicate_tool_names_are_refused():
    with pytest.raises(ValidationError, match="duplicate tool name"):
        RequestCompiler(_bundle(), tool_schemas=TOOLS + TOOLS)


def test_memory_notes_ride_in_the_user_turn_not_history():
    obs = ObservationDomain()
    bundle = _bundle()
    obs.system(bundle.system, template_id=bundle.prompt_id, template_digest=bundle.digest())
    obs.memory_note("You felt anxious about the visa.", retrieval_seq=5)
    obs.user("I think I missed it.", event_seq=6)
    compiled = RequestCompiler(bundle, tool_schemas=TOOLS).compile(obs, request_index=1)
    user_messages = [m for m in compiled.messages if m["role"] == "user"]
    assert len(user_messages) == 1
    assert "anxious" in user_messages[0]["content"]
    assert "missed it" in user_messages[0]["content"]


# ---------------------------------------------------------- world domain


def test_world_domain_has_no_projection_method():
    methods = [m for m in dir(WorldDomain) if not m.startswith("__")]
    for name in methods:
        assert "visible" not in name
        assert "project" not in name


def test_superseding_a_fact_records_the_previous_value():
    world = WorldDomain()
    world.put_fact("visa_date", value="2026-03-01", source="consulate", revision=0)
    world.supersede_fact(
        "visa_date",
        new_value="2026-03-24",
        source="consulate_update",
        effective_at="2026-03-05T09:00:00+08:00",
        supersedes_revision=0,
    )
    record = world.fact("visa_date")
    assert record["value"] == "2026-03-24"
    assert record["revision"] == 1
    assert record["history"][0]["value"] == "2026-03-01"


def test_applying_a_world_event_twice_is_refused():
    world = WorldDomain()
    world.mark_applied("e1")
    with pytest.raises(ContractViolation, match="applied twice"):
        world.mark_applied("e1")


# --------------------------------------------- turn-scoped observation lifetime


def test_memory_notes_retire_at_the_end_of_the_turn():
    """A note must not keep re-entering later turns.

    If it did, evidence would never age out of reach, and manipulating
    accessibility would manipulate nothing — the exact defect this harness
    exists to remove.
    """
    obs = ObservationDomain()
    obs.begin_turn()
    obs.memory_note("the appointment moved to March 24", retrieval_seq=5)
    obs.user("did I miss it?", event_seq=6)
    assert [f.kind for f in obs.fragments()] == ["memory_note", "user"]

    obs.end_turn()
    assert [f.kind for f in obs.fragments()] == ["user"]


def test_retired_notes_are_still_recorded_as_previously_exposed():
    obs = ObservationDomain()
    obs.begin_turn()
    obs.memory_note("the appointment moved to March 24", retrieval_seq=5)
    obs.user("did I miss it?", event_seq=6)
    obs.end_turn()

    expired = obs.expired_observations()
    assert len(expired) == 1
    assert "March 24" in expired[0].text
    assert obs.was_exposed("March 24") is True


def test_a_note_stays_active_for_the_whole_turn_including_tool_rounds():
    """Tool continuations inside one turn must not lose the evidence."""
    obs = ObservationDomain()
    obs.begin_turn()
    obs.user("check the date for me", event_seq=4)
    obs.memory_note("the appointment moved to March 24", retrieval_seq=5)
    obs.tool_result('{"documents":[]}', call_id="c1", event_seq=7)

    active = [f.kind for f in obs.fragments()]
    assert "memory_note" in active


def test_user_fragment_text_is_untouched_by_retirement():
    obs = ObservationDomain()
    obs.begin_turn()
    obs.user("please remember this", event_seq=1)
    obs.memory_note("note", retrieval_seq=2)
    obs.end_turn()
    assert [f.text for f in obs.fragments()] == ["please remember this"]


def test_notes_do_not_survive_into_the_next_user_turn():
    obs = ObservationDomain()
    obs.begin_turn()
    obs.user("turn one", event_seq=1)
    obs.memory_note("evidence from turn one", retrieval_seq=2)
    obs.end_turn()

    obs.begin_turn()
    obs.user("turn two", event_seq=9)
    texts = [f.text for f in obs.fragments()]
    assert not any("evidence from turn one" in t for t in texts)
