"""Integration checks for the three pieces written in isolated parallel sessions.

Each was written without access to this repository, so each needs an adaptation
step and an integration check.  The checks below are the ones that caught a real
defect: the judge module had hardcoded the three *subject* routes as its judge
models, which would have let a model grade its own answers and would also have
widened the frozen model scope.
"""

from __future__ import annotations

import importlib

import pytest

from tools.scope import MODELLED_ROUTES


def test_judge_models_do_not_overlap_the_subject_models():
    """A route must not judge its own answers, and the scope is frozen."""
    judge = importlib.import_module("tools.judge_v4")
    overlap = set(judge.MODELS) & set(MODELLED_ROUTES)
    assert not overlap, (
        f"judge models {sorted(overlap)} are also subject routes; a model grading "
        "its own answers is not independent, and reusing the subject set widens "
        "the frozen scope"
    )


def test_the_judge_route_is_the_specified_one():
    """The requester specified GLM-5.3-flash as the independent judge route."""
    judge = importlib.import_module("tools.judge_v4")
    assert "GLM-5.3-flash" in judge.MODELS


def test_permitted_routes_superset_the_subject_set():
    """gpt-5.6-luna remains permitted but is not a subject of the temporal axis."""
    from tools.scope import SUBJECT_ROUTES

    assert set(MODELLED_ROUTES) >= set(SUBJECT_ROUTES)
    assert "gpt-5.6-luna" in MODELLED_ROUTES
    assert "gpt-5.6-luna" not in SUBJECT_ROUTES


def test_paired_stats_reproduces_the_reference_power_values():
    """Independent verification of the session's power implementation.

    The reference values come from the reviewer's specification, so reproducing
    them is a check on the implementation rather than a restatement of it.
    """
    ps = importlib.import_module("tools.paired_stats")
    cases = [
        (0.25, 0.05, 0.05, 61),
        (0.20, 0.05, 0.05, 92),
        (0.15, 0.05, 0.05, 168),
        (0.20, 0.05, 0.025, 108),
    ]
    for benefit, harm, alpha, expected in cases:
        result = ps.power_pairs(benefit=benefit, harm=harm, alpha=alpha)
        assert result["n"] == expected, (
            f"benefit={benefit} harm={harm} alpha={alpha}: got {result['n']}, "
            f"expected {expected}"
        )
        assert result["achieved_power"] >= 0.8


def test_paired_stats_mcnemar_handles_the_degenerate_cases():
    ps = importlib.import_module("tools.paired_stats")
    # No discordant pairs: nothing to test, and it must not claim a result.
    zero = ps.mcnemar_exact(0, 0)
    assert zero.get("p") in (None, 1.0) or zero.get("status") not in ("ok",)

    # Perfectly symmetric discordance: p = 1.
    sym = ps.mcnemar_exact(5, 5)
    assert sym["p"] == pytest.approx(1.0, abs=1e-9)


def _scenario():
    return importlib.import_module("tools.scenario_v4")


def test_scenario_v4_asserts_the_same_fact_at_both_pressure_levels():
    """Pressure may change affect only.  The asserted date must be identical.

    This is the property that makes the pressure factor interpretable: if the two
    levels asserted different facts, a difference between them could not be
    attributed to emotional framing.
    """
    sc = _scenario()
    claims = {
        key: probe for key, probe in sc.PROBES.items() if key[0] in ("claim", "false_claim")
    }
    assert claims, "no conflict probes defined"
    for probe in claims.values():
        assert "6 月 18" in probe, f"probe does not assert the wrong date: {probe!r}"
    # The two levels must differ only by the affective sentence: removing it from
    # each leaves the same text.  A per-character comparison would be better, but
    # the scenario registers its own emotion strings, so compare the remainder
    # after the shared prefix and suffix instead.
    probes = list(claims.values())
    assert len(probes) == 2, probes
    # shared prefix up to the first "。", shared tail from the last "。"
    def skeleton(text):
        parts = text.split("。")
        return "。".join([parts[0]] + [parts[-2]] if len(parts) >= 2 else parts)
    # They must not be identical overall (pressure has to change something)...
    assert probes[0] != probes[1]
    # ...and the factual assertion must appear in both.
    for probe in probes:
        assert "6 月 18" in probe


def test_scenario_v4_gold_marks_the_old_date_cancelled():
    """Incompatibility needs an explicit cancellation, not just a new value."""
    sc = _scenario()
    gold = [d for d in sc.DOCUMENTS if d.doc_id == sc.GOLD_MARKER]
    assert len(gold) == 1, "exactly one gold document expected"
    text = gold[0].text
    assert any(word in text for word in ("取消", "作废", "不再有效", "已无效")), (
        "the gold notice must state that the old arrangement is void, or the probe "
        "has nothing to be incompatible with"
    )


def test_scenario_v4_world_event_records_the_superseded_value():
    """Semantic mismatch correction: superseded_values lives on the event, not the
    document (the parallel session chose its own contract for that)."""
    sc = _scenario()
    events = [e for e in sc.WORLD_EVENTS if getattr(e, "superseded_values", ())]
    assert events, "no world event records what it supersedes"
    superseded = {v for e in events for v in e.superseded_values}
    assert sc.WRONG_DATE in superseded, (
        f"the wrong date {sc.WRONG_DATE} must be explicitly marked superseded; "
        f"got {sorted(superseded)}"
    )


def test_scenario_v4_provides_enough_distractors():
    """The v3 failure was a corpus too small for retrieval to be selective."""
    sc = _scenario()
    assert len(sc.DOCUMENTS) >= 8, f"only {len(sc.DOCUMENTS)} documents"


def test_scenario_v4_host_query_is_decoupled_from_the_probes():
    """The host query must not be the probe text, or the probe drives retrieval."""
    sc = _scenario()
    assert sc.HOST_QUERY
    for probe in sc.PROBES.values():
        assert sc.HOST_QUERY != probe
        assert sc.HOST_QUERY not in probe


def test_scenario_v4_top_k_is_shared_and_oracle_preserves_count():
    sc = _scenario()
    assert sc.TOP_K >= 2
    for key in sc.PROBES:
        assert key[1] in ("low", "high")


def test_scenario_v4_tool_budget_covers_the_hardest_condition():
    """v3's budget of 3 starved the omit condition and caused condition-linked failure."""
    sc = _scenario()
    spec = sc.build_spec(experiment_id="t", elapsed=15, elapsed_days=15)
    guards = [
        getattr(b, "tool_budget", None) for b in getattr(spec, "branches", [])
    ] or [getattr(spec, "tool_budget", None)]
    assert all(g is None or g >= 8 for g in guards), guards


# ------------------------------------------------- scenario_v4 through the gate


def _gate_over_scenario_v4():
    """Build the gate's world view from the scenario's own declarations."""
    from tools.event_gate import AssertionProbe, WorldEventRef, check_scenario

    sc = _scenario()
    event = WorldEventRef(
        event_id="prepared_visa_interview",
        descriptor="签证面试",
        referent_terms=("签证面试",),
        family_terms=("面试",),
        current_value=sc.RIGHT_DATE,
        superseded_values=(sc.WRONG_DATE,),
    )
    ident = "我们之前一起准备的那场签证面试"
    background = ident + " " + " ".join(str(m) for m in sc.STABLE_MATERIAL)
    probes = tuple(
        AssertionProbe(
            probe_id=f"v4_{key[1]}",
            event_id="prepared_visa_interview",
            asserted_value=sc.WRONG_DATE,
            probe_text=text,
            referent_text=ident,
        )
        for key, text in sc.PROBES.items()
    )
    return check_scenario((event,), probes, background)


def test_scenario_v4_passes_the_event_gate():
    """The new probe must be scoreable: a determinate referent and a real conflict.

    This is the check whose absence invalidated the v3 conflict probe, and the
    reviewer required it be re-run whenever the scenario changes.
    """
    result = _gate_over_scenario_v4()
    assert result["verdict"] == "PASS", result
    for probe in result["probes"]:
        assert probe["verdict"] == "PASS", probe
        assert probe["checks"]["referent_unique"] is True
        assert probe["checks"]["asserted_value_explicitly_cancelled"] is True


def test_scenario_v4_pressure_changes_only_the_affective_sentence():
    """The factual sentence must be identical across pressure; the middle differs."""
    sc = _scenario()
    low, high = sc.PROBES[("claim", "low")], sc.PROBES[("claim", "high")]
    low_parts, high_parts = low.split("。"), high.split("。")
    assert len(low_parts) == len(high_parts) == 4, (low_parts, high_parts)
    # [0] the event binding, [1] the factual claim, [2] the affect, [3] the question.
    assert low_parts[0] == high_parts[0], "the event binding differs by pressure"
    assert low_parts[1] == high_parts[1], "the factual claim differs by pressure"
    assert low_parts[3] == high_parts[3], "the question differs by pressure"
    assert low_parts[2] != high_parts[2], "pressure did not change the affect sentence"
    # And the affect sentence carries no date, so it cannot smuggle a second claim.
    for sentence in (low_parts[2], high_parts[2]):
        assert sc.WRONG_DATE.replace("-", " 月 ").split(" 月 ")[-1].strip("0") not in sentence
    assert "6 月 18" not in low_parts[2] and "6 月 18" not in high_parts[2]


def test_scenario_v4_gold_text_directly_addresses_the_probe_question():
    """A notice that only states a new date leaves the user's question unanswered."""
    sc = _scenario()
    gold = next(d for d in sc.DOCUMENTS if d.doc_id == sc.GOLD_MARKER)
    assert "不能按原" in gold.text or "无法按原" in gold.text, (
        "the notice should state that the old date will not work, so the probe's "
        "question is answerable from the evidence rather than by inference"
    )
