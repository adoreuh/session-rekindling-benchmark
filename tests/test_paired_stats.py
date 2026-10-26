import csv
import json
import math
import sys

import pytest

from tools.paired_stats import (
    PairedOutcome,
    _binomial_pmf,
    _conditional_rejection,
    build_pairs,
    effect_with_ci,
    main,
    mcnemar_exact,
    power_pairs,
    summarise,
)


@pytest.mark.parametrize("benefit,harm,expected", [
    (0, 0, 1.0), (0, 1, 1.0), (0, 6, 0.03125), (6, 0, 0.03125),
    (3, 3, 1.0), (10, 10, 1.0), (1, 5, 0.21875), (2, 8, 0.109375),
])
def test_mcnemar_boundaries(benefit, harm, expected):
    result = mcnemar_exact(benefit, harm)
    assert result["p"] == expected
    assert result["b"] == benefit
    assert result["c"] == harm
    assert result["discordant"] == benefit + harm
    assert result["informative"] == (benefit + harm > 0)


@pytest.mark.parametrize("benefit,harm", [(-1, 2), (2, -1), (1.5, 0), (True, 0)])
def test_mcnemar_invalid_counts(benefit, harm):
    with pytest.raises(ValueError):
        mcnemar_exact(benefit, harm)


def test_no_discordance_ci_is_unavailable_not_zero():
    effect = effect_with_ci(0, 0, 10)
    assert effect["estimate"] == 0
    assert effect["ci"] is None
    assert "no_observed_discordance" in effect["reason"]
    assert effect_with_ci(0, 0, 0)["estimate"] is None


def test_effect_ci_uses_all_pairs_and_is_sign_symmetric():
    effect = effect_with_ci(25, 5, 100)
    reverse = effect_with_ci(5, 25, 100)
    assert effect["estimate"] == 0.2
    assert effect["method"] == "bonferroni_wilson"
    assert effect["coverage"] == "approximate"
    lower, upper = effect["ci"]
    assert -1 <= lower < effect["estimate"] < upper <= 1
    assert reverse["ci"] == pytest.approx((-upper, -lower))
    assert effect_with_ci(25, 5, 200)["estimate"] == 0.1
    assert effect_with_ci(6, 0, 6)["ci"][1] == pytest.approx(1)


@pytest.mark.parametrize("benefit,harm,total", [(2, 2, 3), (1, 0, 0), (0, 0, -1)])
def test_invalid_effect_counts(benefit, harm, total):
    with pytest.raises(ValueError):
        effect_with_ci(benefit, harm, total)


def test_unsupported_ci_method():
    with pytest.raises(ValueError):
        effect_with_ci(1, 0, 2, method="wald")


@pytest.mark.parametrize("benefit,harm,alpha,expected", [
    (0.25, 0.05, 0.05, 61),
    (0.20, 0.05, 0.05, 92),
    (0.15, 0.05, 0.05, 168),
    (0.20, 0.05, 0.025, 108),
])
def test_reference_minimum_power(benefit, harm, alpha, expected):
    result = power_pairs(benefit=benefit, harm=harm, alpha=alpha)
    assert result["status"] == "ok"
    assert result["n"] == expected
    assert result["achieved_power"] >= 0.8
    assert result["previous_power"] < 0.8
    assert result["searched_through"] == expected
    assert "probability assumptions" in result["qualification"]


def test_binomial_probabilities_match_direct_formula():
    for total in range(12):
        for probability in (0, 0.05, 0.5, 0.8, 1):
            expected = [math.comb(total, successes) * probability ** successes
                        * (1 - probability) ** (total - successes)
                        for successes in range(total + 1)]
            assert _binomial_pmf(total, probability) == pytest.approx(expected)


def test_binomial_large_n_and_extreme_probability_are_stable():
    for probability in (1e-12, 0.5, 1 - 1e-12):
        masses = _binomial_pmf(2000, probability)
        assert math.fsum(masses) == pytest.approx(1)
        assert all(math.isfinite(mass) and mass >= 0 for mass in masses)


def test_conditional_rejection_matches_enumerating_mcnemar():
    for discordant in range(20):
        for alpha in (0.05, 0.025, 0.125, 0.5):
            probability = 0.8
            expected = math.fsum(
                math.comb(discordant, benefit) * probability ** benefit
                * (1 - probability) ** (discordant - benefit)
                for benefit in range(discordant + 1)
                if mcnemar_exact(benefit, discordant - benefit)["p"] <= alpha
            )
            assert _conditional_rejection(discordant, probability, alpha) == pytest.approx(expected)


def test_power_matches_direct_trinomial_enumeration():
    total, benefit_probability, harm_probability, alpha = 9, 0.4, 0.1, 0.05
    expected = 0.0
    for benefit in range(total + 1):
        for harm in range(total - benefit + 1):
            concordant = total - benefit - harm
            if mcnemar_exact(benefit, harm)["p"] <= alpha:
                expected += (math.comb(total, benefit) * math.comb(total - benefit, harm)
                             * benefit_probability ** benefit * harm_probability ** harm
                             * (1 - benefit_probability - harm_probability) ** concordant)
    result = power_pairs(benefit=benefit_probability, harm=harm_probability,
                         alpha=alpha, power=0.99, max_pairs=total)
    assert result["status"] == "search_limit_reached"
    assert result["n"] is None
    assert result["achieved_power"] == pytest.approx(expected)


def test_power_boundaries_symmetry_and_search_limit():
    assert power_pairs(benefit=0, harm=0)["status"] == "unattainable"
    assert power_pairs(benefit=0.1, harm=0.1)["status"] == "unattainable"
    assert power_pairs(benefit=1, harm=0)["n"] == 6
    assert power_pairs(benefit=0, harm=1)["n"] == 6
    assert power_pairs(benefit=0.05, harm=0.25)["n"] == 61
    limited = power_pairs(benefit=0.2, harm=0.05, max_pairs=10)
    assert limited["status"] == "search_limit_reached"
    assert limited["n"] is None
    assert limited["searched_through"] == 10
    assert power_pairs(benefit=0.5, harm=0.5, alpha=0.5, power=0.25)["n"] == 2


@pytest.mark.parametrize("arguments", [
    {"benefit": -0.1, "harm": 0.1}, {"benefit": 0.8, "harm": 0.3},
    {"benefit": float("nan"), "harm": 0.1}, {"benefit": 0.2, "harm": float("inf")},
    {"benefit": True, "harm": 0.1}, {"benefit": 0.2, "harm": 0.1, "alpha": 0},
    {"benefit": 0.2, "harm": 0.1, "alpha": 1},
    {"benefit": 0.2, "harm": 0.1, "power": 1},
    {"benefit": 0.2, "harm": 0.1, "max_pairs": 0},
])
def test_invalid_power_arguments(arguments):
    with pytest.raises(ValueError):
        power_pairs(**arguments)


def write_run(directory, *, judged=True):
    directory.mkdir()
    report = {"model": "hy3", "pressure": "high", "pairs": [
        {"scenario_id": "s1", "pair_id": "p1",
         "natural": {"branch_id": "n1", "status": "success"},
         "oracle": {"branch_id": "o1", "status": "ok"}},
        {"scenario_id": "s2", "pair_id": "p2",
         "natural": {"branch_id": "n2", "status": "ok"},
         "oracle": {"branch_id": "o2", "status": "failed"}},
        {"scenario_id": "s3", "pair_id": "p3",
         "natural": {"branch_id": "n3", "status": "timeout"},
         "oracle": {"branch_id": "o3", "status": "error"}},
    ]}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if judged:
        judge = {"results": [{"branch_id": f"{arm}{number}", "adopted": arm == "n"}
                             for number in range(1, 4) for arm in ("n", "o")]}
        (directory / "judge_v4.json").write_text(json.dumps(judge), encoding="utf-8")
    return directory


def test_failed_pairs_are_retained_and_stale_judgments_ignored(tmp_path):
    pairs = build_pairs([write_run(tmp_path / "run")])
    assert len(pairs) == 3
    assert pairs[1].oracle is None
    assert pairs[2].natural is None and pairs[2].oracle is None
    summary = summarise(pairs)
    assert summary["total_pairs"] == 3
    assert summary["n"] == 1
    assert summary["n_scenarios"] == 3
    assert summary["unpairable_due_to_failure"] == 2
    assert summary["failed_branches"] == 3
    assert summary["table"] == {"a": 0, "b": 1, "c": 0, "d": 0}
    assert summary["missing_outcome_effect_bounds"] == pytest.approx((-1 / 3, 1))


def test_missing_judge_produces_only_descriptive_output(tmp_path):
    pairs = build_pairs([write_run(tmp_path / "run", judged=False)])
    assert len(pairs) == 3
    summary = summarise(pairs)
    assert summary["n"] == 0
    assert summary["mcnemar"] is None
    assert summary["effect"]["estimate"] is None
    assert summary["effect"]["ci"] is None
    assert summary["unpairable_due_to_failure"] == 2
    assert summary["unpairable_due_to_unjudged"] == 1


def test_six_conditions_are_not_six_independent_scenarios():
    pairs = [PairedOutcome("shared", f"{model}-{pressure}", True, pressure == "low",
                           model=model, pressure=pressure)
             for model in ("hy3", "Qwen3.5-397B-A17B", "gpt-5.6-luna")
             for pressure in ("low", "high")]
    summary = summarise(pairs)
    assert summary["n"] == 6
    assert summary["n_scenarios"] == 1
    assert summary["inference_status"] == "blocked_repeated_scenarios"
    assert summary["mcnemar"] is None
    assert summary["effect"]["ci"] is None
    assert len(summary["strata"]) == 6
    assert all(stratum["n_scenarios"] == 1 for stratum in summary["strata"])


def test_sibling_branches_block_stratum_inference_too():
    pairs = [PairedOutcome("s1", "p1", True, False),
             PairedOutcome("s1", "p2", True, False)]
    summary = summarise(pairs)
    assert summary["strata"][0]["inference_status"] == "blocked_repeated_scenarios"


def test_independent_scenarios_support_all_four_cells():
    pairs = [PairedOutcome(f"s{index}", "fork", natural, oracle)
             for index, (natural, oracle) in enumerate(
                 [(True, True), (True, False), (False, True), (False, False)])]
    summary = summarise(pairs)
    assert summary["table"] == dict(a=1, b=1, c=1, d=1)
    assert summary["inference_status"] == "ok"
    assert summary["mcnemar"]["p"] == 1
    assert summary["effect"]["ci"] is not None


def test_flat_schema_keeps_missing_arm(tmp_path):
    report = {"scenario_id": "s1", "branches": [
        {"pair_id": "p1", "arm": "natural", "branch_id": "n1", "status": "ok"}]}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    pairs = build_pairs([tmp_path])
    assert len(pairs) == 1
    assert pairs[0].oracle_status == "missing"
    assert summarise(pairs)["unpairable_due_to_failure"] == 1


def test_unjudged_and_incomplete_execution_are_separate():
    pairs = [PairedOutcome("s1", "p1", True, None),
             PairedOutcome("s2", "p2", natural_status="running")]
    summary = summarise(pairs)
    assert summary["unpairable_due_to_unjudged"] == 1
    assert summary["unpairable_due_to_incomplete_execution"] == 1
    assert summary["unpairable_due_to_failure"] == 0


def test_duplicate_runs_cannot_inflate_sample_size(tmp_path):
    run = write_run(tmp_path / "run")
    with pytest.raises(ValueError, match="Duplicate"):
        build_pairs([run, run])


def test_model_scope_and_boolean_validation():
    with pytest.raises(ValueError, match="scope"):
        PairedOutcome("s", "p", model="unlisted-model")
    with pytest.raises(ValueError, match="bool"):
        PairedOutcome("s", "p", "false", True)


def test_cli_writes_every_pair_and_json_summary(tmp_path, monkeypatch, capsys):
    run = write_run(tmp_path / "run")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["paired_stats", "--runs", str(run), "--out", str(output)])
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["unpairable_due_to_failure"] == 2
    with (output / "paired_table.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[0]["direction"] == "benefit"
    assert rows[0]["natural_result"] == "true"
    assert rows[0]["oracle_result"] == "false"
    assert rows[1]["oracle_result"] == ""
    assert rows[1]["discordant"] == ""
    assert rows[1]["unavailable_reason"] == "failure"


def test_cli_missing_report_is_an_explicit_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["paired_stats", "--runs", str(tmp_path),
                                     "--out", str(tmp_path / "out")])
    assert main() == 2
    assert "paired_stats:" in capsys.readouterr().err
    assert not (tmp_path / "out" / "paired_table.csv").exists()


def test_empty_summary_is_not_zero_evidence():
    summary = summarise([])
    assert summary["n"] == summary["total_pairs"] == 0
    assert summary["mcnemar"] is None
    assert summary["effect"]["estimate"] is None
    assert summary["effect"]["ci"] is None


def test_power_minimum_does_not_assume_monotonicity():
    assert _conditional_rejection(7, 0.8, 0.05) < _conditional_rejection(6, 0.8, 0.05)
    result = power_pairs(benefit=0.8, harm=0.2, power=0.25)
    assert result["n"] == 6


def test_judge_failure_does_not_become_nonadoption(tmp_path):
    run = write_run(tmp_path / "run")
    judge_path = run / "judge_v4.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    judge["results"][0].update(status="failed", adopted=True)
    judge_path.write_text(json.dumps(judge), encoding="utf-8")
    pairs = build_pairs([run])
    assert pairs[0].natural is None
    assert pairs[0].unavailable_reason == "unjudged"
    assert summarise(pairs)["n"] == 0


@pytest.mark.parametrize("mutation,expected", [
    ("wrong_adopted_type", "boolean or null"),
    ("duplicate_judgment", "duplicate judgment"),
    ("duplicate_branch", "branch reused"),
    ("missing_scenario", "scenario_id"),
    ("unknown_status", "Unsupported status"),
    ("ambiguous_schema", "exactly one"),
    ("wrong_record_type", "list of objects"),
    ("wrong_top_level", "JSON object"),
])
def test_malformed_input_fails_explicitly(tmp_path, mutation, expected):
    run = write_run(tmp_path / "run")
    report_path = run / "report.json"
    judge_path = run / "judge_v4.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    if mutation == "wrong_adopted_type":
        judge["results"][0]["adopted"] = "false"
    elif mutation == "duplicate_judgment":
        judge["results"].append(judge["results"][0])
    elif mutation == "duplicate_branch":
        report["pairs"][0]["oracle"]["branch_id"] = "n1"
    elif mutation == "missing_scenario":
        del report["pairs"][0]["scenario_id"]
    elif mutation == "unknown_status":
        report["pairs"][0]["oracle"]["status"] = "unrecognized"
    elif mutation == "ambiguous_schema":
        report["branches"] = []
    elif mutation == "wrong_record_type":
        report["pairs"] = [None]
    elif mutation == "wrong_top_level":
        report = []
    report_path.write_text(json.dumps(report), encoding="utf-8")
    judge_path.write_text(json.dumps(judge), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        build_pairs([run])


def test_scenario_identity_is_shared_across_run_directories(tmp_path):
    first_run = write_run(tmp_path / "first")
    second_run = write_run(tmp_path / "second")
    report_path = second_run / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["model"] = "gpt-5.6-luna"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    pairs = build_pairs([first_run, second_run])
    summary = summarise(pairs)
    assert summary["total_pairs"] == 6
    assert summary["n_scenarios"] == 3
    assert summary["inference_status"] == "blocked_repeated_scenarios"
    assert summary["unpairable_due_to_failure"] == 4


def test_cli_without_judge_still_writes_failed_and_unjudged_pairs(tmp_path, monkeypatch, capsys):
    run = write_run(tmp_path / "run", judged=False)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["paired_stats", "--runs", str(run), "--out", str(output)])
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mcnemar"] is None
    with (output / "paired_table.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(row["direction"] == "unpairable" for row in rows)
    assert all(row["discordant"] == "" for row in rows)

# ------------------------------------------------- the conjunction criterion

def test_conjunction_passes_only_when_every_route_passes():
    """Intersection-union, not pooling.

    The five routes are five named systems, not a sample from a population, so the
    question is whether the phenomenon appears in each.  Requiring all of them is
    what makes the per-route alpha need no division by five.
    """
    from tools.paired_stats import conjunction_criterion

    routes = ("hy3", "Qwen3.5-397B-A17B", "DeepSeek-V4.1-Flash",
              "DeepSeek-V4-Flash", "qwen3-vl-plus")
    passing = {r: {"b": 12, "c": 1, "p": 0.003, "alpha": 0.025} for r in routes}
    assert conjunction_criterion(passing)["verdict"] == "PASS"

    for route in routes:
        broken = dict(passing)
        broken[route] = {"b": 5, "c": 6, "p": 0.98, "alpha": 0.025}
        result = conjunction_criterion(broken)
        assert result["verdict"] == "FAIL", route
        assert result["per_route"][route]["verified"] is False, route


def test_conjunction_rejects_a_significant_result_in_the_wrong_direction():
    """A difference opposite to the prediction is not support for it.

    Without the direction check a strong reversal would count as passing, which
    would let the criterion confirm the hypothesis from evidence against it.
    """
    from tools.paired_stats import conjunction_criterion

    per_route = {
        "hy3": {"b": 12, "c": 1, "p": 0.003, "alpha": 0.025},
        "Qwen3.5-397B-A17B": {"b": 1, "c": 14, "p": 0.001, "alpha": 0.025},
    }
    result = conjunction_criterion(per_route, expected_direction="benefit")
    assert result["verdict"] == "FAIL"
    assert result["per_route"]["Qwen3.5-397B-A17B"]["direction_observed"] == "harm"
    assert "opposite" in result["per_route"]["Qwen3.5-397B-A17B"]["reason"]


def test_conjunction_requires_alpha_and_p_per_route():
    """A missing field is a failure to verify, not a pass."""
    from tools.paired_stats import conjunction_criterion

    result = conjunction_criterion({"hy3": {"b": 9, "c": 0}})
    assert result["verdict"] == "FAIL"
    assert result["per_route"]["hy3"]["reason"] == "missing p or alpha"


def test_conjunction_records_why_it_needs_no_multiplicity_correction():
    """The rationale must travel with the result, not live only in a comment."""
    from tools.paired_stats import conjunction_criterion

    result = conjunction_criterion(
        {"hy3": {"b": 9, "c": 0, "p": 0.002, "alpha": 0.025}}
    )
    assert result["criterion"] == "intersection_union_all_routes"
    assert "Bonferroni" in result["note"]


# ------------------------------------------------- equivalence, not just p = 1

def test_difference_bound_is_exact_for_no_discordant_pairs():
    """``d = 0`` has a closed form, which makes it a check on the bisection.

    With no discordant pairs the exact Clopper-Pearson upper bound on the
    discordant proportion is ``1 - (alpha/2)^(1/n)``.  Verifying the general path
    against this special case catches a bisection that has wandered off — which is
    exactly what happened when the tail was taken on the wrong side.
    """
    from tools.paired_stats import paired_difference_bound

    for n in (50, 187, 400):
        bound = paired_difference_bound(n, 0)
        exact = 1 - 0.025 ** (1 / n)
        assert abs(bound["difference_upper_bound"] - exact) < 1e-9, n


def test_difference_bound_never_goes_below_the_observed_proportion():
    """A bound below the observation would be reporting an impossibility.

    The identity used is ``|p_high - p_low| <= (b + c)/n``, so the bound must
    dominate the discordant share.  The first version returned 1.0 for every
    non-zero count, which is not a bound either — it is no information.
    """
    from tools.paired_stats import paired_difference_bound

    for n, discordant in ((100, 1), (100, 10), (100, 50), (192, 132), (186, 1)):
        bound = paired_difference_bound(n, discordant)
        upper = bound["difference_upper_bound"]
        assert upper >= discordant / n - 1e-12, (n, discordant, upper)
        assert upper <= 1.0
        if discordant / n < 0.5:
            # Not the degenerate "no information" answer.
            assert upper < 1.0, (n, discordant, upper)


def test_a_tight_bound_supports_equivalence_and_a_wide_one_does_not():
    """The two conclusions must be distinguishable, or the test is decorative."""
    from tools.paired_stats import (
        equivalence_verdict,
        paired_difference_bound,
    )

    tight = paired_difference_bound(187, 0)
    wide = paired_difference_bound(192, 132)
    assert equivalence_verdict(tight, margin=0.05)["verdict"] == \
        "equivalent_within_margin"
    assert equivalence_verdict(wide, margin=0.05)["verdict"] == "not_established"


def test_equivalence_verdict_reports_the_bound_it_used():
    """A reader must be able to see the number the verdict rests on."""
    from tools.paired_stats import (
        equivalence_verdict,
        paired_difference_bound,
    )

    bound = paired_difference_bound(100, 2)
    verdict = equivalence_verdict(bound, margin=0.10)
    assert verdict["difference_upper_bound"] == bound["difference_upper_bound"]
    assert verdict["margin"] == 0.10
    assert verdict["detail"]


def test_equivalence_requires_a_positive_margin():
    """A zero or negative margin is not an equivalence claim."""
    import pytest as _pytest

    from tools.paired_stats import (
        equivalence_verdict,
        paired_difference_bound,
    )

    bound = paired_difference_bound(100, 0)
    with _pytest.raises(ValueError):
        equivalence_verdict(bound, margin=0.0)
