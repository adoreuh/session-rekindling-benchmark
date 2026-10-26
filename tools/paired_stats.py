"""Paired adoption statistics, using only the Python 3.11 standard library.

Input adapter contract (no imports from the experiment harness):

report.json::

    {"scenario_id": "visa_01", "model": "hy3", "pressure": "high",
     "pairs": [{"pair_id": "fork_01",
                "natural": {"branch_id": "n", "status": "ok"},
                "oracle": {"branch_id": "o", "status": "ok"}}]}

judge_v4.json (optional)::

    {"results": [{"branch_id": "n", "adopted": true},
                 {"branch_id": "o", "adopted": false}]}

``adopted`` means adoption of the user's WRONG date, not evidence adoption.
It must be a JSON boolean or null; absent/null means unjudged. Judge records
with a non-success status cannot supply an outcome. Report execution failures
override any judgment, so stale judgments cannot turn failures into successes.

Pairs may override scenario_id/model/pressure. Alternatively, report.json may
contain ``branches`` instead of ``pairs``: flat branch objects with pair_id,
arm (natural/oracle), branch_id, status, and optional metadata overrides.
Branch IDs must be unique within each run. Missing arms remain in the output.
An entirely absent pair cannot be recovered: the report must enumerate planned
pairs, including failed pairs. Other schemas must be adapted at this boundary.

Only one pair per scenario supports inference in a given analysis. Repeated
scenarios block pooled inference, even across run directories or model/pressure
conditions. Stratified results are separate comparisons, NOT independent
replications. No automatic multiple-comparison correction is applied.

The effect is P(benefit) - P(harm). Its approximate 95% CI subtracts two
Bonferroni-adjusted 97.5% Wilson intervals for b/n and c/n. This intentionally
avoids assuming these multinomial categories are independent, or fixing the
observed discordance rate as if it were known. Wilson coverage is approximate;
this is not an exact 95% interval. With no observed discordance the interval
is withheld by reporting convention, rather than fabricated as [0, 0].

CLI: python tools/paired_stats.py --runs RUN [RUN ...] --out OUTPUT
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping


ALLOWED_MODELS = frozenset({"hy3", "Qwen3.5-397B-A17B", "gpt-5.6-luna"})
SUCCESS_STATUSES = frozenset({"ok", "success", "completed"})
FAILURE_STATUSES = frozenset(
    {"failed", "error", "timeout", "cancelled", "canceled", "missing"}
)
KNOWN_STATUSES = SUCCESS_STATUSES | FAILURE_STATUSES | {"pending", "running"}


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _probability(value: float, name: str, *, interior: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability")
    if interior and not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return value


def _text(value: object, name: str, *, optional: bool = False) -> str:
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if optional else ''}string")
    return value


def _status(value: object) -> str:
    status = _text(value, "status").lower()
    if status not in KNOWN_STATUSES:
        raise ValueError(f"Unsupported status: {value!r}")
    return status


@dataclass(frozen=True)
class PairedOutcome:
    scenario: str
    pair: str
    natural: bool | None = None
    oracle: bool | None = None
    model: str = ""
    pressure: str = ""
    natural_status: str = "ok"
    oracle_status: str = "ok"
    run_dir: str = ""
    natural_branch_id: str = ""
    oracle_branch_id: str = ""

    def __post_init__(self) -> None:
        _text(self.scenario, "scenario")
        _text(self.pair, "pair")
        _text(self.model, "model", optional=True)
        _text(self.pressure, "pressure", optional=True)
        if self.model and self.model not in ALLOWED_MODELS:
            raise ValueError(f"Model outside the fixed scope: {self.model!r}")
        for arm in ("natural", "oracle"):
            outcome = getattr(self, arm)
            if outcome is not None and not isinstance(outcome, bool):
                raise ValueError(f"{arm} must be bool or None")
            status = _status(getattr(self, f"{arm}_status"))
            object.__setattr__(self, f"{arm}_status", status)

    @property
    def failed(self) -> bool:
        return any(status in FAILURE_STATUSES for status in
                   (self.natural_status, self.oracle_status))

    @property
    def complete(self) -> bool:
        return (self.natural_status in SUCCESS_STATUSES
                and self.oracle_status in SUCCESS_STATUSES
                and self.natural is not None and self.oracle is not None)

    @property
    def unavailable_reason(self) -> str | None:
        if self.failed:
            return "failure"
        if any(status not in SUCCESS_STATUSES for status in
               (self.natural_status, self.oracle_status)):
            return "incomplete_execution"
        return None if self.complete else "unjudged"

    @property
    def direction(self) -> str:
        if not self.complete:
            return "unpairable"
        if self.natural and not self.oracle:
            return "benefit"
        if self.oracle and not self.natural:
            return "harm"
        return "both_adopt" if self.natural else "neither_adopts"


def mcnemar_exact(b: int, c: int) -> dict:
    """Two-sided conditional binomial test, including both tails (not mid-p)."""
    _count(b, "b")
    _count(c, "c")
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, successes)
                   for successes in range(min(b, c) + 1))
        p_value = min(1.0, (2 * tail) / (1 << discordant))
    return {"b": b, "c": c, "discordant": discordant, "p": p_value,
            "method": "two_sided_exact_conditional_mcnemar",
            "informative": discordant > 0}


def _wilson(successes: int, total: int, z_score: float) -> tuple[float, float]:
    proportion = successes / total
    adjustment = z_score * z_score / total
    center = (proportion + adjustment / 2) / (1 + adjustment)
    radius = z_score * math.sqrt(
        proportion * (1 - proportion) / total
        + z_score * z_score / (4 * total * total)
    ) / (1 + adjustment)
    return max(0.0, center - radius), min(1.0, center + radius)


def effect_with_ci(b: int, c: int, n: int, *, method: str = "wilson") -> dict:
    """Use the full paired multinomial denominator, not only discordant pairs."""
    for name, value in (("b", b), ("c", c), ("n", n)):
        _count(value, name)
    if b + c > n:
        raise ValueError("b + c cannot exceed n")
    if method != "wilson":
        raise ValueError("Only method='wilson' is supported")
    result = {
        "estimate": (b - c) / n if n else None,
        "n": n, "ci": None, "confidence": 0.95,
        "method": "bonferroni_wilson", "coverage": "approximate",
        "reason": None,
        "interpretation": "positive means a net reduction in wrong-date adoption",
    }
    if n == 0:
        result["reason"] = "no_complete_pairs"
    elif b + c == 0:
        result["reason"] = "no_observed_discordance; interval not estimable by reporting convention"
    else:
        z_score = NormalDist().inv_cdf(1 - (1 - 0.95) / 4)
        benefit_lower, benefit_upper = _wilson(b, n, z_score)
        harm_lower, harm_upper = _wilson(c, n, z_score)
        result["ci"] = (max(-1.0, benefit_lower - harm_upper),
                        min(1.0, benefit_upper - harm_lower))
    return result


def _binomial_pmf(total: int, probability: float) -> list[float]:
    """Normalize mode-relative binomial masses to avoid extreme-tail underflow."""
    masses = [0.0] * (total + 1)
    if probability == 0:
        masses[0] = 1.0
        return masses
    if probability == 1:
        masses[total] = 1.0
        return masses
    mode = min(total, int((total + 1) * probability))
    masses[mode] = 1.0
    for successes in range(mode, 0, -1):
        masses[successes - 1] = (masses[successes] * successes
                                / (total - successes + 1)
                                * (1 - probability) / probability)
    for successes in range(mode, total):
        masses[successes + 1] = (masses[successes] * (total - successes)
                                / (successes + 1)
                                * probability / (1 - probability))
    normalization = math.fsum(masses)
    return [mass / normalization for mass in masses]


def _conditional_rejection(discordant: int, probability: float, alpha: float) -> float:
    if discordant == 0:
        return 0.0
    numerator, denominator = alpha.as_integer_ratio()
    threshold = numerator * (1 << discordant)
    tail = 0
    cutoff = -1
    coefficient = math.comb(discordant, 0)
    for successes in range((discordant - 1) // 2 + 1):
        tail += coefficient
        if 2 * tail * denominator > threshold:
            break
        cutoff = successes
        coefficient = coefficient * (discordant - successes) // (successes + 1)
    if cutoff < 0:
        return 0.0
    masses = _binomial_pmf(discordant, probability)
    return min(1.0, math.fsum(masses[:cutoff + 1])
               + math.fsum(masses[discordant - cutoff:]))


def power_pairs(*, benefit: float, harm: float, alpha: float = 0.05,
                power: float = 0.8, max_pairs: int = 10000) -> dict:
    """Find the first n attaining unconditional power of the conditional test.

    Sum over D ~ Binomial(n, benefit+harm), then both exact rejection tails of
    B | D ~ Binomial(D, benefit/(benefit+harm)). Enumeration uses floating-point
    binomial probabilities, never a normal approximation or simulation.
    Sequential search is necessary because discrete-test power need not be
    monotone. A finite search limit is disclosed, never called impossibility.
    """
    benefit = _probability(benefit, "benefit")
    harm = _probability(harm, "harm")
    alpha = _probability(alpha, "alpha", interior=True)
    power = _probability(power, "power", interior=True)
    _count(max_pairs, "max_pairs")
    if max_pairs == 0:
        raise ValueError("max_pairs must be positive")
    discordance_probability = benefit + harm
    if discordance_probability > 1:
        raise ValueError("benefit + harm cannot exceed one")
    result = {
        "n": None, "achieved_power": None, "previous_power": None,
        "benefit": benefit, "harm": harm, "alpha": alpha, "target_power": power,
        "max_pairs": max_pairs, "searched_through": 0,
        "status": "search_limit_reached",
        "test": "two_sided_exact_conditional_mcnemar",
        "qualification": "Minimum independent scenario pairs, conditional on the selected "
                         "test and these probability assumptions; no failure inflation.",
    }
    if discordance_probability == 0 or (benefit == harm and power > alpha):
        result["status"] = "unattainable"
        result["reason"] = ("no_discordance" if discordance_probability == 0
                            else "null_rejection_probability_cannot_exceed_alpha")
        return result
    conditional = [0.0]
    previous = 0.0
    for total in range(1, max_pairs + 1):
        conditional.append(_conditional_rejection(
            total, benefit / discordance_probability, alpha))
        discordance_masses = _binomial_pmf(total, discordance_probability)
        achieved = min(1.0, math.fsum(mass * rejection for mass, rejection in
                                    zip(discordance_masses, conditional)))
        result.update(searched_through=total, achieved_power=achieved,
                      previous_power=previous)
        if achieved >= power:
            result.update(n=total, status="ok")
            return result
        previous = achieved
    return result


# ---------------------------------------------------------------------------
# The conjunction criterion, for the case of several pre-specified subjects
# ---------------------------------------------------------------------------

def conjunction_criterion(per_route: Mapping, *,
                          expected_direction: str = "benefit") -> dict:
    """Combine per-route paired tests by requiring every route to pass.

    Why a conjunction rather than a pooled test, and why no extra correction.

    The five routes are **not** a random sample from a population of models; they
    are five named systems the study is about.  Pooling them would answer "what is
    the average effect across these five", which is a different and less
    interpretable question — and it would treat one scenario's outcome on five
    models as five independent observations, which it is not.

    Requiring all five to pass in the pre-specified direction answers the question
    actually being asked: does the phenomenon appear in each of these systems.  It
    is an intersection-union test, and that structure is why the per-route alpha
    needs no Bonferroni division by five: for the conjunction to be falsely
    accepted, *every* route must be falsely accepted, so the conjunction's size is
    bounded by the smallest of theirs rather than by their sum.

    The reverse test — "at least one route shows it" — is a different question and
    would need multiplicity control.  It is not implemented, because implementing it
    here would make the choice of question a matter of which argument is passed.
    """
    if not per_route:
        raise ValueError("conjunction criterion needs at least one route")
    if expected_direction not in ("benefit", "harm"):
        raise ValueError(
            f"expected_direction must be 'benefit' or 'harm', got "
            f"{expected_direction!r}; 'benefit' means the first condition of the "
            "pair is the one expected to change the outcome"
        )

    details = {}
    passed = True
    for route, entry in per_route.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"route {route!r}: expected a mapping of test results")
        b = entry.get("b", 0)
        c = entry.get("c", 0)
        p_value = entry.get("p")
        alpha = entry.get("alpha")
        if p_value is None or alpha is None:
            details[route] = {"verified": False, "reason": "missing p or alpha"}
            passed = False
            continue
        # The direction must match what was predicted: a significant difference in
        # the opposite direction is not support for the hypothesis.
        direction_ok = (b > c) if expected_direction == "benefit" else (c > b)
        ok = bool(p_value < alpha) and direction_ok
        details[route] = {
            "verified": ok,
            "b": b, "c": c, "p": p_value, "alpha": alpha,
            "direction_expected": expected_direction,
            "direction_observed": "benefit" if b > c else ("harm" if c > b else "tie"),
            "reason": ("" if ok else
                       "not significant" if p_value >= alpha else
                       "significant in the opposite direction"),
        }
        passed = passed and ok

    return {
        "criterion": "intersection_union_all_routes",
        "routes": len(per_route),
        "expected_direction": expected_direction,
        "verdict": "PASS" if passed else "FAIL",
        "per_route": details,
        "note": (
            "No Bonferroni division by the number of routes: for the conjunction to "
            "be falsely accepted every route must be falsely accepted, so its size "
            "is bounded by the smallest per-route alpha. The reverse disjunction "
            "would need multiplicity control and is a different question."
        ),
    }


# ---------------------------------------------------------------------------
# Equivalence: bounding a difference, rather than failing to detect one
# ---------------------------------------------------------------------------

def paired_difference_bound(n: int, discordant: int, *,
                            confidence: float = 0.95) -> dict:
    """A conservative confidence bound on a paired difference of proportions.

    Why this is needed for a "no effect" claim.  Reporting ``p = 1.0`` on a paired
    test says the data are compatible with no difference; it does **not** say the
    difference is small.  With few discordant pairs the test has little power, so
    "not significant" and "no effect" are different statements and only the second
    is worth publishing about a boater channel.  Establishing the second requires an
    upper bound on the difference — an equivalence bound — with a margin set in
    advance.

    The identity that makes this exact and cheap.  For paired binary outcomes with
    ``b`` discordant in one direction and ``c`` in the other::

        p_high - p_low = (b - c) / n          and        |b - c| <= b + c

    so ``|p_high - p_low| <= (b + c) / n``, the discordant proportion.  A
    Clopper-Pearson upper bound on that proportion therefore bounds the difference,
    conservatively, from a count of discordant pairs alone.  No simulation and no
    distributional assumption beyond the pairing.

    Returns the bound and, when ``margin`` is supplied by the caller, whether the
    difference is inside it.
    """
    _count(n, "n")
    _count(discordant, "discordant")
    if discordant > n:
        raise ValueError("discordant cannot exceed n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    tail = (1.0 - confidence) / 2.0
    if discordant == 0:
        # No discordant pairs: the exact upper bound is 1 - tail^(1/n).
        upper = 1.0 - math.exp(math.log(tail) / n) if n else 1.0
    else:
        # Bisection on the binomial tail, to avoid depending on an inverse
        # incomplete beta function that the standard library does not provide.
        # Clopper-Pearson upper bound: the largest p whose *lower* tail still
        # reaches the required mass.  The upper tail is the wrong side here — at
        # p = 1 it is identically 1, so a bisection on it walks straight to 1 and
        # reports a useless bound (which is what the first version did, for every
        # non-zero discordant count).
        def lower_tail(p_value: float) -> float:
            return math.fsum(
                math.comb(n, k) * (p_value ** k) * ((1 - p_value) ** (n - k))
                for k in range(0, discordant + 1)
            )

        low, high = discordant / n, 1.0
        for _ in range(200):
            middle = (low + high) / 2.0
            if lower_tail(middle) > tail:
                low = middle
            else:
                high = middle
        upper = high
    return {
        "n": n,
        "discordant": discordant,
        "confidence": confidence,
        "difference_upper_bound": upper,
        "method": "clopper_pearson_on_discordant_proportion",
        "note": (
            "|p_high - p_low| <= (b + c) / n exactly, so this bound applies to the "
            "paired difference without further assumptions. It is conservative: the "
            "true difference may be much smaller."
        ),
    }


def equivalence_verdict(bound: dict, *, margin: float) -> dict:
    """Whether a bounded difference falls inside a pre-specified margin.

    ``margin`` is the largest difference that would still count as "no meaningful
    change".  It must be chosen before the data are read; a margin picked to fit the
    result is not a test.

    The three outcomes are kept apart because they mean different things: inside the
    margin is a supported equivalence claim, outside it is simply "not established",
    and a bound too wide to decide is a power problem rather than a null result.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    upper = bound["difference_upper_bound"]
    if upper <= margin:
        verdict = "equivalent_within_margin"
        detail = (
            f"the difference is bounded at {upper:.4f}, inside the pre-specified "
            f"margin {margin}; a change larger than the margin is excluded"
        )
    else:
        verdict = "not_established"
        detail = (
            f"the bound {upper:.4f} exceeds the margin {margin}; the data neither "
            "show a difference nor exclude one larger than the margin"
        )
    return {
        "verdict": verdict,
        "margin": margin,
        "difference_upper_bound": upper,
        "detail": detail,
    }


def _load_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _records(value: object, name: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _metadata(record: dict, defaults: dict) -> tuple[str, str, str, str]:
    scenario = _text(record.get("scenario_id", defaults.get("scenario_id")), "scenario_id")
    model = _text(record.get("model", defaults.get("model", "")), "model", optional=True)
    pressure = _text(record.get("pressure", defaults.get("pressure", "")),
                     "pressure", optional=True)
    pair = _text(record.get("pair_id"), "pair_id")
    return scenario, model, pressure, pair


def _report_pairs(report: dict) -> list[tuple[tuple[str, str, str, str], dict]]:
    if ("pairs" in report) == ("branches" in report):
        raise ValueError("report.json must have exactly one of 'pairs' or 'branches'")
    if "pairs" in report:
        return [(_metadata(record, report), record)
                for record in _records(report["pairs"], "pairs")]
    grouped: dict[tuple[str, str, str, str], dict] = {}
    for branch in _records(report["branches"], "branches"):
        key = _metadata(branch, report)
        arm = branch.get("arm")
        if arm not in ("natural", "oracle"):
            raise ValueError("branch arm must be 'natural' or 'oracle'")
        arms = grouped.setdefault(key, {})
        if arm in arms:
            raise ValueError(f"Duplicate {arm} arm for {key!r}")
        arms[arm] = branch
    return list(grouped.items())


def build_pairs(run_dirs: Iterable[str | Path]) -> list[PairedOutcome]:
    """Retain every enumerated pair, including failed, absent and unjudged arms."""
    outcomes = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for run_dir in run_dirs:
        directory = Path(run_dir)
        report = _load_object(directory / "report.json")
        judge_path = directory / "judge_v4.json"
        judgments: dict[str, bool | None] = {}
        if judge_path.exists():
            judge = _load_object(judge_path)
            for record in _records(judge.get("results"), "judge results"):
                branch_id = _text(record.get("branch_id"), "judge branch_id")
                if branch_id in judgments:
                    raise ValueError(f"{directory}: duplicate judgment for {branch_id!r}")
                adopted = record.get("adopted")
                if adopted is not None and not isinstance(adopted, bool):
                    raise ValueError(f"{directory}: adopted must be a boolean or null")
                judge_status = _status(record.get("status", "ok"))
                judgments[branch_id] = adopted if judge_status in SUCCESS_STATUSES else None
        seen_branches: set[str] = set()
        for key, record in _report_pairs(report):
            if key in seen_pairs:
                raise ValueError(f"Duplicate scenario/model/pressure/pair across inputs: {key!r}")
            seen_pairs.add(key)
            scenario, model, pressure, pair = key
            arguments = dict(scenario=scenario, pair=pair, model=model,
                             pressure=pressure, run_dir=str(directory))
            for arm in ("natural", "oracle"):
                branch = record.get(arm)
                if branch is None:
                    arguments.update({arm: None, f"{arm}_status": "missing"})
                    continue
                if not isinstance(branch, dict):
                    raise ValueError(f"{directory}: {arm} must be an object or null")
                branch_id = _text(branch.get("branch_id"), "branch_id")
                if branch_id in seen_branches:
                    raise ValueError(f"{directory}: branch reused by multiple pairs: {branch_id!r}")
                seen_branches.add(branch_id)
                status = _status(branch.get("status"))
                arguments[f"{arm}_branch_id"] = branch_id
                arguments[f"{arm}_status"] = status
                arguments[arm] = judgments.get(branch_id) if status in SUCCESS_STATUSES else None
            outcomes.append(PairedOutcome(**arguments))
    return outcomes


def _summary_group(pairs: list[PairedOutcome]) -> dict:
    complete = [pair for pair in pairs if pair.complete]
    directions = Counter(pair.direction for pair in complete)
    table = {"a": directions["both_adopt"], "b": directions["benefit"],
             "c": directions["harm"], "d": directions["neither_adopts"]}
    scenario_counts = Counter(pair.scenario for pair in pairs)
    repeated = sorted(scenario for scenario, count in scenario_counts.items() if count > 1)
    unavailable = Counter(pair.unavailable_reason for pair in pairs if not pair.complete)
    n = len(complete)
    effect = effect_with_ci(table["b"], table["c"], n)
    inference = "ok" if n else "no_complete_pairs"
    test = mcnemar_exact(table["b"], table["c"]) if n else None
    if repeated:
        inference = "blocked_repeated_scenarios"
        test = None
        effect.update(ci=None, reason="repeated_scenarios; pooled effect is descriptive only")
    missing = len(pairs) - n
    total = len(pairs)
    sensitivity = ((table["b"] - table["c"] - missing) / total,
                   (table["b"] - table["c"] + missing) / total) if total else None
    return {
        "table": table, "b": table["b"], "c": table["c"],
        "discordant": table["b"] + table["c"], "n": n,
        "n_scenarios": len(scenario_counts),
        "n_complete_scenarios": len({pair.scenario for pair in complete}),
        "total_pairs": total, "unpairable": missing,
        "unpairable_due_to_failure": unavailable["failure"],
        "unpairable_due_to_unjudged": unavailable["unjudged"],
        "unpairable_due_to_incomplete_execution": unavailable["incomplete_execution"],
        "failed_branches": sum(status in FAILURE_STATUSES for pair in pairs
                               for status in (pair.natural_status, pair.oracle_status)),
        "repeated_scenarios": repeated, "inference_status": inference,
        "mcnemar": test, "effect": effect,
        "missing_outcome_effect_bounds": sensitivity,
        "missing_outcome_bounds_note": "Descriptive worst-case bounds over ALL enumerated "
                                       "pairs, NOT a confidence interval.",
        "estimand": "complete-pair net reduction in wrong-date adoption; missing outcomes "
                    "are not imputed, and selective failure can bias this estimate",
    }


def summarise(pairs: Iterable[PairedOutcome]) -> dict:
    pairs = list(pairs)
    summary = _summary_group(pairs)
    strata: dict[tuple[str, str], list[PairedOutcome]] = defaultdict(list)
    for pair in pairs:
        strata[(pair.model, pair.pressure)].append(pair)
    summary["strata"] = [dict(model=model, pressure=pressure, **_summary_group(group))
                         for (model, pressure), group in sorted(strata.items())]
    summary["unit"] = "independent scenario, not model, pressure condition, or sibling branch"
    summary["multiplicity"] = ("Stratum p-values are unadjusted; shared scenarios are not "
                               "independent replications across strata.")
    return summary


def _write_csv(pairs: list[PairedOutcome], path: Path) -> None:
    fields = ["scenario", "pair", "model", "pressure", "run_dir",
              "natural_branch_id", "oracle_branch_id", "natural_result", "oracle_result",
              "natural_status", "oracle_status", "discordant", "direction",
              "unavailable_reason"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = {field: getattr(pair, field) for field in fields if hasattr(pair, field)}
            for arm in ("natural", "oracle"):
                value = getattr(pair, arm)
                row[f"{arm}_result"] = "" if value is None else str(value).lower()
            row["discordant"] = str(pair.natural != pair.oracle).lower() if pair.complete else ""
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, metavar="DIR")
    parser.add_argument("--out", required=True, metavar="DIR")
    arguments = parser.parse_args()
    try:
        pairs = build_pairs(arguments.runs)
        summary = summarise(pairs)
        output = Path(arguments.out)
        output.mkdir(parents=True, exist_ok=True)
        _write_csv(pairs, output / "paired_table.csv")
    except (OSError, ValueError) as error:
        print(f"paired_stats: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())