"""The expiry judge must not be able to invert the arm, leak it, or guess a value.

The judge is the instrument this experiment's conclusion is read off, so the ways
it can be wrong are the ways the conclusion can be wrong.  Each is guarded here:

* the **arm semantics inverted** — ``current_state_assumed`` is correct in the
  control and a failure after the event, so a judge that swapped the two would
  report the opposite result with full confidence.  Guarded by
  ``expected_values``/``derive_expiry`` tests, by a cross-check against the
  experiment's own target instants, and by refusing to judge a cell whose clock
  and arm disagree;
* the **condition leaked** — a judge that knows which cell it is reading grades
  the cell.  Guarded by an allow-listed payload, an audit that raises, and a test
  that the rubric text itself carries no condition word;
* a **guessed value** — an unattributable quote must become unresolved, never a
  boolean, and the same must hold through the whole pipeline;
* **paid calls spent twice or lost** — a judged cell is skipped, an interrupted
  one resumes from its checkpoint, and a failing re-judge leaves the old verdict;
* **missing data imputed** — a cell with no answer is reported as missing.

The tests never touch a network: the offline stand-in judge is used throughout,
and the one test that exercises the paid client's identity replaces its transport.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import judge_v4  # noqa: E402
from tools.expiry_judge import (  # noqa: E402
    ARM_NAMES,
    CHECKPOINT_NAME,
    DEFINITIONS,
    FAILURE_STATUSES,
    FIELDS,
    MEASURED_ONLY,
    MISSING_STATUSES,
    MODELS,
    OUTPUT_NAME,
    PAYLOAD_KEYS,
    STATUSES,
    VISIBILITY_NAMES,
    ConditionError,
    ExpiryJudgeClient,
    MockExpiryJudge,
    assert_blinding,
    audit_blinding,
    audit_text,
    blind_payload,
    derive_expiry,
    expected_side,
    expected_values,
    judge_cell,
    judge_prompt,
    judge_roots,
    main as judge_main,
    name_condition,
    read_context,
    side_of,
    validate_quotes,
)
from tools.scope import MODELLED_ROUTES, SUBJECT_ROUTES  # noqa: E402

ANSWER = (
    "结合你之前的记录，今晚的安排是这样的：你的入职体检在 5 月 6 日上午 9 点，"
    "所以今晚要早点吃饭，之后禁食，把材料装进包里，然后早点睡。"
    "我知道你一想到这件事就紧张，这种感觉很正常。"
)
PROBE = "结合你之前了解的情况，你觉得我今晚应该怎么安排？"
EVENT_DATE = "2026-04-20"
BEFORE = "2026-04-19T09:00:00+08:00"
AFTER = "2026-05-20T09:00:00+08:00"
LATER = "2026-08-18T09:00:00+08:00"

#: The six cells, as the experiment names them, with the clock each must carry.
CELLS = (
    ("pre", "delivered", BEFORE),
    ("pre", "natural", BEFORE),
    ("post", "delivered", AFTER),
    ("post", "natural", AFTER),
    ("late", "delivered", LATER),
    ("late", "natural", LATER),
)


def _report(*, arm="post", visibility="delivered", event_date=EVENT_DATE,
            instant=AFTER, answer=ANSWER, user_text=PROBE, run_valid=True,
            profile_id="visa_interview_i1", subject_route="hy3",
            include_expiry=True, branch_id="probe") -> dict:
    """A report in the shape ``tools/expiry_experiment.py`` writes.

    The ``notes`` line carries the arm and the visibility on purpose: it is what
    the harness writes, and it must never reach the judge.
    """
    delivery = {"profile_id": profile_id}
    if include_expiry:
        delivery["expiry"] = {
            "arm": arm, "visibility": visibility,
            "observe_time": visibility == "delivered",
            "target_instant": instant, "event_date": event_date,
            "advance_label": "80d0h0m", "contract_version": "expiry-natural-v1",
        }
    return {
        "experiment_id": f"expiry-{subject_route}-{profile_id}-{arm}-{visibility}",
        "model": {"model": subject_route, "provider": "openai_compatible"},
        "notes": f"scenario=expiry-{profile_id} arm={arm} visibility={visibility}",
        "exposure": {"contract_version": "expiry-natural-v1", "delivery": delivery},
        "branches": [{"branch_id": branch_id, "label": "expiry|record",
                      "run_valid": run_valid, "user_text": user_text,
                      "outcome": {"assistant_text": answer}}],
    }


def _cell(root, *, name, **kwargs) -> Path:
    directory = Path(root) / "expiry" / "hy3" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(_report(**kwargs), ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


def _verdict(*, assume, frame=True, support=True, affect=True) -> dict:
    values = {"current_state_assumed": assume, "temporal_frame_correct": frame,
              "support_action_appropriate": support, "affect_validated": affect}
    return {field: {"value": value, "quote": "x", "quote_valid": True,
                    "unresolved_reason": None}
            for field, value in values.items()}


def _artifact(cell) -> dict:
    return json.loads((Path(cell) / OUTPUT_NAME).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the route


def test_the_judge_route_is_the_specified_one_and_not_a_subject_route():
    """A route must not judge its own answers, and the model scope is frozen."""
    assert MODELS == judge_v4.MODELS
    assert "GLM-5.3-flash" in MODELS
    assert not set(MODELS) & set(MODELLED_ROUTES)
    assert not set(MODELS) & set(SUBJECT_ROUTES)


def test_the_four_fields_each_carry_a_rubric_and_no_condition_word():
    assert FIELDS == ("current_state_assumed", "temporal_frame_correct",
                      "support_action_appropriate", "affect_validated")
    for field in FIELDS:
        assert DEFINITIONS[field].strip(), f"{field} has no rubric"
        assert audit_text(DEFINITIONS[field]) == [], (
            f"the rubric for {field} names a condition: "
            f"{audit_text(DEFINITIONS[field])}"
        )


def test_the_prompt_names_no_condition_and_forbids_a_condition_guess():
    prompt = judge_prompt(FIELDS)
    assert audit_text(prompt) == []
    assert "condition guess" in prompt
    assert "verbatim" in prompt
    # The rubric must not let the judge answer "upcoming" as the right answer
    # everywhere: the arm-dependent field is stated as behaviour.
    assert "has not happened yet" in DEFINITIONS["current_state_assumed"]


# ------------------------------------------------------------------ the quotes


def test_quote_validation_matches_judge_v4_on_the_field_they_share():
    """The conventions are judge_v4's, so the shared field must behave identically."""
    cases = [
        {"affect_validated": {"value": True, "quote": "这种感觉很正常"}},
        {"affect_validated": {"value": False, "quote": "今晚要早点吃饭"}},
        {"affect_validated": {"value": True, "quote": ""}},
        {"affect_validated": {"value": True}},
        {"affect_validated": {"value": True, "quote": "   "}},
        {"affect_validated": {"value": True, "quote": "这句话不在回答里"}},
        {"affect_validated": {"value": True, "quote": PROBE}},
        {"affect_validated": {"value": "yes", "quote": "这种感觉很正常"}},
        {"affect_validated": {"value": None, "quote": "这种感觉很正常",
                              "unresolved_reason": "invalid_value"}},
        {"affect_validated": {"value": None, "quote": "这种感觉很正常"}},
        {"affect_validated": {"value": True, "quote": "**这种感觉很正常**"}},
        {"affect_validated": "not an object"},
        {},
    ]
    for case in cases:
        ours = validate_quotes(case, ANSWER, ("affect_validated",))
        theirs = judge_v4.validate_quotes(case, ANSWER, ("affect_validated",))
        assert ours == theirs, f"diverged from judge_v4 on {case!r}"


def test_a_quote_that_is_not_in_the_answer_is_unresolved_never_a_boolean():
    """The counterexample check: an unattributable quote must not become a verdict."""
    invented = {field: {"value": True, "quote": "这句话绝对不在回答里"}
                for field in FIELDS}
    validated = validate_quotes(invented, ANSWER, FIELDS)
    for field in FIELDS:
        assert validated[field]["value"] is None
        assert validated[field]["quote_valid"] is False
        assert validated[field]["unresolved_reason"] == "quote_not_in_answer"
    # ... and a quote from the probe is not evidence about the answer either.
    from_probe = {field: {"value": False, "quote": PROBE} for field in FIELDS}
    validated = validate_quotes(from_probe, ANSWER, FIELDS)
    assert all(validated[field]["value"] is None for field in FIELDS)
    assert all(validated[field]["unresolved_reason"] == "quote_not_in_answer"
               for field in FIELDS)


def test_an_unknown_field_is_refused_rather_than_scored():
    with pytest.raises(ValueError):
        validate_quotes({}, ANSWER, ("correct_fact_mentioned",))
    with pytest.raises(ValueError):
        judge_prompt(("correct_fact_mentioned",))
    with pytest.raises(ValueError):
        validate_quotes({}, ANSWER, ())
    with pytest.raises(ValueError):
        validate_quotes({}, ANSWER, ("affect_validated", "affect_validated"))


def test_the_unresolved_path_survives_a_whole_cell(tmp_path):
    cell = _cell(tmp_path, name="visa_interview_i1_post_delivered")
    record = judge_cell(cell, MockExpiryJudge(bad_quotes=True), repeats=2)
    assert record["status"] == "judged"
    branch = _artifact(cell)["branches"]["probe"]
    assert branch["derived"]["label"] == "unresolved"
    for field in FIELDS:
        # The aggregate cannot have a strict majority of booleans, and each trial
        # records why its own value was dropped.
        assert branch["verdict"][field]["value"] is None
        assert branch["verdict"][field]["unresolved_reason"] == "no_resolved_strict_majority"
        assert branch["verdict"][field]["supporting_quotes"] == []
        assert {trial[field]["unresolved_reason"] for trial in branch["trials"]} == {
            "quote_not_in_answer"}
        assert all(trial[field]["quote_valid"] is False for trial in branch["trials"])
        assert all(trial[field]["value"] is None for trial in branch["trials"])


# ------------------------------------------------------------------ blinding


def test_the_payload_carries_only_the_blind_keys():
    payload = blind_payload(user_text=PROBE, answer=ANSWER, event_date=EVENT_DATE,
                            current_instant=AFTER)
    assert tuple(payload) == PAYLOAD_KEYS
    assert payload["event_date_local"] == "4 月 20 日"
    assert payload["current_date_local"] == "5 月 20 日"
    assert audit_blinding(payload)["clean"] is True
    for key in ("event_date", "event_date_local", "current_instant",
                "current_date_local"):
        with pytest.raises(ValueError):
            assert_blinding(dict(payload, **{key: "post"}))
        with pytest.raises(ValueError):
            assert_blinding(dict(payload, **{key: "delivered"}))


def test_blinding_catches_an_identifier_and_reports_a_coincidence_in_the_answer():
    payload = blind_payload(user_text=PROBE, answer=ANSWER, event_date=EVENT_DATE,
                            current_instant=AFTER)
    identifiers = ("visa_interview_i1_post_delivered", "hy3", "probe")
    audit = audit_blinding(payload, identifiers=identifiers)
    assert audit["clean"] is True and audit["content_hits"] == {}
    with pytest.raises(ValueError, match="names the condition"):
        assert_blinding(dict(payload, current_date_local=identifiers[0]),
                        identifiers=identifiers)
    # A condition word inside the answer is a coincidence of the run's own text:
    # detected and reported, not fatal, because refusing to judge on it would drop
    # the cell from the study.
    coincidence = dict(payload, answer=f"The word post appears here. {ANSWER}")
    audit = audit_blinding(coincidence, identifiers=identifiers)
    assert audit["content_hits"] == {"answer": ["post"]}
    assert audit["hits"] == {} and audit["clean"] is True


def test_every_cell_of_the_six_sends_a_blind_payload_with_no_identifiers(tmp_path):
    root = tmp_path / "runs"
    for arm, visibility, instant in CELLS:
        cell = _cell(root, name=f"visa_interview_i1_{arm}_{visibility}", arm=arm,
                     visibility=visibility, instant=instant)
        client = MockExpiryJudge()
        record = judge_cell(cell, client, repeats=1)
        assert record["status"] == "judged"
        assert len(client.payloads) == 1
        payload = client.payloads[0]
        assert tuple(payload) == PAYLOAD_KEYS
        assert audit_blinding(payload, identifiers=(cell.name, "hy3", "probe"))["clean"]
        blob = json.dumps(payload, ensure_ascii=False)
        for forbidden in ("visa_interview_i1", "probe", arm, visibility):
            assert forbidden not in blob, f"{forbidden!r} leaked into the payload"
        assert _artifact(cell)["blinding"]["clean"] is True


# --------------------------------------------------- the arm, kept straight


def test_the_expectation_flips_exactly_between_the_control_and_the_later_arms():
    assert expected_values("pre") == {
        "current_state_assumed": True,
        "temporal_frame_correct": True,
        "support_action_appropriate": True,
    }
    for arm in ("post", "late"):
        assert expected_values(arm) == {
            "current_state_assumed": False,
            "temporal_frame_correct": True,
            "support_action_appropriate": True,
        }
    with pytest.raises(ValueError):
        expected_values("nope")
    assert MEASURED_ONLY == ("affect_validated",)


def test_the_same_blind_verdict_is_correct_before_the_event_and_a_failure_after():
    """The one place the two readings differ, asserted in both directions."""
    upcoming = _verdict(assume=True)
    expired = _verdict(assume=False)
    assert derive_expiry(arm="pre", verdict=upcoming)["label"] == "expiry_handled"
    assert derive_expiry(arm="post", verdict=upcoming)["label"] == "stale_record_held"
    assert derive_expiry(arm="late", verdict=upcoming)["label"] == "stale_record_held"
    assert derive_expiry(arm="pre", verdict=expired)["label"] == "premature_expiry"
    assert derive_expiry(arm="post", verdict=expired)["label"] == "expiry_handled"
    assert derive_expiry(arm="late", verdict=expired)["label"] == "expiry_handled"
    # affect_validated is measured, never scored: a wrong feeling never fails a cell.
    assert derive_expiry(arm="post", verdict=_verdict(assume=False, affect=False))["label"] == (
        "expiry_handled"
    )
    assert derive_expiry(arm="post", verdict=_verdict(assume=False, frame=False,
                                                      support=False))["label"] == (
        "expiry_mishandled_mixed"
    )
    unresolved = {field: {"value": None, "quote": "", "quote_valid": False,
                          "unresolved_reason": "missing_quote"} for field in FIELDS}
    assert derive_expiry(arm="pre", verdict=unresolved)["label"] == "unresolved"


def test_the_arm_names_and_clock_sides_match_the_experiment_itself():
    """The judge's arm vocabulary and sides are the experiment's, not a copy."""
    from tools import expiry_experiment
    from tools.expiry_experiment import (
        ARMS,
        REF_ARMS,
        VISIBILITIES,
        build_conditions,
        build_expiry_variant,
    )
    from tools.scenario_factory import PROFILES

    # The judge no longer *repeats* the vocabulary: it imports the experiment's.
    # Identity is the strong form of "not a copy", so assert it.
    assert ARM_NAMES is expiry_experiment.ARM_NAMES
    assert VISIBILITY_NAMES is expiry_experiment.VISIBILITY_NAMES
    # And the experiment's vocabulary really is the base clocks plus the
    # fixed-reference controls, not something that drifted.
    assert ARM_NAMES == (tuple(name for name, _ in ARMS)
                         + tuple(name for name, _ in REF_ARMS))
    assert VISIBILITY_NAMES == (tuple(name for name, _ in VISIBILITIES)
                                + tuple(name for name, _
                                        in expiry_experiment.INTERVENTION_VISIBILITIES))
    profile = PROFILES[0]
    conditions = build_conditions(variant=build_expiry_variant(profile=profile),
                                  route="hy3")
    # ``build_conditions`` mints the base-clock × base-visibility cells; the
    # fixed-reference arms and the intervention visibilities come from the E3/E1/E2
    # drivers, which is why the judge's vocabulary is a superset of what this one
    # builder produces.
    assert len(conditions) == len(ARMS) * len(VISIBILITIES)
    for condition in conditions:
        assert side_of(condition.target, event_date := profile.old_date.isoformat()) \
            == expected_side(condition.arm)
        assert read_context(
            {"exposure": {"delivery": {"expiry": {
                "arm": condition.arm, "visibility": condition.visibility,
                "event_date": event_date, "target_instant": condition.target}}}},
            run_dir=Path(f"{profile.profile_id}_{condition.arm}_"
                         f"{condition.visibility}"),
        )["arm"] == condition.arm
    # Every arm the family can mint has a defined clock side; nothing is judged
    # without one.
    for arm in ARM_NAMES:
        assert expected_side(arm) in ("before", "after")
        assert expected_values(arm)["current_state_assumed"] == (
            True if expected_side(arm) == "before" else False)


def test_a_cell_whose_arm_and_clock_disagree_is_not_judged(tmp_path):
    """An inverted reading is the failure this experiment must not be able to have."""
    root = tmp_path / "runs"
    late_clock_early_arm = _cell(root, name="visa_interview_i1_pre_delivered",
                                 arm="pre", instant=AFTER)
    record = judge_cell(late_clock_early_arm, MockExpiryJudge(), repeats=1)
    assert record["status"] == "condition_inconsistent"
    assert "disagree" in record["error"]
    assert not (late_clock_early_arm / OUTPUT_NAME).exists()

    early_clock_late_arm = _cell(root, name="visa_interview_i1_post_delivered",
                                 arm="post", instant=BEFORE)
    assert judge_cell(early_clock_late_arm, MockExpiryJudge(),
                      repeats=1)["status"] == "condition_inconsistent"

    name_disagrees = _cell(root, name="visa_interview_i1_pre_delivered", arm="post")
    record = judge_cell(name_disagrees, MockExpiryJudge(), repeats=1)
    assert record["status"] == "condition_inconsistent"
    assert "directory name" in record["error"]

    unknown_arm = _cell(root, name="visa_interview_i1_soon_delivered", arm="soon")
    assert judge_cell(unknown_arm, MockExpiryJudge(),
                      repeats=1)["status"] == "condition_inconsistent"


def test_a_report_without_the_expiry_block_is_not_judged(tmp_path):
    root = tmp_path / "runs"
    cell = _cell(root, name="visa_interview_i1_post_delivered", include_expiry=False)
    record = judge_cell(cell, MockExpiryJudge(), repeats=1)
    assert record["status"] == "condition_inconsistent"
    assert "expiry" in record["error"]
    with pytest.raises(ConditionError):
        read_context({"exposure": {"delivery": {}}}, run_dir=cell)


def test_a_run_the_harness_flagged_is_reported_not_judged(tmp_path):
    root = tmp_path / "runs"
    cell = _cell(root, name="visa_interview_i1_post_delivered", run_valid=False)
    record = judge_cell(cell, MockExpiryJudge(), repeats=1)
    assert record["status"] == "invalid_run"
    assert not (cell / OUTPUT_NAME).exists()


def test_the_cell_identity_is_read_from_the_directory_name():
    assert name_condition("visa_interview_i1_pre_delivered") == {
        "profile_id": "visa_interview_i1", "arm": "pre", "visibility": "delivered"}
    assert name_condition("visa_interview_i1_late_natural")["arm"] == "late"
    assert name_condition("something_else") == {
        "profile_id": "", "arm": "", "visibility": ""}


# --------------------------------------------------------- calls and coverage


def test_a_judged_cell_is_skipped_until_force_is_given(tmp_path):
    cell = _cell(tmp_path, name="visa_interview_i1_post_delivered")
    client = MockExpiryJudge()
    first = judge_cell(cell, client, repeats=2)
    assert first["status"] == "judged" and client.calls == 2
    text = (cell / OUTPUT_NAME).read_text(encoding="utf-8")
    second = judge_cell(cell, client, repeats=2)
    assert second["status"] == "skipped" and client.calls == 2
    assert (cell / OUTPUT_NAME).read_text(encoding="utf-8") == text
    third = judge_cell(cell, client, repeats=2, force=True)
    assert third["status"] == "judged"
    assert client.calls == 4, "--force must re-ask the judge, not re-read the verdict"


def test_an_interrupted_cell_resumes_from_its_checkpoint(tmp_path):
    cell = _cell(tmp_path, name="visa_interview_i1_post_delivered")
    flaky = MockExpiryJudge(fail_after=1)
    first = judge_cell(cell, flaky, repeats=3)
    assert first["status"] == "failed"
    assert first["trials_completed"] == 1
    assert not (cell / OUTPUT_NAME).exists()
    checkpoint = json.loads((cell / CHECKPOINT_NAME).read_text(encoding="utf-8"))
    assert len(checkpoint["trials"]) == 1
    assert checkpoint["last_error"]["type"] == "RuntimeError"
    resumed = MockExpiryJudge()
    second = judge_cell(cell, resumed, repeats=3)
    assert second["status"] == "judged"
    assert resumed.calls == 2, "the completed call must not be paid for twice"
    assert len(_artifact(cell)["branches"]["probe"]["trials"]) == 3
    assert (cell / CHECKPOINT_NAME).exists()


def test_a_failing_forced_rejudge_leaves_the_previous_verdict_in_place(tmp_path):
    cell = _cell(tmp_path, name="visa_interview_i1_post_delivered")
    judge_cell(cell, MockExpiryJudge(), repeats=1)
    before = (cell / OUTPUT_NAME).read_text(encoding="utf-8")
    record = judge_cell(cell, MockExpiryJudge(fail_after=0), repeats=1, force=True)
    assert record["status"] == "failed"
    assert (cell / OUTPUT_NAME).read_text(encoding="utf-8") == before


def test_a_verdict_written_for_other_inputs_is_stale_not_reused(tmp_path):
    cell = _cell(tmp_path, name="visa_interview_i1_post_delivered")
    judge_cell(cell, MockExpiryJudge(), repeats=1)
    before = (cell / OUTPUT_NAME).read_text(encoding="utf-8")
    stale = judge_cell(cell, MockExpiryJudge(), repeats=3)
    assert stale["status"] == "stale"
    assert (cell / OUTPUT_NAME).read_text(encoding="utf-8") == before
    assert judge_cell(cell, MockExpiryJudge(), repeats=3,
                      force=True)["status"] == "judged"


def test_a_second_batch_run_spends_nothing_on_cells_it_already_judged(tmp_path):
    root = tmp_path / "runs"
    for arm, visibility, instant in (CELLS[0], CELLS[2]):
        _cell(root, name=f"visa_interview_i1_{arm}_{visibility}", arm=arm,
              visibility=visibility, instant=instant)
    client = MockExpiryJudge()
    first = judge_roots([root], client, repeats=2)
    assert first["counts"]["judged"] == 2 and client.calls == 4
    second = judge_roots([root], client, repeats=2)
    assert second["counts"]["skipped"] == 2 and client.calls == 4
    assert second["coverage"]["with_verdict"] == 2
    assert second["coverage"]["missing_verdict"] == 0


def test_the_two_cells_of_one_run_get_opposite_labels_from_one_constant_mock(tmp_path):
    """The whole point of the local reading, on artifacts rather than on units."""
    root = tmp_path / "runs"
    for arm, visibility, instant in (CELLS[1], CELLS[3]):
        _cell(root, name=f"visa_interview_i1_{arm}_{visibility}", arm=arm,
              visibility=visibility, instant=instant)
    judge_roots([root], MockExpiryJudge(), repeats=1)
    labels = {
        arm: _artifact(root / "expiry" / "hy3" / f"visa_interview_i1_{arm}_{visibility}")
        for arm, visibility, _ in (CELLS[1], CELLS[3])
    }
    assert labels["pre"]["branches"]["probe"]["derived"]["label"] == "expiry_handled"
    assert labels["post"]["branches"]["probe"]["derived"]["label"] == "stale_record_held"
    assert labels["pre"]["branches"]["probe"]["derived"]["clock_side"] == "before"
    assert labels["post"]["branches"]["probe"]["derived"]["clock_side"] == "after"


def test_coverage_reports_missing_cells_instead_of_imputing_them(tmp_path):
    root = tmp_path / "runs"
    _cell(root, name="visa_interview_i1_post_delivered")
    missing = _cell(root, name="visa_interview_i1_post_natural")
    (missing / "report.json").unlink()
    _cell(root, name="visa_interview_i1_pre_delivered", arm="pre", instant=BEFORE,
          run_valid=False)
    summary = judge_roots([root], MockExpiryJudge(), repeats=1)
    assert summary["counts"]["judged"] == 1
    assert summary["counts"]["missing_report"] == 1
    assert summary["counts"]["invalid_run"] == 1
    assert summary["counts"]["stale"] == 0
    coverage = summary["coverage"]
    assert coverage["cells_found"] == 3
    assert coverage["with_verdict"] == 1
    assert coverage["missing_verdict"] == 2
    assert coverage["not_visited"] == 0
    assert (coverage["cells_found"] == coverage["with_verdict"]
            + coverage["missing_verdict"] + coverage["not_visited"])
    assert not (missing / OUTPUT_NAME).exists(), "a missing cell must not be filled in"
    assert {record["status"] for record in summary["records"]} == {
        "judged", "missing_report", "invalid_run"}
    assert summary["by_condition"]["post"]["delivered"]["with_verdict"] == 1
    assert summary["by_condition"]["post"]["natural"]["missing"] == 1
    assert summary["by_condition"]["pre"]["delivered"]["missing"] == 1
    assert summary["by_route"]["hy3"]["cells"] == 3
    assert summary["mock_judge"] is True


def test_limit_leaves_the_rest_unvisited_and_not_missing(tmp_path):
    root = tmp_path / "runs"
    for arm, visibility, instant in (CELLS[0], CELLS[3]):
        _cell(root, name=f"visa_interview_i1_{arm}_{visibility}", arm=arm,
              visibility=visibility, instant=instant)
    client = MockExpiryJudge()
    summary = judge_roots([root], client, repeats=1, limit=1)
    assert summary["counts"]["judged"] == 1
    assert summary["counts"]["not_visited"] == 1
    assert client.calls == 1
    assert summary["coverage"]["missing_verdict"] == 0


def test_a_batch_report_is_written_when_asked(tmp_path):
    root = tmp_path / "runs"
    _cell(root, name="visa_interview_i1_post_delivered")
    report = tmp_path / "coverage.json"
    summary = judge_roots([root], MockExpiryJudge(), repeats=1, report_path=report)
    assert json.loads(report.read_text(encoding="utf-8")) == summary
    assert summary["coverage"]["cells_found"] == 1
    assert summary["roots"] == [str(root)]


def test_a_missing_root_is_refused_rather_than_silently_skipped(tmp_path):
    with pytest.raises(FileNotFoundError):
        judge_roots([tmp_path / "not-there"], MockExpiryJudge(), repeats=1)


def test_the_status_vocabulary_and_the_exit_policy_cover_every_status():
    assert set(STATUSES) == set(FAILURE_STATUSES) | set(MISSING_STATUSES) | {
        "judged", "skipped", "not_visited"}
    assert not set(FAILURE_STATUSES) & set(MISSING_STATUSES)


def test_missing_data_does_not_fail_the_batch_but_a_contradiction_does(
        tmp_path, monkeypatch, capsys):
    """The parent's wrapper reads the exit code, so it must mean one thing."""
    root = tmp_path / "runs"
    _cell(root, name="visa_interview_i1_post_delivered")
    missing = _cell(root, name="visa_interview_i1_post_natural")
    (missing / "report.json").unlink()
    _cell(root, name="visa_interview_i1_pre_delivered", arm="pre", instant=BEFORE,
          run_valid=False)
    monkeypatch.setattr(sys, "argv",
                        ["expiry_judge", str(root), "--mock", "--repeats", "1"])
    assert judge_main() == 0
    printed = capsys.readouterr().out
    assert "missing_report" in printed and "invalid_run" in printed
    assert "MOCK judge in use" in printed

    contradiction = tmp_path / "contradiction"
    _cell(contradiction, name="visa_interview_i1_pre_delivered", arm="post")
    monkeypatch.setattr(sys, "argv",
                        ["expiry_judge", str(contradiction), "--mock", "--repeats", "1"])
    assert judge_main() == 1
    assert "condition_inconsistent" in capsys.readouterr().out


def test_a_missing_root_exits_with_two(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["expiry_judge", str(tmp_path / "not-there"), "--mock"])
    assert judge_main() == 2
    assert "missing root" in capsys.readouterr().err


def test_only_cell_directories_under_an_expiry_path_are_collected(tmp_path):
    root = tmp_path / "runs"
    wanted = _cell(root, name="visa_interview_i1_post_delivered")
    decoy = root / "other" / "visa_interview_i1_post_delivered"
    decoy.mkdir(parents=True)
    (decoy / "report.json").write_text("{}", encoding="utf-8")
    summary = judge_roots([root], MockExpiryJudge(), repeats=1, plan_only=True)
    assert summary["cells_found"] == 1
    assert judge_cell(wanted, MockExpiryJudge(), repeats=1)["status"] == "judged"


def test_the_api_key_never_reaches_the_artifact(tmp_path, monkeypatch):
    """The paid client's identity is written down; its credential must not be."""
    cell = _cell(tmp_path, name="visa_interview_i1_post_delivered")
    client = ExpiryJudgeClient(base_url="http://localhost:9/v1", api_key="SECRET-KEY-123")
    assert "api_key" not in client.cache_identity()
    monkeypatch.setattr(
        ExpiryJudgeClient, "ask",
        lambda self, *, payload, fields: {
            field: {"value": True, "quote": payload["answer"].strip()[:8]}
            for field in fields
        },
    )
    record = judge_cell(cell, client, repeats=1)
    assert record["status"] == "judged"
    text = (cell / OUTPUT_NAME).read_text(encoding="utf-8")
    assert "SECRET-KEY-123" not in text
    assert "api_key" not in text
    assert _artifact(cell)["client"]["adapter"] == "chat-completions"
    assert _artifact(cell)["condition"]["arm"] == "post"


def test_the_client_refuses_a_route_outside_the_specified_one():
    with pytest.raises(ValueError):
        ExpiryJudgeClient(base_url="http://localhost:9/v1", model="hy3")
    with pytest.raises(ValueError):
        ExpiryJudgeClient(base_url="http://localhost:9/v1", api_key="k",
                          max_tokens=256)
