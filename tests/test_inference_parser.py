import json

from inference import (
    MAX_TOKENS,
    build_fallback_action,
    build_user_prompt,
    extract_model_decision,
    rank_candidates,
    select_action,
)
from sakha.env import SakhaEnvironment


def test_extract_model_decision_parses_json_block():
    payload = extract_model_decision(
        '{"eligible_actions": [{"id": "A1", "patient_id": 1, "eligible": true}], "chosen_action_id": "A1", "chosen_patient_id": 1}'
    )
    assert payload is not None
    assert payload["chosen_action_id"] == "A1"
    assert payload["chosen_patient_id"] == 1


def test_extract_model_decision_returns_none_for_malformed_output():
    payload = extract_model_decision("not valid json output")
    assert payload is None


def test_extract_model_decision_parses_fenced_json():
    payload = extract_model_decision(
        'Reasoning text\n```json\n{"chosen_action_id":"A1","chosen_patient_id":1}\n```'
    )
    assert payload is not None
    assert payload["chosen_action_id"] == "A1"
    assert payload["chosen_patient_id"] == 1


def test_extract_model_decision_returns_none_for_truncated_json():
    payload = extract_model_decision('{"chosen_action_id":"A1","chosen_patient_id":')
    assert payload is None


def test_select_action_falls_back_to_pending_queue_head():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    expected = build_fallback_action(obs)
    action, _scratchpad = select_action(obs, model_payload=None)
    assert getattr(action.action_type, "value", action.action_type) == getattr(
        expected.action_type, "value", expected.action_type
    )
    assert action.patient_id == expected.patient_id


def test_select_action_falls_back_on_invalid_patient():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    expected = build_fallback_action(obs)
    action, _scratchpad = select_action(
        obs,
        model_payload={"chosen_action_id": "A3", "chosen_patient_id": 999},
    )
    assert getattr(action.action_type, "value", action.action_type) == getattr(
        expected.action_type, "value", expected.action_type
    )


def test_select_action_extracts_scratchpad():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    head_task = obs.ward_state.pending_tasks[0]
    action_id = next(
        key
        for key, value in {
            "A0": "review_patient",
            "A1": "medication_round",
            "A2": "check_vitals",
            "A3": "escalate",
            "A4": "alert_doctor",
            "A5": "update_chart",
            "A6": "prepare_discharge",
            "A7": "ward_sweep",
        }.items()
        if value == head_task.required_action
    )
    action, scratchpad = select_action(
        obs,
        model_payload={
            "chosen_action_id": action_id,
            "chosen_patient_id": head_task.patient_id,
            "scratchpad": "Round bed first",
        },
    )
    assert scratchpad == "Round bed first"
    assert getattr(action.action_type, "value", action.action_type) == head_task.required_action


def test_select_action_handles_missing_scratchpad():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    head_task = obs.ward_state.pending_tasks[0]
    action, scratchpad = select_action(
        obs,
        model_payload={"chosen_action_id": "A0", "chosen_patient_id": head_task.patient_id},
    )
    assert scratchpad is None
    assert action.patient_id == head_task.patient_id


def _extract_state_json(prompt: str) -> dict:
    lines = prompt.splitlines()
    assert lines[0] == "STATE JSON:"
    return json.loads(lines[1])


def test_build_user_prompt_operational_profile_payload_shape():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)

    prompt = build_user_prompt(
        obs,
        step=1,
        history=["A0(patient=1)"],
        prompt_profile="operational_realism",
    )
    payload = _extract_state_json(prompt)

    assert "pending_count" in payload
    assert "pending_tasks" not in payload
    assert "action_result" in payload
    assert isinstance(payload["patients"], list)
    assert payload["patients"]

    patient = payload["patients"][0]
    expected_keys = {
        "bed",
        "diag",
        "meds",
        "med_step",
        "vitals",
        "vit_step",
        "esc",
        "rev",
        "rev_step",
        "inc",
        "discharge",
        "adm_req",
        "adm_reviewed",
        "adm_documented",
        "last_vitals",
    }
    assert set(patient.keys()) == expected_keys


def test_build_user_prompt_strict_bedside_excludes_pending_signals():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)

    prompt = build_user_prompt(
        obs,
        step=1,
        history=[],
        prompt_profile="strict_bedside",
    )
    payload = _extract_state_json(prompt)

    assert "pending_count" not in payload
    assert "pending_tasks" not in payload


def test_build_user_prompt_full_legacy_includes_pending_tasks():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)

    prompt = build_user_prompt(
        obs,
        step=1,
        history=[],
        prompt_profile="full_legacy",
    )
    payload = _extract_state_json(prompt)

    assert "pending_count" in payload
    assert "pending_tasks" in payload
    assert isinstance(payload["pending_tasks"], list)
    assert len(payload["pending_tasks"]) == len(obs.ward_state.pending_tasks)
    patient = payload["patients"][0]
    assert "bed_id" in patient
    assert "bed" not in patient


def test_build_user_prompt_invalid_profile_fails_fast():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)

    try:
        build_user_prompt(
            obs,
            step=1,
            history=[],
            prompt_profile="invalid_profile",
        )
        raise AssertionError("Expected ValueError for invalid profile")
    except ValueError:
        pass


def test_max_tokens_target_reduced():
    assert MAX_TOKENS == 64


def test_rank_candidates_independent_of_pending_tasks():
    env = SakhaEnvironment(patient_count=8, task="medium")
    obs = env.reset(seed=42)

    ranked_before = rank_candidates(obs)
    obs.ward_state.pending_tasks = []
    ranked_after = rank_candidates(obs)

    assert ranked_after
    assert ranked_before == ranked_after


def test_rank_candidates_prioritizes_incident_workflow_order():
    env = SakhaEnvironment(patient_count=8, task="medium")
    obs = env.reset(seed=42)

    patient = obs.ward_state.patients[0]
    patient.active_incident_id = 123
    patient.incident_deadline_step = 4
    patient.incident_checked = False
    patient.incident_alerted = False
    patient.incident_escalated = False
    patient.incident_documented = False

    top = rank_candidates(obs)[0]
    assert top[1] == "A2"

    patient.incident_checked = True
    top = rank_candidates(obs)[0]
    assert top[1] == "A4"

    patient.incident_alerted = True
    top = rank_candidates(obs)[0]
    assert top[1] == "A3"

    patient.incident_escalated = True
    top = rank_candidates(obs)[0]
    assert top[1] == "A5"


def test_rank_candidates_tie_breaks_by_due_then_bed_then_action_id():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)

    first = obs.ward_state.patients[0]
    second = obs.ward_state.patients[1]

    first.review_required = True
    first.last_reviewed_step = -1
    first.bed_id = 2

    second.review_required = True
    second.last_reviewed_step = -1
    second.bed_id = 1

    ranked = rank_candidates(obs)
    review_actions = [r for r in ranked if r[1] == "A0"]
    assert review_actions[0][2] == 1
    assert review_actions[1][2] == 2


def test_rank_candidates_orders_by_due_step_for_same_bucket():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)

    first = obs.ward_state.patients[0]
    second = obs.ward_state.patients[1]

    first.vitals_due = True
    second.vitals_due = True
    first.vitals_due_by_step = 7
    second.vitals_due_by_step = 3

    ranked = rank_candidates(obs)
    vitals_actions = [r for r in ranked if r[1] == "A2" and r[2] in {first.bed_id, second.bed_id}]
    assert vitals_actions[0][2] == second.bed_id
    assert vitals_actions[1][2] == first.bed_id


def test_rank_candidates_is_stable_for_same_observation():
    env = SakhaEnvironment(patient_count=8, task="medium")
    obs = env.reset(seed=42)
    first = rank_candidates(obs)
    second = rank_candidates(obs)
    assert first == second
