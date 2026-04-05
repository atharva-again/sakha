from inference import build_fallback_action, extract_model_decision, select_action
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
