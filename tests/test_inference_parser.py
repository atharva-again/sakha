from inference import (
    build_fallback_action,
    extract_model_decision,
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


def test_select_action_falls_back_on_none_payload():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    obs.ward_state.patients[0].escalation_level = 2
    obs.ward_state.patients[0].vitals_due = True
    action, scratchpad = select_action(obs, model_payload=None)
    assert getattr(action.action_type, "value", action.action_type) == "escalate"


def test_select_action_falls_back_on_invalid_patient():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    action, scratchpad = select_action(
        obs,
        model_payload={"chosen_action_id": "A3", "chosen_patient_id": 999},
    )
    assert getattr(action.action_type, "value", action.action_type) in {
        "administer_medicine",
        "check_vitals",
        "escalate",
    }


def test_select_action_extracts_scratchpad():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    action, scratchpad = select_action(
        obs,
        model_payload={
            "chosen_action_id": "A1",
            "chosen_patient_id": 1,
            "scratchpad": "P1 needs meds",
        },
    )
    assert scratchpad == "P1 needs meds"
    assert getattr(action.action_type, "value", action.action_type) == "administer_medicine"


def test_select_action_handles_missing_scratchpad():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    action, scratchpad = select_action(
        obs,
        model_payload={"chosen_action_id": "A1", "chosen_patient_id": 1},
    )
    assert scratchpad is None
    assert getattr(action.action_type, "value", action.action_type) == "administer_medicine"
