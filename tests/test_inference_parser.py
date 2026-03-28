from inference import (
    build_fallback_action,
    choose_balanced_action,
    extract_model_decision,
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


def test_choose_balanced_action_prefers_critical_escalation():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    obs.ward_state.patients[0].escalation_level = 2
    action = choose_balanced_action(obs, history=[], model_payload=None)
    assert getattr(action.action_type, "value", action.action_type) == "escalate"


def test_choose_balanced_action_falls_back_safely_on_invalid_payload():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    action = choose_balanced_action(
        obs,
        history=[],
        model_payload={"chosen_action_id": "A3", "chosen_patient_id": 999},
    )
    assert getattr(action.action_type, "value", action.action_type) in {
        "administer_medicine",
        "check_vitals",
        "escalate",
    }
