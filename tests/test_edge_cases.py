import pytest

from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction


def test_invalid_action_type_rejected():
    env = SakhaEnvironment(patient_count=5, task="easy")
    env.reset(seed=42)
    with pytest.raises(ValueError) as exc_info:
        SakhaAction.model_validate({"action_type": "invalid_action"})
    assert "Invalid action_type" in str(exc_info.value)


def test_valid_action_type_accepted():
    action = SakhaAction(action_type=ActionType.REVIEW_PATIENT, patient_id=1)
    assert action.action_type == ActionType.REVIEW_PATIENT


def test_invalid_patient_id_returns_action_result():
    env = SakhaEnvironment(patient_count=5, task="easy")
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=999))
    assert obs.action_result is not None
    assert obs.action_result.status == "invalid"
    assert obs.reward == -0.02


def test_noop_is_penalized_with_pending_work():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.NOOP))
    assert obs.reward == -0.03
    assert obs.action_result is not None
    assert obs.action_result.status == "no_effect"


def test_out_of_range_patient_id_gets_invalid_penalty():
    env = SakhaEnvironment(patient_count=5, task="easy")
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=100))
    assert obs.reward == -0.02
    assert obs.action_result is not None
    assert obs.action_result.status == "invalid"


def test_review_action_is_available_from_start():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    assert any(
        task.required_action == ActionType.REVIEW_PATIENT for task in obs.ward_state.pending_tasks
    )
