import pytest
from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction, ActionType


def test_invalid_action_type_rejected():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)
    with pytest.raises(ValueError) as exc_info:
        SakhaAction(action_type="invalid_action")
    assert "Invalid action_type" in str(exc_info.value)


def test_valid_action_type_accepted():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)
    action = SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=1)
    assert action.action_type == ActionType.ADMINISTER_MEDICINE


def test_invalid_patient_id_returns_action_result():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=999))
    assert obs.action_result is not None
    assert obs.action_result.status == "invalid"


def test_noop_is_neutral():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.NOOP))
    assert obs.reward == 0.0
    assert obs.action_result is not None
    assert obs.action_result.status == "no_effect"


def test_out_of_range_patient_id_no_penalty():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=100))
    assert obs.reward == 0.0
    assert obs.action_result is not None
    assert obs.action_result.status == "invalid"


def test_action_validation_with_enum():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.ESCALATE, patient_id=1))
    assert obs is not None
