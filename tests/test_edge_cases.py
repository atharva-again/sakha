from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction


def test_invalid_patient_id_is_absorbed_with_penalty():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=999))
    assert obs.reward == -0.05


def test_noop_penalty_is_negative():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.NOOP))
    assert obs.reward == -0.05
