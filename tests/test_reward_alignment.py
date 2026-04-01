from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task
from sakha.models import ActionType, SakhaAction


def test_reward_on_time_medication():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=1))
    assert obs.reward > 0.0


def test_reward_aligns_with_grader():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    trajectory = [obs]
    for i in range(5):
        obs = env.step(SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=i + 1))
        trajectory.append(obs)
    assert score_easy_task(trajectory) > 0.0
