from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task
from sakha.models import ActionType, SakhaAction


def _run_easy_trajectory(seed: int, actions: list[ActionType]) -> list:
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    for action_type in actions:
        obs = env.step(SakhaAction(action_type=action_type, patient_id=1))
        trajectory.append(obs)
    return trajectory


def test_easy_grader_returns_0_to_1():
    trajectory = _run_easy_trajectory(42, [ActionType.NOOP] * 10)
    score = score_easy_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_easy_grader_good_trajectory_scores_high():
    actions = [ActionType.ADMINISTER_MEDICINE, ActionType.CHECK_VITALS] * 5
    trajectory = _run_easy_trajectory(42, actions)
    score = score_easy_task(trajectory)
    assert score > 0.0


def test_easy_grader_bad_trajectory_scores_low():
    trajectory = _run_easy_trajectory(42, [ActionType.NOOP] * 20)
    score = score_easy_task(trajectory)
    assert score < 0.5
