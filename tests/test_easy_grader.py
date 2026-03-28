import pytest
from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction
from sakha.graders import score_easy_task


def _run_easy_trajectory(seed: int, actions: list[str]) -> list:
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    for action_type in actions:
        obs = env.step(SakhaAction(action_type=action_type, patient_id=1))
        trajectory.append(obs)
    return trajectory


def test_easy_grader_returns_0_to_1():
    trajectory = _run_easy_trajectory(42, ["noop"] * 10)
    score = score_easy_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_easy_grader_good_trajectory_scores_high():
    actions = ["administer_medicine", "check_vitals"] * 5
    trajectory = _run_easy_trajectory(42, actions)
    score = score_easy_task(trajectory)
    assert score > 0.0


def test_easy_grader_bad_trajectory_scores_low():
    trajectory = _run_easy_trajectory(42, ["noop"] * 20)
    score = score_easy_task(trajectory)
    assert score < 0.5


def test_easy_grader_deterministic():
    t1 = _run_easy_trajectory(42, ["noop"] * 10)
    t2 = _run_easy_trajectory(42, ["noop"] * 10)
    assert score_easy_task(t1) == score_easy_task(t2)
