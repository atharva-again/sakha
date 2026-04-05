from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task
from sakha.models import ActionType, SakhaAction


def _run_easy_trajectory(seed: int, use_queue_head: bool, steps: int = 12) -> list:
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=seed)
    trajectory = [obs]
    for _ in range(steps):
        if use_queue_head:
            if obs.ward_state.pending_tasks:
                task = obs.ward_state.pending_tasks[0]
                action = SakhaAction(action_type=task.required_action, patient_id=task.patient_id)
            else:
                action = SakhaAction(action_type=ActionType.NOOP, patient_id=None)
        else:
            action = SakhaAction(action_type=ActionType.NOOP, patient_id=None)
        obs = env.step(action)
        trajectory.append(obs)
    return trajectory


def test_easy_grader_returns_0_to_1():
    trajectory = _run_easy_trajectory(42, use_queue_head=False)
    score = score_easy_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_easy_grader_good_trajectory_scores_higher_than_noop():
    weak = _run_easy_trajectory(42, use_queue_head=False)
    strong = _run_easy_trajectory(42, use_queue_head=True)
    assert score_easy_task(strong) > score_easy_task(weak)


def test_easy_grader_bad_trajectory_scores_low():
    trajectory = _run_easy_trajectory(42, use_queue_head=False, steps=20)
    score = score_easy_task(trajectory)
    assert score < 0.5


def test_easy_grader_deterministic():
    t1 = _run_easy_trajectory(42, use_queue_head=False)
    t2 = _run_easy_trajectory(42, use_queue_head=False)
    assert score_easy_task(t1) == score_easy_task(t2)
