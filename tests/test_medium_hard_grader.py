from sakha.env import SakhaEnvironment
from sakha.graders import score_hard_task, score_medium_task
from sakha.models import ActionType, SakhaAction


def _run_trajectory(task: str, seed: int, use_queue_head: bool, steps: int) -> list:
    patient_count = 8 if task == "medium" else 18
    env = SakhaEnvironment(patient_count=patient_count, task=task)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    for _ in range(steps):
        if use_queue_head and obs.ward_state.pending_tasks:
            pending = obs.ward_state.pending_tasks[0]
            action = SakhaAction(action_type=pending.required_action, patient_id=pending.patient_id)
        else:
            action = SakhaAction(action_type=ActionType.NOOP, patient_id=None)
        obs = env.step(action)
        trajectory.append(obs)
    return trajectory


def test_medium_grader_returns_0_to_1():
    trajectory = _run_trajectory("medium", 42, use_queue_head=False, steps=10)
    score = score_medium_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_medium_grader_handles_conflicts():
    trajectory = _run_trajectory("medium", 42, use_queue_head=True, steps=12)
    score = score_medium_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_hard_grader_returns_0_to_1():
    trajectory = _run_trajectory("hard", 42, use_queue_head=False, steps=10)
    score = score_hard_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_hard_grader_penalizes_missed_escalation():
    trajectory_no_escalate = _run_trajectory("hard", 42, use_queue_head=False, steps=20)
    trajectory_with_queue = _run_trajectory("hard", 42, use_queue_head=True, steps=20)
    score_no = score_hard_task(trajectory_no_escalate)
    score_yes = score_hard_task(trajectory_with_queue)
    assert score_yes >= score_no


def test_hard_grader_weak_vs_strong_gap():
    weak = _run_trajectory("hard", 42, use_queue_head=False, steps=30)
    strong = _run_trajectory("hard", 42, use_queue_head=True, steps=30)
    score_weak = score_hard_task(weak)
    score_strong = score_hard_task(strong)
    assert score_strong > score_weak


def test_medium_grader_deterministic():
    t1 = _run_trajectory("medium", 42, use_queue_head=False, steps=10)
    t2 = _run_trajectory("medium", 42, use_queue_head=False, steps=10)
    assert score_medium_task(t1) == score_medium_task(t2)


def test_hard_grader_deterministic():
    t1 = _run_trajectory("hard", 42, use_queue_head=False, steps=10)
    t2 = _run_trajectory("hard", 42, use_queue_head=False, steps=10)
    assert score_hard_task(t1) == score_hard_task(t2)
