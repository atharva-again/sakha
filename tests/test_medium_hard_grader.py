import pytest
from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction
from sakha.graders import score_medium_task, score_hard_task


def _run_trajectory(seed: int, actions: list[tuple[str, int | None]]) -> list:
    env = SakhaEnvironment(patient_count=8)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    for action_type, patient_id in actions:
        obs = env.step(SakhaAction(action_type=action_type, patient_id=patient_id))
        trajectory.append(obs)
    return trajectory


def test_medium_grader_returns_0_to_1():
    trajectory = _run_trajectory(42, [("noop", None)] * 10)
    score = score_medium_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_medium_grader_handles_conflicts():
    actions = [
        ("administer_medicine", 1),
        ("check_vitals", 2),
        ("administer_medicine", 3),
    ]
    trajectory = _run_trajectory(42, actions)
    score = score_medium_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_hard_grader_returns_0_to_1():
    trajectory = _run_trajectory(42, [("noop", None)] * 10)
    score = score_hard_task(trajectory)
    assert 0.0 <= score <= 1.0


def test_hard_grader_penalizes_missed_escalation():
    trajectory_no_escalate = _run_trajectory(42, [("noop", None)] * 20)
    trajectory_with_escalate = _run_trajectory(42, [("escalate", 1)] + [("noop", None)] * 19)
    score_no = score_hard_task(trajectory_no_escalate)
    score_yes = score_hard_task(trajectory_with_escalate)
    assert score_yes >= score_no


def test_hard_grader_weak_vs_strong_gap():
    weak = _run_trajectory(42, [("noop", None)] * 30)
    strong_actions = []
    for _ in range(6):
        strong_actions.extend(
            [
                ("administer_medicine", 1),
                ("check_vitals", 6),
                ("escalate", 6),
                ("check_vitals", 5),
                ("administer_medicine", 4),
                ("escalate", 5),
            ]
        )
    strong = _run_trajectory(42, strong_actions)
    score_weak = score_hard_task(weak)
    score_strong = score_hard_task(strong)
    assert score_strong > score_weak


def test_medium_grader_deterministic():
    t1 = _run_trajectory(42, [("noop", None)] * 10)
    t2 = _run_trajectory(42, [("noop", None)] * 10)
    assert score_medium_task(t1) == score_medium_task(t2)


def test_hard_grader_deterministic():
    t1 = _run_trajectory(42, [("noop", None)] * 10)
    t2 = _run_trajectory(42, [("noop", None)] * 10)
    assert score_hard_task(t1) == score_hard_task(t2)
