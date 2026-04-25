"""Golden parity tests for rubric migration.

These tests verify that the rubric-integrated SakhaEnvironment produces
bit-for-bit identical rewards and grader scores compared to the golden
fixtures captured before rubric integration.
"""

import json
from pathlib import Path

import pytest

from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import SakhaAction

TASK_GRADERS = {
    "easy": score_easy_task,
    "medium": score_medium_task,
    "hard": score_hard_task,
}
PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(task: str):
    path = FIXTURE_DIR / f"rubric_golden_{task}.json"
    with open(path) as f:
        return json.load(f)


def _replay_episode(env: SakhaEnvironment, task: str, seed: int, expected_rewards: list):
    pc = PATIENT_COUNTS[task]
    obs = env.reset(seed=seed)
    trajectory = [obs]
    replay_rewards = []

    for step_idx, expected_reward in enumerate(expected_rewards):
        if obs.ward_state.pending_tasks:
            task_obj = obs.ward_state.pending_tasks[0]
            action = SakhaAction(
                action_type=task_obj.required_action, patient_id=task_obj.patient_id
            )
        else:
            action = SakhaAction(action_type="noop", patient_id=None)

        obs = env.step(action)
        trajectory.append(obs)
        replay_rewards.append(obs.reward)

        assert obs.reward == pytest.approx(expected_reward, abs=1e-4), (
            f"Step {step_idx} reward mismatch: expected {expected_reward}, got {obs.reward}"
        )

    return trajectory, replay_rewards


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_rubric_reward_parity(task: str):
    fixture = _load_fixture(task)
    env = SakhaEnvironment(patient_count=PATIENT_COUNTS[task], task=task)

    for ep_data in fixture["episodes_data"]:
        seed = ep_data["seed"]
        expected_rewards = ep_data["step_rewards"]
        _replay_episode(env, task, seed, expected_rewards)


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_rubric_grader_score_parity(task: str):
    fixture = _load_fixture(task)
    grader = TASK_GRADERS[task]
    env = SakhaEnvironment(patient_count=PATIENT_COUNTS[task], task=task)

    for ep_data in fixture["episodes_data"]:
        seed = ep_data["seed"]
        expected_grader_score = ep_data["grader_score"]
        trajectory, _ = _replay_episode(env, task, seed, ep_data["step_rewards"])
        actual_grader_score = grader(trajectory)
        assert actual_grader_score == pytest.approx(expected_grader_score, abs=1e-4), (
            f"Grader score mismatch for seed {seed}: expected {expected_grader_score}, got {actual_grader_score}"
        )


def test_rubric_reset_clears_state():
    env = SakhaEnvironment(task="easy")
    env.reset(seed=42)
    env.step(SakhaAction(action_type="noop", patient_id=None))

    # Simulate rubric state by setting a flag on a sub-rubric
    env.rubric.rubric_0.last_score = 999.0
    env.rubric.rubric_1.last_score = 999.0
    env.rubric.rubric_2.last_score = 999.0

    env.reset(seed=43)

    # After reset, rubric scores should be cleared (None or 0.0)
    assert env.rubric.rubric_0.last_score in (None, 0.0)
    assert env.rubric.rubric_1.last_score in (None, 0.0)
    assert env.rubric.rubric_2.last_score in (None, 0.0)


def test_env_step_structured_output():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "inference.py",
            "--tasks",
            "easy",
            "--episodes",
            "1",
            "--seed",
            "42",
            "--deterministic-baseline",
            "--max-steps",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    stdout = result.stdout + result.stderr
    assert "[START]" in stdout, "Missing [START] block in stdout"
    assert "[STEP]" in stdout, "Missing [STEP] block in stdout"
    assert "[END]" in stdout, "Missing [END] block in stdout"
