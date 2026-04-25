"""Capture pre-migration golden reward fixtures for rubric parity testing."""

import json
import sys
from pathlib import Path

from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import SakhaAction

TASK_GRADERS = {
    "easy": score_easy_task,
    "medium": score_medium_task,
    "hard": score_hard_task,
}
PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}


def priority_policy(obs, step, pc):
    if obs.ward_state.pending_tasks:
        task = obs.ward_state.pending_tasks[0]
        return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)
    return SakhaAction(action_type="noop", patient_id=None)


def capture_fixtures(task: str, seed: int, episodes: int, max_steps: int = 96) -> dict:
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]
    episodes_data = []
    grader_scores = []
    total_rewards = []

    for ep in range(episodes):
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed + ep)
        trajectory = [obs]
        step_rewards = []

        for step in range(max_steps):
            action = priority_policy(obs, step, pc)
            obs = env.step(action)
            trajectory.append(obs)
            step_rewards.append(obs.reward)
            if obs.done:
                break

        grader_score = grader(trajectory)
        total_reward = sum(step_rewards)
        grader_scores.append(grader_score)
        total_rewards.append(total_reward)

        episodes_data.append(
            {
                "episode_index": ep,
                "seed": seed + ep,
                "step_rewards": step_rewards,
                "grader_score": round(grader_score, 6),
                "total_reward": round(total_reward, 6),
            }
        )

    return {
        "task": task,
        "policy": "priority",
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "episodes_data": episodes_data,
        "summary": {
            "mean_grader_score": round(sum(grader_scores) / len(grader_scores), 6),
            "std_grader_score": round(
                (
                    sum((s - sum(grader_scores) / len(grader_scores)) ** 2 for s in grader_scores)
                    / len(grader_scores)
                )
                ** 0.5,
                6,
            ),
            "mean_total_reward": round(sum(total_rewards) / len(total_rewards), 6),
            "std_total_reward": round(
                (
                    sum((r - sum(total_rewards) / len(total_rewards)) ** 2 for r in total_rewards)
                    / len(total_rewards)
                )
                ** 0.5,
                6,
            ),
        },
    }


def main():
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    for task in ["easy", "medium", "hard"]:
        print(f"Capturing golden fixtures for {task}...", file=sys.stderr)
        data = capture_fixtures(task, seed=42, episodes=50)
        out_path = fixtures_dir / f"rubric_golden_{task}.json"
        out_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"  Saved {out_path}", file=sys.stderr)
        print(
            f"  mean_grader_score={data['summary']['mean_grader_score']} "
            f"mean_total_reward={data['summary']['mean_total_reward']}",
            file=sys.stderr,
        )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
