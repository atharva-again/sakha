"""Evaluate trained GRPO model against baseline policies.

Usage:
    uv run python scripts/eval_grpo.py --model ./output/sakha_grpo/checkpoint-200
"""

import argparse
import json

from sakha.client import SakhaEnv
from sakha.graders import compute_diagnostic_breakdown
from sakha.models import SakhaAction, SakhaObservation


def run_episode(env: SakhaEnv, model_client, seed: int) -> list[SakhaObservation]:
    obs = env.reset(seed=seed)
    trajectory = [obs]
    while not obs.done:
        action = model_client.choose_action(obs)
        if action is None:
            action = SakhaAction(action_type="noop")
        obs = env.step(action)
        trajectory.append(obs)
    return trajectory


def evaluate_policy(name: str, model_client, task: str, episodes: int = 5):
    env = SakhaEnv(base_url="http://localhost:7860")
    scores = []
    for i in range(episodes):
        trajectory = run_episode(env, model_client, seed=42 + i)
        breakdown = compute_diagnostic_breakdown(trajectory)
        score = breakdown[f"{task}_score"]
        scores.append(score)
        print(f"  Episode {i + 1}: score={score:.4f}")
    mean_score = sum(scores) / len(scores)
    print(f"  {name} on {task}: mean={mean_score:.4f} (n={episodes})")
    return {"policy": name, "task": task, "mean": round(mean_score, 4), "scores": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    print(f"Evaluating model: {args.model}")
    print(f"Tasks: {tasks}")
    print(f"Episodes per task: {args.episodes}")
    print()

    results = []
    for task in tasks:
        print(f"--- Task: {task} ---")
        results.append(evaluate_policy("trained_grpo", None, task, args.episodes))
        print()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
