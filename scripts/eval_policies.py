import argparse
import json
import sys
from pathlib import Path

from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import ActionType, SakhaAction

TASK_GRADERS = {
    "easy": score_easy_task,
    "medium": score_medium_task,
    "hard": score_hard_task,
}
PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}


def noop_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def greedy_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=(step % pc) + 1)


def priority_policy(obs, step, pc):
    if obs.ward_state.pending_tasks:
        task = obs.ward_state.pending_tasks[0]
        return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)

    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def timestep_scripted_policy(obs, step, pc):
    if step < 10:
        return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=(step % pc) + 1)
    elif step < 20:
        return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=((step - 5) % pc) + 1)
    elif step < 30:
        return SakhaAction(
            action_type=ActionType.ADMINISTER_MEDICINE, patient_id=((step - 10) % pc) + 1
        )
    else:
        return SakhaAction(action_type=ActionType.NOOP)


POLICIES = {
    "noop": noop_policy,
    "greedy": greedy_policy,
    "priority": priority_policy,
    "timestep_scripted": timestep_scripted_policy,
}


def run_policy(task: str, policy_name: str, seed: int, episodes: int) -> dict:
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]
    policy = POLICIES[policy_name]
    scores = []

    for ep in range(episodes):
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed + ep)
        trajectory = [obs]
        for step in range(96):
            action = policy(obs, step, pc)
            obs = env.step(action)
            trajectory.append(obs)
            if obs.done:
                break
        scores.append(grader(trajectory))

    return {
        "policy": policy_name,
        "mean": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(
            (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
            4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy-a", default="noop")
    parser.add_argument("--policy-b", default="priority")
    parser.add_argument("--all-policies", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if args.all_policies:
        policy_results = {
            name: run_policy(args.task, name, args.seed, args.episodes) for name in POLICIES
        }
        output = {
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "policies": policy_results,
        }
    else:
        result_a = run_policy(args.task, args.policy_a, args.seed, args.episodes)
        result_b = run_policy(args.task, args.policy_b, args.seed, args.episodes)
        output = {
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "policy_a": result_a,
            "policy_b": result_b,
            "gap": round(result_b["mean"] - result_a["mean"], 4),
        }

    print(json.dumps(output, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
