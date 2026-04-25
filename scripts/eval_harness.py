"""Reproducible evaluation harness for Sakha baseline and trained policies."""

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


def _parse_seed_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in value.split(",")]


def noop_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def greedy_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=(step % pc) + 1)


def priority_policy(obs, step, pc):
    if obs.ward_state.pending_tasks:
        task = obs.ward_state.pending_tasks[0]
        return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def deterministic_policy(obs, step, pc):
    return priority_policy(obs, step, pc)


def trained_policy(obs, step, pc, checkpoint_path: str | None = None):
    raise NotImplementedError(
        "Trained policy requires a valid checkpoint and model loading. "
        "Pass --checkpoint PATH to a saved GRPO model checkpoint."
    )


POLICIES = {
    "noop": noop_policy,
    "greedy": greedy_policy,
    "priority": priority_policy,
    "deterministic": deterministic_policy,
}


def run_single_episode(env: SakhaEnvironment, policy, task: str, seed: int, max_steps: int) -> dict:
    pc = PATIENT_COUNTS[task]
    obs = env.reset(seed=seed)
    step_rewards = []
    for step in range(max_steps):
        action = policy(obs, step, pc)
        obs = env.step(action)
        step_rewards.append(obs.reward)
        if obs.done:
            break

    metrics = env.episode_metrics
    total_reward = sum(step_rewards)
    trajectory = []  # grader needs trajectory; rebuild it
    obs = env.reset(seed=seed)
    trajectory.append(obs)
    for step in range(max_steps):
        action = policy(obs, step, pc)
        obs = env.step(action)
        trajectory.append(obs)
        if obs.done:
            break

    grader = TASK_GRADERS[task]
    grader_score = grader(trajectory)

    return {
        "seed": seed,
        "grader_score": round(grader_score, 6),
        "total_reward": round(total_reward, 6),
        "critical_incidents_resolved": metrics.critical_incidents_resolved,
        "critical_incidents_total": metrics.critical_incidents_total,
        "critical_incidents_missed": metrics.critical_incidents_missed,
        "medication_tasks_completed": metrics.medication_tasks_completed,
        "medication_tasks_on_time": metrics.medication_tasks_on_time,
        "vitals_tasks_completed": metrics.vitals_tasks_completed,
        "vitals_tasks_on_time": metrics.vitals_tasks_on_time,
        "overdue_tasks": metrics.overdue_tasks,
        "noop_steps": metrics.noop_steps,
        "discharges_prepared": metrics.discharges_prepared,
    }


def run_eval(
    task: str,
    policy_name: str,
    seeds: list[int],
    max_steps: int,
    checkpoint: str | None = None,
) -> dict:
    pc = PATIENT_COUNTS[task]

    if policy_name == "trained":
        if not checkpoint:
            print("Error: --checkpoint is required when --policy trained", file=sys.stderr)
            sys.exit(1)
        policy = lambda obs, step, pc: trained_policy(obs, step, pc, checkpoint)
    else:
        policy = POLICIES[policy_name]

    episodes = []
    for seed in seeds:
        env = SakhaEnvironment(patient_count=pc, task=task)
        ep = run_single_episode(env, policy, task, seed, max_steps)
        episodes.append(ep)

    def _mean(key: str) -> float:
        return sum(e[key] for e in episodes) / len(episodes)

    def _std(key: str) -> float:
        m = _mean(key)
        return (sum((e[key] - m) ** 2 for e in episodes) / len(episodes)) ** 0.5

    summary = {
        "mean_grader_score": round(_mean("grader_score"), 6),
        "std_grader_score": round(_std("grader_score"), 6),
        "mean_total_reward": round(_mean("total_reward"), 6),
        "std_total_reward": round(_std("total_reward"), 6),
        "mean_critical_incidents_resolved": round(_mean("critical_incidents_resolved"), 6),
        "mean_critical_incidents_missed": round(_mean("critical_incidents_missed"), 6),
        "mean_overdue_tasks": round(_mean("overdue_tasks"), 6),
        "mean_noop_steps": round(_mean("noop_steps"), 6),
        "mean_discharges_prepared": round(_mean("discharges_prepared"), 6),
    }

    return {
        "task": task,
        "policy": policy_name,
        "max_steps": max_steps,
        "seeds": seeds,
        "episodes": episodes,
        "summary": summary,
    }


def _print_markdown_table(result: dict) -> None:
    print(f"\n## Eval Results: {result['task']} / {result['policy']}\n")
    print("| Metric | Mean | Std |")
    print("|--------|------|-----|")
    s = result["summary"]
    print(f"| Grader Score | {s['mean_grader_score']:.4f} | {s['std_grader_score']:.4f} |")
    print(f"| Total Reward | {s['mean_total_reward']:.4f} | {s['std_total_reward']:.4f} |")
    print(f"| Critical Resolved | {s['mean_critical_incidents_resolved']:.2f} | - |")
    print(f"| Critical Missed | {s['mean_critical_incidents_missed']:.2f} | - |")
    print(f"| Overdue Tasks | {s['mean_overdue_tasks']:.2f} | - |")
    print(f"| NOOP Steps | {s['mean_noop_steps']:.2f} | - |")
    print(f"| Discharges | {s['mean_discharges_prepared']:.2f} | - |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--seeds", default="42-71", help="Seed range, e.g. 42-71 or 42,43,44")
    parser.add_argument(
        "--policy",
        default="priority",
        choices=["deterministic", "priority", "noop", "greedy", "trained"],
    )
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    seeds = _parse_seed_range(args.seeds)
    result = run_eval(
        task=args.task,
        policy_name=args.policy,
        seeds=seeds,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint,
    )

    _print_markdown_table(result)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
