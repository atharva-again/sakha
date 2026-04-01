import argparse
import json
import math
import statistics
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_policies import run_policy

TASKS = ("easy", "medium", "hard")


def _ci95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    std = statistics.pstdev(values)
    return 1.96 * std / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducibility report across seeds")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--policy", default="priority")
    parser.add_argument("--output-json", default="artifacts/reproducibility_report.json")
    args = parser.parse_args()

    runs = []
    for task in TASKS:
        task_seed_means = []
        for i in range(args.seed_count):
            seed = args.base_seed + i
            result = run_policy(task, args.policy, seed, args.episodes, with_details=True)
            task_seed_means.append(result["mean"])
            runs.append(
                {
                    "task": task,
                    "seed": seed,
                    "episodes": args.episodes,
                    "policy": args.policy,
                    "mean": result["mean"],
                    "std": result["std"],
                    "ci95": result["ci95"],
                }
            )

        runs.append(
            {
                "task": task,
                "aggregate": {
                    "policy": args.policy,
                    "seed_count": args.seed_count,
                    "episodes": args.episodes,
                    "mean_of_means": round(statistics.fmean(task_seed_means), 4),
                    "min_mean": round(min(task_seed_means), 4),
                    "max_mean": round(max(task_seed_means), 4),
                    "std_of_means": round(statistics.pstdev(task_seed_means), 4),
                    "ci95_of_means": round(_ci95(task_seed_means), 4),
                },
            }
        )

    payload = {
        "policy": args.policy,
        "base_seed": args.base_seed,
        "seed_count": args.seed_count,
        "episodes": args.episodes,
        "tasks": list(TASKS),
        "results": runs,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
