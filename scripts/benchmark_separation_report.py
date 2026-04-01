import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_policies import run_policy

DEFAULT_POLICIES = [
    "random",
    "noop",
    "greedy",
    "timestep_scripted",
    "vitals_spam",
    "escalation_tunnel",
    "priority",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark policy separation report")
    parser.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--output-json", default="artifacts/benchmark_separation_report.json")
    args = parser.parse_args()

    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    results = [run_policy(args.task, policy, args.seed, args.episodes) for policy in policies]
    ranked = sorted(results, key=lambda item: item["mean"], reverse=True)

    separation = []
    for idx in range(len(ranked) - 1):
        upper = ranked[idx]
        lower = ranked[idx + 1]
        separation.append(
            {
                "higher": upper["policy"],
                "lower": lower["policy"],
                "gap": round(upper["mean"] - lower["mean"], 4),
            }
        )

    payload = {
        "task": args.task,
        "seed": args.seed,
        "episodes": args.episodes,
        "ranked_results": ranked,
        "separation": separation,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
