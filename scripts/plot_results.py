"""Generate training evidence plots from baseline and training artifacts."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline(task: str) -> dict | None:
    path = ARTIFACTS_DIR / f"baseline_{task}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_training_results() -> list[dict]:
    results = []
    grpo_dir = ARTIFACTS_DIR / "grpo"
    if not grpo_dir.exists():
        return results
    for subdir in grpo_dir.iterdir():
        if subdir.is_dir():
            result_file = subdir / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    if data.get("status") != "not_run":
                        results.append(data)
    return results


def plot_reward_curve(baselines: dict[str, dict], training: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    if training:
        episodes = list(range(1, len(training) + 1))
        rewards = [r.get("mean_reward", 0) for r in training]
        ax.plot(episodes, rewards, "o-", color="green", linewidth=2, label="Training Reward")
        ax.fill_between(
            episodes,
            [r - 0.1 for r in rewards],
            [r + 0.1 for r in rewards],
            alpha=0.2,
            color="green",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No training data available\nRun training to generate reward curves",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )

    ax.set_xlabel("Training Episode / Checkpoint")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Training Reward Curve")
    ax.grid(True, alpha=0.3)
    if training:
        ax.legend()
    fig.savefig(PLOTS_DIR / "reward_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grader_score_curve(baselines: dict[str, dict], training: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    if training:
        episodes = list(range(1, len(training) + 1))
        scores = [r.get("mean_grader_score", 0) for r in training]
        ax.plot(episodes, scores, "o-", color="blue", linewidth=2, label="Grader Score")
        ax.fill_between(
            episodes,
            [max(0, s - 0.05) for s in scores],
            [min(1, s + 0.05) for s in scores],
            alpha=0.2,
            color="blue",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No training data available\nRun training to generate grader score curves",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )

    ax.set_xlabel("Training Episode / Checkpoint")
    ax.set_ylabel("Mean Grader Score")
    ax.set_title("Training Grader Score Curve")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if training:
        ax.legend()
    fig.savefig(PLOTS_DIR / "grader_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_before_after(baselines: dict[str, dict]) -> None:
    tasks = ["easy", "medium", "hard"]
    baseline_scores = []
    trained_scores = []
    labels = []

    for task in tasks:
        if task in baselines:
            baseline_scores.append(baselines[task]["summary"]["mean_grader_score"])
            trained_scores.append(0.0)  # placeholder
            labels.append(task.capitalize())

    if not labels:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No baseline data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        fig.savefig(PLOTS_DIR / "before_after.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, baseline_scores, width, label="Baseline", color="#e74c3c")
    bars2 = ax.bar(x + width / 2, trained_scores, width, label="Trained (TBD)", color="#2ecc71")

    ax.set_ylabel("Mean Grader Score")
    ax.set_title("Before vs After: Baseline vs Trained Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(PLOTS_DIR / "before_after.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_comparison(baselines: dict[str, dict]) -> None:
    tasks = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        if task in baselines:
            episodes = baselines[task]["episodes"]
            scores = [e["grader_score"] for e in episodes]
            rewards = [e["total_reward"] for e in episodes]

            ax.scatter(scores, rewards, alpha=0.6, s=50)
            ax.set_xlabel("Grader Score")
            ax.set_ylabel("Total Reward")
            ax.set_title(f"{task.capitalize()} Task")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
            ax.set_title(f"{task.capitalize()} Task")

    fig.suptitle("Per-Task Baseline Distribution: Grader Score vs Total Reward")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "per_task_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading baseline data...", file=sys.stderr)
    baselines = {}
    for task in ["easy", "medium", "hard"]:
        data = load_baseline(task)
        if data:
            baselines[task] = data
            print(f"  {task}: {len(data['episodes'])} episodes", file=sys.stderr)

    print("Loading training data...", file=sys.stderr)
    training = load_training_results()
    print(f"  Found {len(training)} training result files", file=sys.stderr)

    print("Generating plots...", file=sys.stderr)
    plot_reward_curve(baselines, training)
    plot_grader_score_curve(baselines, training)
    plot_before_after(baselines)
    plot_per_task_comparison(baselines)

    for png in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {png.name}: {png.stat().st_size / 1024:.1f} KB", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
