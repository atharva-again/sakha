"""
GRPO Training Script for Sakha Environment

Training script using HF TRL's GRPOTrainer for RL training on the Sakha
hospital ward assistant environment.

Usage:
    python scripts/train_grpo.py --mode smoke
    python scripts/train_grpo.py --mode demo --task hard
    python scripts/train_grpo.py --mode train --task hard --episodes 5000
    python scripts/train_grpo.py --help

Install dependencies:
    pip install uv
    uv pip install --system "transformers>=5.2.0" "trl[quantization]" peft accelerate bitsandbytes
"""

import argparse
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


MODE_PRESETS = {
    "smoke": {
        "episodes": 3,
        "eval_fraction": 0.0,
        "save_strategy": "no",
        "logging_steps": 1,
        "report_to": [],
    },
    "demo": {
        "episodes": 200,
        "eval_fraction": 0.1,
        "save_strategy": "steps",
        "save_steps": 50,
        "logging_steps": 5,
        "report_to": ["wandb"],
    },
    "train": {
        "episodes": 5000,
        "eval_fraction": 0.15,
        "save_strategy": "steps",
        "save_steps": 100,
        "logging_steps": 10,
        "report_to": ["wandb"],
    },
}


def _import_trl_deps():
    """Lazy import to allow wrapper testing without full TRL installed."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    return Dataset, GRPOConfig, GRPOTrainer


class SakhaEnvWrapper:
    """Environment wrapper for TRL's GRPOTrainer.

    TRL exposes public methods as tools - each method becomes a callable action.
    The environment must have:
    - reset() method (REQUIRED)
    - reward attribute (updated after each action)
    """

    def __init__(self, task: str = "hard", seed: int | None = None):
        from sakha.env import SakhaEnvironment
        from sakha.models import SakhaAction, ActionType

        self.task = task
        self.seed = seed
        self.env = SakhaEnvironment(task=task)
        self.reward = 0.0
        self._action_type = ActionType
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._metrics: dict[str, Any] = {}

    def reset(self, **kwargs) -> str:
        """Reset the environment. REQUIRED method for TRL.

        Returns observation string that gets appended to the conversation.
        """
        if self.seed is not None:
            random.seed(self.seed)
        obs = self.env.reset()
        self.reward = 0.0
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._metrics = {
            "actions_taken": 0,
            "critical_events": 0,
            "routine_completed": 0,
        }
        return self._format_observation(obs)

    def _format_observation(self, obs) -> str:
        """Format observation as string for tool result."""
        pending = obs.ward_state.pending_tasks[:5] if obs.ward_state.pending_tasks else []
        tasks_str = (
            ", ".join(f"{t.required_action}({t.patient_id})" for t in pending) or "No pending tasks"
        )
        return (
            f"Step {obs.ward_state.current_step}, "
            f"Patients: {len(obs.ward_state.patients)}, "
            f"Pending: {obs.pending_count}, "
            f"Tasks: {tasks_str}"
        )

    def _step(self, action) -> str:
        """Execute action and track metrics."""
        from sakha.models import SakhaAction

        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        self._episode_reward += self.reward
        self._episode_steps += 1
        self._metrics["actions_taken"] += 1

        if obs.action_result:
            if "critical" in (obs.action_result.detail or "").lower():
                self._metrics["critical_events"] += 1
            if "completed" in (obs.action_result.detail or "").lower():
                self._metrics["routine_completed"] += 1

        return obs

    def review_patient(self, patient_id: int) -> str:
        """Review patient at bed ID.

        Args:
            patient_id: The bed number of the patient to review.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.REVIEW_PATIENT, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def administer_medicine(self, patient_id: int) -> str:
        """Administer medication to patient.

        Args:
            patient_id: The bed number of the patient to administer medicine to.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(
            action_type=self._action_type.ADMINISTER_MEDICINE, patient_id=patient_id
        )
        obs = self._step(action)
        detail = obs.action_result.detail if obs.action_result else ""
        return f"{self._format_observation(obs)} | {detail}"

    def check_vitals(self, patient_id: int) -> str:
        """Check vitals for patient.

        Args:
            patient_id: The bed number of the patient to check vitals for.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.CHECK_VITALS, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def alert_doctor(self, patient_id: int) -> str:
        """Alert doctor for patient.

        Args:
            patient_id: The bed number of the patient to alert the doctor about.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.ALERT_DOCTOR, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def escalate(self, patient_id: int) -> str:
        """Escalate patient incident.

        Args:
            patient_id: The bed number of the patient to escalate.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.ESCALATE, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def update_chart(self, patient_id: int) -> str:
        """Update patient chart.

        Args:
            patient_id: The bed number of the patient whose chart to update.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.UPDATE_CHART, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def prepare_discharge(self, patient_id: int) -> str:
        """Prepare patient for discharge.

        Args:
            patient_id: The bed number of the patient to prepare for discharge.
        """
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.PREPARE_DISCHARGE, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def medication_round(self) -> str:
        """Complete medication round for all patients with due medications."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.MEDICATION_ROUND, patient_id=None)
        obs = self._step(action)
        detail = obs.action_result.detail if obs.action_result else ""
        return f"{self._format_observation(obs)} | {detail}"

    def ward_sweep(self) -> str:
        """Complete ward sweep and coordination check."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.WARD_SWEEP, patient_id=None)
        obs = self._step(action)
        return self._format_observation(obs)

    def noop(self) -> str:
        """Take no action (pass time)."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.NOOP, patient_id=None)
        obs = self._step(action)
        return self._format_observation(obs)

    def get_metrics(self) -> dict[str, Any]:
        """Return accumulated episode metrics."""
        return {
            "episode_reward": self._episode_reward,
            "episode_steps": self._episode_steps,
            **self._metrics,
        }


def reward_func(environments, **kwargs):
    """Reward function for GRPO.

    Must use **kwargs to handle extra parameters from TRL.
    """
    return [getattr(env, "reward", 0.0) for env in environments]


def create_prompt(task: str, episode_id: int = 0) -> list[dict]:
    """Create prompt in Messages format for dataset."""
    system_msg = (
        "You are a hospital ward assistant managing patients. "
        "Available actions: review_patient(patient_id), check_vitals(patient_id), "
        "administer_medicine(patient_id), alert_doctor(patient_id), escalate(patient_id), "
        "update_chart(patient_id), prepare_discharge(patient_id), medication_round(), "
        "ward_sweep(), noop(). "
        f"Task difficulty: {task}. "
        f"Episode: {episode_id}. "
        "Choose the best action based on pending tasks and patient needs."
    )
    return [{"role": "system", "content": system_msg}]


def get_git_sha() -> str:
    """Get current git SHA for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_metrics(metrics: list[dict], output_path: Path) -> None:
    """Save metrics to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Sakha environment")
    parser.add_argument(
        "--mode",
        default="smoke",
        choices=["smoke", "demo", "train"],
        help="Training mode: smoke (3 eps, no eval), demo (200 eps, eval), train (5000 eps, full).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model to train (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--task",
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Task difficulty (default: hard)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episode count (default: from mode preset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/grpo"),
        help="Output directory for checkpoints and metrics (default: artifacts/grpo)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=96,
        help="Max steps per episode (default: 96 for full shift)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--report-to",
        nargs="+",
        default=None,
        help="Override reporting targets (e.g., wandb tensorboard). Default: from mode preset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output-dir",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for GRPO training (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size for training (default: 4)",
    )
    args = parser.parse_args()

    preset = MODE_PRESETS[args.mode]
    episodes = args.episodes if args.episodes is not None else preset["episodes"]
    eval_fraction = preset["eval_fraction"]
    report_to = args.report_to if args.report_to is not None else preset["report_to"]

    Dataset, GRPOConfig, GRPOTrainer = _import_trl_deps()

    prompts = [create_prompt(args.task, i) for i in range(episodes)]
    dataset = Dataset.from_dict({"prompt": prompts})

    if eval_fraction > 0 and episodes > 1:
        split = dataset.train_test_split(test_size=eval_fraction, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sakha_grpo_{args.mode}_{args.task}_{timestamp}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "mode": args.mode,
        "model": args.model,
        "task": args.task,
        "episodes": episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "eval_fraction": eval_fraction,
        "git_sha": get_git_sha(),
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    resume_checkpoint = None
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {resume_checkpoint}")

    grpo_config = GRPOConfig(
        num_train_epochs=1,
        num_generations=4,
        max_completion_length=512,
        learning_rate=args.learning_rate,
        logging_steps=preset["logging_steps"],
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        save_strategy=preset["save_strategy"],
        save_steps=preset.get("save_steps", 500),
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=preset.get("save_steps", 500) if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_reward",
        greater_is_better=True,
        report_to=report_to,
        run_name=run_name,
        output_dir=str(output_dir),
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_func,
        args=grpo_config,
        environment_factory=lambda: SakhaEnvWrapper(task=args.task, seed=args.seed),
    )

    print(f"Starting training: mode={args.mode}, episodes={episodes}, task={args.task}")
    print(f"Output directory: {output_dir}")
    if eval_dataset is not None:
        print(f"Train episodes: {len(train_dataset)}, Eval episodes: {len(eval_dataset)}")

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    metrics = trainer.evaluate() if eval_dataset is not None else {}

    results = {
        "run_name": run_name,
        "task": args.task,
        "model": args.model,
        "mode": args.mode,
        "episodes": episodes,
        "max_steps": args.max_steps,
        "final_metrics": metrics,
        "config": config,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
