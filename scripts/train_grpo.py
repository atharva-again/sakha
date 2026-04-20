"""
GRPO Training Script for Sakha Environment

Minimal training script using HF TRL's GRPOTrainer to demonstrate RL training
on the Sakha hospital ward assistant environment.

Usage:
    python scripts/train_grpo.py --model Qwen/Qwen3-0.6B --task hard --episodes 3
    python scripts/train_grpo.py --help

Install dependencies:
    pip install uv
    uv pip install --system "transformers>=5.2.0" "trl[quantization]" peft accelerate bitsandbytes
"""

import argparse
import json
from pathlib import Path


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

    def __init__(self, task: str = "hard"):
        from sakha.env import SakhaEnvironment
        from sakha.models import SakhaAction, ActionType

        self.task = task
        self.env = SakhaEnvironment(task=task)
        self.reward = 0.0
        self._action_type = ActionType

    def reset(self, **kwargs) -> str:
        """Reset the environment. REQUIRED method for TRL.

        Returns observation string that gets appended to the conversation.
        """
        obs = self.env.reset()
        self.reward = 0.0
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

    def review_patient(self, patient_id: int) -> str:
        """Review patient at bed ID."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.REVIEW_PATIENT, patient_id=patient_id)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def administer_medicine(self, patient_id: int) -> str:
        """Administer medication to patient."""
        from sakha.models import SakhaAction

        action = SakhaAction(
            action_type=self._action_type.ADMINISTER_MEDICINE, patient_id=patient_id
        )
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        detail = obs.action_result.detail if obs.action_result else ""
        return f"{self._format_observation(obs)} | {detail}"

    def check_vitals(self, patient_id: int) -> str:
        """Check vitals for patient."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.CHECK_VITALS, patient_id=patient_id)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def alert_doctor(self, patient_id: int) -> str:
        """Alert doctor for patient."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.ALERT_DOCTOR, patient_id=patient_id)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def escalate(self, patient_id: int) -> str:
        """Escalate patient incident."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.ESCALATE, patient_id=patient_id)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def update_chart(self, patient_id: int) -> str:
        """Update patient chart."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.UPDATE_CHART, patient_id=patient_id)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def prepare_discharge(self, patient_id: int) -> str:
        """Prepare patient for discharge."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.PREPARE_DISCHARGE, patient_id=patient_id)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def medication_round(self) -> str:
        """Complete medication round for all patients with due medications."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.MEDICATION_ROUND, patient_id=None)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        detail = obs.action_result.detail if obs.action_result else ""
        return f"{self._format_observation(obs)} | {detail}"

    def ward_sweep(self) -> str:
        """Complete ward sweep and coordination check."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.WARD_SWEEP, patient_id=None)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)

    def noop(self) -> str:
        """Take no action (pass time)."""
        from sakha.models import SakhaAction

        action = SakhaAction(action_type=self._action_type.NOOP, patient_id=None)
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        return self._format_observation(obs)


def reward_func(environments, **kwargs):
    """Reward function for GRPO.

    Must use **kwargs to handle extra parameters from TRL.
    """
    return [getattr(env, "reward", 0.0) for env in environments]


def create_prompt(task: str) -> list[dict]:
    """Create prompt in Messages format for dataset."""
    system_msg = (
        "You are a hospital ward assistant managing patients. "
        "Available actions: review_patient(patient_id), check_vitals(patient_id), "
        "administer_medicine(patient_id), alert_doctor(patient_id), escalate(patient_id), "
        "update_chart(patient_id), prepare_discharge(patient_id), medication_round(), "
        "ward_sweep(), noop(). "
        f"Task difficulty: {task}. "
        "Choose the best action based on pending tasks and patient needs."
    )
    return [{"role": "system", "content": system_msg}]


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Sakha environment")
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
        default=3,
        help="Number of training episodes (default: 3)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Max steps per episode (default: 24 for dev)",
    )
    args = parser.parse_args()

    Dataset, GRPOConfig, GRPOTrainer = _import_trl_deps()

    prompt_template = create_prompt(args.task)
    dataset = Dataset.from_dict({"prompt": [prompt_template] * args.episodes})

    grpo_config = GRPOConfig(
        num_train_epochs=1,
        num_generations=4,
        max_completion_length=512,
        learning_rate=1e-5,
        logging_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=grpo_config,
        environment_factory=lambda: SakhaEnvWrapper(task=args.task),
    )

    print(f"Starting training: {args.episodes} episodes, task={args.task}")
    trainer.train()

    metrics = trainer.evaluate()
    results = {
        "task": args.task,
        "model": args.model,
        "episodes": args.episodes,
        "final_metrics": metrics,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output_json}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
