"""GRPO training for Sakha — Kaggle notebook version.

Run this on Kaggle with GPU enabled (P100 or T4).
The Sakha env server must be running in the background.
"""

import os
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trl>=1.0.0", "accelerate"])

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, "/kaggle/working/sakha/src")

from sakha.grpo_env import SakhaToolEnv


def reward_func(environments: list[SakhaToolEnv], **kwargs) -> list[float]:
    return [env.reward for env in environments]


def build_dataset() -> Dataset:
    easy = (
        "You are managing a small hospital ward with 5-8 patients. "
        "Keep patients safe by checking vitals when due, "
        "administering medications on time, and escalating critical patients. "
        "Use the available tools to manage the ward effectively. "
        "Avoid wasting time — every action counts."
    )
    medium = (
        "You are managing a busy hospital ward with 5-10 patients. "
        "Patients arrive throughout your shift. Prioritize care: "
        "check vitals when due, administer medications before deadlines, "
        "escalate deteriorating patients immediately, and document your actions. "
        "Complete a handover before your shift ends. "
        "Avoid wasting time — every action counts."
    )
    hard = (
        "You are managing a critical hospital ward with 8-15 patients. "
        "The ward is understaffed and patients arrive frequently. "
        "Triage carefully: handle deterioration events within the escalation window, "
        "administer all medications on time, check vitals on schedule, "
        "document patient status, and complete a quality handover. "
        "Missed escalations are safety violations. "
        "Avoid wasting time — every action counts."
    )

    prompts = []
    for _ in range(20):
        prompts.append([{"role": "user", "content": easy}])
    for _ in range(20):
        prompts.append([{"role": "user", "content": medium}])
    for _ in range(20):
        prompts.append([{"role": "user", "content": hard}])

    return Dataset.from_dict({"prompt": prompts})


def main():
    env_url = os.environ.get("SAKHA_ENV_URL", "http://localhost:7860")
    os.environ["SAKHA_ENV_URL"] = env_url

    dataset = build_dataset()

    config = GRPOConfig(
        num_generations=4,
        max_completion_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        max_steps=200,
        logging_steps=10,
        save_steps=50,
        output_dir="/kaggle/working/sakha_grpo",
        report_to="none",
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B",
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=config,
        environment_factory=SakhaToolEnv,
    )

    trainer.train()
    trainer.save_model("/kaggle/working/sakha_grpo_final")


if __name__ == "__main__":
    main()
