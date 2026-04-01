"""Sakha GRPO Training — Modal.

Run with: modal run modal_train.py
Requires: modal setup (auth) and $30 free credit.
"""

import modal

app = modal.App("sakha-grpo-training")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=/root/.local/bin:$PATH && "
        "uv pip install --system "
        '"git+https://github.com/meta-pytorch/OpenEnv.git" '
        '"pydantic>=2.0" "fastapi>=0.100" "uvicorn>=0.20" '
        '"openai>=1.0" "python-dotenv>=1.0" "tenacity>=8.0" '
        '"torch>=2.5" "accelerate>=1.13" "datasets>=3.0" '
        '"trl>=1.0.0" "jmespath" '
        '"transformers==5.4.0" && '
        "git clone -b grpo-training https://github.com/atharva-again/sakha.git /root/sakha && "
        "uv pip install --system -e /root/sakha",
        env={"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"},
    )
    .add_local_python_source("sakha", ignore=["*.pyc", "__pycache__"])
)

model_vol = modal.Volume.from_name("sakha-models", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="T4",
    timeout=60 * 60 * 4,
    volumes={"/models": model_vol},
)
def train():
    import os
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/root/sakha/src")

    server_proc = subprocess.Popen(
        [sys.executable, "/root/sakha/server/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(10)
    print("Server started, waiting for it to be ready...")

    os.environ["SAKHA_ENV_URL"] = "http://localhost:7860"
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer

    from sakha.grpo_env import SakhaToolEnv

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

    SYSTEM_PROMPT = (
        "You are a hospital ward assistant managing patient care. "
        "Use the available tools to interact with patients. "
        "Always use the appropriate tool when action is needed. "
        "Prioritize safety: escalate critical patients immediately, "
        "administer medications before deadlines, and check vitals on schedule. "
        "Only act on admitted patients. "
        "Complete a handover before your shift ends."
    )

    easy = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "You are managing a small hospital ward with 5 patients. "
         "Your task: Keep patients safe by administering medications on time and checking vitals when due. "
         "Complete your shift by ensuring all medications are given and vitals checked."},
    ]
    medium = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "You are managing a busy hospital ward with 5-10 patients. "
         "Patients arrive throughout your shift. Prioritize care: "
         "check vitals when due, administer medications before deadlines, "
         "escalate deteriorating patients immediately, and document your actions. "
         "Complete a handover before your shift ends."},
    ]
    hard = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "You are managing a critical hospital ward with 8-15 patients. "
         "The ward is understaffed and patients arrive frequently. "
         "Triage carefully: handle deterioration events within the escalation window, "
         "administer all medications on time, check vitals on schedule, "
         "document patient status, and complete a quality handover. "
         "Missed escalations are safety violations."},
    ]

    prompts = []
    for _ in range(20):
        prompts.append(easy)
    for _ in range(20):
        prompts.append(medium)
    for _ in range(20):
        prompts.append(hard)

    dataset = Dataset.from_dict({"prompt": prompts})

    def reward_func(environments, **kwargs):
        return [env.reward for env in environments]

    config = GRPOConfig(
        num_generations=4,
        max_completion_length=1024,
        max_tool_calling_iterations=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        max_steps=200,
        logging_steps=10,
        save_steps=50,
        output_dir="/models/sakha_grpo",
        report_to="none",
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen3.5-4B",
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=config,
        environment_factory=SakhaToolEnv,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("/models/sakha_grpo_final")
    model_vol.commit()

    print("Training complete. Model saved to /models/sakha_grpo_final")


@app.local_entrypoint()
def main():
    train.remote()
