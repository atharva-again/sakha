"""Sakha GRPO Training — Kaggle GPU Script.

Run this on Kaggle with GPU enabled (P100 or T4, 16GB VRAM).
Copy-paste each cell section into a Kaggle notebook, or run the whole script.

Expected runtime: ~1.5-2 hours for 200 steps on P100.
"""

# =============================================================================
# CELL 1: Clone repo and install dependencies
# =============================================================================
# !git clone https://github.com/atharva-again/sakha.git /kaggle/working/sakha
# %cd /kaggle/working/sakha
# !pip install -q trl>=1.0.0 accelerate

import os
import subprocess
import sys
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trl>=1.0.0", "accelerate"])

# =============================================================================
# CELL 2: Start the Sakha environment server in background
# =============================================================================
sys.path.insert(0, "/kaggle/working/sakha/src")

server_proc = subprocess.Popen(
    [sys.executable, "server/app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(5)

import urllib.request

try:
    urllib.request.urlopen("http://localhost:7860/health")
    print("Server is running on port 7860")
except Exception as e:
    print(f"Server not ready: {e}")

# =============================================================================
# CELL 3: Import training components
# =============================================================================
os.environ["SAKHA_ENV_URL"] = "http://localhost:7860"

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from sakha.grpo_env import SakhaToolEnv

# =============================================================================
# CELL 4: Build training dataset
# =============================================================================
easy = (
    "You are managing a small hospital ward with 5-8 patients. "
    "Keep patients safe by checking vitals when due, "
    "administering medications on time, and escalating critical patients. "
    "Use the available tools to manage the ward effectively. "
    "Avoid wasting time \u2014 every action counts."
)
medium = (
    "You are managing a busy hospital ward with 5-10 patients. "
    "Patients arrive throughout your shift. Prioritize care: "
    "check vitals when due, administer medications before deadlines, "
    "escalate deteriorating patients immediately, and document your actions. "
    "Complete a handover before your shift ends. "
    "Avoid wasting time \u2014 every action counts."
)
hard = (
    "You are managing a critical hospital ward with 8-15 patients. "
    "The ward is understaffed and patients arrive frequently. "
    "Triage carefully: handle deterioration events within the escalation window, "
    "administer all medications on time, check vitals on schedule, "
    "document patient status, and complete a quality handover. "
    "Missed escalations are safety violations. "
    "Avoid wasting time \u2014 every action counts."
)

prompts = []
for _ in range(20):
    prompts.append([{"role": "user", "content": easy}])
for _ in range(20):
    prompts.append([{"role": "user", "content": medium}])
for _ in range(20):
    prompts.append([{"role": "user", "content": hard}])

dataset = Dataset.from_dict({"prompt": prompts})
print(f"Dataset size: {len(dataset)} prompts")


# =============================================================================
# CELL 5: Define reward function
# =============================================================================
def reward_func(environments: list[SakhaToolEnv], **kwargs) -> list[float]:
    return [env.reward for env in environments]


# =============================================================================
# CELL 6: Configure and run training
# =============================================================================
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

# =============================================================================
# CELL 7: Verify output
# =============================================================================
print("Trained model files:")
for f in os.listdir("/kaggle/working/sakha_grpo_final"):
    path = os.path.join("/kaggle/working/sakha_grpo_final", f)
    size = os.path.getsize(path) / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")
