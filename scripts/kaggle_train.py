"""Sakha GRPO Training — Kaggle GPU Script.

Run this on Kaggle with GPU enabled (P100 or T4, 16GB VRAM).
Copy-paste each cell section into a Kaggle notebook, or run the whole script.

Expected runtime: ~1.5-2 hours for 200 steps on P100.
"""

# =============================================================================
# CELL 1: Clone repo and install dependencies
# =============================================================================
!git clone -b grpo-training https://github.com/atharva-again/sakha.git /kaggle/working/sakha
%cd /kaggle/working/sakha

!pip install --upgrade "transformers>=5.2.0"
!pip install -e .

# =============================================================================
# CELL 2: Start the Sakha environment server in background
# =============================================================================
import os
import subprocess
import sys
import time

server_proc = subprocess.Popen(
    [sys.executable, "server/app.py"],
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
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

from sakha.grpo_env import SakhaToolEnv

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

# =============================================================================
# CELL 4: Build training dataset
# =============================================================================
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
    max_completion_length=1024,  # Lower to avoid regex hang (#5415)
    max_tool_calling_iterations=10,  # Prevent long tool chains
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_steps=200,
    logging_steps=10,
    save_steps=50,
    output_dir="/kaggle/working/sakha_grpo",
    report_to="none",
    chat_template_kwargs={"enable_thinking": False},
    log_completions=True,
    use_vllm=True,
    vllm_mode="server",  # Avoids colocate issue (#5269)
    vllm_gpu_memory_utilization=0.4,
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
trainer.save_model("/kaggle/working/sakha_grpo_final")

# =============================================================================
# CELL 7: Verify output
# =============================================================================
print("Trained model files:")
for f in os.listdir("/kaggle/working/sakha_grpo_final"):
    path = os.path.join("/kaggle/working/sakha_grpo_final", f)
    size = os.path.getsize(path) / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")
