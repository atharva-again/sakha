"""
Sakha GRPO Training Script
1:1 conversion from notebooks/sakha_grpo_training.ipynb

Install compatible dependencies (Colab):
    pip install unsloth vllm datasets
    pip install --upgrade --force-reinstall --no-deps unsloth unsloth_zoo
"""

import sys
import os

# Prevent vLLM from reconfiguring logging (crashes Jupyter/Colab's OutStream)
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import re
import json
import gc
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction, ActionType
from sakha.graders import score_easy_task, score_medium_task, score_hard_task

# ============================================================
# Notebook Cell: 3. Configure Training
# ============================================================
# Training configuration
MODEL = "Qwen/Qwen3-0.6B"          # Model to train (0.6B fits on T4)
TASK = "hard"                      # Task difficulty: easy | medium | hard
EPISODES = 200                       # Number of training episodes
MAX_STEPS = 96                       # Max steps per episode (96 = full 8hr shift)
SEED = 42                            # Random seed for reproducibility

# Unsloth config (set USE_UNSLOTH=True for 4-bit training on T4)
USE_UNSLOTH = True                   # Use Unsloth for memory-efficient training
LOAD_IN_4BIT = True                  # 4-bit quantization (critical for T4)

# GRPO specific
NUM_GENERATIONS = 4                  # Responses per prompt
LEARNING_RATE = 1e-5                 # Learning rate
MAX_COMPLETION_LENGTH = 512          # Max tokens per completion

# Checkpoint directory
CHECKPOINT_DIR = Path("./grpo_output")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training config:")
print(f"  Model: {MODEL}")
print(f"  Task: {TASK}")
print(f"  Episodes: {EPISODES}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Use Unsloth: {USE_UNSLOTH}")
print(f"  4-bit quant: {LOAD_IN_4BIT}")

# ============================================================
# Notebook Cell: 4. Define Environment Wrapper for TRL
# ============================================================

class SakhaEnvWrapper:
    """Environment wrapper for TRL's GRPOTrainer."""

    def __init__(self, task: str = "hard", seed: int | None = None):
        self.task = task
        self.seed = seed
        self.env = SakhaEnvironment(task=task)
        self.reward = 0.0
        self._action_type = ActionType
        self._episode_reward = 0.0
        self._episode_steps = 0

    def reset(self, **kwargs) -> str:
        """Reset the environment."""
        if self.seed is not None:
            import random
            random.seed(self.seed)
        obs = self.env.reset()
        self.reward = 0.0
        self._episode_reward = 0.0
        self._episode_steps = 0
        return self._format_observation(obs)

    def _format_observation(self, obs) -> str:
        """Format observation as string for tool result."""
        pending = obs.ward_state.pending_tasks[:5] if obs.ward_state.pending_tasks else []
        tasks_str = ", ".join(f"{t.required_action}({t.patient_id})" for t in pending) or "No pending tasks"
        return (
            f"Step {obs.ward_state.current_step}, "
            f"Patients: {len(obs.ward_state.patients)}, "
            f"Pending: {obs.pending_count}, "
            f"Tasks: {tasks_str}"
        )

    def _step(self, action) -> str:
        """Execute action and track metrics."""
        obs = self.env.step(action)
        self.reward = obs.reward or 0.0
        self._episode_reward += self.reward
        self._episode_steps += 1
        return obs

    def review_patient(self, patient_id: int) -> str:
        """Review patient at bed ID."""
        action = SakhaAction(action_type=self._action_type.REVIEW_PATIENT, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def administer_medicine(self, patient_id: int) -> str:
        """Administer medication to patient."""
        action = SakhaAction(action_type=self._action_type.ADMINISTER_MEDICINE, patient_id=patient_id)
        obs = self._step(action)
        detail = obs.action_result.detail if obs.action_result else ""
        return f"{self._format_observation(obs)} | {detail}"

    def check_vitals(self, patient_id: int) -> str:
        """Check vitals for patient."""
        action = SakhaAction(action_type=self._action_type.CHECK_VITALS, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def alert_doctor(self, patient_id: int) -> str:
        """Alert doctor for patient."""
        action = SakhaAction(action_type=self._action_type.ALERT_DOCTOR, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def escalate(self, patient_id: int) -> str:
        """Escalate patient incident."""
        action = SakhaAction(action_type=self._action_type.ESCALATE, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def update_chart(self, patient_id: int) -> str:
        """Update patient chart."""
        action = SakhaAction(action_type=self._action_type.UPDATE_CHART, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def prepare_discharge(self, patient_id: int) -> str:
        """Prepare patient for discharge."""
        action = SakhaAction(action_type=self._action_type.PREPARE_DISCHARGE, patient_id=patient_id)
        obs = self._step(action)
        return self._format_observation(obs)

    def medication_round(self) -> str:
        """Complete medication round for all patients with due medications."""
        action = SakhaAction(action_type=self._action_type.MEDICATION_ROUND, patient_id=None)
        obs = self._step(action)
        detail = obs.action_result.detail if obs.action_result else ""
        return f"{self._format_observation(obs)} | {detail}"

    def ward_sweep(self) -> str:
        """Complete ward sweep and coordination check."""
        action = SakhaAction(action_type=self._action_type.WARD_SWEEP, patient_id=None)
        obs = self._step(action)
        return self._format_observation(obs)

    def noop(self) -> str:
        """Take no action (pass time)."""
        action = SakhaAction(action_type=self._action_type.NOOP, patient_id=None)
        obs = self._step(action)
        return self._format_observation(obs)

    def get_metrics(self):
        """Return accumulated episode metrics."""
        return {
            "episode_reward": self._episode_reward,
            "episode_steps": self._episode_steps,
        }

def reward_func(prompts, completions, **kwargs):
    """Reward function for GRPO. TRL passes prompts and completions as kwargs."""
    rewards = []
    for completion in completions:
        content = completion[-1]["content"] if isinstance(completion, list) else completion
        content = str(content).strip().lower()
        # Reward action-like responses, penalize empty/nonsense
        match = re.search(r"(\w+)\s*\(\s*(\d+)?\s*\)", content)
        if match and match.group(1) in ACTION_NAME_MAP:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

# ============================================================
# Notebook Cell: 5. Create Dataset and Configure GRPO
# ============================================================

def create_prompt(task: str, episode_id: int = 0):
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
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Begin your shift. Review the ward and decide your first action."}
    ]

# Build dataset
prompts = [create_prompt(TASK, i) for i in range(EPISODES)]
dataset = Dataset.from_dict({"prompt": prompts})

print(f"Dataset size: {len(dataset)} prompts")

# ============================================================
# Notebook Cell: 6. Pre-Training Evaluation — Base LLM
# ============================================================
# RUN THIS FIRST to avoid Unsloth monkey-patch conflict

# Action name mapping for eval
ACTION_NAME_MAP = {
    "review_patient": ActionType.REVIEW_PATIENT,
    "check_vitals": ActionType.CHECK_VITALS,
    "administer_medicine": ActionType.ADMINISTER_MEDICINE,
    "alert_doctor": ActionType.ALERT_DOCTOR,
    "escalate": ActionType.ESCALATE,
    "update_chart": ActionType.UPDATE_CHART,
    "prepare_discharge": ActionType.PREPARE_DISCHARGE,
    "medication_round": ActionType.MEDICATION_ROUND,
    "ward_sweep": ActionType.WARD_SWEEP,
    "noop": ActionType.NOOP,
}

SYSTEM_PROMPT = (
    "You are a hospital ward assistant managing patients. "
    "Available actions: review_patient(patient_id), check_vitals(patient_id), "
    "administer_medicine(patient_id), alert_doctor(patient_id), escalate(patient_id), "
    "update_chart(patient_id), prepare_discharge(patient_id), medication_round(), "
    "ward_sweep(), noop(). "
    "Choose the best action based on pending tasks and patient needs."
)

def build_eval_prompt(obs):
    pending = obs.ward_state.pending_tasks[:5] if obs.ward_state.pending_tasks else []
    tasks_str = ", ".join(f"{t.required_action}({t.patient_id})" for t in pending) or "No pending tasks"
    return (
        f"Step {obs.ward_state.current_step}. "
        f"Patients: {len(obs.ward_state.patients)}. "
        f"Pending: {obs.pending_count}. "
        f"Tasks: {tasks_str}. "
        f"What action do you take?"
    )

def parse_action_response(response):
    response = response.strip().lower()
    match = re.search(r"(\w+)\s*\(\s*(\d+)?\s*\)", response)
    if match:
        name = match.group(1)
        patient_id = int(match.group(2)) if match.group(2) else None
        if name in ACTION_NAME_MAP:
            return SakhaAction(action_type=ACTION_NAME_MAP[name], patient_id=patient_id)
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)

def save_eval_cache(results, cache_path):
    """Save evaluation results to JSON for resume across sessions."""
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Eval results saved to {cache_path}")

def load_eval_cache(cache_path):
    """Load cached evaluation results if available."""
    if cache_path.exists():
        with open(cache_path, "r") as f:
            results = json.load(f)
        print(f"⚡ Loaded cached eval results from {cache_path}")
        return results
    return None

def run_llm_eval_batched(task, model, tokenizer, seeds, max_steps):
    """Vectorized evaluation: runs all seeds in parallel per step for ~5x speedup."""
    from sakha.graders import score_easy_task, score_medium_task, score_hard_task
    device = next(model.parameters()).device
    print(f"Running batched eval on device: {device} | {len(seeds)} seeds × {max_steps} steps")

    TASK_GRADERS = {"easy": score_easy_task, "medium": score_medium_task, "hard": score_hard_task}
    PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]

    # Initialize all environments at once
    envs = []
    observations = []
    trajectories = []
    step_rewards_all = []
    for seed in seeds:
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed)
        envs.append(env)
        observations.append(obs)
        trajectories.append([obs])
        step_rewards_all.append([])

    active_mask = [True] * len(seeds)
    pbar = tqdm(total=max_steps, desc=f"Eval {task} (batched, {len(seeds)} seeds)")

    for step in range(max_steps):
        # Collect indices of environments that are still running
        active_indices = [i for i, active in enumerate(active_mask) if active]
        if not active_indices:
            pbar.update(max_steps - step)
            break

        # Build prompts for all active environments
        prompt_texts = []
        for i in active_indices:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_eval_prompt(observations[i])},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # Batch tokenize with left-padding
        inputs = tokenizer(
            prompt_texts, return_tensors="pt", padding=True
        ).to(device)

        # Single batched generation call
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Parse responses and step environments
        input_lens = inputs["attention_mask"].sum(dim=1)
        newly_done = []
        for batch_idx, env_idx in enumerate(active_indices):
            input_len = input_lens[batch_idx].item()
            response = tokenizer.decode(
                outputs[batch_idx][input_len:], skip_special_tokens=True
            )
            action = parse_action_response(response)
            obs = envs[env_idx].step(action)
            observations[env_idx] = obs
            trajectories[env_idx].append(obs)
            step_rewards_all[env_idx].append(obs.reward)

            if obs.done:
                active_mask[env_idx] = False
                newly_done.append(env_idx)

        # Log
        active_count = sum(active_mask)
        sample_idx = active_indices[0]
        sample_action = parse_action_response(
            tokenizer.decode(outputs[0][input_lens[0].item():], skip_special_tokens=True)
        )
        pbar.set_postfix({
            "active": f"{active_count}/{len(seeds)}",
            "action": sample_action.action_type.name,
        })
        for idx in newly_done:
            pbar.write(f"  🏁 Seed {seeds[idx]} finished at step {step}")
        pbar.update(1)

    pbar.close()

    # Compute per-episode results
    episodes = []
    for i, seed in enumerate(seeds):
        grader_score = grader(trajectories[i])
        metrics = envs[i].episode_metrics
        ep = {
            "seed": seed, "grader_score": grader_score,
            "total_reward": sum(step_rewards_all[i]),
            "critical_incidents_resolved": metrics.critical_incidents_resolved,
            "critical_incidents_missed": metrics.critical_incidents_missed,
            "overdue_tasks": metrics.overdue_tasks,
            "noop_steps": metrics.noop_steps,
            "discharges_prepared": metrics.discharges_prepared,
        }
        episodes.append(ep)
        print(f"  Seed {seed:3}: Reward {ep['total_reward']:6.2f} | Grader {grader_score:.4f} | Steps {len(trajectories[i])-1}")

    def mean(key):
        return sum(e[key] for e in episodes) / len(episodes)
    return {
        "task": task, "policy": "base_llm", "episodes": episodes,
        "summary": {
            "mean_grader_score": mean("grader_score"),
            "mean_total_reward": mean("total_reward"),
            "mean_critical_incidents_resolved": mean("critical_incidents_resolved"),
            "mean_critical_incidents_missed": mean("critical_incidents_missed"),
            "mean_overdue_tasks": mean("overdue_tasks"),
            "mean_noop_steps": mean("noop_steps"),
            "mean_discharges_prepared": mean("discharges_prepared"),
        }
    }

# --- Base Eval: Check cache first, then run if needed ---
BASE_CACHE = CHECKPOINT_DIR / "base_eval_cache.json"
base_results = load_eval_cache(BASE_CACHE)

if base_results is None:
    print("Loading base Qwen3-0.6B for zero-shot evaluation...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, padding_side="left"
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    EVAL_SEEDS = list(range(42, 52))
    base_results = run_llm_eval_batched(TASK, base_model, base_tokenizer, EVAL_SEEDS, MAX_STEPS)
    save_eval_cache(base_results, BASE_CACHE)

    # Free VRAM for training
    del base_model
    del base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
else:
    EVAL_SEEDS = list(range(42, 52))

print(f"Base LLM mean grader score: {base_results['summary']['mean_grader_score']:.4f}")

# ============================================================
# Notebook Cell: 7. Load Model with Unsloth (Optional)
# ============================================================

if USE_UNSLOTH:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_COMPLETION_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=True,
        gpu_memory_utilization=0.7,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    print(f"Unsloth model loaded for training: {MODEL}")
else:
    model = MODEL
    tokenizer = None

# ============================================================
# Notebook Cell: 8. Configure GRPO (cont.)
# ============================================================

from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    num_train_epochs=1,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    learning_rate=LEARNING_RATE,
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    report_to=[],
    output_dir="./grpo_output",
    seed=SEED,
    remove_unused_columns=False,
    use_vllm=USE_UNSLOTH,
    vllm_gpu_memory_utilization=0.3 if USE_UNSLOTH else None,
)

# ============================================================
# Notebook Cell: 9. Trainer Setup & Training
# ============================================================

trainer_kwargs = {
    "train_dataset": dataset,
    "reward_funcs": reward_func,
    "args": grpo_config,
    "environment_factory": lambda: SakhaEnvWrapper(task=TASK, seed=SEED),
}

if USE_UNSLOTH:
    trainer_kwargs["model"] = model
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["model"] = MODEL

trainer = GRPOTrainer(**trainer_kwargs)
print(f"Starting training: {EPISODES} episodes, task={TASK}")
trainer.train()

# ============================================================
# Notebook Cell: 10. Post-Training Evaluation — Trained LLM
# ============================================================

TRAINED_CACHE = CHECKPOINT_DIR / "trained_eval_cache.json"
trained_results = load_eval_cache(TRAINED_CACHE)

if trained_results is None:
    from peft import PeftModel
    ckpts = sorted(Path("./grpo_output").glob("checkpoint-*"))
    if not ckpts:
        print("No checkpoints found. Skipping trained eval.")
        trained_results = None
    else:
        CHECKPOINT_PATH = str(ckpts[-1])
        print(f"Loading trained checkpoint: {CHECKPOINT_PATH}")

        if USE_UNSLOTH:
            from unsloth import FastLanguageModel
            trained_model, trained_tokenizer = FastLanguageModel.from_pretrained(
                model_name=CHECKPOINT_PATH,
                max_seq_length=MAX_COMPLETION_LENGTH,
                load_in_4bit=LOAD_IN_4BIT,
                fast_inference=True,
                gpu_memory_utilization=0.7,
            )
            FastLanguageModel.for_inference(trained_model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            trained_tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
            base_for_trained = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            trained_model = PeftModel.from_pretrained(base_for_trained, CHECKPOINT_PATH)
            trained_model = trained_model.merge_and_unload()
            trained_model.eval()

        if trained_tokenizer.pad_token is None:
            trained_tokenizer.pad_token = trained_tokenizer.eos_token

        print("Running evaluation on trained model...")
        trained_results = run_llm_eval_batched(TASK, trained_model, trained_tokenizer, EVAL_SEEDS, MAX_STEPS)
        trained_results["policy"] = "trained_llm"
        save_eval_cache(trained_results, TRAINED_CACHE)
        print(f"Trained mean grader score: {trained_results['summary']['mean_grader_score']:.4f}")

# ============================================================
# Notebook Cell: 11. Base vs Trained Comparison
# ============================================================
if trained_results is not None:
    b = base_results["summary"]
    t = trained_results["summary"]

    print("\n## Base LLM vs Trained LLM Comparison\n")
    print("| Metric | Base LLM | Trained | Delta |")
    print("|--------|----------|---------|-------|")
    print(f"| Grader Score | {b['mean_grader_score']:.4f} | {t['mean_grader_score']:.4f} | {t['mean_grader_score'] - b['mean_grader_score']:+.4f} |")
    print(f"| Total Reward | {b['mean_total_reward']:.4f} | {t['mean_total_reward']:.4f} | {t['mean_total_reward'] - b['mean_total_reward']:+.4f} |")
    print(f"| Critical Resolved | {b['mean_critical_incidents_resolved']:.2f} | {t['mean_critical_incidents_resolved']:.2f} | {t['mean_critical_incidents_resolved'] - b['mean_critical_incidents_resolved']:+.2f} |")
    print(f"| Critical Missed | {b['mean_critical_incidents_missed']:.2f} | {t['mean_critical_incidents_missed']:.2f} | {t['mean_critical_incidents_missed'] - b['mean_critical_incidents_missed']:+.2f} |")
    print(f"| Overdue Tasks | {b['mean_overdue_tasks']:.2f} | {t['mean_overdue_tasks']:.2f} | {t['mean_overdue_tasks'] - b['mean_overdue_tasks']:+.2f} |")
    print(f"| NOOP Steps | {b['mean_noop_steps']:.2f} | {t['mean_noop_steps']:.2f} | {t['mean_noop_steps'] - b['mean_noop_steps']:+.2f} |")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["Grader Score", "Total Reward", "Critical Resolved", "Overdue Tasks"]
    base_vals = [b['mean_grader_score'], b['mean_total_reward'], b['mean_critical_incidents_resolved'], b['mean_overdue_tasks']]
    trained_vals = [t['mean_grader_score'], t['mean_total_reward'], t['mean_critical_incidents_resolved'], t['mean_overdue_tasks']]
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], base_vals, width, label='Base LLM', color='#e74c3c')
    ax.bar([i + width/2 for i in x], trained_vals, width, label='Trained LLM', color='#2ecc71')
    ax.set_ylabel('Value')
    ax.set_title('Base vs Trained LLM Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./grpo_output/base_vs_trained.png", dpi=150)
    print("Plot saved to ./grpo_output/base_vs_trained.png")
else:
    print("Skipping comparison — no trained eval results available.")
