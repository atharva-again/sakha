"""
Sakha GRPO Training Script
1:1 conversion from notebooks/sakha_grpo_training.ipynb

Install compatible dependencies (Colab):
    pip install unsloth vllm datasets
    pip install --upgrade --force-reinstall --no-deps unsloth unsloth_zoo
"""

import os

# Prevent vLLM from reconfiguring logging (crashes Jupyter/Colab's OutStream)
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import re
import json
import gc
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sakha.env import SakhaEnvironment
from sakha.grpo_training import (
    ACTION_NAME_MAP,
    build_grpo_prompt,
    build_state_aligned_examples,
    score_completion_action,
)
from sakha.models import SakhaAction, ActionType
from sakha.graders import score_easy_task, score_medium_task, score_hard_task

# ============================================================
# Notebook Cell: 3. Configure Training
# ============================================================
# Training configuration
MODEL = "Qwen/Qwen3-0.6B"          # Model to train (0.6B fits on T4)
TASK = "hard"                      # Task difficulty: easy | medium | hard
EPISODES = 200                       # Training examples (seed × state × policy). 200 ≈ 50 optimizer steps.
MAX_STEPS = 96                       # Max steps per episode (96 = full 8hr shift)
SEED = 42                            # Random seed for reproducibility

# Unsloth config (set USE_UNSLOTH=True for 4-bit training on T4)
USE_UNSLOTH = True                   # Use Unsloth for memory-efficient training
LOAD_IN_4BIT = True                  # 4-bit quantization (critical for T4)

# GRPO specific
NUM_GENERATIONS = 4                  # Responses per prompt
LEARNING_RATE = 1e-5                 # Learning rate
MAX_COMPLETION_LENGTH = 512          # Max tokens per completion
MAX_SEQ_LENGTH = 2048                # Prompt + completion context for vLLM/Unsloth

# Checkpoint directory.
# Set SAKHA_OUTPUT_DIR (e.g. "/content/drive/MyDrive/sakha_grpo" in Colab once you've
# mounted Drive) so checkpoints and eval caches survive a session restart. If unset,
# we auto-detect a mounted Drive folder; otherwise we fall back to ./grpo_output.
def _default_output_dir() -> Path:
    env_dir = os.environ.get("SAKHA_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return drive_root / "sakha_grpo"
    return Path("./grpo_output")


CHECKPOINT_DIR = _default_output_dir()
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"  Checkpoint dir: {CHECKPOINT_DIR}")

print(f"Training config:")
print(f"  Model: {MODEL}")
print(f"  Task: {TASK}")
print(f"  Episodes: {EPISODES}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  Use Unsloth: {USE_UNSLOTH}")
print(f"  4-bit quant: {LOAD_IN_4BIT}")

def _completion_text(completion):
    if isinstance(completion, list):
        return str(completion[-1]["content"])
    return str(completion)


def _metadata_value(values, completion_idx: int, completion_count: int, default):
    """Select metadata whether TRL passes it per prompt or per generation."""
    if values is None:
        return default
    if not isinstance(values, (list, tuple)):
        return values
    if len(values) == completion_count:
        return values[completion_idx]
    if len(values) == 1:
        return values[0]
    prompt_idx = min(completion_idx // NUM_GENERATIONS, len(values) - 1)
    return values[prompt_idx]


def reward_func(prompts, completions, **kwargs):
    """State-aligned reward: parse the action, reconstruct prompt state, step once."""
    rewards = []
    completion_count = len(completions)
    for idx, completion in enumerate(completions):
        seed = int(_metadata_value(kwargs.get("seed"), idx, completion_count, SEED))
        replay_actions_json = _metadata_value(
            kwargs.get("replay_actions_json"), idx, completion_count, "[]"
        )
        rewards.append(
            score_completion_action(
                _completion_text(completion),
                task=TASK,
                seed=seed,
                replay_actions_json=str(replay_actions_json),
            )
        )
    return rewards

# ============================================================
# Notebook Cell: 5. Create Dataset and Configure GRPO
# ============================================================

def create_prompt(task: str, episode_id: int = 0):
    env = SakhaEnvironment(task=task)
    obs = env.reset(seed=SEED + episode_id)
    return build_grpo_prompt(obs, task=task, episode_id=episode_id)

# Build dataset
training_examples = build_state_aligned_examples(
    task=TASK,
    episodes=EPISODES,
    seed=SEED,
    max_steps=MAX_STEPS,
)
dataset = Dataset.from_dict(training_examples)

print(f"Dataset size: {len(dataset)} prompts")

# ============================================================
# Notebook Cell: 6. Pre-Training Evaluation — Base LLM
# ============================================================
# RUN THIS FIRST to avoid Unsloth monkey-patch conflict

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

        # Build prompts using the SAME builder as training so the model sees the
        # exact distribution it was optimized on.
        prompt_texts = []
        for i in active_indices:
            messages = build_grpo_prompt(
                observations[i], task=task, episode_id=seeds[i]
            )
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

# Decide up front whether we'll need to actually train. If a usable checkpoint
# already exists we skip the (expensive) Unsloth/vLLM load too, which keeps
# session-restart re-runs fast and frees VRAM for the trained-eval step.
_pretraining_ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))
_force_retrain = os.environ.get("SAKHA_FORCE_RETRAIN") == "1"
_resume_training = os.environ.get("SAKHA_RESUME_TRAINING") == "1"
_will_train = (not _pretraining_ckpts) or _force_retrain or _resume_training

if _will_train and USE_UNSLOTH:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=True,
        gpu_memory_utilization=0.6,
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
elif _will_train:
    model = MODEL
    tokenizer = None
else:
    print("Existing checkpoints found, skipping model load until trained-eval step.")
    model = None
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
    output_dir=str(CHECKPOINT_DIR),
    overwrite_output_dir=False,
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
}

if USE_UNSLOTH:
    trainer_kwargs["model"] = model
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["model"] = MODEL

# --- Training control flow ---
# Default behavior: if checkpoints already exist in CHECKPOINT_DIR, skip training and
# jump straight to trained-eval (so a session restart doesn't redo expensive RL steps).
#   SAKHA_FORCE_RETRAIN=1   -> train fresh (resumes from latest if present)
#   SAKHA_RESUME_TRAINING=1 -> continue training from the latest checkpoint
if not _will_train:
    print(
        f"Found {len(_pretraining_ckpts)} existing checkpoint(s) in {CHECKPOINT_DIR}. "
        "Skipping training (set SAKHA_FORCE_RETRAIN=1 or SAKHA_RESUME_TRAINING=1 to override)."
    )
else:
    trainer = GRPOTrainer(**trainer_kwargs)
    resume_path = (
        str(_pretraining_ckpts[-1])
        if (_pretraining_ckpts and _resume_training)
        else None
    )
    if resume_path:
        print(f"Resuming training from {resume_path}")
    else:
        print(f"Starting training: {EPISODES} episodes, task={TASK}")
    trainer.train(resume_from_checkpoint=resume_path)

# ============================================================
# Notebook Cell: 10. Post-Training Evaluation — Trained LLM
# ============================================================

TRAINED_CACHE = CHECKPOINT_DIR / "trained_eval_cache.json"
trained_results = load_eval_cache(TRAINED_CACHE)

if trained_results is None:
    ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))
    if not ckpts:
        print("No checkpoints found. Skipping trained eval.")
        trained_results = None
    else:
        CHECKPOINT_PATH = str(ckpts[-1])
        print(f"Loading trained checkpoint: {CHECKPOINT_PATH}")

        # Free training-time resources before loading the eval model.
        try:
            del trainer
        except NameError:
            pass
        try:
            del model
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # Always evaluate via plain transformers + PEFT. This matches the base-eval
        # code path exactly (so the comparison is apples-to-apples) and avoids the
        # "lm_head.weight ... newly initialized" warning that the Unsloth/vLLM path
        # surfaces for tied-embedding models like Qwen3.
        from peft import PeftModel
        trained_tokenizer = AutoTokenizer.from_pretrained(
            MODEL, trust_remote_code=True, padding_side="left"
        )
        base_for_trained = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
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

        del trained_model
        del base_for_trained
        del trained_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

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
    plot_path = CHECKPOINT_DIR / "base_vs_trained.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
else:
    print("Skipping comparison — no trained eval results available.")
