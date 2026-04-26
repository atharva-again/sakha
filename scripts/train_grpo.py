"""
Sakha GRPO Training Script
1:1 conversion from sakha_grpo_training.ipynb

Install compatible dependencies (Colab):
    pip install unsloth vllm datasets
    pip install --upgrade --force-reinstall --no-deps unsloth unsloth_zoo
"""

import os
import re

# Prevent vLLM from reconfiguring logging (crashes Jupyter/Colab's OutStream)
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import json
import gc
import torch
import datetime as _dt
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction, ActionType
from sakha.grpo_training import (
    build_grpo_prompt,
    build_state_aligned_examples,
    parse_action_response_with_status,
    score_completion_action,
)
from sakha.graders import score_easy_task, score_medium_task, score_hard_task

# ============================================================
# Notebook Cell: 3. Configure Training
# ============================================================
# Training configuration.
# Qwen3-1.7B over 0.6B: the smaller model converges on "format compliance"
# without solving the task under any realistic GRPO budget. 1.7B in 4-bit
# fits T4 (1.2GB weights + LoRA + vLLM rollouts) and gives a much higher
# zero-shot baseline to start RL from.
MODEL = "Qwen/Qwen3-1.7B"  # Model to train (1.7B in 4-bit fits T4)
TASK = "hard"  # Task difficulty: easy | medium | hard
# 120 examples / (per_device 2 × accum 2) = 30 optimizer updates. On T4 with
# 1.7B + 4 generations + 384-token completions this finishes inside ~25
# minutes, leaving budget for a real base-vs-trained eval.
EPISODES = 120
MAX_STEPS = 96  # Max steps per episode (96 = full 8hr shift)
SEED = 42  # Random seed for reproducibility

# Unsloth config (set USE_UNSLOTH=True for 4-bit training on T4).
# Dropping Unsloth on T4 is not viable: without its vLLM-backed rollouts
# each GRPO step takes 3-5x longer, which collapses the achievable number
# of optimizer updates inside the training window.
USE_UNSLOTH = True  # Use Unsloth for memory-efficient training
LOAD_IN_4BIT = True  # 4-bit quantization (critical for T4)

# GRPO specific.
# Tuning rationale on a 1.7B base under a T4 budget:
#   * NUM_GENERATIONS=4 — the parse-failure / format-reward fixes already
#     cleaned the reward signal, so we no longer need group=8 to dampen
#     advantage variance. 4 keeps step time ~50% lower so we get more
#     optimizer updates inside the budget, which matters more.
#   * MAX_COMPLETION_LENGTH=384 — 1.7B is decisive enough that 384 covers
#     a full <think>...</think> + action call without truncation. 512 only
#     wastes step time at this model scale.
#   * LEARNING_RATE=3e-5 — compensates for the relatively short optimizer
#     schedule (~30 updates). Safe under LoRA r=16 regularization.
NUM_GENERATIONS = 4  # Responses per prompt
LEARNING_RATE = 3e-5  # Learning rate
MAX_COMPLETION_LENGTH = 384  # Max tokens per completion during GRPO training
MAX_SEQ_LENGTH = 2048  # Prompt + completion context for vLLM/Unsloth
EVAL_MAX_NEW_TOKENS = 384  # Max tokens during eval (must match training budget)
SAVE_STEPS = 30  # Checkpoint cadence (~1 checkpoint per epoch at episodes=120)

# Eval scope. Base eval runs once per (model, eval_config) tuple and is
# cached on disk — first run pays ~10 min for 3 seeds × 48 steps with 1.7B;
# every subsequent run is free. Bump SEEDS_COUNT or MAX_STEPS for tighter
# error bars at the cost of a one-time cache rebuild. All three knobs are
# folded into the cache key so changes invalidate cleanly.
EVAL_SEEDS_COUNT = 3  # Number of seeds for base/trained comparison
EVAL_SEED_START = 42  # First seed (seeds = [start, start+1, ...])
EVAL_MAX_STEPS = 48  # Eval episode length (half shift is enough to show learning)

# --- Experiment tracking (Weights & Biases) ---
# Hackathon judges require experimental tracking to be enabled for any training
# run. We pull WANDB_API_KEY from Colab secrets when available; otherwise we
# downgrade to offline mode so the run still produces logs that can be synced
# later with `wandb sync`.
try:
    from google.colab import userdata as _colab_userdata  # type: ignore

    _wandb_key = _colab_userdata.get("WANDB_API_KEY")
    if _wandb_key:
        os.environ["WANDB_API_KEY"] = _wandb_key
except Exception:
    pass

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sakha-grpo")
WANDB_RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME",
    f"{TASK}-{MODEL.split('/')[-1]}-ep{EPISODES}-{_dt.datetime.now():%Y%m%d-%H%M%S}",
)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_RUN_NAME"] = WANDB_RUN_NAME
if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") is None:
    os.environ["WANDB_MODE"] = "offline"
    print("WARNING: WANDB_API_KEY not set — running wandb in offline mode.")
    print("   Logs will be written under wandb/ inside the checkpoint dir;")
    print("   run `wandb sync <run-dir>` after training to push them.")


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

print(f"Training config:")
print(f"  Model: {MODEL}")
print(f"  Task: {TASK}")
print(f"  Episodes: {EPISODES}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  Use Unsloth: {USE_UNSLOTH}")
print(f"  4-bit quant: {LOAD_IN_4BIT}")
print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
print(
    f"  Eval seeds: {EVAL_SEEDS_COUNT} × {EVAL_MAX_STEPS} steps (max_new_tokens={EVAL_MAX_NEW_TOKENS})"
)
print(f"  W&B project: {WANDB_PROJECT}")
print(f"  W&B run name: {WANDB_RUN_NAME}")
print(f"  W&B mode: {os.environ.get('WANDB_MODE', 'online')}")


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


def strip_think(text: str) -> str:
    """Qwen3 emits <think>...</think> blocks. Strip them before parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def reward_func(prompts, completions, **kwargs):
    """State-aligned reward: parse the action, reconstruct prompt state, step once."""
    rewards = []
    completion_count = len(completions)
    for idx, completion in enumerate(completions):
        seed = int(_metadata_value(kwargs.get("seed"), idx, completion_count, SEED))
        replay_actions_json = _metadata_value(
            kwargs.get("replay_actions_json"), idx, completion_count, "[]"
        )
        text = strip_think(_completion_text(completion))
        rewards.append(
            score_completion_action(
                text,
                task=TASK,
                seed=seed,
                replay_actions_json=str(replay_actions_json),
            )
        )
    return rewards


def format_reward(completions, **kwargs):
    """Small bonus for outputs the actual scorer can parse.

    Previously this used a lax `\\w+\\(...)$` regex that rewarded any
    parens-shaped trailing token, including outputs the reward scorer
    later rejected as parse failures (`check_vitals(11, p=50, due=0)`).
    The two reward signals contradicted each other — the model converged
    on a format the grader can't read. Use the canonical parser so format
    and content rewards point in the same direction.
    """
    rewards = []
    for completion in completions:
        text = strip_think(_completion_text(completion))
        _, parsed_ok = parse_action_response_with_status(text)
        rewards.append(0.05 if parsed_ok else 0.0)
    return rewards


# ============================================================
# Notebook Cell: 5. Create Dataset and Configure GRPO
# ============================================================


def create_prompt(task: str, episode_id: int = 0):
    env = SakhaEnvironment(task=task)
    obs = env.reset(seed=SEED + episode_id)
    return build_grpo_prompt(obs, task=task, episode_id=episode_id)


# Build dataset.
# Use EVAL_MAX_STEPS (not MAX_STEPS) as the cap: state_steps that exceed the
# eval horizon are dropped, so training mass concentrates on states the
# trained model will actually see at evaluation time.
training_examples = build_state_aligned_examples(
    task=TASK,
    episodes=EPISODES,
    seed=SEED,
    max_steps=EVAL_MAX_STEPS,
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
    tasks_str = (
        ", ".join(f"{t.required_action}({t.patient_id})" for t in pending) or "No pending tasks"
    )
    return (
        f"Step {obs.ward_state.current_step}. "
        f"Patients: {len(obs.ward_state.patients)}. "
        f"Pending: {obs.pending_count}. "
        f"Tasks: {tasks_str}. "
        f"What action do you take?"
    )


def parse_action_response(response):
    """Thin wrapper kept for callers that don't care about the parse status flag."""
    action, _ = parse_action_response_with_status(strip_think(response))
    return action


def save_eval_cache(results, cache_path):
    """Save evaluation results to JSON for resume across sessions."""
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Eval results saved to {cache_path}")


def load_eval_cache(
    cache_path,
    *,
    eval_max_new_tokens=None,
    eval_seeds=None,
    eval_max_steps=None,
):
    """Load cached evaluation results if the eval config still matches.

    Returns None on missing cache OR on any drift in (token budget, seed set,
    max steps) so changing any eval knob automatically invalidates the cache
    instead of silently serving stale numbers from a different scope.
    """
    if not cache_path.exists():
        return None
    with open(cache_path, "r") as f:
        results = json.load(f)
    cfg = results.get("eval_config", {})
    cached_tokens = cfg.get("max_new_tokens")
    cached_seeds = cfg.get("seeds")
    cached_steps = cfg.get("max_steps")
    if eval_max_new_tokens is not None and cached_tokens != eval_max_new_tokens:
        print(
            f"WARNING: Cache token budget {cached_tokens} ≠ current {eval_max_new_tokens}; "
            f"re-running eval"
        )
        return None
    if eval_seeds is not None and cached_seeds != list(eval_seeds):
        print(
            f"WARNING: Cache seed set {cached_seeds} ≠ current {list(eval_seeds)}; re-running eval"
        )
        return None
    if eval_max_steps is not None and cached_steps != eval_max_steps:
        print(
            f"WARNING: Cache max_steps {cached_steps} ≠ current {eval_max_steps}; re-running eval"
        )
        return None
    print(f"Loaded cached eval results from {cache_path}")
    return results


def run_llm_eval_batched(task, model, tokenizer, seeds, max_steps):
    """Vectorized evaluation: runs all seeds in parallel per step for ~5x speedup."""
    from sakha.graders import score_easy_task, score_medium_task, score_hard_task

    device = next(model.parameters()).device
    print(f"Running batched eval on device: {device} | {len(seeds)} seeds × {max_steps} steps")

    TASK_GRADERS = {"easy": score_easy_task, "medium": score_medium_task, "hard": score_hard_task}
    PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]

    # Capture the first N failing completions per seed so we can tell
    # truncated-thinking from no-answer-emitted from format slop without
    # having to re-run with debug prints sprinkled in.
    FAILURE_SAMPLE_LIMIT = 3

    # Initialize all environments at once
    envs = []
    observations = []
    trajectories = []
    step_rewards_all = []
    parse_failures = []
    failure_samples = []
    for seed in seeds:
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed)
        envs.append(env)
        observations.append(obs)
        trajectories.append([obs])
        step_rewards_all.append([])
        parse_failures.append(0)
        failure_samples.append([])

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
            messages = build_grpo_prompt(observations[i], task=task, episode_id=seeds[i])
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # Batch tokenize with left-padding
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)

        # Single batched generation call
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Generation always starts at the padded prompt length, which is the
        # same for every row in the batch. Slicing per-row by
        # ``attention_mask.sum()`` (the *unpadded* prompt length) is wrong
        # under left-padding: it leaks the last `pad_i` tokens of the prompt
        # into the decoded "response", which then shows up as fragments like
        # "_incidents:\n- None\nNext action:\nassistant\n" prefixing the
        # actual generation. That noise breaks the parser and made the
        # trained model look catastrophically worse than baseline.
        prompt_total_len = inputs["input_ids"].shape[1]
        newly_done = []
        for batch_idx, env_idx in enumerate(active_indices):
            response = tokenizer.decode(
                outputs[batch_idx][prompt_total_len:], skip_special_tokens=True
            )
            response = strip_think(response)
            action, parsed_ok = parse_action_response_with_status(response)
            if not parsed_ok:
                parse_failures[env_idx] += 1
                if len(failure_samples[env_idx]) < FAILURE_SAMPLE_LIMIT:
                    failure_samples[env_idx].append(response[:800])
            obs = envs[env_idx].step(action)
            observations[env_idx] = obs
            trajectories[env_idx].append(obs)
            step_rewards_all[env_idx].append(obs.reward)

            if obs.done:
                active_mask[env_idx] = False
                newly_done.append(env_idx)

        # Log
        active_count = sum(active_mask)
        sample_action = parse_action_response(
            tokenizer.decode(outputs[0][prompt_total_len:], skip_special_tokens=True)
        )
        pbar.set_postfix(
            {
                "active": f"{active_count}/{len(seeds)}",
                "action": sample_action.action_type.name,
            }
        )
        for idx in newly_done:
            pbar.write(f"  Seed {seeds[idx]} finished at step {step}")
        pbar.update(1)

    pbar.close()

    # Compute per-episode results
    episodes = []
    for i, seed in enumerate(seeds):
        grader_score = grader(trajectories[i])
        metrics = envs[i].episode_metrics
        ep = {
            "seed": seed,
            "grader_score": grader_score,
            "total_reward": sum(step_rewards_all[i]),
            "critical_incidents_resolved": metrics.critical_incidents_resolved,
            "critical_incidents_missed": metrics.critical_incidents_missed,
            "overdue_tasks": metrics.overdue_tasks,
            "noop_steps": metrics.noop_steps,
            "parse_failures": parse_failures[i],
            "sample_failures": failure_samples[i],
            "discharges_prepared": metrics.discharges_prepared,
        }
        episodes.append(ep)
        print(
            f"  Seed {seed:3}: Reward {ep['total_reward']:6.2f} | "
            f"Grader {grader_score:.4f} | Steps {len(trajectories[i]) - 1} | "
            f"Parse failures {parse_failures[i]}"
        )

    if any(failure_samples):
        print("\nSample parse-failure completions (first per affected seed):")
        for i, seed in enumerate(seeds):
            if not failure_samples[i]:
                continue
            preview = failure_samples[i][0].replace("\n", " [newline] ")[:200]
            print(f"  seed {seed}: {preview!r}")

    def mean(key):
        return sum(e[key] for e in episodes) / len(episodes)

    return {
        "task": task,
        "policy": "base_llm",
        "episodes": episodes,
        "eval_config": {
            "max_new_tokens": EVAL_MAX_NEW_TOKENS,
            "seeds": list(seeds),
            "max_steps": max_steps,
        },
        "summary": {
            "mean_grader_score": mean("grader_score"),
            "mean_total_reward": mean("total_reward"),
            "mean_critical_incidents_resolved": mean("critical_incidents_resolved"),
            "mean_critical_incidents_missed": mean("critical_incidents_missed"),
            "mean_overdue_tasks": mean("overdue_tasks"),
            "mean_noop_steps": mean("noop_steps"),
            "mean_discharges_prepared": mean("discharges_prepared"),
        },
    }


# --- Fast Deterministic Baseline (runs in seconds) ---
def run_deterministic_eval(task, seeds, max_steps):
    """Rule-based policy eval for instant floor comparison."""
    TASK_GRADERS = {"easy": score_easy_task, "medium": score_medium_task, "hard": score_hard_task}
    PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]

    def _deterministic_action(obs):
        ws = obs.ward_state
        pending = ws.pending_tasks
        critical = [t for t in pending if getattr(t, "is_critical", False)]
        if critical:
            t = critical[0]
            return SakhaAction(
                action_type=ActionType[t.required_action.upper()], patient_id=t.patient_id
            )
        if pending:
            t = pending[0]
            return SakhaAction(
                action_type=ActionType[t.required_action.upper()], patient_id=t.patient_id
            )
        return SakhaAction(action_type=ActionType.NOOP)

    episodes = []
    for seed in seeds:
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed)
        traj = [obs]
        for step in range(max_steps):
            action = _deterministic_action(obs)
            obs = env.step(action)
            traj.append(obs)
            if obs.done:
                break
        score = grader(traj)
        episodes.append({"seed": seed, "grader_score": score})
        print(f"  Seed {seed}: Grader {score:.4f}")

    mean_score = sum(e["grader_score"] for e in episodes) / len(episodes)
    return {
        "task": task,
        "policy": "deterministic",
        "mean_grader_score": mean_score,
        "episodes": episodes,
    }


print("\n--- Deterministic Baseline ---")
det_seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SEEDS_COUNT))
det_results = run_deterministic_eval(TASK, det_seeds, EVAL_MAX_STEPS)
print(f"Deterministic baseline mean: {det_results['mean_grader_score']:.4f}\n")

# --- Base Eval: Check cache first, then run if needed ---
BASE_CACHE = CHECKPOINT_DIR / "base_eval_cache.json"
EVAL_SEEDS = list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SEEDS_COUNT))
base_results = load_eval_cache(
    BASE_CACHE,
    eval_max_new_tokens=EVAL_MAX_NEW_TOKENS,
    eval_seeds=EVAL_SEEDS,
    eval_max_steps=EVAL_MAX_STEPS,
)

def _eval_quant_kwargs():
    """Match training quantization (NF4 + double quant + fp16 compute) for eval."""
    if LOAD_IN_4BIT:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        }
    return {"torch_dtype": torch.float16}


if base_results is None:
    print(
        f"Loading base {MODEL.split('/')[-1]} for zero-shot evaluation (4-bit={LOAD_IN_4BIT})..."
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, padding_side="left"
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        trust_remote_code=True,
        **_eval_quant_kwargs(),
    )
    base_model.eval()

    base_results = run_llm_eval_batched(
        TASK, base_model, base_tokenizer, EVAL_SEEDS, EVAL_MAX_STEPS
    )
    save_eval_cache(base_results, BASE_CACHE)

    del base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
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
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    report_to=["wandb"],
    run_name=WANDB_RUN_NAME,
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
    "reward_funcs": [reward_func, format_reward],
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
    resume_path = str(_pretraining_ckpts[-1]) if (_pretraining_ckpts and _resume_training) else None
    if resume_path:
        print(f"Resuming training from {resume_path}")
    else:
        print(f"Starting training: {EPISODES} episodes, task={TASK}")
    trainer.train(resume_from_checkpoint=resume_path)

# ============================================================
# Notebook Cell: 10. Post-Training Evaluation — Trained LLM
# ============================================================

TRAINED_CACHE = CHECKPOINT_DIR / "trained_eval_cache.json"
trained_results = load_eval_cache(TRAINED_CACHE, eval_max_new_tokens=EVAL_MAX_NEW_TOKENS)

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

        # Load via Unsloth (not plain transformers + PeftModel). The LoRA
        # weights were trained on Unsloth-patched layers; loading them onto
        # a plain transformers model and calling merge_and_unload() causes
        # "AttributeError: 'Qwen3Attention' object has no attribute 'apply_qkv'"
        # because the Unsloth patching injects custom forward methods that
        # the plain model doesn't have.
        #
        # ``fast_inference=True`` enables vLLM-backed generate during the
        # eval loop. Without it the eval falls back to plain HF generate
        # and a 48-step x 3-seed run on hard takes ~21 minutes; vLLM cuts
        # that to ~3-5 minutes and frees the time budget for actual
        # training. The trainer is already deleted above, so we can take
        # most of the GPU for inference here.
        from unsloth import FastLanguageModel as _FLM

        trained_model, trained_tokenizer = _FLM.from_pretrained(
            model_name=CHECKPOINT_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
            fast_inference=True,
            gpu_memory_utilization=0.7,
        )
        _FLM.for_inference(trained_model)

        if trained_tokenizer.pad_token is None:
            trained_tokenizer.pad_token = trained_tokenizer.eos_token

        print("Running evaluation on trained model...")
        trained_results = run_llm_eval_batched(
            TASK, trained_model, trained_tokenizer, EVAL_SEEDS, EVAL_MAX_STEPS
        )
        trained_results["policy"] = "trained_llm"
        save_eval_cache(trained_results, TRAINED_CACHE)
        print(f"Trained mean grader score: {trained_results['summary']['mean_grader_score']:.4f}")

        del trained_model
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
    print(
        f"| Grader Score | {b['mean_grader_score']:.4f} | {t['mean_grader_score']:.4f} | {t['mean_grader_score'] - b['mean_grader_score']:+.4f} |"
    )
    print(
        f"| Total Reward | {b['mean_total_reward']:.4f} | {t['mean_total_reward']:.4f} | {t['mean_total_reward'] - b['mean_total_reward']:+.4f} |"
    )
    print(
        f"| Critical Resolved | {b['mean_critical_incidents_resolved']:.2f} | {t['mean_critical_incidents_resolved']:.2f} | {t['mean_critical_incidents_resolved'] - b['mean_critical_incidents_resolved']:+.2f} |"
    )
    print(
        f"| Critical Missed | {b['mean_critical_incidents_missed']:.2f} | {t['mean_critical_incidents_missed']:.2f} | {t['mean_critical_incidents_missed'] - b['mean_critical_incidents_missed']:+.2f} |"
    )
    print(
        f"| Overdue Tasks | {b['mean_overdue_tasks']:.2f} | {t['mean_overdue_tasks']:.2f} | {t['mean_overdue_tasks'] - b['mean_overdue_tasks']:+.2f} |"
    )
    print(
        f"| NOOP Steps | {b['mean_noop_steps']:.2f} | {t['mean_noop_steps']:.2f} | {t['mean_noop_steps'] - b['mean_noop_steps']:+.2f} |"
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["Grader Score", "Total Reward", "Critical Resolved", "Overdue Tasks"]
    base_vals = [
        b["mean_grader_score"],
        b["mean_total_reward"],
        b["mean_critical_incidents_resolved"],
        b["mean_overdue_tasks"],
    ]
    trained_vals = [
        t["mean_grader_score"],
        t["mean_total_reward"],
        t["mean_critical_incidents_resolved"],
        t["mean_overdue_tasks"],
    ]
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base LLM", color="#e74c3c")
    ax.bar([i + width / 2 for i in x], trained_vals, width, label="Trained LLM", color="#2ecc71")
    ax.set_ylabel("Value")
    ax.set_title("Base vs Trained LLM Performance")
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
