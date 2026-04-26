---
title: Sakha - Hospital Ward Assistant
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Sakha — can an LLM survive a full nursing shift?

In an Indian general ward, a single attendant can be responsible for 18 beds for 8 straight hours. Medications stack up every few minutes. Vitals come due. A patient deteriorates without warning. A discharge is waiting on paperwork. They don't fail because they don't know *what* to do — they fail because they have to keep doing it, correctly, for 96 five-minute slots in a row.

**Sakha** is an OpenEnv benchmark *and* a GRPO-ready training environment that reproduces that pressure deterministically, so we can ask one question of an agent:

> *Can you maintain coordination across an entire 8-hour ward shift while the ward keeps interrupting you?*

Today, even strong LLMs say no — and Sakha is built both to measure that gap and to give you the tools to close it.

**Links:** [Hosted playground](https://huggingface.co/spaces/atharva-again/sakha) · [HF files](https://huggingface.co/spaces/atharva-again/sakha/tree/main) · [Blog](https://huggingface.co/spaces/atharva-again/sakha/blob/main/BLOG.md) · [Benchmark spec](https://huggingface.co/spaces/atharva-again/sakha/blob/main/docs/benchmark_spec.md)

---

## The problem

LLMs are good at *one* hard decision. They are unreliable at *the hundredth* easy one in a row. Most agent benchmarks reward a single coherent plan — pick the right page, write the right SQL, finish the puzzle. Real ward work is not like that.

A shift stacks three pressures on top of each other:

1. **Routine work that never stops** — medications and vitals come due on a fixed cadence per patient.
2. **Critical interrupts that arrive without warning** — deteriorations follow a workflow (`check_vitals → alert_doctor → escalate → update_chart`), not a single click.
3. **Throughput** — admissions, discharges, and bedside rounding still have to happen between the above.

Sakha is, as far as we know, the first OpenEnv benchmark targeting that exact failure mode — *sustained prioritization under load* — in a high-stakes professional setting where zoning out has a concrete cost.

## The environment

A shift is **96 deterministic steps × 5 minutes = 8 hours**. The episode never ends early; idleness is a choice the agent has to defend.

**What the agent sees.** A `WardState` with the patient roster, a deterministic `pending_tasks` queue (each task carries `required_action`, `priority`, `due_step`, and a short summary), what's currently due (`medications_due`, `vitals_due`), the ward's `escalation_level`, and `last_vitals` for any bed it has actually visited recently — plus `pending_count`, `time_remaining_minutes`, and the result of its last action.

**What it can do.** Ten typed actions covering bedside care, routine sweeps, and the deterioration workflow:

`review_patient` · `administer_medicine` · `check_vitals` · `alert_doctor` · `escalate` · `update_chart` · `ward_sweep` · `medication_round` · `prepare_discharge` · `noop`

Each is parameterized by `patient_id`, optional `medicine_id`, and an optional `reason_code` for justifying escalations.

**What it's rewarded for.** Dense signal across the full trajectory:

- positive reward for routine work completed on time
- larger reward for advancing or closing critical workflows in order
- penalties for overdue care, invalid actions, and `noop` while real work is pending

The reward is shaped, not sparse — every step tells the agent something.

**Three difficulty steps.**

| Task | Beds | Pressure | What it measures |
|---|---|---|---|
| Easy | 5 | recurring meds + vitals | bedside matching and schedule discipline |
| Medium | 8 | + admissions + deterministic deteriorations | switching contexts without losing routine work |
| Hard | 18 | + 5 deteriorations + discharges across the shift | sustained coordination under concurrent pressure |

Full task semantics, expected failure modes, and grader internals live in the [benchmark spec](https://huggingface.co/spaces/atharva-again/sakha/blob/main/docs/benchmark_spec.md).

## Results

A deterministic priority-queue baseline — *explicitly aware of the task structure* — is enough to ace Easy, but loses ground fast as the ward gets bigger:

| Task | Deterministic baseline (mean over seeds) | Trained policy |
|---|---|---|
| Easy | **0.8048** | TBD |
| Medium | **0.4905** | TBD |
| Hard | **0.3147** | TBD |

The drop from **0.80 → 0.31** between Easy and Hard is the gap. Hard is where reactive heuristics break: the queue keeps changing under the policy's feet, and any greedy approach starts dropping critical workflows half-finished. That's the headroom learned policies have to claim.

![Training reward curve](artifacts/plots/reward_curve.png)
![Before vs after](artifacts/plots/before_after.png)
![Per-task comparison](artifacts/plots/per_task_comparison.png)

*Trained-policy numbers are produced by [`notebooks/sakha_grpo_training.ipynb`](https://huggingface.co/spaces/atharva-again/sakha/blob/main/notebooks/sakha_grpo_training.ipynb) on Colab T4 / HF Spaces GPU; the plots above are from the included GRPO run.*

## Train your own policy

Sakha is not just an eval harness — it ships a **complete GRPO training recipe** that fits a free Colab T4 (or an HF Spaces GPU) in under 30 minutes. The same env code that grades your agent is also what supplies the training reward, so there is no eval/train distribution gap to chase.

- **State-aligned dataset.** [`build_state_aligned_examples`](https://huggingface.co/spaces/atharva-again/sakha/blob/main/src/sakha/grpo_training.py) replays trajectories with three different policies (queue head, noisy queue head, random pending) to reach 14 state checkpoints across the shift. Training mass concentrates on states the model actually sees at eval time — not just `t=0` snapshots.
- **Dense per-step reward.** [`score_completion_action`](https://huggingface.co/spaces/atharva-again/sakha/blob/main/src/sakha/grpo_training.py) parses the model's action, reconstructs the env to the prompt state via a recorded replay, steps it once, and returns a scaled + clipped env reward. Parse failures cost `-1.0` (strictly more painful than any legitimate misstep), and a small companion `format_reward` keeps format-compliance and content-quality signals pointing in the same direction instead of contradicting each other.
- **T4-fitting stack.** Qwen3-1.7B + 4-bit + LoRA r=16, TRL `GRPOTrainer` with Unsloth + vLLM-backed rollouts. Without vLLM each GRPO step is 3–5× slower and the achievable optimizer-update count collapses inside the training window.
- **Cheap iteration.** Base-vs-trained eval is cached on disk keyed by `(max_new_tokens, seeds, max_steps)` and invalidates automatically when any knob changes — no silent stale numbers between runs.
- **W&B out of the box.** Online if `WANDB_API_KEY` is set; otherwise it logs offline and you can `wandb sync <run-dir>` after.
- **Resume-aware.** Existing checkpoints skip retraining by default; `SAKHA_FORCE_RETRAIN=1` or `SAKHA_RESUME_TRAINING=1` override.

**Run it.** The Colab notebook is the easiest path:

```bash
# Open in Colab (T4) or HF Spaces GPU
notebooks/sakha_grpo_training.ipynb

# Or run the equivalent script directly
uv run python scripts/train_grpo.py
```

If you want a different base model, task, or budget, edit the constants at the top of `scripts/train_grpo.py` (`MODEL`, `TASK`, `EPISODES`, `EVAL_MAX_STEPS`, …) — they are deliberately at module scope so it stays a single readable file.

## Why it matters

- **For RL / agent researchers** — a deterministic, reproducible benchmark *and* training environment for the failure mode that frontier LLMs actually exhibit in agentic workflows: not "did you reason correctly once" but "did you stay coherent for an hour".
- **For healthcare technologists** — a sandbox to evaluate assistive-coordination policies *before* they touch a real attendant or a real patient. Sakha is explicitly attendant-assist: graders reward correct workflow execution, not clinical judgement.
- **For training-stack folks** — clean-room rewards (dense, deterministic, seeded), typed I/O, OpenEnv compliance, a state-aligned GRPO recipe that fits a free T4, and baselines you can beat in an afternoon.

If your agent can hold up across an 8-hour shift on Hard, that's a meaningful capability claim — far more than a single-task win rate.

## Try it in 60 seconds

**Hosted playground** (no install): [`huggingface.co/spaces/atharva-again/sakha`](https://huggingface.co/spaces/atharva-again/sakha) → click **Reset**, fill in an action, click **Step**. Each step is 5 in-shift minutes; the panel on the right shows what the agent would see.

**Local baseline run:**

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"

uv run python inference.py \
    --tasks easy,medium,hard --seed 42 --episodes 3 \
    --deterministic-baseline \
    --output-json baseline_results.json
```

You should see structured `[START] / [STEP] / [END]` blocks in stdout for every episode.

---

## Reference

<details>
<summary><b>Action and observation schemas</b></summary>

**Action**

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | One of `review_patient`, `administer_medicine`, `check_vitals`, `alert_doctor`, `escalate`, `update_chart`, `ward_sweep`, `medication_round`, `prepare_discharge`, `noop` |
| `patient_id` | `int \| None` | 1-indexed bed ID |
| `medicine_id` | `str \| None` | Optional medicine identifier |
| `reason_code` | `str \| None` | Optional justification for escalation or documentation |

**Observation**

| Field | Type | Description |
|---|---|---|
| `ward_state` | `WardState` | Patient roster plus deterministic `pending_tasks` queue |
| `pending_count` | `int` | Number of pending workflow items |
| `time_remaining_minutes` | `int` | Minutes left in the 8-hour shift |
| `action_result` | `ActionResult \| None` | Status and detail for the last action |
| `metadata` | `dict` | OpenEnv/base observation metadata |

</details>

<details>
<summary><b>More commands (tests, eval, GRPO, Docker)</b></summary>

```bash
uv run pytest tests/ -v

uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 20 \
    --policy-a noop --policy-b priority

uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 10 \
    --all-policies --output-json hard_seed_sweep.json

uv run python scripts/train_grpo.py
SAKHA_RESUME_TRAINING=1 uv run python scripts/train_grpo.py
SAKHA_FORCE_RETRAIN=1    uv run python scripts/train_grpo.py

docker build -t sakha-env .
docker run --rm -p 8000:8000 sakha-env
```

</details>

<details>
<summary><b>Playground walkthrough</b></summary>

1. **Initialize** — click **Reset** to start a seeded 8-hour shift. Episodes always run the full shift; the env never ends early just because the ward is briefly quiet.
2. **Observe** — the **Status Panel** shows `WardState`. Key fields:
   - `pending_tasks` — deterministic queue with `required_action`, `priority`, `due_step`, summary.
   - `medications_due` / `vitals_due` — currently due routine work.
   - `escalation_level` — active critical state.
   - `last_vitals` — only visible after recent bedside contact or active deterioration.
3. **Act** — fill the action fields and click **Step** (each action = 5 minutes).
4. **Inspect** — **Get state** returns the full raw internal state of all patients at any time.

</details>

<details>
<summary><b>What Sakha is and isn't</b></summary>

**Sakha is:**
- an OpenEnv-compatible benchmark environment
- a reproducible evaluation harness for LLM and scripted agents
- a working GRPO training environment (state-aligned dataset, dense per-step reward, T4-fitting stack)
- a foundation for further learned-policy work

**Sakha is not:**
- a clinically deployable decision-maker
- a substitute for nurses or doctors
- a diagnosis or treatment recommender — it is attendant-assist only

Critical incidents are framed as workflow problems, not diagnosis problems. Escalation alone is insufficient; the intended workflow is `check_vitals → alert_doctor → escalate → update_chart`.

</details>

<details>
<summary><b>Evaluation guidance</b></summary>

When interpreting scores, focus on:

- separation between weak and stronger policies
- hard-task difficulty relative to easy/medium
- exploit resistance (single-action loops should not dominate)
- seed stability and runtime reproducibility

For judge-facing benchmark semantics, see the [benchmark spec](https://huggingface.co/spaces/atharva-again/sakha/blob/main/docs/benchmark_spec.md).

</details>

## License

Apache 2.0.
