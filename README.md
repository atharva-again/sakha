---
title: Sakha - Hospital Ward Assistant
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Sakha – Hospital Ward Shift OpenEnv Environment

## How to Use the Playground

The interactive playground lets you step through a full 8-hour ward shift.

1. **Initialize**: Click **Reset** to start a seeded 8-hour shift. Every task runs the full shift length; the environment never ends early just because the ward is temporarily quiet.
2. **Observe**: Check the **Status Panel** for `WardState`. Look for:
   - `pending_tasks`: deterministic task queue with `required_action`, `priority`, `due_step`, and a short summary.
   - `medications_due` / `vitals_due`: currently due routine work.
   - `escalation_level`: active critical state.
   - `last_vitals`: only visible after recent bedside contact or active deterioration.
3. **Act**: Fill the action fields and click **Step** (each action takes 5 minutes):
   - **Action Type**: One of `review_patient`, `administer_medicine`, `check_vitals`, `alert_doctor`, `escalate`, `document_findings`, `prepare_discharge`, or `noop`.
   - **Patient Id**: The bed number (e.g. `1`, `2`, up to `18`).
   - **Medicine Id**: (Optional) The name of the medicine being given.
   - **Reason Code**: (Optional) Use this when escalating or documenting a decision trail.
4. **Evaluate**: Use **Get state** to see the full raw internal state of all patients at any time.

### Understanding the Metrics

- **Done: False**: The shift is still active. Each step takes 5 minutes and every task runs the full 96-step shift.
- **Step Reward**:
  - positive reward for completing routine work on time
  - larger positive reward for advancing or closing critical workflows
  - negative reward for overdue care, invalid actions, and `noop` while work is pending

## Motivation

In Indian hospitals, a single nurse or ward attendant may handle a large patient load with routine care, deterioration events, admissions, discharge prep, and repeated bedside rounding happening in the same shift. **Sakha** is a deterministic OpenEnv benchmark for evaluating whether an agent can manage that workflow under time pressure.

Sakha provides:

- the environment, rewards, and graders
- baseline inference and scripted policy evaluation
- a minimal GRPO training script and Colab notebook for fine-tuning LLM policies

See https://github.com/atharva-again/sakha/blob/main/docs/benchmark_spec.md for the benchmark definition.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of: `review_patient`, `administer_medicine`, `check_vitals`, `alert_doctor`, `escalate`, `document_findings`, `prepare_discharge`, `noop` |
| `patient_id` | `int \| None` | Target bed ID (1-indexed) |
| `medicine_id` | `str \| None` | Optional medicine identifier |
| `reason_code` | `str \| None` | Optional justification string for escalation or documentation |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ward_state` | `WardState` | Patient roster plus deterministic `pending_tasks` queue |
| `pending_count` | `int` | Number of pending workflow items |
| `time_remaining_minutes` | `int` | Minutes left in the 8-hour shift |
| `action_result` | `ActionResult \| None` | Status and detail for the last action |
| `metadata` | `dict` | OpenEnv/base observation metadata |

## Tasks

### Easy: Routine Shift Discipline
Manage 5 patients through a full shift with recurring medication and vitals deadlines.

- **Grader**: rewards routine completion, timeliness, and productive action quality
- **Capability**: bedside matching and schedule discipline without relying on `noop`

### Medium: Interruptions and Prioritization
Manage 8 patients with recurring routine work, admissions, and deterministic deterioration events.

- **Grader**: combines routine performance with critical-workflow completion and timeliness
- **Capability**: switching between routine care and urgent interruptions without losing the ward

### Hard: Full Ward Coordination
Manage 18 patients with repeated routine work, 5 deterioration events, admissions, and discharge opportunities across the entire shift.

- **Grader**: rewards routine care, incident resolution, timeliness, and ward throughput while penalizing backlog
- **Capability**: sustained coordination under concurrent pressure instead of one-off crisis clicks

For benchmark intent, expected failure modes, and interpretation guidance, see https://github.com/atharva-again/sakha/blob/main/docs/benchmark_spec.md.

## Setup

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Usage

### Run tests
```bash
uv run pytest tests/ -v
```

### Run baseline inference
```bash
uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 5
```

### Run deterministic reproducible baseline with artifact output
```bash
uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 3 --deterministic-baseline --output-json baseline_results.json
```

### Compare policies
```bash
uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 20 --policy-a noop --policy-b priority
```

### Run seed-sweep benchmark report
```bash
uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 10 --all-policies --output-json hard_seed_sweep.json
```

### Run GRPO training (local)
```bash
uv run python scripts/train_grpo.py --mode smoke
uv run python scripts/train_grpo.py --mode demo --task hard --episodes 200
```

### Open the Colab notebook
Open `notebooks/sakha_grpo_training.ipynb` in Google Colab for a GPU-backed training run.

### Docker
```bash
docker build -t sakha-env .
docker run --rm -p 8000:8000 sakha-env
```

## Baseline Scores

These are benchmark reference scores, not claims of optimal policy quality.

| Task | Policy | Mean Score |
|------|--------|------------|
| Easy | deterministic queue policy | 0.4735 |
| Medium | deterministic queue policy | 0.3542 |
| Hard | deterministic queue policy | 0.2195 |

## Reward Design

- dense reward for on-time routine care
- stronger reward for completing critical workflows in order
- penalties for overdue care, invalid actions, and pending-work `noop`
- all tasks continue until the end of the shift instead of terminating early

## Safety Constraints

- Critical incidents are designed as workflow problems, not diagnosis problems
- Escalation alone is insufficient; the intended workflow is `check_vitals -> alert_doctor -> escalate -> document_findings`
- No diagnosis or treatment recommendations — attendant-assist only

## What This Repository Is

Sakha is:

- an OpenEnv-compatible benchmark environment
- a reproducible evaluation harness for LLM and scripted agents
- a foundation for future learned-policy work

Sakha is not:

- a full PPO/DQN/RL training stack
- a clinically deployable decision-maker
- a substitute for nurses or doctors

## Evaluation Guidance

When interpreting scores, focus on:

- separation between weak and stronger policies
- hard-task difficulty relative to easy/medium
- exploit resistance (single-action loops should not dominate)
- seed stability and runtime reproducibility

For judge-facing benchmark semantics, use https://github.com/atharva-again/sakha/blob/main/docs/benchmark_spec.md.

## License

Apache 2.0
