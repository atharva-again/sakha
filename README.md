---
title: Sakha - Hospital Ward Assistant
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Sakha – Ward Assistant OpenEnv Evaluation Environment

## How to Use the Playground

The interactive playground allows you to manually step through a ward shift.

1. **Initialize**: Click **Reset** to start a new 8-hour shift. This will generate 18 beds with initial needs (meds, vitals, or critical status).
2. **Observe**: Check the **Status Panel** for `WardState`. Look for:
   - `medications_due`: List of pending meds (e.g. `["paracetamol"]`).
   - `vitals_due`: `true` means a check is needed.
   - `escalation_level`: `2` or higher means the patient is critical.
3. **Act**: Fill the action fields and click **Step** (each action takes 5 minutes):
   - **Action Type**: One of `administer_medicine`, `check_vitals`, `escalate`, or `noop`.
   - **Patient Id**: The bed number (e.g. `1`, `2`, up to `18`).
   - **Medicine Id**: (Optional) The name of the medicine being given.
   - **Reason Code**: (Optional) Use this when escalating to provide justification (e.g. "BP 90/60, SPO2 92"). This is logged for telemetry and transparency.
4. **Evaluate**: Use **Get state** to see the full raw internal state of all patients at any time.

### Understanding the Metrics

- **Done: False**: The 8-hour shift is still active. Each step takes 5 minutes. The shift ends (Done: True) after 96 steps (480 minutes).
- **Step Reward**:
  - `+0.10`: Successfully administered a medication.
  - `+0.05`: Successfully checked vitals.
  - `+0.15`: Successfully escalated a critical patient.
  - `-0.05`: Missed a required escalation or took an invalid action (e.g., `noop` when tasks are pending).

## Motivation

In Indian hospitals, a single attendant often manages 15–18 patients. Medication timing errors and missed vitals checks create real operational risk. **Sakha** is a deterministic OpenEnv benchmark for evaluating hospital ward assistant agents on medication delivery, vitals checks, and escalation under time pressure.

Sakha is intentionally positioned as an **evaluation environment**:

- it provides the environment, rewards, and graders
- it includes baseline inference and scripted policy evaluation
- it does **not** include a full RL training or policy-learning stack

See https://github.com/atharva-again/sakha/blob/main/docs/benchmark_spec.md for the benchmark definition.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of: `administer_medicine`, `check_vitals`, `escalate`, `noop` |
| `patient_id` | `int \| None` | Target bed ID (1-indexed) |
| `medicine_id` | `str \| None` | Optional medicine identifier |
| `reason_code` | `str \| None` | Optional escalation reason (for logging/transparency only) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ward_state` | `WardState` | 18 patients with vitals, medications due, escalation levels |
| `pending_count` | `int` | Number of patients with pending tasks |
| `time_remaining_minutes` | `int` | Minutes left in the 8-hour shift |
| `metadata` | `dict` | OpenEnv/base observation metadata |

## Tasks

### Easy: Schedule Adherence
Keep medications and vitals on schedule for 5 patients. No conflicts.

- **Grader**: 0.5 × medication_score + 0.5 × vitals_score
- **Capability**: basic action-to-need matching

### Medium: Conflict Prioritization
Handle 8 patients with overlapping needs and resource conflicts.

- **Grader**: 0.4 × medication + 0.3 × vitals + 0.3 × conflict_resolution
- **Capability**: prioritization when multiple task types compete

### Hard: Crisis Ward Management
Manage 18 patients with deterioration events, escalation requirements, and competing deadlines.

- **Grader**: 0.3 × medication + 0.2 × vitals + 0.25 × escalation + 0.25 × deterioration_handling − missed_escalation_penalty
- **Capability**: crisis detection and escalation under broader ward load

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

### Docker
```bash
docker build -t sakha-env .
docker run --rm -p 8000:8000 sakha-env
```

## Baseline Scores

These are benchmark reference scores, not claims of optimal policy quality.

| Task | Policy | Mean Score |
|------|--------|------------|
| Easy | priority | 1.0000 |
| Medium | priority | 0.9250 |
| Hard | priority | 0.5000 |

## Reward Design

- +0.1 per medication administered
- +0.05 per vitals check completed
- +0.15 per successful escalation
- −0.05 per missed escalation
- negative reward for useless or invalid action choices

## Safety Constraints

- Escalation required for patients with `escalation_level >= 2`
- Missed escalation penalty: −0.05 per patient (in step reward and grader)
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
