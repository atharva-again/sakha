# Sakha – Ward Assistant OpenEnv Evaluation Environment

## Motivation

In Indian hospitals, a single attendant often manages 15–18 patients. Medication timing errors and missed vitals checks create real operational risk. **Sakha** is a deterministic OpenEnv benchmark for evaluating hospital ward assistant agents on medication delivery, vitals checks, and escalation under time pressure.

Sakha is intentionally positioned as an **evaluation environment**:

- it provides the environment, rewards, and graders
- it includes baseline inference and scripted policy evaluation
- it does **not** include a full RL training or policy-learning stack

See [`benchmark_spec.md`](./benchmark_spec.md) for the benchmark definition.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of: `administer_medicine`, `check_vitals`, `escalate`, `noop` |
| `patient_id` | `int \| None` | Target bed ID (1-indexed) |
| `medicine_id` | `str \| None` | Optional medicine identifier |
| `reason_code` | `str \| None` | Optional escalation reason |

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

For benchmark intent, expected failure modes, and interpretation guidance, see [`benchmark_spec.md`](./benchmark_spec.md).

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

For judge-facing benchmark semantics, use [`benchmark_spec.md`](./benchmark_spec.md).

## License

Apache 2.0
