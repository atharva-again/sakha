# Sakha Winning Plan (.sisyphus)

Date: 2026-03-27

## Objective

Deliver a hackathon-winning OpenEnv environment for Sakha by maximizing score under rubric constraints while de-risking disqualification.

## Strategic framing

Treat Sakha as a **deterministic ward-prioritization benchmark** (not a broad hospital simulator).

- One shared deterministic engine
- Three conflict-escalating tasks (Easy/Medium/Hard)
- Small structured graded action space (enum + IDs)
- Dense reward and final grader aligned by one hidden priority model

## 5-day execution sequence with go/no-go gates

### Day 0 — Bootstrap + TDD harness

- Create Python project scaffold:
  - `pyproject.toml`
  - package root (e.g., `sakha_env/`)
  - `tests/` and `scripts/`
  - base fixtures directory `tests/fixtures/`
- Install and pin core dependencies (OpenEnv-compatible stack, test tooling).
- Add minimal CI/local command targets for lint/test/validate.
- Define deterministic serialization format for trajectory logs used by tests.

Go/No-Go:
- project scaffold exists and imports resolve
- test runner executes a smoke test in CI/local

QA scenario (executable):
- Commands:
  - `python -m pytest tests/test_smoke.py -q`
  - `python -m pytest --collect-only -q`
- Pass criteria:
  - smoke test passes
  - test discovery finds expected test modules with no import errors.

### Day 1 — Spec compliance core

- Implement typed models and environment methods `reset`, `step`, `state`
- Add deterministic replay primitive (event tape / checkpoint timeline)
- Add `openenv.yaml`

TDD order:
1. Write failing `tests/test_env_contract.py` for `reset/step/state` schema contract.
2. Implement minimal env logic to pass tests.
3. Refactor without changing observable behavior.

Go/No-Go:
- `openenv validate` passes on minimal case
- same seed + same action sequence => same trajectory

QA scenario (executable):
- Command: `python -m pytest tests/test_determinism.py -q`
- Input: seed `42`, fixed action script `tests/fixtures/easy_actions.json`
- Pass criteria: byte-identical serialized trajectory logs across 2 runs; no test failures.

### Day 2 — Easy task + deterministic grader

- Build Easy task (medication/vitals schedule adherence)
- Implement deterministic 0.0–1.0 grader
- Add baseline safety constraints and invalid-action handling

TDD order:
1. Write failing `tests/test_easy_grader.py` + fixed fixtures in `tests/fixtures/easy_*.json`.
2. Implement easy-task transitions + grader.
3. Add edge-case tests (invalid action, out-of-window medication).

Go/No-Go:
- repeated runs produce identical scores

QA scenario (executable):
- Command: `python -m pytest tests/test_easy_grader.py -q`
- Input: 3 fixed trajectories (good/medium/bad)
- Pass criteria: identical scores across repeated runs; all scores in `[0.0, 1.0]`; ordering `good > medium > bad`.

### Day 3 — Medium/Hard via conflicts

- Medium: overlaps + refusals + stock shortage
- Hard: deterioration + competing overdue tasks + escalation
- Validate score spread weak vs strong trajectories

TDD order:
1. Write failing tests for medium/hard progression and escalation triggers.
2. Implement scenario logic and scoring.
3. Add regression tests for previously observed failure trajectories.

Go/No-Go:
- hard task clearly separates weak and strong policies

QA scenario (executable):
- Command: `python scripts/eval_policies.py --task hard --seed 42 --episodes 20`
- Input: `weak_policy` (random/greedy-low-risk) vs `strong_policy` (priority-aware heuristic)
- Pass criteria: mean score gap `>= 0.20`; standard deviation stable (no collapse to constant scores).

### Day 4 — reward shaping + baseline inference

- Implement dense anti-gaming reward components:
  - timeliness, risk reduction, safety validity, wasted-step penalties
- Ensure reward aligns with final grader
- Add root `inference.py` with env var compatibility

Deterministic baseline mode (explicit):
- `inference.py` must support `--deterministic-baseline` using a local heuristic policy (no remote sampling variance).
- Remote model mode defaults: `temperature=0`, fixed `seed=42`, pinned `MODEL_NAME`.
- Reproducibility gate uses deterministic-baseline mode for strict equality checks.

TDD order:
1. Write failing `tests/test_reward_alignment.py` and `tests/test_baseline_repro.py`.
2. Implement reward terms and deterministic baseline runner.
3. Validate anti-gaming with explicit noop-loop fixtures.

Go/No-Go:
- no-op/loop farming fails to produce good score
- baseline run is reproducible

QA scenario (executable):
- Command: `python inference.py --tasks easy,medium,hard --seed 42 --episodes 5 --deterministic-baseline`
- Env vars required: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` or `API_KEY` (also accept `OPENAI_API_KEY` fallback).
- Pass criteria: two back-to-back runs produce matching per-task averages (tolerance `<= 1e-6` for deterministic mode).

### Day 5 — deploy + docs + submission hardening

- Docker build/run
- HF Space response verification
- README with required sections and baseline scores
- final exploit/disqualification checks

CLI contracts to lock:
- `scripts/eval_policies.py`
  - required args: `--task`, `--seed`, `--episodes`, `--policy-a`, `--policy-b`
  - output: JSON summary with `mean_a`, `mean_b`, `gap`, `std_a`, `std_b`
- `scripts/check_hf_endpoint.py`
  - required args: `--url`, optional `--mode {local,hf}` (default `hf`)
  - checks: HTTP 200, `reset` response schema, one valid `step` round-trip.

Go/No-Go:
- all checklist gates pass; no missing required artifacts

QA scenario (executable):
- Commands:
  - `openenv validate`
  - `docker build -t sakha-env .`
  - `docker run --rm -p 7860:7860 sakha-env`
  - `python scripts/check_hf_endpoint.py --url "http://localhost:7860" --mode local`
  - `python scripts/check_hf_endpoint.py --url "$HF_SPACE_URL" --mode hf`
- Pass criteria:
  - validator exit code `0`
  - docker build exit code `0`
  - container starts without crash
  - local + HF endpoint both return HTTP `200` and valid `reset`/`step` schemas.

## Rubric alignment

- Real-world utility (30%): operational ward prioritization outcomes
- Task/grader quality (25%): deterministic graders + difficulty progression
- Environment design (20%): clear spaces + meaningful dense reward
- Code/spec compliance (15%): OpenEnv + Docker + HF + reproducible inference
- Creativity (10%): India-specific constraints encoded in scenarios

## Scope guardrails

- Prefer conflicting deadlines over adding side subsystems
- Avoid free-text in graded actions
- Keep medical role to attendant-assist/safety escalation, not diagnosis

## Atomic commit strategy (ultrawork-safe)

1. `chore(scaffold): bootstrap pyproject, package layout, tests, scripts`
2. `feat(core): implement openenv models and env contract`
3. `feat(easy): add easy scenario and deterministic grader`
4. `feat(conflicts): add medium/hard scenarios and escalation logic`
5. `feat(reward): add dense anti-gaming reward + alignment checks`
6. `feat(baseline): add inference.py deterministic + remote modes`
7. `build(deploy): add docker + hf endpoint checks`
8. `docs(readme): add usage, tasks, and baseline scores`

Rule: each commit must leave tests green for touched scope and include/refresh fixtures relevant to that scope.
