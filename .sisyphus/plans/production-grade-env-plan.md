# Sakha Production-Grade Environment Implementation Plan

Date: 2026-03-28
Owner: Hephaestus
Scope: Upgrade the current Sakha ward environment from deterministic prototype to production-grade benchmark and deployable OpenEnv service.

## Goal

Ship a benchmark-quality environment that is:

- train/eval aligned for RL and LLM agents
- reproducible but not script-memorizable
- difficult to game through metadata or open-loop policies
- fully compliant with OpenEnv + hackathon deployment requirements
- observable, testable, and operable in production-like deployment

## Current-State Summary

The current implementation already has a usable base:

- core environment in `src/sakha/env.py`
- typed models in `src/sakha/models.py`
- deterministic graders in `src/sakha/graders.py`
- eval/inference entrypoints in `scripts/eval_policies.py` and `inference.py`
- baseline tests for contract, determinism, and grader behavior

The main gaps to close are:

1. reward and final grader are not aligned, especially for missed escalations
2. observation leaks scorer-facing cumulative metadata
3. deterioration schedule and reset are overly deterministic and invite memorization
4. action semantics are too forgiving for invalid, redundant, or harmful actions
5. state/API boundaries are loose (`state.current_time` drift, mutable ward references)
6. deployment and reproducibility paths need stronger CI and production checks

## Success Criteria

This plan is complete only when all of the following are true:

- step reward is an incremental decomposition of final task scoring
- observation is policy-safe and no longer exposes direct score proxies
- same seed reproduces the same episode, but different seeds produce meaningful bounded variation
- weak, open-loop, and exploit policies score substantially below robust priority-aware baselines
- full local validation passes: package install, syntax checks, tests, Docker build, OpenEnv validation, and endpoint smoke checks
- README and benchmark docs reflect the new reward, task, and baseline behavior

## Workstreams

### Workstream A — Simulator Integrity

Focus: make the environment internally correct, safe, and stable.

Files primarily touched:

- `src/sakha/env.py`
- `src/sakha/models.py`
- `tests/test_env_contract.py`
- `tests/test_determinism.py`

Required changes:

- Introduce a single canonical episode state model so ward time and state time cannot drift.
- Return copied or immutable observation payloads instead of handing out internal mutable references.
- Make action validation explicit and structured: invalid patient, redundant medicine, useless escalate, late/no-op churn.
- Add explicit info/debug channels for counters and diagnostics that should not be policy-visible.
- Keep reset and step behavior deterministic per seed.

Exit gate:

- same seed + same action script => identical serialized trajectory
- external mutation of returned observation cannot alter internal env state
- state(), reset(), and step() agree on current time and terminal status

### Workstream B — Reward and Grader Alignment

Focus: make training objective match benchmark objective.

Files primarily touched:

- `src/sakha/env.py`
- `src/sakha/graders.py`
- `tests/test_reward_alignment.py`
- `tests/test_medium_hard_grader.py`
- `tests/test_easy_grader.py`

Required changes:

- Redesign per-step reward as the incremental delta of the same latent scoring logic used by final graders.
- Add negative reward for missed escalations, delayed critical response, invalid actions, and repeated low-value churn.
- Preserve partial credit where appropriate, but remove reward paths that let agents accumulate score without improving final outcomes.
- Ensure hard-task penalties are reflected both in episode reward and final grader.

Exit gate:

- reward monotonically tracks better final trajectories under controlled policy comparisons
- exploit policies do not outperform clinically sensible policies on reward or final score
- regression tests cover positive and negative alignment

### Workstream C — Benchmark Validity and Difficulty Shaping

Focus: make the environment evaluate closed-loop reasoning rather than timestep memorization.

Files primarily touched:

- `src/sakha/env.py`
- `src/sakha/graders.py`
- `scripts/eval_policies.py`
- new fixtures under `tests/fixtures/`

Required changes:

- Replace the fixed deterioration schedule with seeded bounded variation.
- Add scenario templates for easy, medium, and hard instead of varying only patient count.
- Encode richer but still deterministic-per-seed conflict structure: medication timing windows, overlapping vitals, escalation urgency, limited slack.
- Keep full observability of clinically relevant state, but remove hidden answer keys and score counters.
- Add explicit exploit checks for open-loop policies that act only by timestep or patient index.

Exit gate:

- hard task has clear score spread across noop, greedy, timestep-scripted, and priority-aware baselines
- baseline variance is low for the same seed bundle and meaningful across policy classes
- each task has a distinct failure mode and difficulty curve

### Workstream D — Baseline, Evaluation, and Reproducibility

Focus: make baselines reproducible, fast, and submission-safe.

Files primarily touched:

- `inference.py`
- `scripts/eval_policies.py`
- `tests/test_baseline_repro.py`
- `README.md`

Required changes:

- Add a deterministic local baseline mode in `inference.py` that does not depend on remote LLM variance.
- Keep remote LLM mode for benchmark demos, but gate reproducibility tests on deterministic baseline mode.
- Emit structured run artifacts including env version, seed, task, policy/model, and aggregate metrics.
- Publish calibrated baseline numbers for all tasks after reward/task redesign.

Exit gate:

- two deterministic baseline runs produce identical per-task results
- remote mode remains optional and does not block CI
- eval scripts clearly separate benchmark scoring from ad hoc experimentation

### Workstream E — Production Hardening and Deployment

Focus: make the system deployable, inspectable, and reliable.

Files primarily touched:

- `server/app.py`
- `openenv.yaml`
- `pyproject.toml`
- `README.md`
- `Dockerfile`
- `scripts/check_hf_endpoint.py`
- CI config if added

Required changes:

- Add a full validation pipeline: package install, tests, OpenEnv validation, endpoint smoke tests, Docker build.
- Ensure the server returns stable schemas and useful error messages.
- Add structured logs for episode start/end, step failures, and validation errors.
- Freeze benchmark metadata: version, task definitions, baseline policy versions, config hashes.
- Document production runbooks for local launch, Docker launch, and HF Space verification.

Exit gate:

- `docker build` succeeds
- local server starts cleanly and supports reset/step smoke checks
- OpenEnv validation passes
- documented deployment path is reproducible on a clean machine

## Phase Plan

## Phase 1 — Core Integrity Refactor

Duration: 1 day

Tasks:

- Refactor env state ownership and time bookkeeping.
- Separate policy-visible observation from internal/debug metrics.
- Make observation construction defensive against external mutation.
- Add tests for state consistency and immutability boundaries.

Definition of done:

- env contract tests still pass
- new mutation/state consistency tests pass
- no behavior regressions for deterministic reset/step semantics

QA scenario:

- Commands:
  - `uv run --extra dev pytest tests/test_env_contract.py tests/test_determinism.py -v`
- Pass criteria:
  - all selected tests pass
  - determinism tests confirm identical trajectories for identical seeds
  - new immutability/state-consistency assertions pass with no mutation leak

## Phase 2 — Reward/Grader Unification

Duration: 1 day

Tasks:

- Define shared scoring primitives used by both step reward and final graders.
- Add explicit penalties for safety violations and inaction in critical scenarios.
- Update grader and reward tests to assert alignment under positive and negative cases.

Definition of done:

- reward alignment tests cover missed escalations and exploit trajectories
- hard-task score no longer contradicts step incentives

QA scenario:

- Commands:
  - `uv run --extra dev pytest tests/test_reward_alignment.py tests/test_easy_grader.py tests/test_medium_hard_grader.py -v`
- Pass criteria:
  - all selected tests pass
  - tests explicitly cover positive reward alignment and missed-escalation penalty alignment
  - exploit or churn trajectories do not beat clinically sensible trajectories

## Phase 3 — Scenario Redesign and Anti-Memorization

Duration: 1–2 days

Tasks:

- Replace fixed deterioration script with seeded scenario generation.
- Introduce richer easy/medium/hard templates.
- Add open-loop exploit baselines and compare against closed-loop heuristics.

Definition of done:

- scripted-by-timestep policies no longer look artificially strong
- same-seed determinism remains intact
- score spread across policy classes is demonstrably meaningful

QA scenario:

- Commands:
  - `uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 20 --policy-a noop --policy-b priority`
  - `uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 20 --policy-a greedy --policy-b priority`
  - `uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 20 --policy-a timestep_scripted --policy-b priority`
- Pass criteria:
  - all commands exit with code 0
  - `priority` clearly beats `noop`, `greedy`, and `timestep_scripted`
  - repeated runs over the same seed suite produce stable means/stds

## Phase 4 — Baseline and Eval Hardening

Duration: 0.5–1 day

Tasks:

- Add deterministic baseline mode to `inference.py`.
- Update `test_baseline_repro.py` to use deterministic mode only.
- Record and publish refreshed baseline metrics.

Definition of done:

- reproducibility tests run quickly and do not require live API calls
- docs and scripts agree on benchmark usage

QA scenario:

- Commands:
  - `uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 3 --deterministic-baseline`
  - `uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 3 --deterministic-baseline`
  - `uv run --extra dev pytest tests/test_baseline_repro.py -v`
- Pass criteria:
  - the two deterministic baseline runs emit identical per-task results
  - `test_baseline_repro.py` passes without network access
  - remote-model mode remains optional and is not required for CI success

## Phase 5 — Deployment and Ops Hardening

Duration: 1 day

Tasks:

- Validate server schema and endpoint behavior.
- Add deploy smoke checks and Docker verification.
- Update docs and release notes for benchmark version `v2`.

Definition of done:

- clean install + test + Docker + endpoint validation path is documented and repeatable

QA scenario:

- Commands:
  - `uv run python -m compileall src server scripts inference.py`
  - `uv run openenv validate .`
  - `uv run python server/app.py`
  - `uv run openenv validate --url http://localhost:8000`
  - `uv run python scripts/check_hf_endpoint.py --url http://localhost:8000 --mode local`
  - `docker build -t sakha-env .`
- Pass criteria:
  - compile step exits 0
  - local directory validation passes
  - local server starts cleanly on port 8000
  - runtime OpenEnv validation against `http://localhost:8000` passes
  - endpoint smoke check exits 0 and confirms valid reset schema
  - Docker build exits 0

## Recommended Implementation Order by File

1. `src/sakha/models.py`
   - split policy observation fields from debug/info fields
   - tighten model semantics where needed

2. `src/sakha/env.py`
   - centralize state
   - implement reward unification and scenario generation
   - make action validation explicit

3. `src/sakha/graders.py`
   - refactor to reuse shared scoring primitives

4. `tests/test_reward_alignment.py`
   - add negative and exploit-alignment coverage

5. `tests/test_determinism.py`, `tests/test_env_contract.py`
   - add immutability/state-consistency checks

6. `scripts/eval_policies.py`
   - add exploit/scripted baselines and richer reporting

7. `inference.py`
   - deterministic local baseline mode, artifact output, reproducible CLI

8. `server/app.py`, `scripts/check_hf_endpoint.py`, `README.md`, `openenv.yaml`
   - deployment correctness and documentation sync

## Validation Matrix

After each phase, run the smallest relevant verification set first, then full validation.

### Fast checks

- environment contract tests
- determinism tests
- reward alignment tests
- task grader tests

### Full local validation

- `uv run --extra dev pytest tests/ -v`
- `uv run python -m compileall src server scripts inference.py`
- `uv run openenv validate .`
- `uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 3 --deterministic-baseline`
- rerun the same deterministic baseline command and compare output for exact match
- `docker build -t sakha-env .`
- start server with `uv run python server/app.py`
- `uv run openenv validate --url http://localhost:8000`
- `uv run python scripts/check_hf_endpoint.py --url http://localhost:8000 --mode local`

Pass criteria:

- compile step exits 0
- full pytest suite exits 0
- both local and runtime OpenEnv validation pass
- deterministic baseline outputs match exactly across two runs
- endpoint smoke check exits 0
- Docker build exits 0

### Benchmark validation

- compare `noop`, `greedy`, `timestep-scripted`, and `priority-aware` policies
- inspect score mean/std over a fixed seed suite
- confirm no exploit policy beats intended baselines

## Risks and Mitigations

### Risk: too much randomness breaks reproducibility

Mitigation:

- use seeded bounded variation only
- freeze scenario generation from a single seed-driven config object

### Risk: removing metadata makes task partially observable

Mitigation:

- keep all clinically actionable state visible
- remove only direct score proxies and debug counters

### Risk: stronger penalties collapse learning signal

Mitigation:

- apply incremental penalties with calibrated magnitudes
- validate reward-score correlation on controlled trajectories before locking weights

### Risk: benchmark changes invalidate old README numbers

Mitigation:

- treat this as benchmark version bump
- regenerate baseline artifacts and update docs together

## Final Deliverables

- production-grade environment core
- aligned graders and rewards
- exploit-resistant task definitions
- deterministic local baseline and optional remote baseline
- deployable Docker + HF Space validation path
- updated README and benchmark version notes

## Immediate Next Step

Start with Workstream A + B together: refactor observation/state boundaries and unify reward/grader logic first. Those changes unlock everything else and remove the biggest production and benchmark risks in the current codebase.
