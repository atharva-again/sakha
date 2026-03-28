# Sakha Evaluation Environment Hardening Plan

Date: 2026-03-28
Scope: Close the remaining gaps needed to make Sakha a strong, judge-ready evaluation environment.

## Goal

Strengthen Sakha as a benchmark-quality evaluation environment for agentic systems by improving benchmark clarity, hard-task credibility, anti-exploit robustness, baseline methodology, and evaluator-facing documentation.

## Current Position

Sakha already has:

- a functioning OpenEnv environment
- 3 graded tasks
- reward shaping
- baseline inference and deterministic baselines
- Docker / validation path

What is still missing is not “learning,” but benchmark rigor and evaluator confidence.

## Success Criteria

This plan is complete when:

- each task has a crisp benchmark spec and capability target
- hard-task difficulty is evidenced across seeds and policy classes
- exploit trajectories and brittle heuristics are formally tested
- baseline reporting includes variance, runtime, and seed-sweep summaries
- evaluator docs clearly frame Sakha as an evaluation environment
- prompt/inference behavior is transparent and benchmark logic remains environment-led

## Workstreams

### Workstream A — Benchmark Specification

Goal: make the benchmark legible to judges and future users.

Deliverables:

- A benchmark spec section in README or dedicated `benchmark_spec.md`
- For each task: objective, measured capability, failure modes, scoring logic, expected difficulty
- Clear explanation of what easy / medium / hard are intended to test

Files likely touched:

- `README.md`
- new `benchmark_spec.md` or `docs/benchmark_spec.md`
- possibly `sakha_proposal.md`

Validation:

- A reviewer can explain each task and its metric without reading code
- Task definitions and grader logic are consistent

### Workstream B — Hard-Task Credibility

Goal: prove that the hard task is actually a meaningful evaluation challenge.

Deliverables:

- Seed-sweep evaluation for easy/medium/hard
- Comparison across policy classes: weak, prompt-only LLM, deterministic baseline, exploit baselines
- Score distribution summary (mean/std/min/max)
- Short write-up of why hard is hard

Files likely touched:

- `scripts/eval_policies.py`
- `README.md`
- possibly new `scripts/eval_seed_sweep.py`

Validation:

- Hard task shows clear separation across policy classes
- Hard is not trivially solved by one repetitive action class
- Score variance is reported and reasonable

### Workstream C — Anti-Exploit & Robustness Testing

Goal: prove the benchmark resists cheap shortcut strategies.

Deliverables:

- Tests for repeated-action collapse
- Tests for malformed model outputs / parser fallbacks
- Tests for exploit heuristics (always medicate, always escalate, fixed patient loop, timestep scripts)
- Tests showing exploit policies underperform state-aware policies

Files likely touched:

- `tests/test_reward_alignment.py`
- `tests/test_medium_hard_grader.py`
- new `tests/test_exploit_resistance.py`
- new `tests/test_inference_parser.py`

Validation:

- Exploit tests pass
- Shortcut policies do not achieve suspiciously strong scores
- Inference path degrades safely under malformed outputs

### Workstream D — Baseline Methodology & Reporting

Goal: make baseline results trustworthy and easy to compare.

Deliverables:

- Deterministic baseline report across multiple seeds
- LLM baseline report with runtime and variance
- One stable baseline command for judge reproduction
- Output artifact format (JSON/JSONL/CSV) with task, seed, score, reward, runtime, model, config

Files likely touched:

- `inference.py`
- `scripts/eval_policies.py`
- `tests/test_baseline_repro.py`
- optionally new `scripts/report_baselines.py`

Validation:

- Deterministic baseline is reproducible
- LLM baseline produces reportable per-seed metrics
- Runtime budget against hackathon constraints is documented

### Workstream E — Environment / Agent Boundary Cleanup

Goal: ensure benchmark logic lives in the environment and graders, not hidden in fragile prompt patches.

Deliverables:

- Clear statement of what is benchmark logic vs baseline-agent logic
- Minimize agent-side “rescue” behavior that changes what is being measured
- Keep inference-time arbitration clinically justified and documented, or simplify it if it obscures evaluation

Files likely touched:

- `inference.py`
- `README.md`
- docs describing baseline methodology

Validation:

- A judge can distinguish environment quality from baseline-agent heuristics
- The environment remains meaningful even if a different agent is used

### Workstream F — Evaluator-Facing Documentation

Goal: make the submission feel intentional, not incomplete.

Deliverables:

- README framing Sakha as an evaluation environment
- Explicit statement that this repo is environment + baseline, not a full training stack
- Section on intended research use: evaluate LLM agents, scripted policies, future RL agents
- Known limitations and interpretation notes for scores

Files likely touched:

- `README.md`
- `sakha_proposal.md`

Validation:

- The repo’s purpose is clear within the first page of docs
- Judges can understand what artifact is being submitted and why it is valuable

## Recommended Execution Order

### Phase 1 — Benchmark Spec + Documentation

Do first because it clarifies the artifact before deeper tuning.

Tasks:

- Write benchmark spec
- Update README framing
- Align task/grader wording with code

### Phase 2 — Exploit Resistance & Tests

Do second because it hardens the environment itself.

Tasks:

- Add exploit-resistance tests
- Add malformed-output / parser tests
- Add repeated-action-collapse tests

### Phase 3 — Hard-Task Evidence

Do third because it produces the strongest proof for judges.

Tasks:

- Run seed sweeps
- Compare weak / exploit / deterministic / LLM baselines
- Publish score distributions

### Phase 4 — Baseline Reporting

Do fourth so the evidence becomes reproducible and judge-friendly.

Tasks:

- Emit structured artifacts
- Add per-seed summaries and runtime info
- Document stable benchmark commands

### Phase 5 — Boundary Cleanup

Do last because it depends on what earlier evaluation reveals.

Tasks:

- Review inference-time arbitration
- Keep or simplify only what remains clinically/evaluatively justified

## Concrete Task List

1. Add `benchmark_spec.md`
2. Rewrite README sections for benchmark framing
3. Add exploit-resistance tests
4. Add parser / malformed-output tests
5. Extend policy eval scripts with seed sweeps and summaries
6. Add baseline result artifact output
7. Document runtime, variance, and interpretation guidance
8. Review and trim baseline-agent arbitration if it distorts evaluation

## Verification Matrix

### Documentation checks

- README explains artifact purpose clearly
- Benchmark spec matches code behavior

### Test checks

- `uv run --extra dev pytest tests/ -v`
- exploit-resistance tests pass
- parser robustness tests pass

### Benchmark checks

- deterministic baseline reproducible across seeds
- weak and exploit policies clearly underperform stronger baselines
- hard task shows meaningful spread

### Submission checks

- OpenEnv validation passes
- Docker build succeeds
- baseline command completes within runtime budget

## Final Framing for Submission

Sakha should be presented as:

> A benchmark-quality hospital ward assistant evaluation environment for testing LLM agents, scripted policies, and future learned policies.

Not as:

> A complete end-to-end learning system.

That framing is accurate, defensible, and aligned with the hackathon checklist.
