# Sakha Benchmark Specification

## Artifact Type

Sakha is a benchmark-quality evaluation environment for hospital ward assistant agents. This repository provides:

- the environment
- typed actions and observations
- deterministic graders
- a baseline inference runner
- scripted policy evaluation tools

This repository does **not** include a full learning or RL training stack.

## Benchmark Intent

Sakha measures whether an agent can choose the right ward action for the right patient under competing demands.

Core capabilities under evaluation:

- schedule adherence
- triage under competing needs
- escalation correctness
- handling deterioration events
- avoiding shortcut policies that repeat a single action class

## Task Definitions

### Easy — Schedule Adherence

Scenario:

- 5 patients
- routine meds and vitals needs
- no meaningful crisis complexity

Primary capability measured:

- basic matching of action to patient need

Expected failure modes:

- repeating one action type
- ignoring vitals while only medicating
- targeting invalid or already-served patients

Scoring intent:

- balanced credit for medication completion and vitals completion

### Medium — Conflict Prioritization

Scenario:

- 8 patients
- overlapping task needs
- more conflict between patient needs and time budget

Primary capability measured:

- choosing the right next action when multiple task types are available

Expected failure modes:

- single-action collapse
- weak handling of simultaneous med/vitals demands
- shallow heuristics that ignore patient state

Scoring intent:

- credit for meds, vitals, and conflict handling quality

### Hard — Crisis Ward Management

Scenario:

- 18 patients
- seeded deterioration events
- escalation requirements plus routine load

Primary capability measured:

- crisis detection and escalation without abandoning routine ward tasks

Expected failure modes:

- missing deteriorations
- delayed escalation
- repetitive low-value action loops
- inability to trade off urgent and routine demands

Scoring intent:

- reward correct escalation and deterioration handling
- penalize missed escalations
- retain credit for broader ward completion quality

## Difficulty Design

- Easy checks basic task matching.
- Medium checks prioritization under overlap.
- Hard checks whether the agent can handle acute deterioration while still managing the ward.

The benchmark should not be interpreted as diagnosis or treatment recommendation. It is an attendant-assist prioritization benchmark.

## Grader Properties

- deterministic
- scores in the range 0.0–1.0
- stable under fixed seed and fixed trajectory
- designed to separate weak, exploitative, and stronger policies

## Baseline Interpretation

The deterministic baseline is a reproducible reference policy, not a claim of optimal clinical practice.

The LLM baseline is a judge-facing sanity check showing that a standard agent can interact with the environment. It should be interpreted with seed variance and runtime in mind.

## Known Limitations

- this repo is an evaluation environment, not a training system
- baseline-agent logic may evolve faster than the environment itself
- prompt-sensitive LLM behavior should not be confused with environment quality
- the benchmark is intentionally simplified relative to real hospital operations
