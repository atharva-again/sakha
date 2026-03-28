# Sakha Findings (Sisyphus Artifact)

Date: 2026-03-27

## Current repo status

The repository is currently documentation-first, with no environment implementation yet.

- Present: `sakha_proposal.md`, `hackathon_checklist.md`, `BRAINSTORMING_CHECKLIST.md`, `AGENTS.md`
- Missing: Python env code, `openenv.yaml`, `Dockerfile`, `inference.py`, tests, required README

## Winning strategy (synthesized)

1. Build a deterministic ward-assistant **prioritization benchmark**, not an over-broad simulator.
2. Use one shared engine and 3 conflict-escalating tasks.
3. Keep graded action space structured and minimal (enum + IDs).
4. Align dense reward and final grader through the same hidden priority logic.
5. Prioritize pass/fail compliance gates first, then scoring optimization.

## Highest risks

- Failing deploy/compliance gates due to scope creep.
- Non-deterministic grader behavior.
- Reward gaming via loops/no-op behavior.
- Hard task lacking score separation between weak and strong policies.

## Mandatory gates

- OpenEnv interface (`reset`, `step`, `state`) with typed models
- `openenv validate` passes
- 3+ deterministic 0.0–1.0 graders
- root `inference.py` reproducible baseline
- Docker build/run works
- HF Space deploys and responds
