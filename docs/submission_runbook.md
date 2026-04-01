# Sakha Submission Runbook

Use this runbook to eliminate submission-time failures.

## Environment Constraints

- Runtime target: under 20 minutes for baseline script
- Machine envelope: 2 vCPU, 8 GB memory
- Python: 3.12

## One-command readiness validation

```bash
uv run sakha submit-check
```

This runs:

- `sakha ci` (format/lint/typecheck/tests)
- `openenv validate`
- deterministic baseline inference across easy/medium/hard
- benchmark separation report generation
- reproducibility report generation
- docker build/run + endpoint smoke check

Artifacts are written to `artifacts/`.

## Frozen Submission Profile

Use this exact command as the final pre-submit gate:

```bash
uv run sakha submit-check
```

Expected core outcomes:

- `sakha ci` exits 0
- `openenv validate` exits 0
- deterministic baseline JSON written to `artifacts/baseline_submit_check.json`
- separation JSON written to `artifacts/benchmark_separation_report.json`
- reproducibility JSON written to `artifacts/reproducibility_report.json`
- docker build/run + local `/reset` smoke passes (when docker available)

## Judge-Facing Evidence Files

- `artifacts/baseline_submit_check.json`
- `artifacts/benchmark_separation_report.json`
- `artifacts/reproducibility_report.json`
- `artifacts/submit_check_report.json`

## HF Endpoint Smoke Check

```bash
uv run python scripts/check_hf_endpoint.py --url "https://<space>.hf.space" --mode hf
```

Expected outcome: `OK: hf endpoint returned valid reset response`.
