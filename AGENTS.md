## Sakha agent quick guide

### What to read first
- `README.md` for run/eval commands.
- `docs/hackathon_checklist.md` for submission gates (especially structured stdout requirement).
- `openenv.yaml` for runtime truth (`server.app:app`, port `7860`).

### Canonical commands (use `uv`, not plain `pip`)
- Setup:
  - `uv python install 3.12`
  - `uv venv --python 3.12`
  - `uv pip install -e ".[dev]"`
- Fast checks via project CLI (`src/sakha/__main__.py`):
  - `uv run sakha lint` (ruff check on `src/`)
  - `uv run sakha format` (ruff format on `src/`)
  - `uv run sakha typecheck` (`ty check src/`)
  - `uv run sakha check` (lint -> typecheck)
  - `uv run sakha all` (lint -> format -> typecheck)
- Tests:
  - `uv run pytest tests/ -v`
  - Single file: `uv run pytest tests/test_inference_parser.py -v`
- Baseline/eval:
  - `uv run python inference.py --tasks easy,medium,hard --seed 42 --episodes 3 --deterministic-baseline --output-json baseline_results.json`
  - `uv run python scripts/eval_policies.py --task hard --seed 42 --episodes 10 --all-policies --output-json hard_seed_sweep.json`

### Architecture map (only the parts that affect edits)
- Core environment logic: `src/sakha/env.py` (`SakhaEnvironment`).
- Typed models and action enums: `src/sakha/models.py`.
- Inference runner and baseline policy: `inference.py`.
- Structured run logging format (`[START]/[STEP]/[END]`): `src/sakha/formatters.py`.
- HTTP/OpenEnv app: `server/app.py` (redirects `/` -> `/web`).

### Non-obvious repo gotchas
- **Action-name mismatch exists between prose docs and code**: code uses `update_chart`, `ward_sweep`, and `medication_round` (see `ActionType` in `models.py`), while README text may mention older names like `document_findings`.
- Do not break compact formatter stdout blocks in `formatters.py`; hackathon validation expects parseable `[START]`, `[STEP]`, `[END]` lines printed to **stdout** with `flush=True`.
- Keep validator runs on compact formatter output (default `--format compact`); structured blocks are emitted by `CompactFormatter`.
- For each episode, emit at least one `[START]`, one or more `[STEP]`, and one `[END]`.
- `inference.py` defaults to `--max-steps 24` for dev speed; full-shift behavior in env is 96 steps (`SHIFT_STEPS`). Use explicit `--max-steps` when validating behavior.
- LLM mode requires `HF_TOKEN`; deterministic mode works without it via `--deterministic-baseline`.
- `PROMPT_PROFILE` is validated strictly: `operational_realism | strict_bedside | full_legacy`.
- Keep `inference.py` at repo root and keep OpenAI-client env wiring (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) intact.

### Validation order before finishing work
1. Targeted tests for changed area.
2. `uv run pytest tests/ -v`.
3. If `inference.py` or formatting output changed, run:
   - `uv run python inference.py --tasks easy --episodes 1 --seed 42 --deterministic-baseline --max-steps 5`
   - Confirm stdout includes `[START]`, `[STEP]`, `[END]` lines.

### Keep-out / low-signal paths
- Do not edit generated cache/artifact paths unless task explicitly requires it: `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `artifacts/`.
