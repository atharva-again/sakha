This document explains hackathon requirements and other important details.

# Hackathon Requirements and Details

## Functional Requirements

### Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

### OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.

### Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.

### Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

### Baseline inference script

Uses the OpenAI-compatible API client to run a model against the environment. Reads API credentials from environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`). Produces a reproducible baseline score on all 3 tasks.

## Non-Functional Requirements

### Deploys to a Hugging Face Space

Environment must run as a containerized HF Space tagged with openenv.

### Containerized execution

Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.

### Documentation

README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.

## Evaluation Criteria

### Real-world utility (30%)

- 0–5: Toy/artificial problem with no practical application
- 6–15: Valid domain but shallow modeling of the real task
- 16–25: Good domain modeling, would be useful for agent evaluation
- 26–30: Excellent — fills a real gap, immediate value for the RL/agent community

### Task & grader quality (25%)

- 3+ tasks with difficulty range?
- Graders produce scores between 0.0–1.0?
- Graders deterministic and reproducible?
- Hard task genuinely challenges frontier models?

### Environment design (20%)

- reset() produces clean state?
- Action/observation types well-designed and documented?
- Reward function provides useful varying signal (not just sparse)?
- Episode boundaries sensible?

### Code quality & spec compliance (15%)

- openenv validate passes?
- docker build && docker run works?
- HF Space deploys and responds?
- Baseline script runs and reproduces scores?

## How Judging Works

### Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

### Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

#### Phase 2 Fail-Fast Structured Stdout Requirement

Phase 2 can fail immediately if validator cannot parse structured run blocks from `inference.py` stdout.

Required stdout blocks (example shape):

- `[START] task=<TASK> episode=<N> seed=<SEED> mode=<llm|deterministic> max_steps=<M>`
- `[STEP] task=<TASK> episode=<N> step=<K> action=<ACTION> patient_id=<ID|None> reward=<FLOAT> done=<true|false> status=<STATUS>`
- `[END] task=<TASK> episode=<N> seed=<SEED> score=<FLOAT> steps=<COUNT> done=<true|false>`

Rules:

1. Print these lines to **stdout** (not stderr)
2. Use `print(..., flush=True)`
3. Do not suppress or redirect stdout inside `inference.py`
4. Emit at least one START, one or more STEP, and one END per episode

### Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

### Disqualification Criteria

- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

## Sample Inference Script

```
"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import base64
import textwrap
from io import BytesIO
from typing import List, Optional, Dict

from openai import OpenAI
import numpy as np
from PIL import Image

from browsergym_env import BrowserGymAction, BrowserGymEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
MAX_STEPS = 8
MAX_DOM_CHARS = 3500
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "noop()"

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a web browser through BrowserGym.
    Reply with exactly one action string.
    The action must be a valid BrowserGym command such as:
    - noop()
    - click('<BID>')
    - type('selector', 'text to enter')
    - fill('selector', 'text to enter')
    - send_keys('Enter')
    - scroll('down')
    Use single quotes around string arguments.
    When clicking, use the BrowserGym element IDs (BIDs) listed in the user message.
    If you are unsure, respond with noop().
    Do not include explanations or additional text.
    """
).strip()


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def extract_screenshot_uri(observation) -> Optional[str]:
    if observation.screenshot is None:
        return None
    screen_array = np.array(observation.screenshot, dtype=np.uint8)
    image = Image.fromarray(screen_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{data_uri}"


def extract_clickable_elements(observation) -> List[Dict[str, str]]:
    """Collect BrowserGym element IDs that can be clicked."""

    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}

    clickables: List[Dict[str, str]] = []
    for bid, props in extra_props.items():
        if not props.get("clickable"):
            continue

        bbox = props.get("bbox") or []
        bbox_str = ", ".join(bbox) if bbox else "?"
        clickables.append(
            {
                "bid": str(bid),
                "bbox": bbox_str,
            }
        )

    # Keep a stable ordering for readability
    clickables.sort(key=lambda item: item["bid"])
    return clickables


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    goal = observation.goal or "(not provided)"
    url = observation.url or "(unknown)"
    error_note = "Yes" if observation.last_action_error else "No"

    clickables = extract_clickable_elements(observation)
    if clickables:
        actions_hint = "\n".join(
            f"    - {item['bid']} (bbox: {item['bbox']})" for item in clickables
        )
    else:
        actions_hint = "    (none detected)"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Current URL: {url}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}
        Available clickable element IDs: {actions_hint}
        Reply with exactly one BrowserGym action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    # Prefer the first line that looks like an action string
    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            # Collapse internal whitespace
            action = re.sub(r"\s+", " ", action)
            # If the model tried to click by natural-language description while we
            # only exposed numeric BrowserGym IDs, fallback to the single detected ID.
            return action

    # Fall back to searching the whole response
    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = BrowserGymEnv.from_docker_image(
        image="browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": "click-test",
        },
    )

    history: List[str] = []

    try:
        result = env.reset()
        observation = result.observation
        print(f"Episode goal: {observation.goal}")

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print("Environment signalled done. Stopping early.")
                break

            user_prompt = build_user_prompt(step, observation, history)
            user_content = [{"type": "text", "text": user_prompt}]
            screenshot_uri = extract_screenshot_uri(observation)
            if screenshot_uri:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_uri},
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            # pylint: disable=broad-except
            except Exception as exc:  # noqa: BLE001
                failure_msg = f"Model request failed ({exc}). Using fallback action."
                print(failure_msg)
                response_text = FALLBACK_ACTION

            action_str = parse_model_action(response_text)
            print(f"Step {step}: model suggested -> {action_str}")

            result = env.step(BrowserGymAction(action_str=action_str))
            observation = result.observation

            reward = result.reward or 0.0
            error_flag = " ERROR" if observation.last_action_error else ""
            history_line = (
                f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
            )
            history.append(history_line)
            print(
                "  Reward: "
                f"{reward:+.2f} | Done: {result.done} | Last action error: "
                f"{observation.last_action_error}"
            )

            if result.done:
                print("Episode complete.")
                break

        else:
            print(f"Reached max steps ({MAX_STEPS}).")

    finally:
        env.close()


if __name__ == "__main__":
    main()
```

## Pre-Submission Checklist

All must pass or you're disqualified.

### Sakha Quick Checklist (the 5 UI checkboxes)

Use this section instead of screenshots before every submission:

1. **Read and follow sample `inference.py` strictly**
   - Keep `inference.py` in repo root.
   - Use `from openai import OpenAI`.
   - Initialize with env-driven config: `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`.

2. **Environment variables present in `inference.py`**
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
   - Optional only if needed by your environment wrapper: `LOCAL_IMAGE_NAME`

3. **Defaults only for API_BASE_URL and MODEL_NAME (not HF_TOKEN)**
   - ✅ allowed defaults:
     - `API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")`
     - `MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")`
   - ✅ no default token:
     - `HF_TOKEN = os.getenv("HF_TOKEN")`

4. **All LLM calls use the configured OpenAI client**
   - All generation calls must be made through that initialized `client` object.
   - Do not mix in other SDK clients for model inference.

5. **Structured stdout format is exact (`START` / `STEP` / `END`)**
   - Emit parseable blocks to stdout for every episode:
     - `[START] task=<TASK> episode=<N> seed=<SEED> mode=<llm|deterministic> max_steps=<M>`
     - `[STEP] task=<TASK> episode=<N> step=<K> action=<ACTION> patient_id=<ID|None> reward=<FLOAT> done=<true|false> status=<STATUS>`
     - `[END] task=<TASK> episode=<N> seed=<SEED> score=<FLOAT> steps=<COUNT> done=<true|false>`
   - Use `print(..., flush=True)`.

### Validation Gates

1. **HF Space deploys**
   - Automated ping to the Space URL — must return 200 and respond to `/reset`

2. **OpenEnv spec compliance**
   - Validate openenv.yaml, typed models, step()/reset()/state() endpoints

3. **Dockerfile builds**
   - Automated docker build on the submitted repo

4. **Baseline reproduces**
   - Run the submitted inference script — must complete without error and produce scores
   - Verify stdout contains parseable `[START]/[STEP]/[END]` blocks

5. **3+ tasks with graders**
   - Enumerate tasks, run each grader, verify scores in 0.0–1.0 range

### Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:

- `API_BASE_URL` — The API endpoint for the LLM
- `MODEL_NAME` — The model identifier to use for inference
- `HF_TOKEN` — Your Hugging Face / API key

**The inference script must be named `inference.py` and placed in the root directory of the project.**

**Participants must use OpenAI Client for all LLM calls using above variables.**

### Infra Restrictions

- Runtime of inference script should be **less than 20 minutes**
- Make sure your env and inference can run on a machine with **vCPU=2, memory=8GB**

### Validator

Run the pre-submission validation script before submitting.

Quick local sanity run for Phase 2 block formatting:

```bash
uv run python inference.py --tasks easy --episodes 1 --seed 42 --deterministic-baseline --max-steps 5
```

Expected: stdout contains lines starting with `[START]`, `[STEP]`, and `[END]`.

---

# Additional Links

- Hackathon Homepage: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon
