"""
Sakha Ward Assistant — Baseline Inference Script
=================================================

Environment variables (MANDATORY):
    API_BASE_URL    The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME      Model identifier (default: microsoft/Phi-3-mini-4k-instruct)
    HF_TOKEN        Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_your_token_here"
    export MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
    python inference.py --tasks easy,medium,hard --seed 42 --episodes 3
"""

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import ActionType, SakhaAction, SakhaObservation

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sakha")


def call_llm(client, messages):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0"))

MAX_STEPS = 96
TEMPERATURE = 0.2
MAX_TOKENS = 150
FALLBACK_ACTION = SakhaAction(action_type=ActionType.REVIEW_PATIENT, patient_id=1)
ACTION_ID_TO_TYPE = {
    "A0": ActionType.REVIEW_PATIENT,
    "A1": ActionType.MEDICATION_ROUND,
    "A2": ActionType.CHECK_VITALS,
    "A3": ActionType.ESCALATE,
    "A4": ActionType.ALERT_DOCTOR,
    "A5": ActionType.UPDATE_CHART,
    "A6": ActionType.PREPARE_DISCHARGE,
    "A7": ActionType.WARD_SWEEP,
}

TASK_GRADERS = {
    "easy": score_easy_task,
    "medium": score_medium_task,
    "hard": score_hard_task,
}


def build_user_prompt(
    observation: SakhaObservation,
    step: int,
    history: list[str],
    scratchpad: str | None = None,
) -> str:
    ward = observation.ward_state

    state_payload = {
        "step": step,
        "time_remaining_minutes": observation.time_remaining_minutes,
        "pending_count": observation.pending_count,
        "pending_tasks": [task.model_dump(mode="json") for task in ward.pending_tasks[:12]],
        "patients": [p.model_dump() for p in ward.patients],
        "recent_actions": history[-3:] if history else [],
    }

    prompt_parts = [
        "STATE JSON:",
        json.dumps(state_payload, separators=(",", ":")),
    ]

    if observation.action_result:
        prompt_parts.append(
            f"ACTION RESULT: {observation.action_result.status} — {observation.action_result.detail}"
        )

    if scratchpad:
        prompt_parts.append(f"YOUR PREVIOUS NOTES: {scratchpad}")

    prompt_parts.append("Produce the required JSON decision now.")

    return textwrap.dedent("\n".join(prompt_parts)).strip()


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a hospital ward assistant AI managing a full ward shift.

    The state includes a deterministic pending_tasks queue. Prefer the highest-priority task unless a recent action result shows that the workflow step is invalid.
    Critical incidents must be handled in this order:
    check_vitals -> alert_doctor -> escalate -> update_chart.
    review_patient is for routine rounding and recovering bedside context when details are stale.
    medication_round is a ward-level med pass covering all currently due medication tasks.
    ward_sweep is a lightweight indirect ward-management task for quiet periods.

    ACTION IDS:
    - A0 = review_patient(patient_id)
    - A1 = medication_round()
    - A2 = check_vitals(patient_id)
    - A3 = escalate(patient_id)
    - A4 = alert_doctor(patient_id)
    - A5 = update_chart(patient_id)
    - A6 = prepare_discharge(patient_id)
    - A7 = ward_sweep()

    REQUIRED OUTPUT FORMAT:
    You must provide your reasoning followed by the JSON action in a fenced code block.

    Example:
    Reasoning: Patient 4 has an escalation level of 2 and vitals are due. I will check vitals first to document the deterioration before escalating.
    ```json
    {
      "chosen_action_id": "A2",
      "chosen_patient_id": 4,
      "scratchpad": "P3 temp 39.5 → escalated. P7 needs vitals next."
    }
    ```

    The "scratchpad" field is optional — use it to track your notes across steps.
    Only one JSON block is allowed.
    """
).strip()

FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
FALLBACK_JSON_PATTERN = re.compile(r"(\{.*\})", re.DOTALL)


def build_fallback_action(observation: SakhaObservation) -> SakhaAction:
    if observation.ward_state.pending_tasks:
        task = observation.ward_state.pending_tasks[0]
        return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def extract_model_decision(response_text: str) -> dict | None:
    if not response_text:
        return None

    text = response_text.strip()
    match = FENCED_JSON_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match_fallback = FALLBACK_JSON_PATTERN.search(text)
    if match_fallback:
        try:
            return json.loads(match_fallback.group(1))
        except json.JSONDecodeError:
            pass

    return None


def get_patient(observation: SakhaObservation, patient_id: int):
    return next(
        (p for p in observation.ward_state.patients if p.bed_id == patient_id),
        None,
    )


def get_eligible_candidates(observation: SakhaObservation) -> dict[str, list[int | None]]:
    candidates: dict[str, list[int | None]] = {action_id: [] for action_id in ACTION_ID_TO_TYPE}
    for task in observation.ward_state.pending_tasks:
        action_id = next(
            (key for key, value in ACTION_ID_TO_TYPE.items() if value == task.required_action),
            None,
        )
        if action_id is not None:
            candidates[action_id].append(task.patient_id)
    return candidates


def rank_candidates(observation: SakhaObservation) -> list[tuple[int, str, int | None]]:
    ranked: list[tuple[int, str, int | None]] = []
    for task in observation.ward_state.pending_tasks:
        action_id = next(
            (key for key, value in ACTION_ID_TO_TYPE.items() if value == task.required_action),
            None,
        )
        if action_id is None:
            continue
        urgency_bonus = 20 if task.overdue else 0
        ranked.append((task.priority + urgency_bonus, action_id, task.patient_id))
    ranked.sort(key=lambda item: (-item[0], item[2] if item[2] is not None else -1, item[1]))
    return ranked


def select_action(
    observation: SakhaObservation, model_payload: dict | None
) -> tuple[SakhaAction, str | None]:
    if model_payload is None:
        return build_fallback_action(observation), None

    scratchpad = model_payload.get("scratchpad")
    if not isinstance(scratchpad, str):
        scratchpad = None

    try:
        chosen_action_id = model_payload["chosen_action_id"]
        chosen_patient_id = model_payload["chosen_patient_id"]
        if (
            ACTION_ID_TO_TYPE[chosen_action_id]
            not in {ActionType.WARD_SWEEP, ActionType.MEDICATION_ROUND}
            and get_patient(observation, chosen_patient_id) is None
        ):
            return build_fallback_action(observation), scratchpad
        eligible = get_eligible_candidates(observation)
        if eligible.get(chosen_action_id):
            if chosen_patient_id not in eligible[chosen_action_id]:
                return build_fallback_action(observation), scratchpad
        action = SakhaAction(
            action_type=ACTION_ID_TO_TYPE[chosen_action_id],
            patient_id=chosen_patient_id,
        )
        return action, scratchpad
    except (ValueError, KeyError, TypeError):
        return build_fallback_action(observation), scratchpad


def deterministic_policy(obs: SakhaObservation, step: int, patient_count: int) -> SakhaAction:
    ranked = rank_candidates(obs)
    if ranked:
        _, action_id, patient_id = ranked[0]
        return SakhaAction(
            action_type=ACTION_ID_TO_TYPE[action_id],
            patient_id=patient_id,
        )
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def run_episode(
    client,
    task: str,
    seed: int,
    max_steps: int = 96,
    verbose: bool = False,
    deterministic_baseline: bool = False,
) -> dict:
    started_at = time.perf_counter()
    patient_count = 5 if task == "easy" else (8 if task == "medium" else 18)
    env = SakhaEnvironment(patient_count=patient_count, task=task)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    history: list[str] = []
    scratchpad: str | None = None
    usage_totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Task: {task} | Patients: {patient_count} | Seed: {seed}")
        print(f"{'=' * 60}")

    for step in range(1, max_steps + 1):
        if deterministic_baseline:
            action = deterministic_policy(obs, step, patient_count)
        else:
            user_prompt = build_user_prompt(obs, step, history, scratchpad)
            try:
                completion = call_llm(
                    client,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    usage_totals["requests"] += 1
                    usage_totals["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                    usage_totals["completion_tokens"] += int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                    usage_totals["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
                response_text = completion.choices[0].message.content or "" if completion else ""
            except Exception as exc:
                if verbose:
                    print(f"  LLM error: {exc}")
                response_text = ""
            model_payload = extract_model_decision(response_text)
            action, scratchpad = select_action(obs, model_payload)

            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

        if verbose:
            print(f"  Step {step}: {action.action_type}(patient={action.patient_id})")

        obs = env.step(action)
        trajectory.append(obs)

        reward = obs.reward or 0.0
        action_type_value = getattr(action.action_type, "value", action.action_type)
        action_id = next(
            (key for key, value in ACTION_ID_TO_TYPE.items() if value == action_type_value),
            "UNK",
        )

        if obs.action_result:
            history_entry = f"{action_id}(patient={action.patient_id}) → {obs.action_result.detail}"
        else:
            history_entry = f"{action_id}(patient={action.patient_id}):reward={reward:+.2f}"
        history.append(history_entry)

    grader = TASK_GRADERS[task]
    score = grader(trajectory)

    result = {
        "task": task,
        "seed": seed,
        "steps": len(trajectory) - 1,
        "grader_score": score,
        "done": trajectory[-1].done,
        "runtime_seconds": round(time.perf_counter() - started_at, 4),
        "mode": "deterministic" if deterministic_baseline else "llm",
        "usage": usage_totals,
    }

    if verbose:
        print(f"  Score: {score:.4f}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sakha baseline inference")
    parser.add_argument("--tasks", default="easy,medium,hard", help="Comma-separated task names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=96,
        help="Max steps per episode (default: 96 for full 8-hour shift)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--deterministic-baseline",
        action="store_true",
        help="Use local heuristic policy instead of LLM for reproducibility",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write structured baseline results as JSON",
    )
    args = parser.parse_args()

    deterministic_mode = args.deterministic_baseline
    aggregate_usage = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if not deterministic_mode and not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable, or use --deterministic-baseline")
        print("  export HF_TOKEN='hf_your_token_here'")
        sys.exit(1)

    if deterministic_mode:
        print("MODE: deterministic baseline (local priority policy)")
        client = None
    else:
        print(f"API_BASE_URL: {API_BASE_URL}")
        print(f"MODEL_NAME:   {MODEL_NAME}")
        masked_api_key = f"{'*' * 8}...{API_KEY[-4:]}" if API_KEY else "********"
        print(f"API_KEY:      {masked_api_key}")
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=30.0)

    tasks = args.tasks.split(",")
    all_results = []
    task_summaries = []

    for task in tasks:
        task_scores = []

        for ep in range(args.episodes):
            seed = args.seed + ep
            result = run_episode(
                client,
                task,
                seed,
                max_steps=args.max_steps,
                verbose=args.verbose,
                deterministic_baseline=deterministic_mode,
            )
            all_results.append(result)
            task_scores.append(result["grader_score"])
            if not deterministic_mode:
                usage = result.get("usage") or {}
                for key in aggregate_usage:
                    aggregate_usage[key] += int(usage.get(key, 0) or 0)

        avg_score = sum(task_scores) / len(task_scores)
        task_runtime = round(
            sum(result["runtime_seconds"] for result in all_results if result["task"] == task),
            4,
        )
        print(f"[{task}] avg_score={avg_score:.4f} (over {args.episodes} episodes)")
        task_summaries.append(
            {
                "task": task,
                "episodes": args.episodes,
                "avg_score": round(avg_score, 4),
                "runtime_seconds": task_runtime,
            }
        )

    print("\n=== Full Results ===")
    print(json.dumps(all_results, indent=2))
    if not deterministic_mode and aggregate_usage["requests"] > 0:
        avg_prompt = aggregate_usage["prompt_tokens"] / aggregate_usage["requests"]
        avg_completion = aggregate_usage["completion_tokens"] / aggregate_usage["requests"]
        avg_total = aggregate_usage["total_tokens"] / aggregate_usage["requests"]
        print("\n=== Token Usage ===")
        print(f"Requests:           {aggregate_usage['requests']}")
        print(f"Prompt tokens:      {aggregate_usage['prompt_tokens']}")
        print(f"Completion tokens:  {aggregate_usage['completion_tokens']}")
        print(f"Total tokens:       {aggregate_usage['total_tokens']}")
        print(f"Avg prompt/request: {avg_prompt:.1f}")
        print(f"Avg completion/req: {avg_completion:.1f}")
        print(f"Avg total/request:  {avg_total:.1f}")
    if args.output_json:
        payload = {
            "tasks": task_summaries,
            "episodes": all_results,
            "model_name": MODEL_NAME,
            "mode": "deterministic" if deterministic_mode else "llm",
            "seed": args.seed,
            "max_steps": args.max_steps,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
