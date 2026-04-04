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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import SakhaAction, SakhaObservation

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sakha")


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_llm_with_retry(client, messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion
    except Exception as exc:
        if "429" in str(exc) or "rate_limit" in str(exc).lower():
            print(f"  Rate limited: {exc}")
        raise


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")

MAX_STEPS = 96
TEMPERATURE = 0.2
MAX_TOKENS = 150
FALLBACK_ACTION = SakhaAction(action_type="check_vitals", patient_id=1)
ACTION_ID_TO_TYPE = {
    "A1": "administer_medicine",
    "A2": "check_vitals",
    "A3": "escalate",
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
    You are a hospital ward assistant AI managing patient care.

    Your task is to analyze the current ward state and select the most appropriate action based on the ABCDE clinical priority protocol.

    ABCDE PRIORITIES:
    1. A/B/C (Critical): escalation_level >= 2. Action: check_vitals (to document deterioration) then escalate.
    2. D (Potential): elevated status or abnormal vitals. Action: check_vitals.
    3. E (Routine): medications due. Action: administer_medicine.

    ACTION IDS:
    - A1 = administer_medicine(patient_id)
    - A2 = check_vitals(patient_id)
    - A3 = escalate(patient_id)

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
    for patient in observation.ward_state.patients:
        if patient.escalation_level >= 2:
            return SakhaAction(action_type="escalate", patient_id=patient.bed_id)
    for patient in observation.ward_state.patients:
        if patient.medications_due:
            return SakhaAction(action_type="administer_medicine", patient_id=patient.bed_id)
    for patient in observation.ward_state.patients:
        if patient.vitals_due:
            return SakhaAction(action_type="check_vitals", patient_id=patient.bed_id)
    return FALLBACK_ACTION


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


def get_eligible_candidates(observation: SakhaObservation) -> dict[str, list[int]]:
    return {
        "A3": [p.bed_id for p in observation.ward_state.patients if p.escalation_level >= 2],
        "A1": [p.bed_id for p in observation.ward_state.patients if p.medications_due],
        "A2": [p.bed_id for p in observation.ward_state.patients if p.vitals_due],
    }


def has_abnormal_vitals(patient) -> bool:
    if patient.last_vitals is None:
        return False
    v = patient.last_vitals
    return v.temperature >= 39.0 or v.spo2 < 93 or v.pulse >= 100


def rank_candidates(observation: SakhaObservation) -> list[tuple[int, str, int]]:
    ranked: list[tuple[int, str, int]] = []

    for patient in observation.ward_state.patients:
        bed_id = patient.bed_id
        elevated = patient.escalation_level >= 1
        abnormal = has_abnormal_vitals(patient)

        if patient.escalation_level >= 2:
            if patient.vitals_due:
                ranked.append((2000, "A2", bed_id))
                ranked.append((1500, "A3", bed_id))
            else:
                ranked.append((2500, "A3", bed_id))

        elif patient.vitals_due:
            vitals_score = 800
            if abnormal:
                vitals_score += 400
            if elevated:
                vitals_score += 200
            ranked.append((vitals_score, "A2", bed_id))

        if patient.medications_due:
            meds_score = 600 + len(patient.medications_due) * 20
            if elevated:
                meds_score += 100
            ranked.append((meds_score, "A1", bed_id))

    ranked.sort(key=lambda item: (-item[0], item[2], item[1]))
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
        action = SakhaAction(
            action_type=ACTION_ID_TO_TYPE[model_payload["chosen_action_id"]],
            patient_id=model_payload["chosen_patient_id"],
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
    return SakhaAction(action_type="noop", patient_id=None)


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
                completion = call_llm_with_retry(
                    client,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                response_text = completion.choices[0].message.content or "" if completion else ""
            except Exception as exc:
                if verbose:
                    print(f"  LLM error: {exc}")
                response_text = ""
            model_payload = extract_model_decision(response_text)
            action, scratchpad = select_action(obs, model_payload)

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
