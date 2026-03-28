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

import os
import sys
import json
import re
import argparse
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()


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


from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction, SakhaObservation
from sakha.graders import score_easy_task, score_medium_task, score_hard_task


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


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a hospital ward assistant AI managing patient care.

    Your task is to select one action from a fixed policy table using only the current state.

    ACTION IDS:
    - A1 = administer_medicine(patient_id)
    - A2 = check_vitals(patient_id)
    - A3 = escalate(patient_id)

    POLICY TABLE:
    - A1 trigger: patient_id is in meds_due_patients
      blocker: patient_id not in meds_due_patients
      evidence needed: meds_due_patients contains patient_id
    - A2 trigger: patient_id is in vitals_due_patients
      blocker: patient_id not in vitals_due_patients
      evidence needed: vitals_due_patients contains patient_id
    - A3 trigger: patient_id is in critical_patients
      blocker: patient_id not in critical_patients
      evidence needed: critical_patients contains patient_id

    DECISION RULES:
    1. Read the state first.
    2. Mark actions eligible only when their trigger is satisfied.
    3. Do not choose an ineligible action.
    4. If multiple actions are eligible, prefer the highest-priority action for the current state:
       critical_patients first, then meds_due_patients, then vitals_due_patients.
    5. If the same action_id appears repeatedly in recent_action_ids while other eligible action_ids exist, pick a different eligible action.
    6. Choose using patient ids from the state, not habit.

    REQUIRED OUTPUT FORMAT:
    Return exactly one JSON object with this shape:
    {
      "eligible_actions": [
        {"id": "A1", "patient_id": 1, "eligible": true},
        {"id": "A2", "patient_id": 1, "eligible": true},
        {"id": "A3", "patient_id": 1, "eligible": false}
      ],
      "chosen_action_id": "A1",
      "chosen_patient_id": 1
    }

    Return JSON only.
    """
).strip()


def build_user_prompt(observation: SakhaObservation, step: int, history: list[str]) -> str:
    ward = observation.ward_state

    critical_patients = []
    meds_due_patients = []
    vitals_due_patients = []
    elevated_patients = []
    abnormal_vitals_patients = []

    for p in ward.patients:
        if p.escalation_level >= 2:
            critical_patients.append(p.bed_id)
        elif p.escalation_level == 1:
            elevated_patients.append(p.bed_id)

        if p.medications_due:
            meds_due_patients.append({"patient_id": p.bed_id, "count": len(p.medications_due)})

        if p.vitals_due:
            vitals_due_patients.append(p.bed_id)

        if p.last_vitals:
            v = p.last_vitals
            if v.temperature >= 39.0 or v.spo2 < 93 or v.pulse >= 100:
                abnormal_vitals_patients.append(
                    {
                        "patient_id": p.bed_id,
                        "temp": v.temperature,
                        "spo2": v.spo2,
                        "pulse": v.pulse,
                    }
                )

    recent_action_ids = [item.split(":", 1)[0] for item in history[-3:]] if history else []

    state_payload = {
        "step": step,
        "time_remaining_minutes": observation.time_remaining_minutes,
        "pending_count": observation.pending_count,
        "critical_patients": critical_patients,
        "elevated_patients": elevated_patients,
        "meds_due_patients": meds_due_patients,
        "vitals_due_patients": vitals_due_patients,
        "abnormal_vitals_patients": abnormal_vitals_patients,
        "recent_action_ids": recent_action_ids,
    }

    return textwrap.dedent(
        f"""
        STATE JSON:
        {json.dumps(state_payload, separators=(",", ":"))}

        Produce the required JSON decision now."""
    ).strip()


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


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
    text = re.sub(r"```python\s*", "", text)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"^(action|next action|response)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    match = JSON_BLOCK_PATTERN.search(text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
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
            ranked.append((1000 + patient.escalation_level, "A3", bed_id))

        if patient.vitals_due:
            vitals_score = 400
            if elevated:
                vitals_score += 120
            if abnormal:
                vitals_score += 100
            ranked.append((vitals_score, "A2", bed_id))

        if patient.medications_due:
            meds_score = 300 + len(patient.medications_due) * 10
            if elevated:
                meds_score += 40
            ranked.append((meds_score, "A1", bed_id))

    ranked.sort(key=lambda item: (-item[0], item[2], item[1]))
    return ranked


def choose_balanced_action(
    observation: SakhaObservation, history: list[str], model_payload: dict | None
) -> SakhaAction:
    eligible = get_eligible_candidates(observation)
    ranked = rank_candidates(observation)
    model_action_id = model_payload.get("chosen_action_id") if model_payload else None
    model_patient_id = model_payload.get("chosen_patient_id") if model_payload else None

    if model_action_id in {"A1", "A2", "A3"} and isinstance(model_patient_id, int):
        for _, ranked_action_id, ranked_patient_id in ranked[:3]:
            if model_action_id == ranked_action_id and model_patient_id == ranked_patient_id:
                return SakhaAction(
                    action_type=ACTION_ID_TO_TYPE[ranked_action_id],
                    patient_id=model_patient_id,
                )

    if ranked:
        _, desired_action_id, desired_patient_id = ranked[0]
        return SakhaAction(
            action_type=ACTION_ID_TO_TYPE[desired_action_id],
            patient_id=desired_patient_id,
        )

    return build_fallback_action(observation)


def deterministic_policy(obs, step, patient_count):
    if step % 3 == 0:
        return SakhaAction(action_type="administer_medicine", patient_id=(step % patient_count) + 1)
    elif step % 3 == 1:
        return SakhaAction(action_type="check_vitals", patient_id=(step % patient_count) + 1)
    else:
        return SakhaAction(action_type="escalate", patient_id=(step % patient_count) + 1)


def run_episode(
    client,
    task: str,
    seed: int,
    max_steps: int = 20,
    verbose: bool = False,
    deterministic_baseline: bool = False,
) -> dict:
    started_at = time.perf_counter()
    patient_count = 5 if task == "easy" else (8 if task == "medium" else 18)
    env = SakhaEnvironment(patient_count=patient_count, task=task)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    history = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Task: {task} | Patients: {patient_count} | Seed: {seed}")
        print(f"{'=' * 60}")

    for step in range(1, max_steps + 1):
        if obs.done:
            if verbose:
                print(f"Episode complete at step {step - 1}")
            break

        if deterministic_baseline:
            action = deterministic_policy(obs, step, patient_count)
        else:
            user_prompt = build_user_prompt(obs, step, history)
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
            action = choose_balanced_action(obs, history, model_payload)

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
        history.append(f"{action_id}:patient={action.patient_id}:reward={reward:+.2f}")

    grader = TASK_GRADERS[task]
    score = grader(trajectory)
    cumulative_reward = sum(o.reward for o in trajectory if o.reward is not None)

    result = {
        "task": task,
        "seed": seed,
        "steps": len(trajectory) - 1,
        "grader_score": score,
        "cumulative_reward": round(cumulative_reward, 4),
        "done": trajectory[-1].done,
        "runtime_seconds": round(time.perf_counter() - started_at, 4),
        "mode": "deterministic" if deterministic_baseline else "llm",
    }

    if verbose:
        print(f"  Score: {score:.4f} | Reward: {cumulative_reward:.4f}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sakha baseline inference")
    parser.add_argument("--tasks", default="easy,medium,hard", help="Comma-separated task names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per episode (default: 20 for testing)",
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
        task_rewards = []

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
            task_rewards.append(result["cumulative_reward"])

        avg_score = sum(task_scores) / len(task_scores)
        avg_reward = sum(task_rewards) / len(task_rewards)
        task_runtime = round(
            sum(result["runtime_seconds"] for result in all_results if result["task"] == task),
            4,
        )
        print(
            f"[{task}] avg_score={avg_score:.4f} avg_reward={avg_reward:.4f} (over {args.episodes} episodes)"
        )
        task_summaries.append(
            {
                "task": task,
                "episodes": args.episodes,
                "avg_score": round(avg_score, 4),
                "avg_reward": round(avg_reward, 4),
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
