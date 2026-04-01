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
import os
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
            response_format={"type": "json_object"},
        )
        return completion
    except Exception as exc:
        if "429" in str(exc) or "rate_limit" in str(exc).lower():
            print(f"  Rate limited: {exc}")
        raise


from sakha.env import SakhaEnvironment
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import (
    ActionType,
    PatientStatus,
    SakhaAction,
    SakhaObservation,
)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")

MAX_STEPS = 96
TEMPERATURE = 0.2
MAX_TOKENS = 150
FALLBACK_ACTION = SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=1)
ACTION_ID_TO_TYPE = {
    "A1": ActionType.ADMINISTER_MEDICINE,
    "A2": ActionType.CHECK_VITALS,
    "A3": ActionType.ESCALATE,
    "A4": ActionType.DOCUMENT,
    "A5": ActionType.COMMUNICATE,
    "A6": ActionType.HANDOVER,
}

TASK_GRADERS = {
    "easy": score_easy_task,
    "medium": score_medium_task,
    "hard": score_hard_task,
}


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a hospital ward assistant AI managing patient care in a realistic ward.

    Your task is to select one action using the current state.

    ACTION IDS:
    - A1 = administer_medicine(patient_id)
    - A2 = check_vitals(patient_id)
    - A3 = escalate(patient_id)  - urgent for critical patients
    - A4 = document(patient_id)  - record patient status
    - A5 = communicate()  - coordinate with team
    - A6 = handover()  - shift change briefing

    PRIORITY RULES (in order):
    1. ESCALATE any patient with escalation_level >= 2 (critical)
    2. CHECK_VITALS for patients with vitals due who are critical or elevated
    3. ADMINISTER_MEDICINE for patients with medications due
    4. CHECK_VITALS for any patient with vitals due
    5. DOCUMENT for patients not recently documented
    6. HANDOVER when shift is ending (hour 7+)
    7. NOOP only when nothing urgent

    CONSTRAINTS:
    - Max 3 actions per step (nurse fatigue)
    - Tasks have deadlines - on-time completion scores higher
    - New patients may arrive - check waiting queue
    - Patients may deteriorate - monitor trends

    OUTPUT FORMAT:
    Return exactly one JSON object:
    {
      "chosen_action_id": "A1",
      "chosen_patient_id": 1
    }

    Return JSON only.
    """
).strip()


def build_user_prompt(observation: SakhaObservation, step: int, history: list[str]) -> str:
    ward = observation.ward_state

    admitted_patients = []
    waiting_count = 0
    critical_count = 0

    if hasattr(ward, 'admitted_patients'):
        admitted_patients = list(ward.admitted_patients)
        waiting_count = len(ward.waiting_patients) if hasattr(ward, 'waiting_patients') else 0
        critical_count = len(ward.critical_patients) if hasattr(ward, 'critical_patients') else 0
    else:
        admitted_patients = list(ward.patients) if hasattr(ward, 'patients') else []

    critical_patients = []
    meds_due_patients = []
    vitals_due_patients = []
    elevated_patients = []

    for p in admitted_patients:
        pid = p.patient_id if hasattr(p, 'patient_id') else getattr(p, 'bed_id', 0)
        escalation = getattr(p, 'escalation_level', 0)

        if escalation >= 2:
            critical_patients.append(pid)
        elif escalation >= 1:
            elevated_patients.append(pid)

        meds = getattr(p, 'medications_due', [])
        if meds:
            meds_due_patients.append({"patient_id": pid, "count": len(meds)})

        vitals = getattr(p, 'vitals_due', False)
        if vitals:
            vitals_due_patients.append(pid)

    recent_action_ids = [item.split(":", 1)[0] for item in history[-3:]] if history else []

    state_payload = {
        "step": step,
        "shift_hour": getattr(observation, 'shift_hour', step // 12),
        "nurse_fatigue": round(getattr(observation, 'nurse_fatigue', 0), 2),
        "time_remaining_minutes": observation.time_remaining_minutes,
        "admitted_count": len(admitted_patients),
        "waiting_count": waiting_count,
        "critical_count": critical_count,
        "critical_patients": critical_patients,
        "elevated_patients": elevated_patients,
        "meds_due_patients": meds_due_patients,
        "vitals_due_patients": vitals_due_patients,
        "recent_action_ids": recent_action_ids,
    }

    return textwrap.dedent(
        f"""
        STATE JSON:
        {json.dumps(state_payload, separators=(",", ":"))}

        Choose the best action now."""
    ).strip()


def build_fallback_action(observation: SakhaObservation) -> SakhaAction:
    ward = observation.ward_state
    patients = ward.admitted_patients if hasattr(ward, 'admitted_patients') else ward.patients

    for patient in patients:
        if getattr(patient, 'escalation_level', 0) >= 2:
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.ESCALATE, patient_id=pid)

    for patient in patients:
        if getattr(patient, 'vitals_due', False):
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=pid)

    for patient in patients:
        if getattr(patient, 'medications_due', []):
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=pid)

    return FALLBACK_ACTION


def extract_model_decision(response_text: str) -> dict | None:
    if not response_text:
        return None

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def get_patient(observation: SakhaObservation, patient_id: int):
    ward = observation.ward_state
    patients = ward.admitted_patients if hasattr(ward, 'admitted_patients') else ward.patients
    return next(
        (p for p in patients if getattr(p, 'patient_id', getattr(p, 'bed_id', None)) == patient_id),
        None,
    )


def get_eligible_candidates(observation: SakhaObservation) -> dict[str, list[int]]:
    ward = observation.ward_state
    patients = ward.admitted_patients if hasattr(ward, 'admitted_patients') else ward.patients

    candidates = {
        "A3": [],
        "A1": [],
        "A2": [],
    }

    for p in patients:
        pid = getattr(p, 'patient_id', getattr(p, 'bed_id', 0))

        if getattr(p, 'escalation_level', 0) >= 2:
            candidates["A3"].append(pid)

        if getattr(p, 'medications_due', []):
            candidates["A1"].append(pid)

        if getattr(p, 'vitals_due', False):
            candidates["A2"].append(pid)

    return candidates


def has_abnormal_vitals(patient) -> bool:
    vitals = getattr(patient, 'vitals', None)
    if vitals is None:
        return False
    return vitals.temperature >= 38.5 or vitals.spo2 < 94 or vitals.pulse >= 100


def priority_policy(observation: SakhaObservation) -> SakhaAction:
    ward = observation.ward_state
    patients = ward.admitted_patients if hasattr(ward, 'admitted_patients') else ward.patients

    for patient in patients:
        if getattr(patient, 'escalation_level', 0) >= 2:
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.ESCALATE, patient_id=pid)

    for patient in patients:
        if getattr(patient, 'vitals_due', False) and getattr(patient, 'escalation_level', 0) >= 1:
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=pid)

    for patient in patients:
        if getattr(patient, 'vitals_due', False) and has_abnormal_vitals(patient):
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=pid)

    for patient in patients:
        if getattr(patient, 'medications_due', []):
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=pid)

    checked_this_round = set()
    for patient in patients:
        if getattr(patient, 'vitals_due', False):
            pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))
            if pid not in checked_this_round:
                checked_this_round.add(pid)
                return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=pid)

    shift_hour = getattr(observation, 'shift_hour', 0)
    if shift_hour >= 7:
        nurse = getattr(ward, 'nurse', None)
        if nurse and not getattr(nurse, 'handover_completed', True):
            return SakhaAction(action_type=ActionType.HANDOVER)

    return SakhaAction(action_type=ActionType.NOOP)


def choose_balanced_action(
    observation: SakhaObservation, history: list[str], model_payload: dict | None
) -> SakhaAction:
    eligible = get_eligible_candidates(observation)

    if model_payload is not None:
        model_action_id = model_payload.get("chosen_action_id")
        model_patient_id = model_payload.get("chosen_patient_id")
        if model_action_id in ACTION_ID_TO_TYPE and isinstance(model_patient_id, int):
            if model_patient_id in eligible.get(model_action_id, []):
                return SakhaAction(
                    action_type=ACTION_ID_TO_TYPE[model_action_id],
                    patient_id=model_patient_id,
                )
            return SakhaAction(action_type=ActionType.NOOP)

    return priority_policy(observation)


def deterministic_policy(obs, step, patient_count, history: list[str]):
    action = priority_policy(obs)

    if history:
        recent = history[-3:]
        action_key = next((k for k, v in ACTION_ID_TO_TYPE.items() if v == action.action_type), None)
        if action_key:
            action_key = f"{action_key}:patient={action.patient_id}"
            same_count = sum(1 for h in recent if action_key in h)

            if same_count >= 2:
                ward = obs.ward_state
                patients = ward.admitted_patients if hasattr(ward, 'admitted_patients') else ward.patients
                for patient in patients:
                    pid = getattr(patient, 'patient_id', getattr(patient, 'bed_id', 0))
                    if pid == action.patient_id:
                        continue
                    if action.action_type == ActionType.CHECK_VITALS and getattr(patient, 'vitals_due', False):
                        return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=pid)
                    if action.action_type == ActionType.ADMINISTER_MEDICINE and getattr(patient, 'medications_due', []):
                        return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=pid)

    return action


def run_episode(
    client,
    task: str,
    seed: int,
    max_steps: int = 20,
    verbose: bool = False,
    deterministic_baseline: bool = False,
) -> dict:
    started_at = time.perf_counter()
    env = SakhaEnvironment(patient_count=0, task=task)
    obs = env.reset(seed=seed)
    trajectory = [obs]
    history = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Task: {task} | Seed: {seed}")
        print(f"{'=' * 60}")

    for step in range(1, max_steps + 1):
        if obs.done:
            if verbose:
                print(f"Episode complete at step {step - 1}")
            break

        if deterministic_baseline:
            action = deterministic_policy(obs, step, 0, history)
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

        try:
            obs = env.step(action)
        except Exception as exc:
            print(f"  Step {step} failed: {exc}, falling back to noop")
            obs = env.step(SakhaAction(action_type=ActionType.NOOP))
        trajectory.append(obs)

        reward = obs.reward or 0.0
        action_type_value = getattr(action.action_type, "value", action.action_type)
        action_id = next(
            (
                key
                for key, value in ACTION_ID_TO_TYPE.items()
                if getattr(value, "value", value) == action_type_value
            ),
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
        default=96,
        help="Max steps per episode (default: 96 = full 8-hour shift)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument(
        "--deterministic-baseline",
        action="store_true",
        help="Use local ranked-priority policy instead of LLM for reproducibility",
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
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tasks": task_summaries,
            "episodes": all_results,
            "model_name": MODEL_NAME,
            "mode": "deterministic" if deterministic_mode else "llm",
            "seed": args.seed,
            "max_steps": args.max_steps,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
