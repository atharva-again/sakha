"""
Sakha Ward Assistant — Baseline Inference Script
=================================================

Environment variables (MANDATORY):
    API_BASE_URL    The API endpoint for the LLM (default: https://api.groq.com/openai/v1)
    MODEL_NAME      Model identifier (default: llama-3.1-8b-instant)
    HF_TOKEN        Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export HF_TOKEN="hf_your_token_here"
    export MODEL_NAME="llama-3.1-8b-instant"
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
from sakha.formatters import (
    CompactFormatter,
    EpisodeResult,
    Formatter,
    JSONFormatter,
    StepData,
    get_formatter,
)
from sakha.graders import score_easy_task, score_hard_task, score_medium_task
from sakha.models import ActionType, SakhaAction, SakhaObservation

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for logger_name in ("httpx", "openai", "httpcore", "httpcore.http11"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger("sakha")


def call_llm(client, messages):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0"))
PROMPT_PROFILE = os.getenv("PROMPT_PROFILE", "operational_realism").strip()
ALLOWED_PROMPT_PROFILES = {"operational_realism", "strict_bedside", "full_legacy"}

MAX_STEPS = 96
TEMPERATURE = 0.0
MAX_TOKENS = 64
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

MEDICATION_GRACE_STEPS = 2
VITALS_GRACE_STEPS = 1


def serialize_compact_patient(patient) -> dict[str, object]:
    last_vitals = getattr(patient, "last_vitals", None)
    last_vitals_payload = None
    if last_vitals is not None:
        last_vitals_payload = {
            "bp_sys": int(getattr(last_vitals, "blood_pressure_sys", 0) or 0),
            "bp_dia": int(getattr(last_vitals, "blood_pressure_dia", 0) or 0),
            "temp": float(getattr(last_vitals, "temperature", 0.0) or 0.0),
            "spo2": int(getattr(last_vitals, "spo2", 0) or 0),
            "pulse": int(getattr(last_vitals, "pulse", 0) or 0),
        }

    return {
        "bed": int(getattr(patient, "bed_id", -1) or -1),
        "diag": str(getattr(patient, "diagnosis", "")),
        "meds": len(getattr(patient, "medications_due", []) or []),
        "med_step": int(getattr(patient, "medication_due_by_step", -1) or -1),
        "vitals": bool(getattr(patient, "vitals_due", False)),
        "vit_step": int(getattr(patient, "vitals_due_by_step", -1) or -1),
        "esc": int(getattr(patient, "escalation_level", 0) or 0),
        "rev": bool(getattr(patient, "review_required", False)),
        "rev_step": int(getattr(patient, "last_reviewed_step", -1) or -1),
        "inc": int(getattr(patient, "active_incident_id", -1) or -1),
        "discharge": bool(getattr(patient, "discharge_prepared", False)),
        "adm_req": bool(getattr(patient, "admission_review_required", False)),
        "adm_reviewed": bool(getattr(patient, "admission_reviewed", False)),
        "adm_documented": bool(getattr(patient, "admission_documented", False)),
        "last_vitals": last_vitals_payload,
    }


def build_user_prompt(
    observation: SakhaObservation,
    step: int,
    history: list[str],
    prompt_profile: str,
    scratchpad: str | None = None,
) -> str:
    ward = observation.ward_state

    patients_payload = [serialize_compact_patient(p) for p in ward.patients]
    if prompt_profile == "full_legacy":
        patients_payload = [p.model_dump() for p in ward.patients]

    state_payload = {
        "step": step,
        "time_remaining_minutes": observation.time_remaining_minutes,
        "patients": patients_payload,
        "recent_actions": history[-3:] if history else [],
        "action_result": (
            {
                "status": observation.action_result.status,
                "detail": observation.action_result.detail,
            }
            if observation.action_result
            else None
        ),
    }

    if prompt_profile == "operational_realism":
        state_payload["pending_count"] = observation.pending_count
    elif prompt_profile == "strict_bedside":
        pass
    elif prompt_profile == "full_legacy":
        state_payload["pending_count"] = observation.pending_count
        state_payload["pending_tasks"] = [
            task.model_dump(mode="json") for task in ward.pending_tasks
        ]
    else:
        raise ValueError(f"Invalid PROMPT_PROFILE: {prompt_profile}")

    prompt_parts = [
        "STATE JSON:",
        json.dumps(state_payload, separators=(",", ":")),
    ]

    if scratchpad:
        prompt_parts.append(f"YOUR PREVIOUS NOTES: {scratchpad}")

    prompt_parts.append("Produce the required JSON decision now.")

    return textwrap.dedent("\n".join(prompt_parts)).strip()


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a hospital ward assistant AI managing a full ward shift.
    Use only the provided state JSON.
    Critical incidents must be handled in this exact order:
    check_vitals -> alert_doctor -> escalate -> update_chart.

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
    Return JSON only. No prose, no markdown, no code fences.
    Output schema:
    {
      "chosen_action_id": "A0|A1|A2|A3|A4|A5|A6|A7",
      "chosen_patient_id": <integer or null>,
      "scratchpad": <optional string>
    }
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
    scored: list[tuple[int, int, int, str, int | None]] = []

    patients = [
        p for p in observation.ward_state.patients if not getattr(p, "discharge_prepared", False)
    ]

    medication_due_patients = [p for p in patients if bool(getattr(p, "medications_due", []))]
    if medication_due_patients:
        earliest_due = min(
            int(getattr(p, "medication_due_by_step", -1) or -1) for p in medication_due_patients
        )
        overdue = any(
            earliest_due >= 0
            and observation.ward_state.current_step > earliest_due + MEDICATION_GRACE_STEPS
            for _ in [0]
        )
        priority = 510 if overdue else 410
        scored.append((priority, earliest_due if earliest_due >= 0 else 10**9, 10**9, "A1", None))

    for patient in patients:
        bed_id = int(getattr(patient, "bed_id", 10**9) or 10**9)
        incident_id = int(getattr(patient, "active_incident_id", -1) or -1)
        incident_due = int(getattr(patient, "incident_deadline_step", 10**9) or 10**9)
        if incident_id >= 0:
            if not bool(getattr(patient, "incident_checked", False)):
                scored.append((650, incident_due, bed_id, "A2", bed_id))
            elif not bool(getattr(patient, "incident_alerted", False)):
                scored.append((640, incident_due, bed_id, "A4", bed_id))
            elif not bool(getattr(patient, "incident_escalated", False)):
                scored.append((630, incident_due, bed_id, "A3", bed_id))
            elif not bool(getattr(patient, "incident_documented", False)):
                scored.append((620, incident_due + 1, bed_id, "A5", bed_id))
            continue

        if bool(getattr(patient, "vitals_due", False)):
            vitals_due = int(getattr(patient, "vitals_due_by_step", 10**9) or 10**9)
            overdue = (
                vitals_due >= 0
                and observation.ward_state.current_step > vitals_due + VITALS_GRACE_STEPS
            )
            priority = 520 if overdue else 420
            scored.append((priority, vitals_due, bed_id, "A2", bed_id))

        if bool(getattr(patient, "admission_review_required", False)):
            admission_due = int(getattr(patient, "admission_due_step", 10**9) or 10**9)
            if not bool(getattr(patient, "admission_reviewed", False)):
                scored.append((350, admission_due, bed_id, "A0", bed_id))
            elif not bool(getattr(patient, "admission_documented", False)):
                scored.append((340, admission_due, bed_id, "A5", bed_id))

        if bool(getattr(patient, "review_required", False)):
            scored.append((220, observation.ward_state.current_step + 1, bed_id, "A0", bed_id))

        ready_for_discharge = (
            incident_id < 0
            and int(getattr(patient, "escalation_level", 0) or 0) == 0
            and not bool(getattr(patient, "medications_due", []))
            and not bool(getattr(patient, "vitals_due", False))
            and not bool(getattr(patient, "admission_review_required", False))
            and int(getattr(patient, "last_documented_step", -1) or -1)
            >= observation.ward_state.current_step - 10
            and int(getattr(patient, "last_reviewed_step", -1) or -1)
            >= observation.ward_state.current_step - 10
            and observation.ward_state.current_step
            - int(getattr(patient, "admission_step", 0) or 0)
            >= 12
        )
        if ready_for_discharge:
            scored.append((210, observation.ward_state.current_step + 4, bed_id, "A6", bed_id))

    if not scored:
        scored.append((100, observation.ward_state.current_step + 1, 10**9, "A7", None))

    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [
        (priority, action_id, patient_id) for priority, _due, _bed, action_id, patient_id in scored
    ]


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
    episode_index: int,
    max_steps: int = 96,
    verbose: bool = False,
    deterministic_baseline: bool = False,
    prompt_profile: str = "operational_realism",
    formatter: Formatter | None = None,
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
    mode = "deterministic" if deterministic_baseline else "llm"

    run_formatter: Formatter = formatter or CompactFormatter()
    run_formatter.start_episode(task, episode_index, seed, patient_count, max_steps, mode)

    for step in range(1, max_steps + 1):
        if deterministic_baseline:
            action = deterministic_policy(obs, step, patient_count)
        else:
            user_prompt = build_user_prompt(obs, step, history, prompt_profile, scratchpad)
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
                print(f"  LLM error at step {step}: {exc}", flush=True)
                response_text = ""
            model_payload = extract_model_decision(response_text)
            action, scratchpad = select_action(obs, model_payload)

            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

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

        action_name = getattr(action.action_type, "value", action.action_type)
        status = obs.action_result.status if obs.action_result else "none"
        step_data = StepData(
            task=task,
            episode=episode_index,
            step_num=step,
            action_name=str(action_name),
            patient_id=action.patient_id,
            reward=reward,
            status=status,
            done=obs.done,
        )
        run_formatter.step(step_data)

    grader = TASK_GRADERS[task]
    score = grader(trajectory)
    final_patients = trajectory[-1].ward_state.patients if trajectory else []
    critical_incidents_missed = sum(
        int(getattr(patient, "critical_incidents_missed", 0) or 0) for patient in final_patients
    )
    runtime = round(time.perf_counter() - started_at, 4)

    result = {
        "task": task,
        "seed": seed,
        "steps": len(trajectory) - 1,
        "grader_score": score,
        "done": trajectory[-1].done,
        "runtime_seconds": runtime,
        "mode": "deterministic" if deterministic_baseline else "llm",
        "usage": usage_totals,
        "critical_incidents_missed": critical_incidents_missed,
    }

    episode_result = EpisodeResult(
        task=task,
        episode=episode_index,
        seed=seed,
        score=score,
        steps=len(trajectory) - 1,
        done=trajectory[-1].done,
        runtime_seconds=runtime,
        critical_incidents_missed=critical_incidents_missed,
    )
    run_formatter.end_episode(episode_result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sakha baseline inference")
    parser.add_argument("--tasks", default="easy,medium,hard", help="Comma-separated task names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Max steps per episode (default: 24 for fast dev loop)",
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
    parser.add_argument(
        "--format",
        default="compact",
        choices=["compact", "pretty", "json"],
        help="Output format (default: compact)",
    )
    args = parser.parse_args()

    if PROMPT_PROFILE not in ALLOWED_PROMPT_PROFILES:
        print("ERROR: PROMPT_PROFILE must be one of operational_realism|strict_bedside|full_legacy")
        sys.exit(2)

    deterministic_mode = args.deterministic_baseline
    aggregate_usage = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if not deterministic_mode and not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable, or use --deterministic-baseline")
        print("  export HF_TOKEN='hf_your_token_here'")
        sys.exit(1)

    if deterministic_mode:
        print("MODE: deterministic baseline (local priority policy)")
        print(f"PROMPT_PROFILE: {PROMPT_PROFILE}")
        client = None
    else:
        print(f"API_BASE_URL: {API_BASE_URL}")
        print(f"MODEL_NAME:   {MODEL_NAME}")
        print(f"PROMPT_PROFILE: {PROMPT_PROFILE}")
        masked_api_key = f"{'*' * 8}...{HF_TOKEN[-4:]}" if HF_TOKEN else "********"
        print(f"API_KEY:      {masked_api_key}")
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30.0)

    formatter = get_formatter(args.format, args.output_json)

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
                episode_index=ep + 1,
                max_steps=args.max_steps,
                verbose=args.verbose,
                deterministic_baseline=deterministic_mode,
                prompt_profile=PROMPT_PROFILE,
                formatter=formatter,
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
        task_summaries.append(
            {
                "task": task,
                "episodes": args.episodes,
                "avg_score": round(avg_score, 4),
                "runtime_seconds": task_runtime,
            }
        )

    episode_results = [
        EpisodeResult(
            task=r["task"],
            episode=-1,
            seed=r["seed"],
            score=r["grader_score"],
            steps=r["steps"],
            done=r["done"],
            runtime_seconds=r["runtime_seconds"],
            critical_incidents_missed=r["critical_incidents_missed"],
        )
        for r in all_results
    ]
    formatter.summary(episode_results)

    if args.format != "json" and args.output_json:
        payload = {
            "tasks": task_summaries,
            "episodes": all_results,
            "model_name": MODEL_NAME,
            "mode": "deterministic" if deterministic_mode else "llm",
            "prompt_profile": PROMPT_PROFILE,
            "seed": args.seed,
            "max_steps": args.max_steps,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")

    if not args.output_json and not deterministic_mode and aggregate_usage["requests"] > 0:
        print()
        print("═" * 30)
        print(" TOKEN USAGE")
        print("═" * 30)
        print(f"  Requests:      {aggregate_usage['requests']}")
        print(f"  Prompt:       {aggregate_usage['prompt_tokens']:,}")
        print(f"  Completion:   {aggregate_usage['completion_tokens']:,}")
        print(f"  Total:       {aggregate_usage['total_tokens']:,}")


if __name__ == "__main__":
    main()
