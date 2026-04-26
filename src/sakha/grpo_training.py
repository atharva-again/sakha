"""Utilities for state-aligned GRPO training examples and rewards."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from typing import Any

from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction, SakhaObservation

ACTION_NAME_MAP = {
    "review_patient": ActionType.REVIEW_PATIENT,
    "check_vitals": ActionType.CHECK_VITALS,
    "administer_medicine": ActionType.ADMINISTER_MEDICINE,
    "alert_doctor": ActionType.ALERT_DOCTOR,
    "escalate": ActionType.ESCALATE,
    "update_chart": ActionType.UPDATE_CHART,
    "prepare_discharge": ActionType.PREPARE_DISCHARGE,
    "medication_round": ActionType.MEDICATION_ROUND,
    "ward_sweep": ActionType.WARD_SWEEP,
    "noop": ActionType.NOOP,
}

DEFAULT_STATE_STEPS = (0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88)
ACTION_RE = re.compile(r"([a-zA-Z_]+)\s*\(\s*(\d+)?\s*\)")


def parse_action_response_with_status(response: str) -> tuple[SakhaAction, bool]:
    """Parse the first valid Sakha action call and report whether parsing succeeded."""
    response = response.strip().lower()
    match = ACTION_RE.search(response)
    if match:
        name = match.group(1)
        patient_id = int(match.group(2)) if match.group(2) else None
        if name in ACTION_NAME_MAP:
            return SakhaAction(action_type=ACTION_NAME_MAP[name], patient_id=patient_id), True
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None), False


def parse_action_response(response: str) -> SakhaAction:
    """Backward-compatible parser used by eval code."""
    action, _ = parse_action_response_with_status(response)
    return action


def action_to_replay_dict(action: SakhaAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "patient_id": action.patient_id,
        "medicine_id": action.medicine_id,
        "reason_code": action.reason_code,
    }


def action_from_replay_dict(payload: dict[str, Any]) -> SakhaAction:
    return SakhaAction(
        action_type=payload["action_type"],
        patient_id=payload.get("patient_id"),
        medicine_id=payload.get("medicine_id"),
        reason_code=payload.get("reason_code"),
    )


def choose_queue_head_action(obs: SakhaObservation) -> SakhaAction:
    """Simple deterministic policy used only to reach varied training states."""
    if not obs.ward_state.pending_tasks:
        return SakhaAction(action_type=ActionType.NOOP, patient_id=None)
    task = obs.ward_state.pending_tasks[0]
    return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)


def choose_random_pending_action(obs: SakhaObservation, rng: random.Random) -> SakhaAction:
    """Pick a random pending task to reach a different (and noisier) trajectory state."""
    if not obs.ward_state.pending_tasks:
        return SakhaAction(action_type=ActionType.NOOP, patient_id=None)
    task = rng.choice(obs.ward_state.pending_tasks)
    return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)


def choose_noisy_queue_head_action(obs: SakhaObservation, rng: random.Random) -> SakhaAction:
    """Mostly queue head, occasionally NOOP, to expose the model to backlog states."""
    if rng.random() < 0.15:
        return SakhaAction(action_type=ActionType.NOOP, patient_id=None)
    return choose_queue_head_action(obs)


ReplayPolicy = Callable[[SakhaObservation, random.Random], SakhaAction]


def _replay_policy(name: str) -> ReplayPolicy:
    if name == "queue_head":
        return lambda obs, _rng: choose_queue_head_action(obs)
    if name == "random_pending":
        return choose_random_pending_action
    if name == "noisy_queue_head":
        return choose_noisy_queue_head_action
    raise ValueError(f"Unknown replay policy: {name}")


DEFAULT_REPLAY_POLICIES: tuple[str, ...] = (
    "queue_head",
    "noisy_queue_head",
    "random_pending",
)


def build_grpo_prompt(obs: SakhaObservation, *, task: str, episode_id: int) -> list[dict[str, str]]:
    pending = obs.ward_state.pending_tasks[:5] if obs.ward_state.pending_tasks else []
    tasks_str = "\n".join(
        (
            f"- {task_obj.required_action.value}"
            f"({task_obj.patient_id if task_obj.patient_id is not None else ''})"
            f" p={task_obj.priority} due={task_obj.due_step}"
        )
        for task_obj in pending
    ) or "- No pending tasks"

    active_incidents = [
        patient
        for patient in obs.ward_state.patients
        if patient.active_incident_id >= 0 and not patient.discharge_prepared
    ]
    incidents_str = "\n".join(
        (
            f"- pt={patient.bed_id} checked={patient.incident_checked} "
            f"alerted={patient.incident_alerted} escalated={patient.incident_escalated} "
            f"deadline={patient.incident_deadline_step}"
        )
        for patient in active_incidents[:3]
    ) or "- None"

    due_meds = [
        patient.bed_id
        for patient in obs.ward_state.patients
        if patient.medications_due and not patient.discharge_prepared
    ]
    due_vitals = [
        patient.bed_id
        for patient in obs.ward_state.patients
        if patient.vitals_due and not patient.discharge_prepared
    ]

    # /no_think disables Qwen3's <think> reasoning blocks. We want this in BOTH
    # training and eval so the model's policy is "emit one action immediately"
    # in both regimes — no train/eval drift, ~5-8x faster eval, and Qwen3-0.6B's
    # CoT doesn't help on this kind of pattern-match task anyway.
    system_msg = (
        "/no_think You are a hospital ward assistant. Return exactly one action call.\n"
        "Actions: review_patient(id), check_vitals(id), administer_medicine(id), "
        "alert_doctor(id), escalate(id), update_chart(id), prepare_discharge(id), "
        "medication_round(), ward_sweep(), noop()."
    )

    user_msg = (
        f"Task={task} episode={episode_id} step={obs.ward_state.current_step} "
        f"patients={len(obs.ward_state.patients)} pending={obs.pending_count}\n"
        f"meds_due={due_meds or 'none'}\n"
        f"vitals_due={due_vitals or 'none'}\n"
        f"pending_tasks:\n{tasks_str}\n"
        f"active_incidents:\n{incidents_str}\n"
        "Next action:"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_state_aligned_examples(
    *,
    task: str,
    episodes: int,
    seed: int,
    max_steps: int,
    state_steps: tuple[int, ...] = DEFAULT_STATE_STEPS,
    replay_policies: tuple[str, ...] = DEFAULT_REPLAY_POLICIES,
) -> dict[str, list[Any]]:
    """Generate (prompt, seed, target_step, replay_actions) tuples for GRPO training.

    Each "episode" here is one training example: a different (seed, target_step,
    replay_policy) combination so the dataset covers many states the model will
    actually see at evaluation time, not just the same 8 snapshots.
    """
    examples: dict[str, list[Any]] = {
        "prompt": [],
        "seed": [],
        "target_step": [],
        "replay_actions_json": [],
    }

    for example_id in range(episodes):
        episode_seed = seed + example_id
        target_step = min(state_steps[example_id % len(state_steps)], max(0, max_steps - 1))
        policy_name = replay_policies[example_id % len(replay_policies)]
        policy = _replay_policy(policy_name)
        rng = random.Random(episode_seed * 1009 + example_id)

        env = SakhaEnvironment(task=task)
        obs = env.reset(seed=episode_seed)
        replay_actions: list[dict[str, Any]] = []

        for _ in range(target_step):
            if obs.done:
                break
            action = policy(obs, rng)
            replay_actions.append(action_to_replay_dict(action))
            obs = env.step(action)

        examples["prompt"].append(build_grpo_prompt(obs, task=task, episode_id=example_id))
        examples["seed"].append(episode_seed)
        examples["target_step"].append(obs.ward_state.current_step)
        examples["replay_actions_json"].append(json.dumps(replay_actions))

    return examples


def reconstruct_env_state(
    *, task: str, seed: int, replay_actions_json: str
) -> tuple[SakhaEnvironment, SakhaObservation]:
    env = SakhaEnvironment(task=task)
    obs = env.reset(seed=seed)
    for payload in json.loads(replay_actions_json or "[]"):
        obs = env.step(action_from_replay_dict(payload))
        if obs.done:
            break
    return env, obs


def score_completion_action(
    completion: str,
    *,
    task: str,
    seed: int,
    replay_actions_json: str,
    parse_failure_reward: float = -0.2,
    env_reward_scale: float = 10.0,
    format_bonus: float = 0.0,
) -> float:
    """Score a single GRPO completion by stepping the env from the prompt state.

    No format bonus is applied by default: a parseable action that the env
    rejects (or no-effect noop) should not look better than a strong action.
    """
    action, parsed_ok = parse_action_response_with_status(completion)
    if not parsed_ok:
        return parse_failure_reward

    try:
        env, obs = reconstruct_env_state(
            task=task, seed=seed, replay_actions_json=replay_actions_json
        )
        if obs.done:
            return parse_failure_reward
        scored_obs = env.step(action)
    except Exception:
        return parse_failure_reward

    return float((scored_obs.reward or 0.0) * env_reward_scale + format_bonus)
