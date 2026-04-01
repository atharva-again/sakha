import argparse
import json
import math
import random
import sys
from pathlib import Path

from sakha.env import SakhaEnvironment
from sakha.graders import (
    compute_diagnostic_breakdown,
    score_easy_task,
    score_hard_task,
    score_medium_task,
)
from sakha.models import ActionType, SakhaAction

TASK_GRADERS = {
    "easy": score_easy_task,
    "medium": score_medium_task,
    "hard": score_hard_task,
}
PATIENT_COUNTS = {"easy": 5, "medium": 8, "hard": 18}


def noop_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def greedy_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=(step % pc) + 1)


def random_policy(obs, step, pc):
    rng = random.Random((step + 1) * (pc + 7))
    action = rng.choice(
        [
            ActionType.NOOP,
            ActionType.ADMINISTER_MEDICINE,
            ActionType.CHECK_VITALS,
            ActionType.ESCALATE,
        ]
    )
    patient_id = None if action == ActionType.NOOP else rng.randint(1, pc)
    return SakhaAction(action_type=action, patient_id=patient_id)


def vitals_spam_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=((step * 3) % pc) + 1)


def escalation_tunnel_policy(obs, step, pc):
    return SakhaAction(action_type=ActionType.ESCALATE, patient_id=((step * 5) % pc) + 1)


def _abnormal(patient) -> bool:
    vitals = getattr(patient, 'vitals', None) or getattr(patient, 'last_vitals', None)
    if vitals is None:
        return False
    return vitals.temperature >= 39.0 or vitals.spo2 < 93 or vitals.pulse >= 100


def _get_patient_id(patient):
    return getattr(patient, 'patient_id', getattr(patient, 'bed_id', 1))


def priority_policy(obs, step, pc):
    patients = obs.ward_state.patients

    for patient in patients:
        if patient.escalation_level >= 2:
            return SakhaAction(action_type=ActionType.ESCALATE, patient_id=_get_patient_id(patient))

    for patient in patients:
        if patient.medications_due and patient.escalation_level >= 1:
            return SakhaAction(
                action_type=ActionType.ADMINISTER_MEDICINE, patient_id=_get_patient_id(patient)
            )

    for patient in patients:
        if patient.vitals_due and (patient.escalation_level >= 1 or _abnormal(patient)):
            return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=_get_patient_id(patient))

    for patient in patients:
        if patient.medications_due:
            return SakhaAction(
                action_type=ActionType.ADMINISTER_MEDICINE, patient_id=_get_patient_id(patient)
            )

    checked_this_round = set()
    for patient in patients:
        if patient.vitals_due:
            pid = _get_patient_id(patient)
            if pid not in checked_this_round:
                checked_this_round.add(pid)
                return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=pid)

    return SakhaAction(action_type=ActionType.NOOP)


def timestep_scripted_policy(obs, step, pc):
    if step < 10:
        return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=(step % pc) + 1)
    elif step < 20:
        return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=((step - 5) % pc) + 1)
    elif step < 30:
        return SakhaAction(
            action_type=ActionType.ADMINISTER_MEDICINE, patient_id=((step - 10) % pc) + 1
        )
    else:
        return SakhaAction(action_type=ActionType.NOOP)


POLICIES = {
    "noop": noop_policy,
    "random": random_policy,
    "greedy": greedy_policy,
    "vitals_spam": vitals_spam_policy,
    "escalation_tunnel": escalation_tunnel_policy,
    "priority": priority_policy,
    "timestep_scripted": timestep_scripted_policy,
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def run_policy(
    task: str, policy_name: str, seed: int, episodes: int, with_details: bool = False,
    max_steps: int = 96,
) -> dict:
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]
    policy = POLICIES[policy_name]
    scores = []
    diagnostics = []

    for ep in range(episodes):
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed + ep)
        trajectory = [obs]
        for step in range(max_steps):
            action = policy(obs, step, pc)
            obs = env.step(action)
            trajectory.append(obs)
            if obs.done:
                break
        score = grader(trajectory)
        scores.append(score)
        diagnostics.append(compute_diagnostic_breakdown(trajectory))

    mu = _mean(scores)
    std = _std(scores)
    ci95 = 1.96 * std / math.sqrt(max(1, len(scores)))
    avg_diag = {
        "loop_ratio": round(_mean([d["loop_ratio"] for d in diagnostics]), 4),
        "late_escalation_rate": round(_mean([d["late_escalation_rate"] for d in diagnostics]), 4),
        "safety_violation_rate": round(
            _mean([1.0 if d["safety_violation"] else 0.0 for d in diagnostics]), 4
        ),
    }

    result = {
        "policy": policy_name,
        "mean": round(mu, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(std, 4),
        "ci95": round(ci95, 4),
        "diagnostics": avg_diag,
    }

    if with_details:
        result["scores"] = [round(s, 4) for s in scores]

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy-a", default="noop")
    parser.add_argument("--policy-b", default="priority")
    parser.add_argument("--all-policies", action="store_true")
    parser.add_argument("--with-details", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if args.all_policies:
        policy_results = {
            name: run_policy(
                args.task, name, args.seed, args.episodes, with_details=args.with_details
            )
            for name in POLICIES
        }
        output = {
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "policies": policy_results,
        }
    else:
        result_a = run_policy(
            args.task,
            args.policy_a,
            args.seed,
            args.episodes,
            with_details=args.with_details,
        )
        result_b = run_policy(
            args.task,
            args.policy_b,
            args.seed,
            args.episodes,
            with_details=args.with_details,
        )
        output = {
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "policy_a": result_a,
            "policy_b": result_b,
            "gap": round(result_b["mean"] - result_a["mean"], 4),
        }

    print(json.dumps(output, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
