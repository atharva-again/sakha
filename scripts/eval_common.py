"""Shared evaluation constants and policies for Sakha."""

from sakha.graders import score_easy_task, score_medium_task, score_hard_task
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


def priority_policy(obs, step, pc):
    if obs.ward_state.pending_tasks:
        task = obs.ward_state.pending_tasks[0]
        return SakhaAction(action_type=task.required_action, patient_id=task.patient_id)
    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)
