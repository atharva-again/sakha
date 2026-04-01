"""GRPO tool-calling wrapper for Sakha hospital ward environment.

Wraps SakhaEnv into the format TRL's GRPOTrainer expects:
- reset() returns formatted ward state as string
- Each ActionType becomes a tool method with typed args + docstrings
- self.reward accumulates per episode for reward_func extraction
"""

import os
from typing import Any

from sakha.client import SakhaEnv
from sakha.models import SakhaAction, SakhaObservation

ENV_URL = os.environ.get("SAKHA_ENV_URL", "http://localhost:7860")


def _format_ward_state(obs: SakhaObservation) -> str:
    """Convert SakhaObservation into a natural language string for the LLM."""
    ward = obs.ward_state
    lines = []
    lines.append(
        f"WARD STATE — Step {ward.current_step} | {ward.current_time_minutes // 60:02d}:{ward.current_time_minutes % 60:02d}"
    )
    lines.append(
        f"Nurse fatigue: {ward.nurse.fatigue_level:.2f} | Actions remaining: {ward.nurse.max_actions_per_step - ward.nurse.actions_this_step}"
    )
    lines.append("")

    for patient in ward.patients:
        status_icon = {
            "waiting": "⏳",
            "admitted": "🛏️",
            "stable": "✅",
            "critical": "🚨",
            "discharged": "🏠",
        }.get(patient.status.value, "?")
        lines.append(f"─── Patient #{patient.patient_id} ───")
        lines.append(
            f"{status_icon} Status: {patient.status.value.upper()} | Bed: {patient.bed_id or 'None'} | Escalation: {patient.escalation_level}"
        )
        lines.append(f"Diagnosis: {patient.diagnosis}")

        if patient.vitals:
            v = patient.vitals
            lines.append(
                f"Vitals: BP {v.blood_pressure_sys}/{v.blood_pressure_dia} | Temp {v.temperature:.1f}°C | SpO2 {v.spo2} | Pulse {v.pulse}"
            )
        else:
            lines.append("Vitals: Not available")

        due_meds = [m for m in patient.medication_schedules if not m.administered]
        if due_meds:
            med_names = ", ".join(m.medicine_name for m in due_meds[:5])
            lines.append(f"💊 Meds due: {med_names}")

        if patient.escalation_level >= 2:
            lines.append("⚠️ CRITICAL — escalation needed")

        if patient.vitals_due:
            lines.append("📋 Vitals check due")

        lines.append("")

    if obs.time_remaining_minutes < 60:
        lines.append(f"⏰ Shift ending in {obs.time_remaining_minutes} minutes — consider handover")

    return "\n".join(lines)


def _format_action_result(
    action_type: str, patient_id: int | None, result: SakhaObservation
) -> str:
    """Format the result of an action as a concise string."""
    parts = [f"Action: {action_type}"]
    if patient_id is not None:
        parts.append(f"Patient: {patient_id}")
    parts.append(f"Reward: {result.reward:+.2f}")
    parts.append(f"Step: {result.ward_state.current_step}")
    return " | ".join(parts)


class SakhaToolEnv:
    """Sakha environment wrapper for TRL GRPO tool-calling.

    Each public method (except reset) is auto-discovered by TRL as a tool.
    Methods must have typed arguments and docstrings for schema generation.
    """

    def __init__(self) -> None:
        self._env = SakhaEnv(base_url=ENV_URL)
        self.reward: float = 0.0
        self._last_obs: SakhaObservation | None = None

    def reset(self, seed: int | None = None, **kwargs: Any) -> str | None:
        """Reset the environment and return the initial ward state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Formatted ward state as a natural language string.
        """
        self.reward = 0.0
        self._last_obs = self._env.reset(seed=seed)
        return _format_ward_state(self._last_obs)

    def check_vitals(self, patient_id: int) -> str:
        """Check the vital signs of a specific patient.

        Use this to monitor patient health and detect deterioration early.
        Only returns useful results when vitals are actually due.

        Args:
            patient_id: The ID of the patient to check.

        Returns:
            Result of the vitals check including reward and current step.
        """
        result = self._env.step(SakhaAction(action_type="check_vitals", patient_id=patient_id))
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("check_vitals", patient_id, result)

    def administer_medicine(self, patient_id: int) -> str:
        """Administer the next due medication to a patient.

        Give the patient's next scheduled medication. The environment
        automatically selects the correct medicine — you just choose the patient.
        Only works when the patient actually has medication due.

        Args:
            patient_id: The ID of the patient to treat.

        Returns:
            Result of the administration including reward and current step.
        """
        result = self._env.step(
            SakhaAction(
                action_type="administer_medicine",
                patient_id=patient_id,
            )
        )
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("administer_medicine", patient_id, result)

    def escalate(self, patient_id: int) -> str:
        """Escalate a critical patient to higher-level care.

        Use when a patient's condition is deteriorating and requires
        immediate senior intervention. Critical for safety scoring.

        Args:
            patient_id: The ID of the patient to escalate.

        Returns:
            Result of the escalation including reward and current step.
        """
        result = self._env.step(SakhaAction(action_type="escalate", patient_id=patient_id))
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("escalate", patient_id, result)

    def document(self, patient_id: int) -> str:
        """Document the current status of a patient.

        Record clinical notes for the patient's file.
        Useful during downtime to maintain documentation quality.

        Args:
            patient_id: The ID of the patient to document.

        Returns:
            Result of the documentation action including reward and current step.
        """
        result = self._env.step(SakhaAction(action_type="document", patient_id=patient_id))
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("document", patient_id, result)

    def communicate(self) -> str:
        """Communicate with the care team.

        Share information with other staff members.
        Useful during downtime to improve team coordination.

        Returns:
            Result of the communication action including reward and current step.
        """
        result = self._env.step(SakhaAction(action_type="communicate"))
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("communicate", None, result)

    def handover(self) -> str:
        """Complete the shift handover.

        Transfer responsibility to the incoming shift.
        Should be done near the end of the shift for maximum score.

        Returns:
            Result of the handover action including reward and current step.
        """
        result = self._env.step(SakhaAction(action_type="handover"))
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("handover", None, result)

    def noop(self) -> str:
        """Take no action this step.

        Use only when no productive actions are available.
        Each noop incurs a small penalty.

        Returns:
            Result of the noop action including reward and current step.
        """
        result = self._env.step(SakhaAction(action_type="noop"))
        self.reward += result.reward
        self._last_obs = result.observation
        return _format_action_result("noop", None, result)
