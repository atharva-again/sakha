"""GRPO tool-calling wrapper for Sakha hospital ward environment.

Wraps SakhaEnv into the format TRL's GRPOTrainer expects:
- reset() returns formatted ward state as string
- Each ActionType becomes a tool method with typed args + docstrings
- self.reward accumulates per episode for reward_func extraction
"""

import asyncio
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
    action_type: str, patient_id: int | None, obs: SakhaObservation, reward: float
) -> str:
    parts = [f"Action: {action_type}"]
    if patient_id is not None:
        parts.append(f"Patient: {patient_id}")
    parts.append(f"Reward: {reward:+.2f}")
    parts.append(f"Step: {obs.ward_state.current_step}")
    return " | ".join(parts)


class SakhaToolEnv:
    """Sakha environment wrapper for TRL GRPO tool-calling.

    Each public method (except reset) is auto-discovered by TRL as a tool.
    Methods must have typed arguments and docstrings for schema generation.
    """

    def __init__(self) -> None:
        self._env = SakhaEnv(base_url=ENV_URL)
        self._loop: asyncio.AbstractEventLoop | None = None
        self.reward: float = 0.0
        self._last_obs: SakhaObservation | None = None
        self._connected = False

    def _ensure_async(self) -> None:
        if not self._connected:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._env.connect())
            self._connected = True

    def reset(self, seed: int | None = None, **kwargs: Any) -> str | None:
        self._ensure_async()
        self.reward = 0.0
        self._last_obs = self._loop.run_until_complete(self._env.reset(seed=seed))
        return _format_ward_state(self._last_obs)

    def _step(self, action: SakhaAction) -> SakhaObservation:
        result = self._loop.run_until_complete(self._env.step(action))
        return result.observation

    def _step_with_reward(self, action: SakhaAction) -> tuple[float, SakhaObservation]:
        result = self._loop.run_until_complete(self._env.step(action))
        return result.reward, result.observation

    def check_vitals(self, patient_id: int) -> str:
        reward, obs = self._step_with_reward(SakhaAction(action_type="check_vitals", patient_id=patient_id))
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("check_vitals", patient_id, obs, reward)

    def administer_medicine(self, patient_id: int) -> str:
        reward, obs = self._step_with_reward(
            SakhaAction(action_type="administer_medicine", patient_id=patient_id)
        )
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("administer_medicine", patient_id, obs, reward)

    def escalate(self, patient_id: int) -> str:
        reward, obs = self._step_with_reward(SakhaAction(action_type="escalate", patient_id=patient_id))
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("escalate", patient_id, obs, reward)

    def document(self, patient_id: int) -> str:
        reward, obs = self._step_with_reward(SakhaAction(action_type="document", patient_id=patient_id))
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("document", patient_id, obs, reward)

    def communicate(self) -> str:
        reward, obs = self._step_with_reward(SakhaAction(action_type="communicate"))
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("communicate", None, obs, reward)

    def handover(self) -> str:
        reward, obs = self._step_with_reward(SakhaAction(action_type="handover"))
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("handover", None, obs, reward)

    def noop(self) -> str:
        reward, obs = self._step_with_reward(SakhaAction(action_type="noop"))
        self.reward += reward
        self._last_obs = obs
        return _format_action_result("noop", None, obs, reward)
