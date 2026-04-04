from enum import StrEnum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


class ActionType(StrEnum):
    ADMINISTER_MEDICINE = "administer_medicine"
    CHECK_VITALS = "check_vitals"
    ESCALATE = "escalate"
    NOOP = "noop"


class Vitals(BaseModel):
    blood_pressure_sys: int
    blood_pressure_dia: int
    temperature: float
    spo2: int
    pulse: int


class PatientState(BaseModel):
    """Patient state - clinically relevant information for decision making."""

    bed_id: int
    name: str
    diagnosis: str
    medications_due: list[str] = Field(default_factory=list)
    vitals_due: bool = False
    last_vitals: Vitals | None = None
    escalation_level: int = 0


class WardState(BaseModel):
    """Ward state - the core observable environment."""

    patients: list[PatientState] = Field(default_factory=list)
    current_time_minutes: int = 480
    pending_tasks: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("patients", mode="before")
    @classmethod
    def _deep_copy_patients(cls, v):
        """Ensure patients are deep copied to prevent external mutation."""
        if isinstance(v, list):
            return [p.model_copy(deep=True) for p in v]
        return v


class SakhaAction(Action):
    """Action space for the Sakha environment."""

    action_type: ActionType
    patient_id: int | None = None
    medicine_id: str | None = None
    reason_code: str | None = None

    @field_validator("action_type", mode="before")
    @classmethod
    def _validate_action_type(cls, v):
        if isinstance(v, str):
            try:
                return ActionType(v)
            except ValueError as err:
                valid_actions = [a.value for a in ActionType]
                raise ValueError(
                    f"Invalid action_type '{v}'. Must be one of: {valid_actions}"
                ) from err
        return v


class SakhaEpisodeMetrics(BaseModel):
    """
    Episode metrics - internal debug/scoring information.
    NOT exposed to the agent during episodes.
    Used by graders for final scoring.
    """

    episode_id: str = ""
    step: int = 0
    meds_administered: int = 0
    vitals_checked: int = 0
    escalations: int = 0
    missed_escalations: int = 0
    conflicts_resolved: int = 0
    deteriorations_handled: int = 0


class ActionResult(BaseModel):
    """Feedback about what the last action did."""

    status: str  # "success" | "invalid" | "no_effect"
    detail: str = ""


class SakhaObservation(Observation):
    """
    Policy-visible observation for the Sakha environment.
    Contains only information needed for decision making.

    Note: episode_metrics is NOT included in observation to prevent score leakage.
    Graders should access metrics via env.episode_metrics property instead.
    """

    ward_state: WardState = Field(default_factory=WardState)
    pending_count: int = 0
    time_remaining_minutes: int = 480
    action_result: ActionResult | None = None

    @field_validator("ward_state", mode="before")
    @classmethod
    def _deep_copy_ward(cls, v):
        if isinstance(v, WardState):
            return v.model_copy(deep=True)
        elif isinstance(v, dict):
            return WardState.model_validate(v).model_copy(deep=True)
        return v


class SakhaState(State):
    """Internal state tracking for the environment."""

    episode_id: str = ""
    step_count: int = 0
    current_time: int = 480
    patients: list[PatientState] = Field(default_factory=list)
