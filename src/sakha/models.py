from enum import IntEnum, StrEnum

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


class ActionType(StrEnum):
    REVIEW_PATIENT = "review_patient"
    WARD_SWEEP = "ward_sweep"
    MEDICATION_ROUND = "medication_round"
    ADMINISTER_MEDICINE = "administer_medicine"
    CHECK_VITALS = "check_vitals"
    ALERT_DOCTOR = "alert_doctor"
    ESCALATE = "escalate"
    UPDATE_CHART = "update_chart"
    PREPARE_DISCHARGE = "prepare_discharge"
    NOOP = "noop"


class TaskKind(StrEnum):
    REVIEW = "review"
    COORDINATION = "coordination"
    MEDICATION_ROUND = "medication_round"
    MEDICATION = "medication"
    VITALS = "vitals"
    ALERT = "alert"
    ESCALATION = "escalation"
    DOCUMENTATION = "documentation"
    DISCHARGE = "discharge"


class TaskPriority(IntEnum):
    LOW = 20
    ROUTINE = 50
    HIGH = 80
    CRITICAL = 100


class Vitals(BaseModel):
    blood_pressure_sys: int
    blood_pressure_dia: int
    temperature: float
    spo2: int
    pulse: int


class PendingTask(BaseModel):
    task_id: str
    patient_id: int | None
    task_kind: TaskKind
    required_action: ActionType
    due_step: int
    priority: int
    overdue: bool = False
    summary: str = ""


class PatientState(BaseModel):
    bed_id: int
    name: str
    diagnosis: str
    medications_due: list[str] = Field(default_factory=list)
    medication_due_by_step: int = -1
    medication_overdue_counted: bool = False
    next_medication_step: int = -1
    medication_interval_steps: int = 12
    vitals_due: bool = False
    vitals_due_by_step: int = -1
    vitals_overdue_counted: bool = False
    next_vitals_step: int = -1
    vitals_interval_steps: int = 8
    last_vitals: Vitals | None = None
    last_vitals_step: int = -1
    escalation_level: int = 0
    review_required: bool = True
    last_reviewed_step: int = -1
    last_documented_step: int = -1
    discharge_prepared: bool = False
    doctor_alerted: bool = False
    doctor_alert_step: int = -1
    last_escalation_step: int = -1
    admission_step: int = -1
    admission_review_required: bool = False
    admission_reviewed: bool = False
    admission_documented: bool = False
    admission_due_step: int = -1
    admission_overdue_counted: bool = False
    active_incident_id: int = -1
    incident_onset_step: int = -1
    incident_deadline_step: int = -1
    incident_checked: bool = False
    incident_alerted: bool = False
    incident_escalated: bool = False
    incident_documented: bool = False
    incident_overdue_counted: bool = False
    incident_resolved_step: int = -1
    medication_tasks_completed: int = 0
    medication_tasks_on_time: int = 0
    vitals_tasks_completed: int = 0
    vitals_tasks_on_time: int = 0
    reviews_completed: int = 0
    admissions_completed: int = 0
    admissions_on_time: int = 0
    documentation_count: int = 0
    doctor_alert_count: int = 0
    critical_incidents_total: int = 0
    critical_incidents_resolved: int = 0
    critical_incidents_resolved_in_time: int = 0
    critical_incidents_missed: int = 0
    overdue_tasks: int = 0


class WardState(BaseModel):
    patients: list[PatientState] = Field(default_factory=list)
    current_time_minutes: int = 480
    pending_tasks: list[PendingTask] = Field(default_factory=list)
    capacity: int = 18
    beds_occupied: int = 0
    active_incident_count: int = 0
    current_step: int = 0

    @field_validator("patients", mode="before")
    @classmethod
    def _deep_copy_patients(cls, value):
        if isinstance(value, list):
            result = []
            for p in value:
                if isinstance(p, dict):
                    # Convert dict to PatientState, then copy
                    result.append(PatientState.model_validate(p).model_copy(deep=True))
                else:
                    # Already a PatientState object
                    result.append(p.model_copy(deep=True))
            return result
        return value


class SakhaAction(Action):
    action_type: ActionType
    patient_id: int | None = None
    medicine_id: str | None = None
    reason_code: str | None = None

    @field_validator("action_type", mode="before")
    @classmethod
    def _validate_action_type(cls, value):
        if isinstance(value, str):
            try:
                return ActionType(value)
            except ValueError as err:
                valid_actions = [action.value for action in ActionType]
                raise ValueError(
                    f"Invalid action_type '{value}'. Must be one of: {valid_actions}"
                ) from err
        return value


class SakhaEpisodeMetrics(BaseModel):
    episode_id: str = ""
    step: int = 0
    medication_tasks_completed: int = 0
    medication_tasks_on_time: int = 0
    vitals_tasks_completed: int = 0
    vitals_tasks_on_time: int = 0
    reviews_completed: int = 0
    admissions_completed: int = 0
    admissions_on_time: int = 0
    doctor_alerts: int = 0
    documentations: int = 0
    escalations: int = 0
    critical_incidents_total: int = 0
    critical_incidents_resolved: int = 0
    critical_incidents_resolved_in_time: int = 0
    critical_incidents_missed: int = 0
    discharges_prepared: int = 0
    overdue_tasks: int = 0
    invalid_actions: int = 0
    no_effect_actions: int = 0
    noop_steps: int = 0


class ActionResult(BaseModel):
    status: str
    detail: str = ""


class SakhaObservation(Observation):
    ward_state: WardState = Field(default_factory=WardState)
    pending_count: int = 0
    time_remaining_minutes: int = 480
    action_result: ActionResult | None = None
    truncated: bool = False

    @field_validator("ward_state", mode="before")
    @classmethod
    def _deep_copy_ward(cls, value):
        if isinstance(value, WardState):
            return value.model_copy(deep=True)
        if isinstance(value, dict):
            return WardState.model_validate(value).model_copy(deep=True)
        return value


class SakhaState(State):
    episode_id: str = ""
    step_count: int = 0
    current_time: int = 480
    patients: list[PatientState] = Field(default_factory=list)
