"""Sakha Environment - Realistic Hospital Ward Simulation Models"""

from enum import StrEnum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# ACTION TYPES
# =============================================================================


class ActionType(StrEnum):
    """All possible actions in the hospital ward."""

    # Core patient care
    ADMINISTER_MEDICINE = "administer_medicine"
    CHECK_VITALS = "check_vitals"
    ESCALATE = "escalate"

    # Documentation & Communication
    DOCUMENT = "document"
    COMMUNICATE = "communicate"
    HANDOVER = "handover"

    # No operation
    NOOP = "noop"


# =============================================================================
# PATIENT MODELS
# =============================================================================


class PatientStatus(StrEnum):
    """Current status of a patient."""

    WAITING = "waiting"  # In queue, not yet admitted
    ADMITTED = "admitted"  # In bed, receiving care
    STABLE = "stable"  # Admitted, condition stable
    CRITICAL = "critical"  # Admitted, needs immediate attention
    DISCHARGED = "discharged"  # Left the ward


class Vitals(BaseModel):
    """Patient vital signs."""

    blood_pressure_sys: int
    blood_pressure_dia: int
    temperature: float
    spo2: int
    pulse: int


class MedicationSchedule(BaseModel):
    """Scheduled medication administration."""

    medicine_id: str
    medicine_name: str
    scheduled_time: int  # Step number when due
    administered: bool = False
    administered_time: int | None = None
    deadline: int  # Latest step for administration
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=urgent


class VitalsSchedule(BaseModel):
    """Scheduled vitals check."""

    round_id: int
    scheduled_time: int  # Step number when due
    checked: bool = False
    checked_time: int | None = None
    deadline: int  # Latest step for check
    is_recurring: bool = False  # Whether this repeats
    recurrence_interval: int | None = None  # Steps between checks


class DeteriorationEvent(BaseModel):
    """A deterioration event requiring escalation."""

    event_id: str
    triggered_at: int  # Step when triggered
    escalation_window: int  # Steps to respond
    responded: bool = False
    responded_at: int | None = None
    severity: int = 2  # 2=high, 3=critical
    early_warning_detected: bool = False


class PatientState(BaseModel):
    """Complete patient state with scheduling and history."""

    # Identity
    patient_id: int
    bed_id: int | None = None  # None if not admitted
    name: str
    diagnosis: str

    # Status
    status: PatientStatus = PatientStatus.WAITING
    arrival_time: int  # Step when entered queue
    admission_time: int | None = None
    discharge_time: int | None = None

    # Clinical state
    vitals: Vitals | None = None
    vitals_trend: list[Vitals] = Field(default_factory=list)  # Last 5 readings
    escalation_level: int = 0
    deterioration_events: list[DeteriorationEvent] = Field(default_factory=list)

    # Tasks
    medications_due: list[str] = Field(default_factory=list)  # Legacy support
    medication_schedules: list[MedicationSchedule] = Field(default_factory=list)
    vitals_schedules: list[VitalsSchedule] = Field(default_factory=list)

    # Documentation
    last_documented_step: int | None = None
    handoff_notes: str = ""

    @property
    def is_admitted(self) -> bool:
        return self.status in (PatientStatus.ADMITTED, PatientStatus.STABLE, PatientStatus.CRITICAL)

    @property
    def vitals_due(self) -> bool:
        """Check if any vitals check is due."""
        return any(not s.checked for s in self.vitals_schedules)

    @property
    def has_urgent_tasks(self) -> bool:
        """Check for urgent tasks requiring immediate attention."""
        if self.escalation_level >= 2:
            return True
        for med in self.medication_schedules:
            if not med.administered and med.priority >= 3:
                return True
        return False

    @property
    def total_tasks_pending(self) -> int:
        count = 0
        count += len(self.medications_due)
        count += sum(1 for s in self.vitals_schedules if not s.checked)
        count += sum(1 for m in self.medication_schedules if not m.administered)
        return count

    @property
    def last_vitals(self) -> Vitals | None:
        return self.vitals


# =============================================================================
# WARD MODELS
# =============================================================================


class BedStatus(StrEnum):
    """Status of a hospital bed."""

    AVAILABLE = "available"
    OCCUPIED = "occupied"
    CLEANING = "cleaning"


class Bed(BaseModel):
    """A hospital bed."""

    bed_id: int
    status: BedStatus = BedStatus.AVAILABLE
    patient_id: int | None = None


class NurseState(BaseModel):
    """State of the ward nurse."""

    fatigue_level: float = 0.0  # 0.0 to 1.0
    actions_this_step: int = 0
    max_actions_per_step: int = 3
    current_shift_hour: int = 0
    handover_required: bool = False
    handover_completed: bool = False


class WardState(BaseModel):
    """Complete ward state."""

    # Time
    current_step: int = 0
    current_time_minutes: int = 480  # 8:00 AM

    # Capacity
    beds: list[Bed] = Field(default_factory=list)
    max_beds: int = 20
    waiting_queue: list[int] = Field(default_factory=list)  # Patient IDs waiting

    # Patients
    patients: list[PatientState] = Field(default_factory=list)

    # Nurse
    nurse: NurseState = Field(default_factory=NurseState)

    # Scheduling
    upcoming_arrivals: list[dict[str, Any]] = Field(default_factory=list)
    scheduled_rounds: list[int] = Field(default_factory=list)

    # Metrics
    total_admissions: int = 0
    total_discharges: int = 0
    total_tasks_completed: int = 0
    total_tasks_missed: int = 0

    @field_validator("patients", mode="wrap")
    @classmethod
    def _deep_copy_patients(cls, v, handler):
        patients = handler(v)
        return [p.model_copy(deep=True) for p in patients]

    @property
    def admitted_patients(self) -> list[PatientState]:
        """Get only admitted patients."""
        return [p for p in self.patients if p.is_admitted]

    @property
    def waiting_patients(self) -> list[PatientState]:
        """Get patients waiting for admission."""
        return [p for p in self.patients if p.status == PatientStatus.WAITING]

    @property
    def critical_patients(self) -> list[PatientState]:
        """Get patients needing immediate attention."""
        return [p for p in self.admitted_patients if p.has_urgent_tasks]

    @property
    def available_beds(self) -> list[Bed]:
        """Get available beds."""
        return [b for b in self.beds if b.status == BedStatus.AVAILABLE]


# =============================================================================
# ACTION MODEL
# =============================================================================


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


# =============================================================================
# OBSERVATION MODEL
# =============================================================================


class SakhaEpisodeMetrics(BaseModel):
    """Episode metrics for scoring."""

    episode_id: str = ""
    step: int = 0

    # Patient flow
    patients_admitted: int = 0
    patients_discharged: int = 0
    patients_waiting: int = 0

    # Task completion
    meds_administered: int = 0
    meds_on_time: int = 0
    meds_late: int = 0
    meds_missed: int = 0

    vitals_checked: int = 0
    vitals_on_time: int = 0
    vitals_late: int = 0
    vitals_missed: int = 0

    # Escalations
    escalations: int = 0
    escalations_on_time: int = 0
    escalations_late: int = 0
    missed_escalations: int = 0

    # Documentation
    documents_completed: int = 0
    handovers_completed: int = 0
    handovers_quality: float = 1.0

    # Deterioration
    deteriorations_detected_early: int = 0
    deteriorations_handled: int = 0
    adverse_events: int = 0

    # Efficiency
    tasks_completed: int = 0
    tasks_missed: int = 0
    conflicts_resolved: int = 0


class SakhaObservation(Observation):
    """Policy-visible observation."""

    ward_state: WardState = Field(default_factory=WardState)
    pending_count: int = 0
    time_remaining_minutes: int = 480
    shift_hour: int = 0
    nurse_fatigue: float = 0.0

    @field_validator("ward_state", mode="before")
    @classmethod
    def _deep_copy_ward(cls, v):
        if isinstance(v, WardState):
            return v.model_copy(deep=True)
        elif isinstance(v, dict):
            return WardState.model_validate(v).model_copy(deep=True)
        return v


class SakhaState(State):
    """Internal state tracking."""

    episode_id: str = ""
    step_count: int = 0
    current_time: int = 480
    patients: list[PatientState] = Field(default_factory=list)
