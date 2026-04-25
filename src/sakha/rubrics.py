"""Composable rubrics for Sakha environment reward computation.

Each sub-rubric wraps a component of the existing reward logic.
SakhaRubric combines them via WeightedSum with scaling so the
final reward is bit-for-bit identical to the pre-migration manual
computation (within floating-point tolerance).
"""

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum

from sakha.models import ActionType, PatientState, SakhaAction, SakhaObservation

# ---------------------------------------------------------------------------
# Constants mirrored from env.py to keep helpers self-contained
# ---------------------------------------------------------------------------
MEDICATION_GRACE_STEPS = 2
VITALS_GRACE_STEPS = 1
ADMISSION_GRACE_STEPS = 4
REVIEW_REFRESH_STEPS = 36
DEFAULT_WARD_SWEEP_COOLDOWN_STEPS = 12


# ---------------------------------------------------------------------------
# Pure helper functions (no mutation)
# ---------------------------------------------------------------------------
def _validate_patient_id(
    patient_id: int | None, patients: list[PatientState]
) -> PatientState | None:
    if patient_id is None:
        return None
    return next((p for p in patients if p.bed_id == patient_id), None)


def _patients_with_due_medications(patients: list[PatientState]) -> list[PatientState]:
    return [
        patient
        for patient in patients
        if not patient.discharge_prepared and bool(patient.medications_due)
    ]


def _has_pending_work(patients: list[PatientState]) -> bool:
    return any(
        not patient.discharge_prepared
        for patient in patients
        if (
            patient.medications_due
            or patient.vitals_due
            or patient.active_incident_id >= 0
            or patient.review_required
        )
    )


def _compute_routine_care_reward(
    action: SakhaAction,
    step_count: int,
    patients: list[PatientState],
    last_ward_sweep_step: int,
    task_config: dict,
) -> float:
    """Positive routine-care rewards (medication, vitals, review, ward sweep, discharge)."""
    if action.action_type == ActionType.NOOP:
        return 0.0

    if action.action_type == ActionType.MEDICATION_ROUND:
        due_patients = sorted(
            _patients_with_due_medications(patients),
            key=lambda p: (p.medication_due_by_step, p.bed_id),
        )
        if not due_patients:
            return 0.0  # penalty handled by DeadlinePenaltyRubric
        reward = 0.0
        for patient in due_patients:
            if step_count <= patient.medication_due_by_step + MEDICATION_GRACE_STEPS:
                reward += 0.06
            else:
                reward += 0.02
        return reward

    if action.action_type == ActionType.WARD_SWEEP:
        if _has_pending_work(patients):
            return 0.0  # penalty handled elsewhere
        cooldown = int(
            task_config.get("ward_sweep_cooldown_steps", DEFAULT_WARD_SWEEP_COOLDOWN_STEPS)
        )
        if step_count - last_ward_sweep_step < cooldown:
            return 0.0  # penalty handled elsewhere
        return 0.008

    patient = _validate_patient_id(action.patient_id, patients)
    if action.patient_id is not None and patient is None:
        return 0.0  # invalid → penalty rubric
    if patient is None:
        return 0.0  # invalid → penalty rubric

    if action.action_type == ActionType.REVIEW_PATIENT:
        if not patient.review_required and step_count - patient.last_reviewed_step <= 2:
            return 0.0  # no-effect penalty elsewhere
        reward = 0.01
        if patient.admission_review_required and not patient.admission_reviewed:
            reward = 0.03
        return reward

    if action.action_type == ActionType.ADMINISTER_MEDICINE:
        if not patient.medications_due:
            return 0.0  # no-effect penalty elsewhere
        reward = 0.02
        if len(patient.medications_due) == 1:  # last medicine → task completion
            if step_count <= patient.medication_due_by_step + MEDICATION_GRACE_STEPS:
                reward = 0.06
        return reward

    if action.action_type == ActionType.CHECK_VITALS:
        if not patient.vitals_due and patient.active_incident_id < 0:
            return 0.0  # no-effect penalty elsewhere
        reward = 0.02
        if (
            patient.vitals_due_by_step >= 0
            and step_count <= patient.vitals_due_by_step + VITALS_GRACE_STEPS
        ):
            reward = 0.05
        return reward

    if action.action_type == ActionType.UPDATE_CHART:
        recent_assessment = (
            max(
                patient.last_reviewed_step,
                patient.last_vitals_step,
                patient.last_escalation_step,
            )
            >= step_count - 6
        )
        if not recent_assessment:
            return 0.0  # no-effect penalty elsewhere
        if patient.active_incident_id >= 0:
            if not patient.incident_escalated or patient.incident_documented:
                return 0.0  # no-effect penalty elsewhere
            # Incident resolution reward goes to CriticalIncidentRubric
            return 0.0
        elif (
            patient.admission_review_required
            and patient.admission_reviewed
            and not patient.admission_documented
        ):
            return 0.04
        else:
            return 0.02

    if action.action_type == ActionType.PREPARE_DISCHARGE:
        recently_documented = patient.last_documented_step >= step_count - 10
        recently_reviewed = patient.last_reviewed_step >= step_count - 10
        stable = patient.active_incident_id < 0 and patient.escalation_level == 0
        no_pending = not patient.medications_due and not patient.vitals_due
        long_enough = step_count - patient.admission_step >= 12
        if (
            patient.discharge_prepared
            or not stable
            or not no_pending
            or not recently_documented
            or not recently_reviewed
            or not long_enough
        ):
            return 0.0  # penalty elsewhere
        return 0.08

    return 0.0


def _compute_critical_incident_reward(
    action: SakhaAction,
    step_count: int,
    patients: list[PatientState],
) -> float:
    """Positive critical-incident rewards (check vitals during incident, alert, escalate, document incident)."""
    patient = _validate_patient_id(action.patient_id, patients)

    if action.action_type == ActionType.CHECK_VITALS:
        if patient is not None and patient.active_incident_id >= 0:
            # Incident vitals bonus (routine base/on-time handled by RoutineCareRubric)
            return 0.04
        return 0.0

    if action.action_type == ActionType.ALERT_DOCTOR:
        if (
            patient is not None
            and patient.active_incident_id >= 0
            and patient.incident_checked
            and not patient.incident_alerted
        ):
            return 0.03
        return 0.0

    if action.action_type == ActionType.ESCALATE:
        if (
            patient is not None
            and patient.active_incident_id >= 0
            and patient.incident_checked
            and patient.incident_alerted
            and not patient.incident_escalated
        ):
            return 0.08
        return 0.0

    if action.action_type == ActionType.UPDATE_CHART:
        if patient is not None and patient.active_incident_id >= 0:
            recent_assessment = (
                max(
                    patient.last_reviewed_step,
                    patient.last_vitals_step,
                    patient.last_escalation_step,
                )
                >= step_count - 6
            )
            if recent_assessment and patient.incident_escalated and not patient.incident_documented:
                if step_count <= patient.incident_deadline_step:
                    return 0.12
                return 0.05
        return 0.0

    return 0.0


def _compute_deadline_penalty(
    action: SakhaAction,
    step_count: int,
    patients: list[PatientState],
    last_ward_sweep_step: int,
    task_config: dict,
) -> float:
    """All penalties: no-effect, invalid, overdue tasks, noop-with-pending."""
    reward = 0.0
    action_type = action.action_type

    # Noop penalty
    if action_type == ActionType.NOOP:
        if _has_pending_work(patients):
            reward -= 0.03
        # Overdue penalties still apply for noop
        for patient in patients:
            if patient.discharge_prepared:
                continue
            if (
                patient.medications_due
                and not patient.medication_overdue_counted
                and step_count > patient.medication_due_by_step + MEDICATION_GRACE_STEPS
            ):
                reward -= 0.04
            if (
                patient.vitals_due
                and not patient.vitals_overdue_counted
                and step_count > patient.vitals_due_by_step + VITALS_GRACE_STEPS
            ):
                reward -= 0.05
            if (
                patient.active_incident_id >= 0
                and not patient.incident_overdue_counted
                and step_count > patient.incident_deadline_step
            ):
                reward -= 0.12
            if (
                patient.admission_review_required
                and not patient.admission_overdue_counted
                and step_count > patient.admission_due_step
            ):
                reward -= 0.05
        return reward

    # Medication round no-effect
    if action_type == ActionType.MEDICATION_ROUND:
        due_patients = _patients_with_due_medications(patients)
        if not due_patients:
            reward -= 0.01
        # Overdue penalties
        for patient in patients:
            if patient.discharge_prepared:
                continue
            if (
                patient.medications_due
                and not patient.medication_overdue_counted
                and step_count > patient.medication_due_by_step + MEDICATION_GRACE_STEPS
            ):
                reward -= 0.04
            if (
                patient.vitals_due
                and not patient.vitals_overdue_counted
                and step_count > patient.vitals_due_by_step + VITALS_GRACE_STEPS
            ):
                reward -= 0.05
            if (
                patient.active_incident_id >= 0
                and not patient.incident_overdue_counted
                and step_count > patient.incident_deadline_step
            ):
                reward -= 0.12
            if (
                patient.admission_review_required
                and not patient.admission_overdue_counted
                and step_count > patient.admission_due_step
            ):
                reward -= 0.05
        return reward

    # Ward sweep no-effect
    if action_type == ActionType.WARD_SWEEP:
        if _has_pending_work(patients):
            reward -= 0.005
        else:
            cooldown = int(
                task_config.get("ward_sweep_cooldown_steps", DEFAULT_WARD_SWEEP_COOLDOWN_STEPS)
            )
            if step_count - last_ward_sweep_step < cooldown:
                reward -= 0.002
        # Overdue penalties
        for patient in patients:
            if patient.discharge_prepared:
                continue
            if (
                patient.medications_due
                and not patient.medication_overdue_counted
                and step_count > patient.medication_due_by_step + MEDICATION_GRACE_STEPS
            ):
                reward -= 0.04
            if (
                patient.vitals_due
                and not patient.vitals_overdue_counted
                and step_count > patient.vitals_due_by_step + VITALS_GRACE_STEPS
            ):
                reward -= 0.05
            if (
                patient.active_incident_id >= 0
                and not patient.incident_overdue_counted
                and step_count > patient.incident_deadline_step
            ):
                reward -= 0.12
            if (
                patient.admission_review_required
                and not patient.admission_overdue_counted
                and step_count > patient.admission_due_step
            ):
                reward -= 0.05
        return reward

    patient = _validate_patient_id(action.patient_id, patients)

    # Invalid patient_id
    if action.patient_id is not None and patient is None:
        reward -= 0.02
        # Overdue penalties
        for p in patients:
            if p.discharge_prepared:
                continue
            if (
                p.medications_due
                and not p.medication_overdue_counted
                and step_count > p.medication_due_by_step + MEDICATION_GRACE_STEPS
            ):
                reward -= 0.04
            if (
                p.vitals_due
                and not p.vitals_overdue_counted
                and step_count > p.vitals_due_by_step + VITALS_GRACE_STEPS
            ):
                reward -= 0.05
            if (
                p.active_incident_id >= 0
                and not p.incident_overdue_counted
                and step_count > p.incident_deadline_step
            ):
                reward -= 0.12
            if (
                p.admission_review_required
                and not p.admission_overdue_counted
                and step_count > p.admission_due_step
            ):
                reward -= 0.05
        return reward

    if patient is None:
        reward -= 0.02
        for p in patients:
            if p.discharge_prepared:
                continue
            if (
                p.medications_due
                and not p.medication_overdue_counted
                and step_count > p.medication_due_by_step + MEDICATION_GRACE_STEPS
            ):
                reward -= 0.04
            if (
                p.vitals_due
                and not p.vitals_overdue_counted
                and step_count > p.vitals_due_by_step + VITALS_GRACE_STEPS
            ):
                reward -= 0.05
            if (
                p.active_incident_id >= 0
                and not p.incident_overdue_counted
                and step_count > p.incident_deadline_step
            ):
                reward -= 0.12
            if (
                p.admission_review_required
                and not p.admission_overdue_counted
                and step_count > p.admission_due_step
            ):
                reward -= 0.05
        return reward

    # Per-patient action no-effect / invalid penalties
    if action_type == ActionType.REVIEW_PATIENT:
        if not patient.review_required and step_count - patient.last_reviewed_step <= 2:
            reward -= 0.005

    elif action_type == ActionType.ADMINISTER_MEDICINE:
        if not patient.medications_due:
            reward -= 0.01

    elif action_type == ActionType.CHECK_VITALS:
        if not patient.vitals_due and patient.active_incident_id < 0:
            reward -= 0.01

    elif action_type == ActionType.ALERT_DOCTOR:
        if patient.active_incident_id < 0 or not patient.incident_checked:
            reward -= 0.01
        elif patient.incident_alerted:
            reward -= 0.005

    elif action_type == ActionType.ESCALATE:
        if (
            patient.active_incident_id < 0
            or not patient.incident_checked
            or not patient.incident_alerted
        ):
            reward -= 0.015
        elif patient.incident_escalated:
            reward -= 0.005

    elif action_type == ActionType.UPDATE_CHART:
        recent_assessment = (
            max(
                patient.last_reviewed_step,
                patient.last_vitals_step,
                patient.last_escalation_step,
            )
            >= step_count - 6
        )
        if not recent_assessment:
            reward -= 0.01
        elif patient.active_incident_id >= 0:
            if not patient.incident_escalated or patient.incident_documented:
                reward -= 0.01

    elif action_type == ActionType.PREPARE_DISCHARGE:
        recently_documented = patient.last_documented_step >= step_count - 10
        recently_reviewed = patient.last_reviewed_step >= step_count - 10
        stable = patient.active_incident_id < 0 and patient.escalation_level == 0
        no_pending = not patient.medications_due and not patient.vitals_due
        long_enough = step_count - patient.admission_step >= 12
        if (
            patient.discharge_prepared
            or not stable
            or not no_pending
            or not recently_documented
            or not recently_reviewed
            or not long_enough
        ):
            reward -= 0.01

    else:
        reward -= 0.02

    # Overdue penalties (applied for all action types)
    for p in patients:
        if p.discharge_prepared:
            continue
        if (
            p.medications_due
            and not p.medication_overdue_counted
            and step_count > p.medication_due_by_step + MEDICATION_GRACE_STEPS
        ):
            reward -= 0.04
        if (
            p.vitals_due
            and not p.vitals_overdue_counted
            and step_count > p.vitals_due_by_step + VITALS_GRACE_STEPS
        ):
            reward -= 0.05
        if (
            p.active_incident_id >= 0
            and not p.incident_overdue_counted
            and step_count > p.incident_deadline_step
        ):
            reward -= 0.12
        if (
            p.admission_review_required
            and not p.admission_overdue_counted
            and step_count > p.admission_due_step
        ):
            reward -= 0.05

    return reward


# ---------------------------------------------------------------------------
# Rubric classes
# ---------------------------------------------------------------------------
class RoutineCareRubric(Rubric):
    """Wraps medication, vitals, review, ward sweep, and discharge rewards."""

    def reset(self) -> None:
        self.last_score = None

    def forward(self, action, observation) -> float:
        if not isinstance(observation, SakhaObservation):
            return 0.0
        step_count = observation.ward_state.current_step
        patients = observation.ward_state.patients
        last_ward_sweep_step = observation.metadata.get("_last_ward_sweep_step", -999)
        task_config = observation.metadata.get("_task_config", {})
        reward = _compute_routine_care_reward(
            action, step_count, patients, last_ward_sweep_step, task_config
        )
        if reward != 0.0:
            return reward / 0.35
        return 0.0


class CriticalIncidentRubric(Rubric):
    """Wraps escalation and incident-resolution rewards."""

    def reset(self) -> None:
        self.last_score = None

    def forward(self, action, observation) -> float:
        if not isinstance(observation, SakhaObservation):
            return 0.0
        step_count = observation.ward_state.current_step
        patients = observation.ward_state.patients
        reward = _compute_critical_incident_reward(action, step_count, patients)
        if reward != 0.0:
            return reward / 0.40
        return 0.0


class DeadlinePenaltyRubric(Rubric):
    """Wraps overdue penalties and no-effect / invalid action penalties."""

    def reset(self) -> None:
        self.last_score = None

    def forward(self, action, observation) -> float:
        if not isinstance(observation, SakhaObservation):
            return 0.0
        step_count = observation.ward_state.current_step
        patients = observation.ward_state.patients
        last_ward_sweep_step = observation.metadata.get("_last_ward_sweep_step", -999)
        task_config = observation.metadata.get("_task_config", {})
        reward = _compute_deadline_penalty(
            action, step_count, patients, last_ward_sweep_step, task_config
        )
        if reward != 0.0:
            return reward / 0.25
        return 0.0


class SakhaRubric(WeightedSum):
    """Weighted combination of routine care, critical incident, and deadline penalty rubrics.

    Weights: [0.35, 0.40, 0.25]

    Sub-rubrics scale their raw component by 1/weight so the weighted sum
    reproduces the original additive reward exactly.
    """

    def __init__(self):
        super().__init__(
            [RoutineCareRubric(), CriticalIncidentRubric(), DeadlinePenaltyRubric()],
            weights=[0.35, 0.40, 0.25],
        )

    def reset(self) -> None:
        for child in self.children():
            child.reset()
        self.last_score = None
