import logging
import random
from typing import TypedDict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from sakha.models import (
    ActionResult,
    ActionType,
    PatientState,
    PendingTask,
    SakhaAction,
    SakhaEpisodeMetrics,
    SakhaObservation,
    SakhaState,
    TaskKind,
    TaskPriority,
    Vitals,
    WardState,
)

logger = logging.getLogger(__name__)

EIGHT_HOURS_MINUTES = 480
START_TIME_MINUTES = 480
SHIFT_STEPS = EIGHT_HOURS_MINUTES // 5
MEDICATION_GRACE_STEPS = 2
VITALS_GRACE_STEPS = 1
REVIEW_REFRESH_STEPS = 36
ADMISSION_GRACE_STEPS = 4
DEFAULT_WARD_SWEEP_COOLDOWN_STEPS = 12

TASK_CONFIGS = {
    "easy": {
        "patient_count": 5,
        "incident_count": 0,
        "admission_count": 0,
        "incident_grace_steps": 2,
        "ward_sweep_cooldown_steps": 8,
    },
    "medium": {
        "patient_count": 8,
        "incident_count": 2,
        "admission_count": 2,
        "incident_grace_steps": 2,
        "ward_sweep_cooldown_steps": 6,
    },
    "hard": {
        "patient_count": 18,
        "incident_count": 8,
        "admission_count": 7,
        "incident_grace_steps": 1,
        "ward_sweep_cooldown_steps": 4,
    },
}


class DiagnosisProfile(TypedDict):
    diagnosis: str
    medications: list[str]
    med_interval: int
    vitals_interval: int
    baseline_vitals: Vitals


DIAGNOSIS_PROFILES: list[DiagnosisProfile] = [
    {
        "diagnosis": "post_op",
        "medications": ["ceftriaxone"],
        "med_interval": 96,
        "vitals_interval": 48,
        "baseline_vitals": Vitals(
            blood_pressure_sys=122,
            blood_pressure_dia=78,
            temperature=37.1,
            spo2=98,
            pulse=78,
        ),
    },
    {
        "diagnosis": "pneumonia",
        "medications": ["azithromycin", "paracetamol"],
        "med_interval": 72,
        "vitals_interval": 48,
        "baseline_vitals": Vitals(
            blood_pressure_sys=118,
            blood_pressure_dia=74,
            temperature=37.8,
            spo2=95,
            pulse=90,
        ),
    },
    {
        "diagnosis": "heart_failure",
        "medications": ["furosemide"],
        "med_interval": 72,
        "vitals_interval": 48,
        "baseline_vitals": Vitals(
            blood_pressure_sys=110,
            blood_pressure_dia=68,
            temperature=36.8,
            spo2=94,
            pulse=88,
        ),
    },
    {
        "diagnosis": "copd_exacerbation",
        "medications": ["nebulizer"],
        "med_interval": 72,
        "vitals_interval": 48,
        "baseline_vitals": Vitals(
            blood_pressure_sys=126,
            blood_pressure_dia=80,
            temperature=37.0,
            spo2=93,
            pulse=92,
        ),
    },
    {
        "diagnosis": "uti",
        "medications": ["cefuroxime"],
        "med_interval": 96,
        "vitals_interval": 72,
        "baseline_vitals": Vitals(
            blood_pressure_sys=124,
            blood_pressure_dia=76,
            temperature=37.3,
            spo2=98,
            pulse=80,
        ),
    },
]


def _generate_event_steps(rng: random.Random, count: int, minimum_step: int = 4) -> list[int]:
    if count == 0:
        return []
    steps: set[int] = set()
    while len(steps) < count:
        steps.add(rng.randint(minimum_step, SHIFT_STEPS - 6))
    return sorted(steps)


def _copy_vitals(vitals: Vitals) -> Vitals:
    return vitals.model_copy(deep=True)


def _initial_round_due_step(rng: random.Random, *, kind: str) -> int:
    if kind == "medication":
        return rng.choice([0, 6, 12, 18])
    return rng.choice([0, 12, 24])


class SakhaEnvironment(Environment[SakhaAction, SakhaObservation, SakhaState]):
    def __init__(self, patient_count: int = 18, task: str = "hard"):
        super().__init__()
        self._task = task
        config = TASK_CONFIGS.get(task, TASK_CONFIGS["hard"])
        self._task_config = config
        self._patient_count = patient_count or config["patient_count"]
        self._rng = random.Random()
        self._state = SakhaState(episode_id=str(uuid4()), step_count=0)
        self._ward = WardState(patients=[])
        self._episode_metrics = SakhaEpisodeMetrics()
        self._incident_steps: list[int] = []
        self._admission_steps: list[int] = []
        self._next_bed_id = self._patient_count + 1
        self._next_incident_id = 1
        self._last_ward_sweep_step = -int(
            config.get("ward_sweep_cooldown_steps", DEFAULT_WARD_SWEEP_COOLDOWN_STEPS)
        )

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs
    ) -> SakhaObservation:
        self._rng = random.Random(seed) if seed is not None else random.Random()
        config = TASK_CONFIGS.get(self._task, TASK_CONFIGS["hard"])
        self._task_config = config
        eid = episode_id or (f"seed-{seed}" if seed is not None else str(uuid4()))
        self._incident_steps = _generate_event_steps(self._rng, config["incident_count"], 6)
        self._admission_steps = _generate_event_steps(self._rng, config["admission_count"], 10)
        self._next_bed_id = self._patient_count + 1
        self._next_incident_id = 1
        self._last_ward_sweep_step = -int(
            config.get("ward_sweep_cooldown_steps", DEFAULT_WARD_SWEEP_COOLDOWN_STEPS)
        )

        patients = [self._make_patient(bed_id=index + 1) for index in range(self._patient_count)]
        self._ward = WardState(
            patients=patients,
            current_time_minutes=START_TIME_MINUTES,
            capacity=max(self._patient_count, 18),
            beds_occupied=len(patients),
            current_step=0,
        )
        self._state = SakhaState(
            episode_id=eid,
            step_count=0,
            current_time=START_TIME_MINUTES,
            patients=[patient.model_copy(deep=True) for patient in patients],
        )
        self._episode_metrics = SakhaEpisodeMetrics(episode_id=eid, step=0)

        self._materialize_due_work()
        return self._build_observation(done=False, reward=None)

    def step(
        self, action: SakhaAction, timeout_s: float | None = None, **kwargs
    ) -> SakhaObservation:
        self._state.step_count += 1
        self._ward.current_step = self._state.step_count
        self._ward.current_time_minutes += 5
        self._state.current_time = self._ward.current_time_minutes

        self._trigger_events()
        self._materialize_due_work()

        action_result, reward = self._process_action(action)
        reward += self._apply_deadline_penalties(action)
        self._materialize_due_work()
        self._update_metrics_step()

        done = self._state.step_count >= SHIFT_STEPS
        truncated = done
        return self._build_observation(
            done=done,
            truncated=truncated,
            reward=round(reward, 4),
            action_result=action_result,
        )

    def _make_patient(self, bed_id: int) -> PatientState:
        profile = DIAGNOSIS_PROFILES[(bed_id - 1) % len(DIAGNOSIS_PROFILES)]
        patient = PatientState(
            bed_id=bed_id,
            name=f"Patient_{bed_id}",
            diagnosis=profile["diagnosis"],
            medications_due=[],
            next_medication_step=_initial_round_due_step(self._rng, kind="medication"),
            medication_interval_steps=profile["med_interval"],
            next_vitals_step=_initial_round_due_step(self._rng, kind="vitals"),
            vitals_interval_steps=profile["vitals_interval"],
            last_vitals=_copy_vitals(profile["baseline_vitals"]),
            admission_step=0,
        )
        patient.review_required = True
        return patient

    def _trigger_events(self) -> None:
        if self._state.step_count in self._incident_steps:
            self._start_incident()
        if self._state.step_count in self._admission_steps:
            self._admit_new_patient()

    def _start_incident(self) -> None:
        eligible = [
            patient
            for patient in self._ward.patients
            if not patient.discharge_prepared and patient.active_incident_id < 0
        ]
        if not eligible:
            return
        patient = eligible[(self._state.step_count + 3) % len(eligible)]
        patient.active_incident_id = self._next_incident_id
        self._next_incident_id += 1
        patient.critical_incidents_total += 1
        patient.incident_onset_step = self._state.step_count
        patient.incident_deadline_step = min(
            SHIFT_STEPS,
            self._state.step_count + int(self._task_config.get("incident_grace_steps", 2)),
        )
        patient.incident_checked = False
        patient.incident_alerted = False
        patient.incident_escalated = False
        patient.incident_documented = False
        patient.incident_overdue_counted = False
        patient.doctor_alerted = False
        patient.doctor_alert_step = -1
        patient.escalation_level = 2
        patient.review_required = True
        patient.vitals_due = True
        patient.vitals_due_by_step = self._state.step_count
        patient.vitals_overdue_counted = False
        patient.last_vitals = Vitals(
            blood_pressure_sys=88,
            blood_pressure_dia=58,
            temperature=39.3,
            spo2=90,
            pulse=118,
        )

    def _admit_new_patient(self) -> None:
        if self._ward.beds_occupied >= self._ward.capacity:
            return
        patient = self._make_patient(self._next_bed_id)
        patient.admission_step = self._state.step_count
        patient.review_required = True
        patient.admission_review_required = True
        patient.admission_reviewed = False
        patient.admission_documented = False
        patient.admission_due_step = self._state.step_count + ADMISSION_GRACE_STEPS
        patient.admission_overdue_counted = False
        patient.next_medication_step = self._state.step_count + self._rng.randint(1, 3)
        patient.next_vitals_step = self._state.step_count
        self._next_bed_id += 1
        self._ward.patients.append(patient)
        self._ward.beds_occupied += 1

    def _materialize_due_work(self) -> None:
        for patient in self._ward.patients:
            if patient.discharge_prepared:
                continue
            if patient.medications_due == [] and patient.next_medication_step >= 0:
                if self._state.step_count >= patient.next_medication_step:
                    profile = DIAGNOSIS_PROFILES[(patient.bed_id - 1) % len(DIAGNOSIS_PROFILES)]
                    patient.medications_due = list(profile["medications"])
                    patient.medication_due_by_step = patient.next_medication_step
                    patient.medication_overdue_counted = False
            if not patient.vitals_due and patient.next_vitals_step >= 0:
                if self._state.step_count >= patient.next_vitals_step:
                    patient.vitals_due = True
                    patient.vitals_due_by_step = patient.next_vitals_step
                    patient.vitals_overdue_counted = False
            if (
                self._state.step_count - max(patient.last_reviewed_step, patient.admission_step)
                >= REVIEW_REFRESH_STEPS
            ):
                patient.review_required = True

    def _patients_with_due_medications(self) -> list[PatientState]:
        return [
            patient
            for patient in self._ward.patients
            if not patient.discharge_prepared and bool(patient.medications_due)
        ]

    def _validate_patient_id(self, patient_id: int | None) -> PatientState | None:
        if patient_id is None:
            return None
        return next((p for p in self._ward.patients if p.bed_id == patient_id), None)

    def _process_action(self, action: SakhaAction) -> tuple[ActionResult, float]:
        if action.action_type == ActionType.NOOP:
            self._episode_metrics.noop_steps += 1
            return ActionResult(status="no_effect", detail="Shift time passed without action"), 0.0
        if action.action_type == ActionType.MEDICATION_ROUND:
            due_patients = sorted(
                self._patients_with_due_medications(),
                key=lambda patient: (
                    patient.medication_due_by_step,
                    patient.bed_id,
                ),
            )
            if not due_patients:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail="No medication round due right now",
                ), -0.01
            reward = 0.0
            total_administered = 0
            for patient in due_patients:
                total_administered += len(patient.medications_due)
                patient.medications_due = []
                patient.medication_tasks_completed += 1
                self._episode_metrics.medication_tasks_completed += 1
                patient.last_reviewed_step = self._state.step_count
                patient.review_required = False
                if (
                    self._state.step_count
                    <= patient.medication_due_by_step + MEDICATION_GRACE_STEPS
                ):
                    patient.medication_tasks_on_time += 1
                    self._episode_metrics.medication_tasks_on_time += 1
                    reward += 0.06
                else:
                    reward += 0.02
                patient.medication_due_by_step = -1
                patient.medication_overdue_counted = False
                next_step = self._state.step_count + patient.medication_interval_steps
                patient.next_medication_step = next_step if next_step < SHIFT_STEPS else -1
            return ActionResult(
                status="success",
                detail=f"Completed medication round for {len(due_patients)} patients ({total_administered} meds)",
            ), reward
        if action.action_type == ActionType.WARD_SWEEP:
            if self._has_pending_work():
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail="Ward sweep deferred because direct patient work is pending",
                ), -0.005
            if self._state.step_count - self._last_ward_sweep_step < int(
                self._task_config.get(
                    "ward_sweep_cooldown_steps", DEFAULT_WARD_SWEEP_COOLDOWN_STEPS
                )
            ):
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail="Ward sweep already completed recently",
                ), -0.002
            self._last_ward_sweep_step = self._state.step_count
            return ActionResult(
                status="success",
                detail="Completed ward sweep and coordination check",
            ), 0.008
        patient = self._validate_patient_id(action.patient_id)
        if action.patient_id is not None and patient is None:
            self._episode_metrics.invalid_actions += 1
            return ActionResult(
                status="invalid", detail=f"Invalid patient_id {action.patient_id}"
            ), -0.02
        if patient is None:
            self._episode_metrics.invalid_actions += 1
            return ActionResult(status="invalid", detail="Action requires patient_id"), -0.02

        if action.action_type == ActionType.REVIEW_PATIENT:
            if (
                not patient.review_required
                and self._state.step_count - patient.last_reviewed_step <= 2
            ):
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect", detail=f"Patient {patient.bed_id} already reviewed recently"
                ), -0.005
            patient.review_required = False
            patient.last_reviewed_step = self._state.step_count
            patient.reviews_completed += 1
            self._episode_metrics.reviews_completed += 1
            reward = 0.01
            if patient.admission_review_required and not patient.admission_reviewed:
                patient.admission_reviewed = True
                reward = 0.03
            return ActionResult(
                status="success", detail=f"Reviewed patient {patient.bed_id}"
            ), reward

        if action.action_type == ActionType.ADMINISTER_MEDICINE:
            if not patient.medications_due:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect", detail=f"No medication due for patient {patient.bed_id}"
                ), -0.01
            medicine = patient.medications_due.pop(0)
            reward = 0.02
            if not patient.medications_due:
                patient.medication_tasks_completed += 1
                self._episode_metrics.medication_tasks_completed += 1
                if (
                    self._state.step_count
                    <= patient.medication_due_by_step + MEDICATION_GRACE_STEPS
                ):
                    patient.medication_tasks_on_time += 1
                    self._episode_metrics.medication_tasks_on_time += 1
                    reward = 0.06
                patient.medication_due_by_step = -1
                patient.medication_overdue_counted = False
                next_step = self._state.step_count + patient.medication_interval_steps
                patient.next_medication_step = next_step if next_step < SHIFT_STEPS else -1
            return ActionResult(
                status="success", detail=f"Administered {medicine} to patient {patient.bed_id}"
            ), reward

        if action.action_type == ActionType.CHECK_VITALS:
            if not patient.vitals_due and patient.active_incident_id < 0:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect", detail=f"No vitals due for patient {patient.bed_id}"
                ), -0.01
            patient.vitals_due = False
            patient.last_vitals_step = self._state.step_count
            patient.last_reviewed_step = self._state.step_count
            patient.review_required = False
            patient.vitals_tasks_completed += 1
            self._episode_metrics.vitals_tasks_completed += 1
            reward = 0.02
            if (
                patient.vitals_due_by_step >= 0
                and self._state.step_count <= patient.vitals_due_by_step + VITALS_GRACE_STEPS
            ):
                patient.vitals_tasks_on_time += 1
                self._episode_metrics.vitals_tasks_on_time += 1
                reward = 0.05
            if patient.active_incident_id >= 0:
                patient.incident_checked = True
                reward += 0.04
                patient.last_vitals = Vitals(
                    blood_pressure_sys=86,
                    blood_pressure_dia=56,
                    temperature=39.4,
                    spo2=89,
                    pulse=120,
                )
            patient.vitals_due_by_step = -1
            patient.vitals_overdue_counted = False
            next_step = self._state.step_count + patient.vitals_interval_steps
            patient.next_vitals_step = next_step if next_step < SHIFT_STEPS else -1
            return ActionResult(
                status="success", detail=f"Checked vitals for patient {patient.bed_id}"
            ), reward

        if action.action_type == ActionType.ALERT_DOCTOR:
            if patient.active_incident_id < 0 or not patient.incident_checked:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail=f"No assessed incident to alert for patient {patient.bed_id}",
                ), -0.01
            if patient.incident_alerted:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail=f"Doctor already alerted for patient {patient.bed_id}",
                ), -0.005
            patient.incident_alerted = True
            patient.doctor_alerted = True
            patient.doctor_alert_step = self._state.step_count
            patient.doctor_alert_count += 1
            self._episode_metrics.doctor_alerts += 1
            return ActionResult(
                status="success", detail=f"Doctor alerted for patient {patient.bed_id}"
            ), 0.03

        if action.action_type == ActionType.ESCALATE:
            if (
                patient.active_incident_id < 0
                or not patient.incident_checked
                or not patient.incident_alerted
            ):
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail=f"Escalation workflow incomplete for patient {patient.bed_id}",
                ), -0.015
            if patient.incident_escalated:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect", detail=f"Patient {patient.bed_id} already escalated"
                ), -0.005
            patient.incident_escalated = True
            patient.last_escalation_step = self._state.step_count
            self._episode_metrics.escalations += 1
            return ActionResult(
                status="success", detail=f"Escalated patient {patient.bed_id}"
            ), 0.08

        if action.action_type == ActionType.UPDATE_CHART:
            recent_assessment = (
                max(
                    patient.last_reviewed_step,
                    patient.last_vitals_step,
                    patient.last_escalation_step,
                )
                >= self._state.step_count - 6
            )
            if not recent_assessment:
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect",
                    detail=f"No recent assessment to document for patient {patient.bed_id}",
                ), -0.01
            if patient.active_incident_id >= 0:
                if not patient.incident_escalated or patient.incident_documented:
                    self._episode_metrics.no_effect_actions += 1
                    return ActionResult(
                        status="no_effect",
                        detail=f"Incident workflow incomplete for patient {patient.bed_id}",
                    ), -0.01
                patient.incident_documented = True
                patient.incident_resolved_step = self._state.step_count
                patient.critical_incidents_resolved += 1
                self._episode_metrics.critical_incidents_resolved += 1
                reward = 0.05
                if self._state.step_count <= patient.incident_deadline_step:
                    patient.critical_incidents_resolved_in_time += 1
                    self._episode_metrics.critical_incidents_resolved_in_time += 1
                    reward = 0.12
                patient.active_incident_id = -1
                patient.incident_onset_step = -1
                patient.incident_deadline_step = -1
                patient.incident_checked = False
                patient.incident_alerted = False
                patient.incident_escalated = False
                patient.incident_documented = False
                patient.incident_overdue_counted = False
                patient.escalation_level = 0
                patient.doctor_alerted = False
                patient.doctor_alert_step = -1
            elif (
                patient.admission_review_required
                and patient.admission_reviewed
                and not patient.admission_documented
            ):
                reward = 0.04
                patient.admission_documented = True
                patient.admission_review_required = False
                patient.admissions_completed += 1
                self._episode_metrics.admissions_completed += 1
                if self._state.step_count <= patient.admission_due_step:
                    patient.admissions_on_time += 1
                    self._episode_metrics.admissions_on_time += 1
            else:
                reward = 0.02
            patient.last_documented_step = self._state.step_count
            patient.documentation_count += 1
            self._episode_metrics.documentations += 1
            return ActionResult(
                status="success", detail=f"Updated chart for patient {patient.bed_id}"
            ), reward

        if action.action_type == ActionType.PREPARE_DISCHARGE:
            recently_documented = patient.last_documented_step >= self._state.step_count - 10
            recently_reviewed = patient.last_reviewed_step >= self._state.step_count - 10
            stable = patient.active_incident_id < 0 and patient.escalation_level == 0
            no_pending = not patient.medications_due and not patient.vitals_due
            long_enough = self._state.step_count - patient.admission_step >= 12
            if (
                patient.discharge_prepared
                or not stable
                or not no_pending
                or not recently_documented
                or not recently_reviewed
                or not long_enough
            ):
                self._episode_metrics.no_effect_actions += 1
                return ActionResult(
                    status="no_effect", detail=f"Patient {patient.bed_id} not ready for discharge"
                ), -0.01
            patient.discharge_prepared = True
            self._episode_metrics.discharges_prepared += 1
            return ActionResult(
                status="success", detail=f"Prepared discharge for patient {patient.bed_id}"
            ), 0.08

        self._episode_metrics.invalid_actions += 1
        return ActionResult(status="invalid", detail=f"Unknown action {action.action_type}"), -0.02

    def _apply_deadline_penalties(self, action: SakhaAction) -> float:
        reward = 0.0
        pending_before_review = any(
            not patient.discharge_prepared for patient in self._ward.patients
        )
        if action.action_type == ActionType.NOOP and self._has_pending_work():
            reward -= 0.03
        for patient in self._ward.patients:
            if patient.discharge_prepared:
                continue
            if patient.medications_due and not patient.medication_overdue_counted:
                if self._state.step_count > patient.medication_due_by_step + MEDICATION_GRACE_STEPS:
                    patient.medication_overdue_counted = True
                    patient.overdue_tasks += 1
                    self._episode_metrics.overdue_tasks += 1
                    reward -= 0.04
            if patient.vitals_due and not patient.vitals_overdue_counted:
                if self._state.step_count > patient.vitals_due_by_step + VITALS_GRACE_STEPS:
                    patient.vitals_overdue_counted = True
                    patient.overdue_tasks += 1
                    self._episode_metrics.overdue_tasks += 1
                    reward -= 0.05
            if patient.active_incident_id >= 0 and not patient.incident_overdue_counted:
                if self._state.step_count > patient.incident_deadline_step:
                    patient.incident_overdue_counted = True
                    patient.critical_incidents_missed += 1
                    patient.overdue_tasks += 1
                    self._episode_metrics.critical_incidents_missed += 1
                    self._episode_metrics.overdue_tasks += 1
                    reward -= 0.12
            if patient.admission_review_required and not patient.admission_overdue_counted:
                if self._state.step_count > patient.admission_due_step:
                    patient.admission_overdue_counted = True
                    patient.overdue_tasks += 1
                    self._episode_metrics.overdue_tasks += 1
                    reward -= 0.05
        if not pending_before_review:
            return reward
        return reward

    def _build_pending_tasks(self, patients: list[PatientState]) -> list[PendingTask]:
        tasks: list[PendingTask] = []
        for patient in patients:
            if patient.discharge_prepared:
                continue
            if patient.active_incident_id >= 0:
                if not patient.incident_checked:
                    tasks.append(
                        PendingTask(
                            task_id=f"incident-{patient.active_incident_id}-assess",
                            patient_id=patient.bed_id,
                            task_kind=TaskKind.VITALS,
                            required_action=ActionType.CHECK_VITALS,
                            due_step=patient.incident_deadline_step,
                            priority=int(TaskPriority.CRITICAL),
                            overdue=self._state.step_count > patient.incident_deadline_step,
                            summary="Critical patient needs immediate vitals confirmation",
                        )
                    )
                elif not patient.incident_alerted:
                    tasks.append(
                        PendingTask(
                            task_id=f"incident-{patient.active_incident_id}-alert",
                            patient_id=patient.bed_id,
                            task_kind=TaskKind.ALERT,
                            required_action=ActionType.ALERT_DOCTOR,
                            due_step=patient.incident_deadline_step,
                            priority=int(TaskPriority.CRITICAL) - 2,
                            overdue=self._state.step_count > patient.incident_deadline_step,
                            summary="Critical patient assessed; notify doctor now",
                        )
                    )
                elif not patient.incident_escalated:
                    tasks.append(
                        PendingTask(
                            task_id=f"incident-{patient.active_incident_id}-escalate",
                            patient_id=patient.bed_id,
                            task_kind=TaskKind.ESCALATION,
                            required_action=ActionType.ESCALATE,
                            due_step=patient.incident_deadline_step,
                            priority=int(TaskPriority.CRITICAL) - 4,
                            overdue=self._state.step_count > patient.incident_deadline_step,
                            summary="Rapid response escalation pending",
                        )
                    )
                elif not patient.incident_documented:
                    tasks.append(
                        PendingTask(
                            task_id=f"incident-{patient.active_incident_id}-document",
                            patient_id=patient.bed_id,
                            task_kind=TaskKind.DOCUMENTATION,
                            required_action=ActionType.UPDATE_CHART,
                            due_step=patient.incident_deadline_step + 1,
                            priority=int(TaskPriority.HIGH),
                            overdue=self._state.step_count > patient.incident_deadline_step + 1,
                            summary="Close the critical incident with documentation",
                        )
                    )
            if patient.vitals_due and patient.active_incident_id < 0:
                tasks.append(
                    PendingTask(
                        task_id=f"vitals-{patient.bed_id}-{patient.vitals_due_by_step}",
                        patient_id=patient.bed_id,
                        task_kind=TaskKind.VITALS,
                        required_action=ActionType.CHECK_VITALS,
                        due_step=patient.vitals_due_by_step,
                        priority=int(TaskPriority.ROUTINE),
                        overdue=self._state.step_count
                        > patient.vitals_due_by_step + VITALS_GRACE_STEPS,
                        summary="Routine vitals check due",
                    )
                )
            if patient.review_required and patient.active_incident_id < 0:
                tasks.append(
                    PendingTask(
                        task_id=f"review-{patient.bed_id}-{patient.last_reviewed_step}",
                        patient_id=patient.bed_id,
                        task_kind=TaskKind.REVIEW,
                        required_action=ActionType.REVIEW_PATIENT,
                        due_step=self._state.step_count + 1,
                        priority=int(TaskPriority.LOW),
                        overdue=False,
                        summary="Round on patient and refresh bedside context",
                    )
                )
            if patient.admission_review_required:
                if not patient.admission_reviewed:
                    tasks.append(
                        PendingTask(
                            task_id=f"admission-review-{patient.bed_id}-{patient.admission_step}",
                            patient_id=patient.bed_id,
                            task_kind=TaskKind.REVIEW,
                            required_action=ActionType.REVIEW_PATIENT,
                            due_step=patient.admission_due_step,
                            priority=int(TaskPriority.HIGH),
                            overdue=self._state.step_count > patient.admission_due_step,
                            summary="New admission needs initial bedside assessment",
                        )
                    )
                elif not patient.admission_documented:
                    tasks.append(
                        PendingTask(
                            task_id=f"admission-document-{patient.bed_id}-{patient.admission_step}",
                            patient_id=patient.bed_id,
                            task_kind=TaskKind.DOCUMENTATION,
                            required_action=ActionType.UPDATE_CHART,
                            due_step=patient.admission_due_step,
                            priority=int(TaskPriority.HIGH) - 5,
                            overdue=self._state.step_count > patient.admission_due_step,
                            summary="Complete admission documentation after bedside review",
                        )
                    )
            ready_for_discharge = (
                patient.active_incident_id < 0
                and patient.escalation_level == 0
                and not patient.medications_due
                and not patient.vitals_due
                and not patient.discharge_prepared
                and not patient.admission_review_required
                and patient.last_documented_step >= self._state.step_count - 10
                and patient.last_reviewed_step >= self._state.step_count - 10
                and self._state.step_count - patient.admission_step >= 12
            )
            if ready_for_discharge:
                tasks.append(
                    PendingTask(
                        task_id=f"discharge-{patient.bed_id}",
                        patient_id=patient.bed_id,
                        task_kind=TaskKind.DISCHARGE,
                        required_action=ActionType.PREPARE_DISCHARGE,
                        due_step=self._state.step_count + 4,
                        priority=int(TaskPriority.LOW) + 5,
                        overdue=False,
                        summary="Stable patient can be prepared for discharge",
                    )
                )
        medication_due_patients = [patient for patient in patients if patient.medications_due]
        if medication_due_patients:
            earliest_due = min(
                patient.medication_due_by_step for patient in medication_due_patients
            )
            overdue = any(
                self._state.step_count > patient.medication_due_by_step + MEDICATION_GRACE_STEPS
                for patient in medication_due_patients
            )
            meds_due_count = sum(
                len(patient.medications_due) for patient in medication_due_patients
            )
            tasks.append(
                PendingTask(
                    task_id=f"med-round-{self._state.step_count}-{earliest_due}",
                    patient_id=None,
                    task_kind=TaskKind.MEDICATION_ROUND,
                    required_action=ActionType.MEDICATION_ROUND,
                    due_step=earliest_due,
                    priority=int(TaskPriority.ROUTINE) + 5,
                    overdue=overdue,
                    summary=(
                        f"Medication round due for {len(medication_due_patients)} patients "
                        f"({meds_due_count} meds)"
                    ),
                )
            )
        if not tasks:
            active_census = sum(1 for patient in patients if not patient.discharge_prepared)
            if (
                active_census >= 4
                and self._state.step_count < SHIFT_STEPS - 6
                and self._state.step_count - self._last_ward_sweep_step
                >= int(
                    self._task_config.get(
                        "ward_sweep_cooldown_steps", DEFAULT_WARD_SWEEP_COOLDOWN_STEPS
                    )
                )
            ):
                tasks.append(
                    PendingTask(
                        task_id=f"ward-sweep-{self._state.step_count}",
                        patient_id=None,
                        task_kind=TaskKind.COORDINATION,
                        required_action=ActionType.WARD_SWEEP,
                        due_step=self._state.step_count + 1,
                        priority=int(TaskPriority.LOW),
                        overdue=False,
                        summary="Brief ward sweep for coordination, bed board, and silent-risk check",
                    )
                )
        tasks.sort(
            key=lambda task: (
                -task.priority,
                task.due_step,
                task.patient_id if task.patient_id is not None else -1,
                task.task_id,
            )
        )
        return tasks

    def _patient_visible_copy(self, patient: PatientState) -> PatientState:
        visible = patient.model_copy(deep=True)
        recent_contact = (
            max(
                patient.last_reviewed_step,
                patient.last_vitals_step,
                patient.last_documented_step,
            )
            >= self._state.step_count - 4
        )
        if patient.active_incident_id < 0 and not recent_contact:
            visible.last_vitals = None
        return visible

    def _has_pending_work(self) -> bool:
        return any(
            not patient.discharge_prepared
            for patient in self._ward.patients
            if (
                patient.medications_due
                or patient.vitals_due
                or patient.active_incident_id >= 0
                or patient.review_required
            )
        )

    def _update_metrics_step(self) -> None:
        self._episode_metrics.step = self._state.step_count

    def _build_observation(
        self,
        done: bool,
        reward: float | None,
        truncated: bool = False,
        action_result: ActionResult | None = None,
    ) -> SakhaObservation:
        visible_patients = [self._patient_visible_copy(patient) for patient in self._ward.patients]
        pending_tasks = self._build_pending_tasks(visible_patients)
        active_incidents = sum(
            1 for patient in self._ward.patients if patient.active_incident_id >= 0
        )
        ward_state = WardState(
            patients=visible_patients,
            current_time_minutes=self._ward.current_time_minutes,
            pending_tasks=pending_tasks,
            capacity=self._ward.capacity,
            beds_occupied=sum(
                1 for patient in self._ward.patients if not patient.discharge_prepared
            ),
            active_incident_count=active_incidents,
            current_step=self._state.step_count,
        )
        self._ward.pending_tasks = pending_tasks
        self._ward.active_incident_count = active_incidents
        self._state.patients = [patient.model_copy(deep=True) for patient in self._ward.patients]
        time_remaining = max(
            0, START_TIME_MINUTES + EIGHT_HOURS_MINUTES - self._ward.current_time_minutes
        )
        return SakhaObservation(
            done=done,
            reward=reward,
            truncated=truncated,
            ward_state=ward_state,
            pending_count=len(pending_tasks),
            time_remaining_minutes=time_remaining,
            action_result=action_result,
        )

    @property
    def state(self) -> SakhaState:
        return self._state

    @property
    def episode_metrics(self) -> SakhaEpisodeMetrics:
        return self._episode_metrics
