import logging
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from sakha.models import (
    ActionType,
    Bed,
    BedStatus,
    DeteriorationEvent,
    MedicationSchedule,
    NurseState,
    PatientState,
    PatientStatus,
    SakhaAction,
    SakhaEpisodeMetrics,
    SakhaObservation,
    SakhaState,
    Vitals,
    VitalsSchedule,
    WardState,
)

logger = logging.getLogger(__name__)


EIGHT_HOURS_MINUTES = 480
START_TIME_MINUTES = 480
ESCALATION_RESPONSE_WINDOW_STEPS = 2
VITALS_CHECK_INTERVAL = 12  # Every hour (60 min / 5 min per step)
MEDICATION_ROUND_INTERVAL = 16  # Every 80 minutes

TASK_CONFIGS = {
    "easy": {
        "initial_patients": 5,
        "max_patients": 8,
        "arrival_rate": 0.05,
        "deterioration_count": 2,
        "adverse_event_rate": 0.02,
        "max_actions_per_step": 4,
    },
    "medium": {
        "initial_patients": 5,
        "max_patients": 10,
        "arrival_rate": 0.08,
        "deterioration_count": 3,
        "adverse_event_rate": 0.04,
        "max_actions_per_step": 3,
    },
    "hard": {
        "initial_patients": 8,
        "max_patients": 18,
        "arrival_rate": 0.12,
        "deterioration_count": 6,
        "adverse_event_rate": 0.06,
        "max_actions_per_step": 3,
    },
}

DIAGNOSES = [
    "post_surgery", "pneumonia", "fracture", "cardiac", "diabetic",
    "infection", "respiratory", "gastrointestinal", "neurological", "renal",
]

MEDICATIONS = [
    ("paracetamol", "pain_reliever"),
    ("ibuprofen", "anti_inflammatory"),
    ("amoxicillin", "antibiotic"),
    ("metformin", "diabetes"),
    ("lisinopril", "blood_pressure"),
    ("omeprazole", "acid_reflux"),
    ("azithromycin", "antibiotic"),
    ("morphine", "pain_reliever"),
]


def _generate_patient_vitals(rng: random.Random, is_critical: bool = False) -> Vitals:
    if is_critical:
        return Vitals(
            blood_pressure_sys=rng.randint(85, 100),
            blood_pressure_dia=rng.randint(55, 65),
            temperature=round(rng.uniform(38.5, 40.0), 1),
            spo2=rng.randint(88, 93),
            pulse=rng.randint(100, 130),
        )
    return Vitals(
        blood_pressure_sys=rng.randint(110, 130),
        blood_pressure_dia=rng.randint(70, 85),
        temperature=round(rng.uniform(36.4, 37.4), 1),
        spo2=rng.randint(95, 99),
        pulse=rng.randint(65, 90),
    )


def _create_patient(
    patient_id: int,
    arrival_time: int,
    rng: random.Random,
    is_critical: bool = False,
) -> PatientState:
    diagnosis = rng.choice(DIAGNOSES)
    vitals = _generate_patient_vitals(rng, is_critical)

    med_count = rng.randint(1, 4)
    med_schedules = []
    meds_due = []
    for idx in range(med_count):
        med_name, _ = rng.choice(MEDICATIONS)
        meds_due.append(med_name)
        scheduled = arrival_time + rng.randint(2, 8)
        med_schedules.append(
            MedicationSchedule(
                medicine_id=f"med_{patient_id}_{idx}",
                medicine_name=med_name,
                scheduled_time=scheduled,
                deadline=scheduled + 4,
                priority=rng.randint(1, 3),
            )
        )

    vitals_schedules = []
    for round_num in range(4):
        scheduled = arrival_time + (round_num + 1) * VITALS_CHECK_INTERVAL
        vitals_schedules.append(
            VitalsSchedule(
                round_id=round_num,
                scheduled_time=scheduled,
                deadline=scheduled + 3,
                is_recurring=(round_num < 3),
                recurrence_interval=VITALS_CHECK_INTERVAL if round_num < 3 else None,
            )
        )

    return PatientState(
        patient_id=patient_id,
        name=f"Patient_{patient_id}",
        diagnosis=diagnosis,
        status=PatientStatus.WAITING,
        arrival_time=arrival_time,
        vitals=vitals,
        vitals_trend=[vitals],
        escalation_level=2 if is_critical else 0,
        medications_due=meds_due,
        medication_schedules=med_schedules,
        vitals_schedules=vitals_schedules,
    )


class SakhaEnvironment(Environment[SakhaAction, SakhaObservation, SakhaState]):
    def __init__(self, patient_count: int = 18, task: str = "hard"):
        super().__init__()
        self._task = task
        self._config = TASK_CONFIGS.get(task, TASK_CONFIGS["hard"])
        self._requested_patient_count = patient_count
        self._rng = random.Random()
        self._state = SakhaState(episode_id=str(uuid4()), step_count=0)
        self._ward = WardState()
        self._patients_by_id: dict[int, PatientState] = {}
        self._beds_by_id: dict[int, Bed] = {}
        self._metrics = SakhaEpisodeMetrics()
        self._next_patient_id = 1
        self._deterioration_steps: list[int] = []
        self._pending_adverse_events: dict[int, int] = {}
        self._severe_since_step: dict[int, int] = {}
        self._pending_deterioration_handled: set[int] = set()
        self._last_vitals_check: dict[int, int] = {}
        self._last_med_admin: dict[int, int] = {}

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> SakhaObservation:
        self._rng = random.Random(seed) if seed is not None else random.Random()
        eid = episode_id or (f"seed-{seed}" if seed is not None else str(uuid4()))

        self._state = SakhaState(episode_id=eid, step_count=0, current_time=START_TIME_MINUTES)

        max_beds = max(self._requested_patient_count, self._config["max_patients"])
        beds = [Bed(bed_id=i) for i in range(1, max_beds + 1)]
        self._beds_by_id = {b.bed_id: b for b in beds}

        patients = []
        self._patients_by_id = {}
        self._next_patient_id = 1

        initial_count = self._requested_patient_count if self._requested_patient_count > 0 else self._config["initial_patients"]
        for i in range(initial_count):
            is_critical = (i == 0 and self._config["deterioration_count"] > 0)
            patient = _create_patient(self._next_patient_id, 0, self._rng, is_critical)
            self._next_patient_id += 1

            bed = beds[i]
            bed.status = BedStatus.OCCUPIED
            bed.patient_id = patient.patient_id
            patient.bed_id = bed.bed_id
            patient.status = PatientStatus.ADMITTED
            patient.admission_time = 0

            patients.append(patient)
            self._patients_by_id[patient.patient_id] = patient

        deterioration_count = self._config["deterioration_count"]
        self._deterioration_steps = []
        if deterioration_count > 0:
            steps = set()
            while len(steps) < deterioration_count:
                step = self._rng.randint(10, 85)
                steps.add(step)
            self._deterioration_steps = sorted(steps)

        self._pending_adverse_events = {}
        self._severe_since_step = {}
        self._pending_deterioration_handled = set()
        self._last_vitals_check = {}
        self._last_med_admin = {}

        self._ward = WardState(
            current_step=0,
            current_time_minutes=START_TIME_MINUTES,
            beds=beds,
            max_beds=max_beds,
            patients=patients,
            nurse=NurseState(max_actions_per_step=self._config["max_actions_per_step"]),
            total_admissions=len(patients),
        )

        self._patients_by_id = {p.patient_id: p for p in self._ward.patients}
        self._beds_by_id = {b.bed_id: b for b in self._ward.beds}

        self._metrics = SakhaEpisodeMetrics(episode_id=eid)

        return self._build_observation(done=False, truncated=False, reward=None)

    def step(self, action: SakhaAction, timeout_s: float | None = None, **kwargs) -> SakhaObservation:
        self._state.step_count += 1
        self._ward.current_step = self._state.step_count
        self._ward.current_time_minutes += 5
        self._state.current_time = self._ward.current_time_minutes
        self._ward.nurse.current_shift_hour = self._state.step_count // 12
        self._ward.nurse.actions_this_step = 0
        self._ward.nurse.fatigue_level = min(1.0, self._state.step_count / 96.0)

        prev_meds = self._metrics.meds_administered
        prev_vitals = self._metrics.vitals_checked
        prev_escalations = self._metrics.escalations

        self._process_patient_arrivals()
        self._trigger_deterioration()
        self._check_missed_tasks()
        step_reward = self._process_action(action)
        self._update_patient_statuses()
        self._process_discharges()

        delta_meds = self._metrics.meds_administered - prev_meds
        delta_vitals = self._metrics.vitals_checked - prev_vitals
        delta_escalations = self._metrics.escalations - prev_escalations

        action_processed = delta_meds > 0 or delta_vitals > 0 or delta_escalations > 0
        if action.action_type == ActionType.NOOP or not action_processed:
            step_reward = -0.05

        done = self._ward.current_time_minutes >= START_TIME_MINUTES + EIGHT_HOURS_MINUTES
        self._metrics.step = self._state.step_count
        self._metrics.patients_waiting = len(self._ward.waiting_patients)

        return self._build_observation(done=done, truncated=False, reward=step_reward)

    def _process_patient_arrivals(self) -> None:
        if self._rng.random() < self._config["arrival_rate"]:
            available_beds = self._ward.available_beds
            if available_beds:
                patient = _create_patient(
                    self._next_patient_id,
                    self._state.step_count,
                    self._rng,
                )
                self._next_patient_id += 1
                bed = available_beds[0]
                bed.status = BedStatus.OCCUPIED
                bed.patient_id = patient.patient_id
                patient.bed_id = bed.bed_id
                patient.status = PatientStatus.ADMITTED
                patient.admission_time = self._state.step_count
                self._ward.patients.append(patient)
                self._patients_by_id[patient.patient_id] = patient
                self._ward.total_admissions += 1
            else:
                self._ward.waiting_queue.append(self._next_patient_id)
                patient = _create_patient(
                    self._next_patient_id,
                    self._state.step_count,
                    self._rng,
                )
                self._next_patient_id += 1
                self._ward.patients.append(patient)
                self._patients_by_id[patient.patient_id] = patient

    def _trigger_deterioration(self) -> None:
        if self._state.step_count in self._deterioration_steps:
            admitted = self._ward.admitted_patients
            if admitted:
                patient = self._rng.choice(admitted)
                event_id = f"deter_{self._state.step_count}_{patient.patient_id}"
                event = DeteriorationEvent(
                    event_id=event_id,
                    triggered_at=self._state.step_count,
                    escalation_window=ESCALATION_RESPONSE_WINDOW_STEPS,
                    severity=2,
                )
                patient.deterioration_events.append(event)
                patient.escalation_level = max(patient.escalation_level, 2)
                patient.status = PatientStatus.CRITICAL
                patient.vitals = _generate_patient_vitals(self._rng, is_critical=True)
                patient.vitals_trend.append(patient.vitals)
                self._severe_since_step[patient.patient_id] = self._state.step_count
                new_vitals_schedule = VitalsSchedule(
                    round_id=len(patient.vitals_schedules),
                    scheduled_time=self._state.step_count,
                    deadline=self._state.step_count + 2,
                    is_recurring=False,
                )
                patient.vitals_schedules.append(new_vitals_schedule)
                self._metrics.adverse_events += 1

    def _check_missed_tasks(self) -> None:
        current_step = self._state.step_count
        for patient in self._ward.admitted_patients:
            for med in patient.medication_schedules:
                if not med.administered and current_step > med.deadline:
                    self._metrics.meds_missed += 1
                    self._metrics.tasks_missed += 1
                    med.administered = True

            for vitals_check in patient.vitals_schedules:
                if not vitals_check.checked and current_step > vitals_check.deadline:
                    self._metrics.vitals_missed += 1
                    self._metrics.tasks_missed += 1
                    vitals_check.checked = True

            if patient.escalation_level >= 2:
                severe_since = self._severe_since_step.get(patient.patient_id, current_step)
                if current_step - severe_since > ESCALATION_RESPONSE_WINDOW_STEPS:
                    if patient.patient_id not in self._pending_deterioration_handled:
                        self._metrics.missed_escalations += 1
                        self._pending_deterioration_handled.add(patient.patient_id)

    def _process_action(self, action: SakhaAction) -> float:
        reward = 0.0

        if action.action_type == ActionType.NOOP:
            return reward

        if action.patient_id is not None:
            patient = self._patients_by_id.get(action.patient_id)
            if patient is None or not patient.is_admitted:
                return -0.05

        if self._ward.nurse.actions_this_step >= self._ward.nurse.max_actions_per_step:
            return -0.1

        patient = self._patients_by_id.get(action.patient_id) if action.patient_id else None

        if action.action_type == ActionType.ADMINISTER_MEDICINE and patient:
            reward = self._administer_medicine(patient)
        elif action.action_type == ActionType.CHECK_VITALS and patient:
            reward = self._check_vitals(patient)
        elif action.action_type == ActionType.ESCALATE and patient:
            reward = self._escalate(patient)
        elif action.action_type == ActionType.DOCUMENT and patient:
            reward = self._document(patient)
        elif action.action_type == ActionType.COMMUNICATE:
            reward = self._communicate()
        elif action.action_type == ActionType.HANDOVER:
            reward = self._handover()

        self._ward.nurse.actions_this_step += 1
        return reward

    def _administer_medicine(self, patient: PatientState) -> float:
        if not patient.medications_due and not any(
            not m.administered for m in patient.medication_schedules
        ):
            return -0.05

        current_step = self._state.step_count
        on_time = False

        if patient.medications_due:
            patient.medications_due.pop(0)
            self._metrics.meds_administered += 1
            self._metrics.tasks_completed += 1
            on_time = True

        for med in patient.medication_schedules:
            if not med.administered:
                med.administered = True
                med.administered_time = current_step
                self._metrics.meds_administered += 1
                self._metrics.tasks_completed += 1
                if current_step <= med.deadline:
                    self._metrics.meds_on_time += 1
                    on_time = True
                else:
                    self._metrics.meds_late += 1
                break

        self._last_med_admin[patient.patient_id] = current_step
        if patient.escalation_level > 0:
            self._metrics.conflicts_resolved += 1

        return 0.15 if on_time else 0.08

    def _check_vitals(self, patient: PatientState) -> float:
        if not patient.vitals_due:
            return -0.05

        current_step = self._state.step_count
        on_time = False

        for vitals_check in patient.vitals_schedules:
            if not vitals_check.checked:
                vitals_check.checked = True
                vitals_check.checked_time = current_step
                self._metrics.vitals_checked += 1
                self._metrics.tasks_completed += 1
                if current_step <= vitals_check.deadline:
                    self._metrics.vitals_on_time += 1
                    on_time = True
                else:
                    self._metrics.vitals_late += 1

                if patient.patient_id in self._pending_deterioration_handled:
                    self._metrics.deteriorations_handled += 1
                    self._pending_deterioration_handled.discard(patient.patient_id)
                break

        self._last_vitals_check[patient.patient_id] = current_step
        new_vitals = _generate_patient_vitals(self._rng)
        patient.vitals = new_vitals
        patient.vitals_trend.append(new_vitals)
        if len(patient.vitals_trend) > 5:
            patient.vitals_trend = patient.vitals_trend[-5:]

        for event in patient.deterioration_events:
            if not event.early_warning_detected and self._detect_early_warning(patient):
                event.early_warning_detected = True
                self._metrics.deteriorations_detected_early += 1

        return 0.10 if on_time else 0.05

    def _escalate(self, patient: PatientState) -> float:
        if patient.escalation_level < 2:
            return -0.05

        current_step = self._state.step_count
        self._metrics.escalations += 1
        self._metrics.tasks_completed += 1

        severe_since = self._severe_since_step.get(patient.patient_id)
        if severe_since is not None:
            response_time = current_step - severe_since
            if response_time <= ESCALATION_RESPONSE_WINDOW_STEPS:
                self._metrics.escalations_on_time += 1
                reward = 0.20
            else:
                self._metrics.escalations_late += 1
                reward = 0.10
        else:
            reward = 0.15

        patient.escalation_level = 0
        patient.status = PatientStatus.ADMITTED
        self._severe_since_step.pop(patient.patient_id, None)
        self._pending_deterioration_handled.add(patient.patient_id)

        for event in patient.deterioration_events:
            if not event.responded:
                event.responded = True
                event.responded_at = current_step
                break

        return reward

    def _document(self, patient: PatientState) -> float:
        if patient.last_documented_step == self._state.step_count:
            return -0.05

        patient.last_documented_step = self._state.step_count
        self._metrics.documents_completed += 1
        return 0.03

    def _communicate(self) -> float:
        self._metrics.tasks_completed += 1
        return 0.02

    def _handover(self) -> float:
        if self._state.step_count < 48:
            return -0.05

        if self._ward.nurse.handover_completed:
            return -0.05

        undocumented = sum(
            1 for p in self._ward.admitted_patients
            if p.last_documented_step is None
            or self._state.step_count - p.last_documented_step > 20
        )
        total = len(self._ward.admitted_patients)
        quality = 1.0 - (undocumented / max(1, total))

        self._ward.nurse.handover_completed = True
        self._ward.nurse.handover_required = False
        self._metrics.handovers_completed += 1
        self._metrics.handovers_quality = quality

        return 0.2 * quality

    def _update_patient_statuses(self) -> None:
        for patient in self._ward.admitted_patients:
            if patient.escalation_level >= 2:
                patient.status = PatientStatus.CRITICAL
            elif patient.total_tasks_pending == 0 and patient.escalation_level == 0:
                patient.status = PatientStatus.STABLE
            else:
                patient.status = PatientStatus.ADMITTED

    def _process_discharges(self) -> None:
        for patient in list(self._ward.admitted_patients):
            if patient.status == PatientStatus.STABLE:
                all_meds_done = (
                    len(patient.medications_due) == 0
                    and all(m.administered for m in patient.medication_schedules)
                )
                all_vitals_done = all(v.checked for v in patient.vitals_schedules)
                no_active_deterioration = all(
                    e.responded for e in patient.deterioration_events
                ) or len(patient.deterioration_events) == 0

                if all_meds_done and all_vitals_done and no_active_deterioration:
                    if self._rng.random() < 0.1:
                        patient.status = PatientStatus.DISCHARGED
                        patient.discharge_time = self._state.step_count
                        if patient.bed_id:
                            bed = self._beds_by_id.get(patient.bed_id)
                            if bed:
                                bed.status = BedStatus.CLEANING
                        self._ward.total_discharges += 1
                        self._metrics.patients_discharged += 1

        for bed in self._ward.beds:
            if bed.status == BedStatus.CLEANING:
                if self._rng.random() < 0.3:
                    bed.status = BedStatus.AVAILABLE
                    bed.patient_id = None

        for patient_id in list(self._ward.waiting_queue):
            patient = self._patients_by_id.get(patient_id)
            if patient and self._ward.available_beds:
                self._ward.waiting_queue.remove(patient_id)
                bed = self._ward.available_beds[0]
                bed.status = BedStatus.OCCUPIED
                bed.patient_id = patient.patient_id
                patient.bed_id = bed.bed_id
                patient.status = PatientStatus.ADMITTED
                patient.admission_time = self._state.step_count
                self._metrics.patients_admitted += 1

    def _detect_early_warning(self, patient: PatientState) -> bool:
        if len(patient.vitals_trend) < 2:
            return False
        current = patient.vitals_trend[-1]
        previous = patient.vitals_trend[-2]
        temp_rising = current.temperature > previous.temperature + 0.5
        spo2_falling = current.spo2 < previous.spo2 - 2
        pulse_rising = current.pulse > previous.pulse + 15
        return temp_rising or spo2_falling or pulse_rising

    @property
    def state(self) -> SakhaState:
        return self._state

    @property
    def episode_metrics(self) -> SakhaEpisodeMetrics:
        return self._metrics

    def _build_observation(self, done: bool, truncated: bool, reward: float | None) -> SakhaObservation:
        pending = sum(p.total_tasks_pending for p in self._ward.admitted_patients)
        time_remaining = (
            START_TIME_MINUTES + EIGHT_HOURS_MINUTES
        ) - self._ward.current_time_minutes

        return SakhaObservation(
            done=done,
            reward=reward,
            ward_state=self._ward.model_copy(deep=True),
            pending_count=pending,
            time_remaining_minutes=max(0, time_remaining),
            shift_hour=self._ward.nurse.current_shift_hour,
            nurse_fatigue=self._ward.nurse.fatigue_level,
            metadata={},
        )
