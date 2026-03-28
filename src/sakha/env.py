import logging
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from sakha.models import (
    ActionType,
    PatientState,
    SakhaAction,
    SakhaEpisodeMetrics,
    SakhaObservation,
    SakhaState,
    Vitals,
    WardState,
)

logger = logging.getLogger(__name__)


EIGHT_HOURS_MINUTES = 480
START_TIME_MINUTES = 480

TASK_CONFIGS = {
    "easy": {"patient_count": 5, "deterioration_count": 0},
    "medium": {"patient_count": 8, "deterioration_count": 2},
    "hard": {"patient_count": 18, "deterioration_count": 5},
}


def _generate_deterioration_steps(rng: random.Random, count: int, max_steps: int = 96) -> list[int]:
    if count == 0:
        return []
    steps = set()
    for _ in range(count):
        step = rng.randint(5, max(10, max_steps - 10))
        steps.add(step)
    return sorted(steps)


def _make_default_patients(count: int = 18, rng: random.Random | None = None) -> list[PatientState]:
    patients = []
    for i in range(count):
        patients.append(
            PatientState(
                bed_id=i + 1,
                name=f"Patient_{i + 1}",
                diagnosis="general",
                medications_due=["paracetamol", "ibuprofen"] if i % 3 == 0 else ["paracetamol"],
                vitals_due=True,
                last_vitals=Vitals(
                    blood_pressure_sys=120,
                    blood_pressure_dia=80,
                    temperature=37.0,
                    spo2=98,
                    pulse=72,
                ),
                escalation_level=1 if i % 4 == 0 else 0,
            )
        )
    return patients


class SakhaEnvironment(Environment[SakhaAction, SakhaObservation, SakhaState]):
    def __init__(self, patient_count: int = 18, task: str = "hard"):
        super().__init__()
        self._task = task
        task_config = TASK_CONFIGS.get(task, TASK_CONFIGS["hard"])
        self._patient_count = patient_count or task_config["patient_count"]
        self._state = SakhaState(episode_id=str(uuid4()), step_count=0)
        self._ward = WardState(patients=_make_default_patients(self._patient_count))
        self._rng = random.Random()
        self._meds_administered = 0
        self._vitals_checked = 0
        self._escalations = 0
        self._missed_escalations = 0
        self._conflicts_resolved = 0
        self._deteriorations_handled = 0
        self._missed_escalation_patients: set[int] = set()
        self._episode_metrics = SakhaEpisodeMetrics()
        self._deterioration_steps: list[int] = []

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs
    ) -> SakhaObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        eid = episode_id or (f"seed-{seed}" if seed is not None else str(uuid4()))

        task_config = TASK_CONFIGS.get(self._task, TASK_CONFIGS["hard"])
        deteriorations = task_config.get("deterioration_count", 5)
        self._deterioration_steps = _generate_deterioration_steps(self._rng, deteriorations)

        self._state = SakhaState(episode_id=eid, step_count=0, current_time=START_TIME_MINUTES)
        self._ward = WardState(
            patients=_make_default_patients(self._patient_count),
            current_time_minutes=START_TIME_MINUTES,
            pending_tasks=[],
        )
        self._meds_administered = 0
        self._vitals_checked = 0
        self._escalations = 0
        self._missed_escalations = 0
        self._conflicts_resolved = 0
        self._deteriorations_handled = 0
        self._missed_escalation_patients = set()

        self._episode_metrics = SakhaEpisodeMetrics(
            episode_id=eid,
            step=0,
            meds_administered=0,
            vitals_checked=0,
            escalations=0,
            missed_escalations=0,
            conflicts_resolved=0,
            deteriorations_handled=0,
        )

        return self._build_observation(done=False, reward=None)

    def step(
        self, action: SakhaAction, timeout_s: float | None = None, **kwargs
    ) -> SakhaObservation:
        self._state.step_count += 1
        self._ward.current_time_minutes += 5
        self._state.current_time = self._ward.current_time_minutes

        prev_meds = self._meds_administered
        prev_vitals = self._vitals_checked
        prev_escalations = self._escalations
        prev_missed = self._missed_escalations

        self._trigger_events()
        self._process_action(action)
        self._check_missed_escalations()

        delta_meds = self._meds_administered - prev_meds
        delta_vitals = self._vitals_checked - prev_vitals
        delta_escalations = self._escalations - prev_escalations
        delta_missed = self._missed_escalations - prev_missed

        step_reward = 0.0
        step_reward += delta_meds * 0.1
        step_reward += delta_vitals * 0.05
        step_reward += delta_escalations * 0.15

        step_reward -= delta_missed * 0.05

        action_processed = delta_meds > 0 or delta_vitals > 0 or delta_escalations > 0

        if action.action_type == ActionType.NOOP or not action_processed:
            step_reward = -0.05

        done = self._ward.current_time_minutes >= START_TIME_MINUTES + EIGHT_HOURS_MINUTES
        truncated = False

        self._episode_metrics.step = self._state.step_count
        self._episode_metrics.meds_administered = self._meds_administered
        self._episode_metrics.vitals_checked = self._vitals_checked
        self._episode_metrics.escalations = self._escalations
        self._episode_metrics.missed_escalations = self._missed_escalations
        self._episode_metrics.conflicts_resolved = self._conflicts_resolved
        self._episode_metrics.deteriorations_handled = self._deteriorations_handled

        logger.debug(
            f"Step {self._state.step_count}: action={action.action_type}, reward={step_reward:.3f}, "
            f"done={done}, meds={self._meds_administered}, vitals={self._vitals_checked}, "
            f"escalations={self._escalations}, missed={self._missed_escalations}"
        )

        return self._build_observation(done=done, truncated=truncated, reward=step_reward)

    def _trigger_events(self) -> None:
        if self._deterioration_steps and self._state.step_count in self._deterioration_steps:
            target = (self._state.step_count * 7 + 3) % self._patient_count
            for p in self._ward.patients:
                if p.bed_id == target + 1:
                    p.escalation_level = max(p.escalation_level, 2)
                    p.last_vitals = Vitals(
                        blood_pressure_sys=90,
                        blood_pressure_dia=60,
                        temperature=39.5,
                        spo2=92,
                        pulse=110,
                    )
                    break

    def _validate_patient_id(self, patient_id: int | None) -> bool:
        if patient_id is None:
            return False
        return 1 <= patient_id <= self._patient_count

    def _process_action(self, action: SakhaAction) -> None:
        if action.action_type == ActionType.NOOP:
            return

        invalid_action = False
        if action.patient_id is not None and not self._validate_patient_id(action.patient_id):
            logger.warning(f"Invalid patient_id {action.patient_id} - ignoring action")
            invalid_action = True

        if invalid_action:
            return

        if action.action_type == ActionType.ADMINISTER_MEDICINE and action.patient_id is not None:
            for p in self._ward.patients:
                if p.bed_id == action.patient_id and p.medications_due:
                    p.medications_due.pop(0)
                    self._meds_administered += 1
                    if p.escalation_level > 0:
                        self._conflicts_resolved += 1
                    break
        elif action.action_type == ActionType.CHECK_VITALS and action.patient_id is not None:
            for p in self._ward.patients:
                if p.bed_id == action.patient_id:
                    p.vitals_due = False
                    self._vitals_checked += 1
                    if p.escalation_level >= 2:
                        self._deteriorations_handled += 1
                    break
        elif action.action_type == ActionType.ESCALATE and action.patient_id is not None:
            for p in self._ward.patients:
                if p.bed_id == action.patient_id:
                    if p.escalation_level >= 2:
                        self._escalations += 1
                        p.escalation_level = 0
                    break

    def _check_missed_escalations(self) -> None:
        for p in self._ward.patients:
            if p.escalation_level >= 2 and p.bed_id not in self._missed_escalation_patients:
                self._missed_escalation_patients.add(p.bed_id)
                self._missed_escalations += 1

    @property
    def state(self) -> SakhaState:
        return self._state

    @property
    def episode_metrics(self) -> SakhaEpisodeMetrics:
        return self._episode_metrics

    def _build_observation(
        self, done: bool, reward: float | None, truncated: bool = False
    ) -> SakhaObservation:
        pending = sum(1 for p in self._ward.patients if p.medications_due or p.vitals_due)
        time_remaining = (
            START_TIME_MINUTES + EIGHT_HOURS_MINUTES
        ) - self._ward.current_time_minutes
        return SakhaObservation(
            done=done,
            reward=reward,
            ward_state=self._ward.model_copy(deep=True),
            pending_count=pending,
            time_remaining_minutes=max(0, time_remaining),
        )
