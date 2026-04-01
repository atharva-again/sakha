"""Tests for new environment functionality: arrivals, deterioration, vitals tracking."""

from sakha.env import SakhaEnvironment
from sakha.graders import score_hard_task
from sakha.models import ActionType, PatientStatus, SakhaAction


def test_patient_arrivals_during_shift():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    initial_count = len(obs.ward_state.patients)

    for _ in range(50):
        obs = env.step(SakhaAction(action_type=ActionType.NOOP))

    assert len(obs.ward_state.patients) >= initial_count


def test_deterioration_triggers_escalation_requirement():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)

    for step in range(30):
        severe = next((p for p in obs.ward_state.patients if p.escalation_level >= 2), None)
        if severe:
            action = SakhaAction(action_type=ActionType.ESCALATE, patient_id=severe.bed_id)
        else:
            action = SakhaAction(action_type=ActionType.NOOP)
        obs = env.step(action)

    assert env.episode_metrics.escalations > 0


def test_vitals_trend_tracking():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)

    for _ in range(10):
        obs = env.step(SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=1))

    patient = next(p for p in obs.ward_state.patients if p.patient_id == 1)
    assert len(patient.vitals_trend) >= 1


def test_full_episode_completes_without_errors():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    trajectory = [obs]

    for step in range(96):
        severe = next((p for p in obs.ward_state.patients if p.escalation_level >= 2), None)
        if severe:
            action = SakhaAction(action_type=ActionType.ESCALATE, patient_id=severe.bed_id)
        else:
            meds = next((p for p in obs.ward_state.patients if p.medications_due), None)
            if meds:
                action = SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=meds.bed_id)
            else:
                vitals = next((p for p in obs.ward_state.patients if p.vitals_due), None)
                action = (
                    SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=vitals.bed_id)
                    if vitals
                    else SakhaAction(action_type=ActionType.NOOP)
                )
        obs = env.step(action)
        trajectory.append(obs)
        if obs.done:
            break

    score = score_hard_task(trajectory)
    assert 0.0 <= score <= 1.0
