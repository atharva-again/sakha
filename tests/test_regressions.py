from sakha.env import SakhaEnvironment
from sakha.graders import score_hard_task
from sakha.models import ActionType, SakhaAction


def test_check_vitals_only_rewards_when_due():
    env = SakhaEnvironment(patient_count=5)
    env.reset(seed=42)

    first = env.step(SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=1))
    second = env.step(SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=1))

    assert first.reward == 0.1
    assert second.reward == 0.1
    assert env.episode_metrics.vitals_checked == 2


def test_hard_grader_counts_mid_episode_escalation_resolution():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    trajectory = [obs]

    for _step in range(30):
        actionable_meds = next(
            (patient for patient in obs.ward_state.patients if patient.medications_due),
            None,
        )
        severe_patient = next(
            (
                patient
                for patient in obs.ward_state.patients
                if patient.vitals_due
                and patient.last_vitals is not None
                and (
                    patient.last_vitals.temperature >= 39.0
                    or patient.last_vitals.spo2 < 93
                    or patient.last_vitals.pulse >= 100
                )
            ),
            None,
        )
        if severe_patient is not None:
            action = SakhaAction(action_type=ActionType.ESCALATE, patient_id=severe_patient.bed_id)
        elif actionable_meds is not None:
            action = SakhaAction(
                action_type=ActionType.ADMINISTER_MEDICINE,
                patient_id=actionable_meds.bed_id,
            )
        else:
            action = SakhaAction(action_type=ActionType.NOOP)
        obs = env.step(action)
        trajectory.append(obs)
        if obs.done:
            break

    assert env.episode_metrics.escalations > 0
    assert score_hard_task(trajectory) >= 0.0
