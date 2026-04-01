from sakha.env import SakhaEnvironment
from sakha.graders import score_hard_task
from sakha.models import ActionType, SakhaAction


def test_hard_grader_penalizes_missed_escalation():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    trajectory_no_escalate = [obs]
    trajectory_with_escalate = [obs]

    for _ in range(20):
        obs = env.step(SakhaAction(action_type=ActionType.NOOP))
        trajectory_no_escalate.append(obs)

    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    for _ in range(20):
        severe = next((p for p in obs.ward_state.patients if p.escalation_level >= 2), None)
        action = (
            SakhaAction(action_type=ActionType.ESCALATE, patient_id=severe.bed_id)
            if severe is not None
            else SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=1)
        )
        obs = env.step(action)
        trajectory_with_escalate.append(obs)

    assert score_hard_task(trajectory_with_escalate) > score_hard_task(trajectory_no_escalate)


def test_hard_grader_weak_vs_strong_gap():
    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    weak = [obs]
    for _ in range(30):
        obs = env.step(SakhaAction(action_type=ActionType.NOOP))
        weak.append(obs)

    env = SakhaEnvironment(patient_count=18, task="hard")
    obs = env.reset(seed=42)
    strong = [obs]
    for step in range(30):
        severe = next((p for p in obs.ward_state.patients if p.escalation_level >= 2), None)
        if severe:
            action = SakhaAction(action_type=ActionType.ESCALATE, patient_id=severe.bed_id)
        else:
            target = (step % 18) + 1
            action = SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=target)
        obs = env.step(action)
        strong.append(obs)

    assert score_hard_task(strong) > score_hard_task(weak)
