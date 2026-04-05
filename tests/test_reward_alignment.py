from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction


def _take_queue_head(env: SakhaEnvironment, obs):
    if not obs.ward_state.pending_tasks:
        return env.step(SakhaAction(action_type=ActionType.NOOP, patient_id=None))
    task = obs.ward_state.pending_tasks[0]
    return env.step(SakhaAction(action_type=task.required_action, patient_id=task.patient_id))


def test_reward_on_time_medication_or_review_is_positive():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    obs = _take_queue_head(env, obs)
    assert obs.reward is not None
    assert obs.reward > 0.0


def test_reward_noop_is_negative_with_pending_work():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.NOOP, patient_id=None))
    assert obs.reward is not None
    assert obs.reward == -0.03


def test_reward_aligns_with_grader():
    from sakha.graders import score_easy_task

    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    trajectory = [obs]

    for _ in range(8):
        obs = _take_queue_head(env, obs)
        trajectory.append(obs)

    grader_score = score_easy_task(trajectory)
    assert grader_score > 0.0
    assert any(step.reward and step.reward > 0.0 for step in trajectory[1:])


def test_reward_deterministic():
    def run_episode(seed: int) -> list[float | None]:
        env = SakhaEnvironment(patient_count=5, task="easy")
        obs = env.reset(seed=seed)
        rewards = []
        for _ in range(10):
            obs = _take_queue_head(env, obs)
            rewards.append(obs.reward)
        return rewards

    r1 = run_episode(42)
    r2 = run_episode(42)
    assert r1 == r2


def test_shift_does_not_end_before_full_length():
    env = SakhaEnvironment(patient_count=5, task="easy")
    obs = env.reset(seed=42)
    for _ in range(95):
        obs = env.step(SakhaAction(action_type=ActionType.NOOP, patient_id=None))
        assert obs.done is False
    obs = env.step(SakhaAction(action_type=ActionType.NOOP, patient_id=None))
    assert obs.done is True
