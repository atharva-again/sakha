from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction


def test_reward_on_time_medication():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type="administer_medicine", patient_id=1))
    assert obs.reward is not None
    assert obs.reward > 0.0


def test_reward_noop_penalizes_inaction():
    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    obs = env.step(SakhaAction(action_type="noop", patient_id=None))
    assert obs.reward is not None
    assert obs.reward < 0.0


def test_reward_aligns_with_grader():
    from sakha.graders import score_easy_task

    env = SakhaEnvironment(patient_count=5)
    obs = env.reset(seed=42)
    trajectory = [obs]

    for i in range(5):
        obs = env.step(SakhaAction(action_type="administer_medicine", patient_id=i + 1))
        trajectory.append(obs)

    final_reward = trajectory[-1].reward
    grader_score = score_easy_task(trajectory)
    assert final_reward > 0.0
    assert grader_score > 0.0


def test_reward_deterministic():
    def run_episode(seed: int) -> list[float]:
        env = SakhaEnvironment(patient_count=5)
        env.reset(seed=seed)
        rewards = []
        for _ in range(10):
            obs = env.step(SakhaAction(action_type="administer_medicine", patient_id=1))
            rewards.append(obs.reward)
        return rewards

    r1 = run_episode(42)
    r2 = run_episode(42)
    assert r1 == r2
