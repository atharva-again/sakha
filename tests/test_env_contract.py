from openenv.core.env_server.types import Observation

from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction, SakhaObservation


def test_env_reset_returns_observation():
    env = SakhaEnvironment()
    obs = env.reset(seed=42)
    assert isinstance(obs, Observation)
    assert isinstance(obs, SakhaObservation)
    assert obs.done is False


def test_env_step_returns_observation():
    env = SakhaEnvironment()
    env.reset(seed=42)
    obs = env.step(SakhaAction(action_type=ActionType.NOOP))
    assert isinstance(obs, Observation)


def test_env_deterministic_reset():
    env1 = SakhaEnvironment()
    env2 = SakhaEnvironment()
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    assert obs1.model_dump() == obs2.model_dump()


def test_env_observation_does_not_leak_episode_metrics():
    env = SakhaEnvironment()
    obs = env.reset(seed=42)
    assert "episode_metrics" not in obs.metadata
