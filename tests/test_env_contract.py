from openenv.core.env_server.types import Observation, State


def test_sakha_action_has_required_fields():
    from sakha.models import ActionType, SakhaAction

    action = SakhaAction(action_type=ActionType.NOOP, patient_id=None)
    assert hasattr(action, "action_type")
    assert hasattr(action, "patient_id")


def test_sakha_observation_has_required_fields():
    from sakha.models import SakhaObservation

    obs = SakhaObservation()
    assert obs.done is False
    assert obs.reward is None
    assert hasattr(obs, "metadata")
    assert hasattr(obs, "ward_state")


def test_sakha_state_has_required_fields():
    from sakha.models import SakhaState

    state = SakhaState(episode_id="test", step_count=0)
    assert state.episode_id == "test"
    assert state.step_count == 0
    assert hasattr(state, "current_time")


def test_env_has_reset_step_state():
    from sakha.env import SakhaEnvironment

    env = SakhaEnvironment()
    obs = env.reset(seed=42)
    assert isinstance(obs, Observation)
    assert hasattr(env, "state")
    assert isinstance(env.state, State)


def test_env_reset_returns_observation():
    from sakha.env import SakhaEnvironment
    from sakha.models import SakhaObservation

    env = SakhaEnvironment()
    obs = env.reset(seed=42)
    assert isinstance(obs, SakhaObservation)
    assert obs.done is False


def test_env_step_returns_observation():
    from sakha.env import SakhaEnvironment
    from sakha.models import ActionType, SakhaAction, SakhaObservation

    env = SakhaEnvironment()
    env.reset(seed=42)
    action = SakhaAction(action_type=ActionType.NOOP, patient_id=None)
    obs = env.step(action)
    assert isinstance(obs, SakhaObservation)


def test_env_deterministic_reset():
    from sakha.env import SakhaEnvironment

    env1 = SakhaEnvironment()
    env2 = SakhaEnvironment()
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    assert obs1.model_dump() == obs2.model_dump()
