import hashlib
import json

from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction


def test_deterministic_trajectory():
    def run_trajectory(seed: int) -> str:
        env = SakhaEnvironment()
        obs = env.reset(seed=seed)
        log = [obs.model_dump(mode="json")]
        for _i in range(5):
            obs = env.step(SakhaAction(action_type=ActionType.NOOP))
            log.append(obs.model_dump(mode="json"))
        return hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest()

    assert run_trajectory(42) == run_trajectory(42)
    assert run_trajectory(42) != run_trajectory(43)


def test_seed_changes_initial_distribution():
    env_a = SakhaEnvironment(patient_count=18, task="hard")
    env_b = SakhaEnvironment(patient_count=18, task="hard")

    obs_a = env_a.reset(seed=42)
    obs_b = env_b.reset(seed=43)

    meds_a = [len(p.medications_due) for p in obs_a.ward_state.patients]
    meds_b = [len(p.medications_due) for p in obs_b.ward_state.patients]
    escalations_a = [p.escalation_level for p in obs_a.ward_state.patients]
    escalations_b = [p.escalation_level for p in obs_b.ward_state.patients]

    assert meds_a != meds_b or escalations_a != escalations_b
