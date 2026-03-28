import json
import hashlib
from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction


def test_deterministic_trajectory():
    def run_trajectory(seed: int) -> str:
        env = SakhaEnvironment()
        obs = env.reset(seed=seed)
        log = [obs.model_dump(mode="json")]
        for i in range(5):
            action = SakhaAction(action_type="noop", patient_id=None)
            obs = env.step(action)
            log.append(obs.model_dump(mode="json"))
        return hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest()

    hash1 = run_trajectory(42)
    hash2 = run_trajectory(42)
    assert hash1 == hash2, "Same seed must produce identical trajectory"
