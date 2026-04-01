from scripts.eval_policies import run_policy


def test_priority_beats_vitals_spam_on_hard() -> None:
    spam = run_policy("hard", "vitals_spam", seed=42, episodes=1)
    priority = run_policy("hard", "priority", seed=42, episodes=1)
    assert priority["mean"] > spam["mean"]


def test_priority_beats_escalation_tunnel_on_hard() -> None:
    tunnel = run_policy("hard", "escalation_tunnel", seed=42, episodes=1)
    priority = run_policy("hard", "priority", seed=42, episodes=1)
    assert priority["mean"] > tunnel["mean"]
