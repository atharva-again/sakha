from sakha.grpo_training import (
    build_state_aligned_examples,
    choose_queue_head_action,
    parse_action_response_with_status,
    reconstruct_env_state,
    score_completion_action,
)


def _action_call(action) -> str:
    if action.patient_id is None:
        return f"{action.action_type.value}()"
    return f"{action.action_type.value}({action.patient_id})"


def test_parser_distinguishes_explicit_noop_from_parse_failure():
    action, parsed_ok = parse_action_response_with_status("noop()")
    assert parsed_ok is True
    assert action.action_type == "noop"

    fallback, parsed_ok = parse_action_response_with_status("I am not sure what to do")
    assert parsed_ok is False
    assert fallback.action_type == "noop"


def test_state_aligned_example_reconstructs_prompt_step():
    examples = build_state_aligned_examples(
        task="easy", episodes=1, seed=42, max_steps=96, state_steps=(8,)
    )

    _, obs = reconstruct_env_state(
        task="easy",
        seed=examples["seed"][0],
        replay_actions_json=examples["replay_actions_json"][0],
    )

    assert obs.ward_state.current_step == examples["target_step"][0]
    assert f"step={obs.ward_state.current_step}" in examples["prompt"][0][1]["content"]


def test_state_aligned_reward_has_action_variance():
    examples = build_state_aligned_examples(
        task="easy", episodes=1, seed=42, max_steps=96, state_steps=(0,)
    )
    _, obs = reconstruct_env_state(
        task="easy",
        seed=examples["seed"][0],
        replay_actions_json=examples["replay_actions_json"][0],
    )
    good_action = choose_queue_head_action(obs)

    good_reward = score_completion_action(
        _action_call(good_action),
        task="easy",
        seed=examples["seed"][0],
        replay_actions_json=examples["replay_actions_json"][0],
    )
    noop_reward = score_completion_action(
        "noop()",
        task="easy",
        seed=examples["seed"][0],
        replay_actions_json=examples["replay_actions_json"][0],
    )
    parse_failure_reward = score_completion_action(
        "not an action",
        task="easy",
        seed=examples["seed"][0],
        replay_actions_json=examples["replay_actions_json"][0],
    )

    assert good_reward > noop_reward
    assert len({good_reward, noop_reward, parse_failure_reward}) > 1
