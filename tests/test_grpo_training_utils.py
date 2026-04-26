from sakha.grpo_training import (
    build_state_aligned_examples,
    choose_queue_head_action,
    parse_action_response_with_status,
    reconstruct_env_state,
    score_completion_action,
)
from sakha.models import ActionType


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


def test_parser_strips_qwen_thinking_block():
    """Reasoning text containing pseudo-calls like patient(3) or step(5) must
    not shadow the real answer that follows the </think> tag."""
    response = (
        "<think>"
        "Let me check patient(3) first. Right now at step(5) the queue says "
        "review_patient(2) is highest priority but I should reconsider."
        "</think>\n"
        "administer_medicine(3)"
    )
    action, parsed_ok = parse_action_response_with_status(response)
    assert parsed_ok is True
    assert action.action_type == ActionType.ADMINISTER_MEDICINE
    assert action.patient_id == 3


def test_parser_handles_truncated_thinking_block():
    """An unclosed <think> block means the model ran out of budget mid-thought.
    There is no committed answer; we must report parse failure and not pluck
    a stray reasoning artifact like patient(3)."""
    response = "<think>i should probably look at patient(3) but first step(0)"
    _, parsed_ok = parse_action_response_with_status(response)
    assert parsed_ok is False


def test_parser_takes_last_valid_action_when_model_restates():
    """Model often weighs options in prose (`X first, then Y`) before
    committing. The trailing call is the actual answer."""
    response = "I considered review_patient(2) but the final action is check_vitals(7)."
    action, parsed_ok = parse_action_response_with_status(response)
    assert parsed_ok is True
    assert action.action_type == ActionType.CHECK_VITALS
    assert action.patient_id == 7


def test_parser_accepts_kwargs_form():
    response = "administer_medicine(patient_id=3)"
    action, parsed_ok = parse_action_response_with_status(response)
    assert parsed_ok is True
    assert action.action_type == ActionType.ADMINISTER_MEDICINE
    assert action.patient_id == 3


def test_parser_rejects_unknown_verb_even_with_parens():
    """Action names not in the documented set must fail. Loosening this
    (e.g. mapping `administer(3)` to `administer_medicine`) would hide
    real format-compliance regressions in the model."""
    for response in ("administer(3)", "assess(3)", "treat(3)"):
        _, parsed_ok = parse_action_response_with_status(response)
        assert parsed_ok is False, response


def test_parser_rejects_missing_parens():
    """The protocol is `name(args?)`. Bare verbs are a parse failure even
    when unambiguous, so the format penalty stays meaningful during GRPO."""
    for response in ("medication_round", "ward_sweep", "noop"):
        _, parsed_ok = parse_action_response_with_status(response)
        assert parsed_ok is False, response


def test_parser_is_case_insensitive():
    action, parsed_ok = parse_action_response_with_status("Review_Patient(3)")
    assert parsed_ok is True
    assert action.action_type == ActionType.REVIEW_PATIENT
    assert action.patient_id == 3


def test_parser_ignores_distractor_calls_inside_thinking_with_no_answer():
    """A closed <think> block with calls inside but no answer after is a
    parse failure, not a leakage of the reasoning content."""
    response = "<think>look at step(0) and patient(2) carefully</think>"
    _, parsed_ok = parse_action_response_with_status(response)
    assert parsed_ok is False


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
