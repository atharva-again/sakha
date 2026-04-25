import argparse
import json
import re
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from sakha.env import SakhaEnvironment
from sakha.models import ActionType, SakhaAction

from scripts.eval_common import (
    PATIENT_COUNTS,
    TASK_GRADERS,
    greedy_policy,
    noop_policy,
    priority_policy,
)


def timestep_scripted_policy(obs, step, pc):
    if step < 10:
        return SakhaAction(action_type=ActionType.ADMINISTER_MEDICINE, patient_id=(step % pc) + 1)
    elif step < 20:
        return SakhaAction(action_type=ActionType.CHECK_VITALS, patient_id=((step - 5) % pc) + 1)
    elif step < 30:
        return SakhaAction(
            action_type=ActionType.ADMINISTER_MEDICINE, patient_id=((step - 10) % pc) + 1
        )
    else:
        return SakhaAction(action_type=ActionType.NOOP)


POLICIES = {
    "noop": noop_policy,
    "greedy": greedy_policy,
    "priority": priority_policy,
    "timestep_scripted": timestep_scripted_policy,
}
LLM_POLICY_NAMES = {"base_llm", "trained"}


def _parse_seed_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in value.split(",")]


def load_llm_model(model_name_or_path: str, device: str = "cuda"):
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "LLM policies require optional dependencies. Install with: uv pip install torch transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    resolved_device = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"

    model = model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def _extract_json_block(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _parse_action_from_text(text: str, patient_count: int) -> SakhaAction:
    def _coerce_patient_id(value):
        if value is None:
            return None
        if isinstance(value, int):
            return value if 1 <= value <= patient_count else None
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                pid = int(value)
                return pid if 1 <= pid <= patient_count else None
        return None

    def _extract_action_payload(payload: dict | None) -> tuple[str | None, int | None]:
        if not payload:
            return None, None

        action_name = payload.get("action_type") or payload.get("action") or payload.get("name")
        patient_id = payload.get("patient_id")

        if "arguments" in payload and isinstance(payload["arguments"], str):
            nested = _extract_json_block(payload["arguments"])
            if nested:
                nested_action, nested_patient = _extract_action_payload(nested)
                action_name = nested_action or action_name
                patient_id = nested_patient if nested_patient is not None else patient_id

        if "function" in payload and isinstance(payload["function"], dict):
            nested_action, nested_patient = _extract_action_payload(payload["function"])
            action_name = nested_action or action_name
            patient_id = nested_patient if nested_patient is not None else patient_id

        if (
            "tool_calls" in payload
            and isinstance(payload["tool_calls"], list)
            and payload["tool_calls"]
        ):
            first_call = payload["tool_calls"][0]
            if isinstance(first_call, dict):
                nested_action, nested_patient = _extract_action_payload(first_call)
                action_name = nested_action or action_name
                patient_id = nested_patient if nested_patient is not None else patient_id

        if "tool_call" in payload and isinstance(payload["tool_call"], dict):
            nested_action, nested_patient = _extract_action_payload(payload["tool_call"])
            action_name = nested_action or action_name
            patient_id = nested_patient if nested_patient is not None else patient_id

        return action_name, _coerce_patient_id(patient_id)

    direct_match = re.search(r"([a-z_]+)\s*\(\s*(\d+|null|none)?\s*\)", text.lower())
    if direct_match:
        action_str = direct_match.group(1)
        patient_raw = direct_match.group(2)
        patient_id = None if patient_raw in (None, "null", "none") else int(patient_raw)
        if action_str in {a.value for a in ActionType}:
            return SakhaAction(
                action_type=ActionType(action_str), patient_id=_coerce_patient_id(patient_id)
            )

    payload = _extract_json_block(text)
    action_name, patient_id = _extract_action_payload(payload)
    if isinstance(action_name, str):
        normalized = action_name.strip().lower()
        if normalized in {a.value for a in ActionType}:
            return SakhaAction(action_type=ActionType(normalized), patient_id=patient_id)

    fallback_match = re.search(
        r"\b(" + "|".join(re.escape(action.value) for action in ActionType) + r")\b",
        text.lower(),
    )
    if fallback_match:
        action_str = fallback_match.group(1)
        pid_match = re.search(r"\bpatient[_\s-]?id\s*[:=]?\s*(\d+)\b", text.lower())
        parsed_pid = int(pid_match.group(1)) if pid_match else None
        return SakhaAction(
            action_type=ActionType(action_str), patient_id=_coerce_patient_id(parsed_pid)
        )

    return SakhaAction(action_type=ActionType.NOOP, patient_id=None)


def llm_policy_factory(model, tokenizer, device: str):
    valid_actions = [a.value for a in ActionType]
    system_prompt = (
        "You are a ward assistant policy for the Sakha environment. "
        "Return exactly one next action. Available actions: "
        f"{', '.join(valid_actions)}. "
        "Preferred output format: action_name(patient_id). "
        "Alternative accepted format: JSON object with keys action_type and patient_id. "
        "Use patient_id null only for noop."
    )

    def policy(obs, step, pc):
        obs_payload = {
            "step": step,
            "pending_count": obs.pending_count,
            "time_remaining_minutes": obs.time_remaining_minutes,
            "pending_tasks": [task.model_dump() for task in obs.ward_state.pending_tasks],
            "action_result": obs.action_result.model_dump() if obs.action_result else None,
        }
        user_prompt = (
            "Observation:\n"
            + json.dumps(obs_payload, ensure_ascii=False)
            + "\n\nRespond with the single best next action."
        )

        prompt = "<|system|>\n" + system_prompt + "\n<|user|>\n" + user_prompt + "\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        assert torch is not None
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _parse_action_from_text(text, pc)

    return policy


def run_policy(
    task: str,
    policy_name: str,
    seed: int,
    episodes: int,
    model=None,
    tokenizer=None,
    device: str = "cuda",
) -> dict:
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]
    if policy_name in LLM_POLICY_NAMES:
        if model is None or tokenizer is None:
            raise ValueError(f"Policy '{policy_name}' requires loaded model and tokenizer")
        policy = llm_policy_factory(model=model, tokenizer=tokenizer, device=device)
    else:
        policy = POLICIES[policy_name]
    scores = []

    for ep in range(episodes):
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed + ep)
        trajectory = [obs]
        for step in range(96):
            action = policy(obs, step, pc)
            obs = env.step(action)
            trajectory.append(obs)
            if obs.done:
                break
        scores.append(grader(trajectory))

    return {
        "policy": policy_name,
        "mean": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(
            (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
            4,
        ),
    }


def run_eval(
    task: str,
    policy_name: str,
    seeds: list[int],
    max_steps: int,
    model=None,
    tokenizer=None,
    device: str = "cuda",
) -> dict:
    """Run evaluation with detailed per-episode metrics (ported from eval_harness)."""
    pc = PATIENT_COUNTS[task]
    grader = TASK_GRADERS[task]

    if policy_name in LLM_POLICY_NAMES:
        if model is None or tokenizer is None:
            raise ValueError(f"Policy '{policy_name}' requires loaded model and tokenizer")
        policy = llm_policy_factory(model=model, tokenizer=tokenizer, device=device)
    else:
        policy = POLICIES[policy_name]

    episodes = []
    for seed in seeds:
        env = SakhaEnvironment(patient_count=pc, task=task)
        obs = env.reset(seed=seed)
        step_rewards = []
        for step in range(max_steps):
            action = policy(obs, step, pc)
            obs = env.step(action)
            step_rewards.append(obs.reward)
            if obs.done:
                break

        metrics = env.episode_metrics
        total_reward = sum(step_rewards)

        # Rebuild trajectory for grader
        trajectory = []
        obs = env.reset(seed=seed)
        trajectory.append(obs)
        for step in range(max_steps):
            action = policy(obs, step, pc)
            obs = env.step(action)
            trajectory.append(obs)
            if obs.done:
                break

        grader_score = grader(trajectory)

        episodes.append(
            {
                "seed": seed,
                "grader_score": round(grader_score, 6),
                "total_reward": round(total_reward, 6),
                "critical_incidents_resolved": metrics.critical_incidents_resolved,
                "critical_incidents_total": metrics.critical_incidents_total,
                "critical_incidents_missed": metrics.critical_incidents_missed,
                "medication_tasks_completed": metrics.medication_tasks_completed,
                "medication_tasks_on_time": metrics.medication_tasks_on_time,
                "vitals_tasks_completed": metrics.vitals_tasks_completed,
                "vitals_tasks_on_time": metrics.vitals_tasks_on_time,
                "overdue_tasks": metrics.overdue_tasks,
                "noop_steps": metrics.noop_steps,
                "discharges_prepared": metrics.discharges_prepared,
            }
        )

    def _mean(key: str) -> float:
        return sum(e[key] for e in episodes) / len(episodes)

    def _std(key: str) -> float:
        m = _mean(key)
        return (sum((e[key] - m) ** 2 for e in episodes) / len(episodes)) ** 0.5

    summary = {
        "mean_grader_score": round(_mean("grader_score"), 6),
        "std_grader_score": round(_std("grader_score"), 6),
        "mean_total_reward": round(_mean("total_reward"), 6),
        "std_total_reward": round(_std("total_reward"), 6),
        "mean_critical_incidents_resolved": round(_mean("critical_incidents_resolved"), 6),
        "mean_critical_incidents_missed": round(_mean("critical_incidents_missed"), 6),
        "mean_overdue_tasks": round(_mean("overdue_tasks"), 6),
        "mean_noop_steps": round(_mean("noop_steps"), 6),
        "mean_discharges_prepared": round(_mean("discharges_prepared"), 6),
    }

    return {
        "task": task,
        "policy": policy_name,
        "max_steps": max_steps,
        "seeds": seeds,
        "episodes": episodes,
        "summary": summary,
    }


def _print_markdown_table(result: dict) -> None:
    print(f"\n## Eval Results: {result['task']} / {result['policy']}\n")
    print("| Metric | Mean | Std |")
    print("|--------|------|-----|")
    s = result["summary"]
    print(f"| Grader Score | {s['mean_grader_score']:.4f} | {s['std_grader_score']:.4f} |")
    print(f"| Total Reward | {s['mean_total_reward']:.4f} | {s['std_total_reward']:.4f} |")
    print(f"| Critical Resolved | {s['mean_critical_incidents_resolved']:.2f} | - |")
    print(f"| Critical Missed | {s['mean_critical_incidents_missed']:.2f} | - |")
    print(f"| Overdue Tasks | {s['mean_overdue_tasks']:.2f} | - |")
    print(f"| NOOP Steps | {s['mean_noop_steps']:.2f} | - |")
    print(f"| Discharges | {s['mean_discharges_prepared']:.2f} | - |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Seed range, e.g. 42-71 or 42,43,44")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=96)
    all_policy_choices = sorted(set(POLICIES.keys()) | LLM_POLICY_NAMES)
    parser.add_argument("--policy", choices=all_policy_choices)
    parser.add_argument("--policy-a", default="noop", choices=all_policy_choices)
    parser.add_argument("--policy-b", default="priority", choices=all_policy_choices)
    parser.add_argument("--all-policies", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json")
    parser.add_argument("--markdown", action="store_true", help="Print markdown table output")
    args = parser.parse_args()

    # Determine seeds
    if args.seeds is not None:
        seeds = _parse_seed_range(args.seeds)
    else:
        seeds = list(range(args.seed, args.seed + args.episodes))

    requested_policies = set()
    if args.policy:
        requested_policies.add(args.policy)
    if not args.all_policies:
        requested_policies.update({args.policy_a, args.policy_b})

    needs_llm = any(name in LLM_POLICY_NAMES for name in requested_policies)
    model = None
    tokenizer = None
    resolved_device = args.device

    if needs_llm:
        if not args.model_path:
            raise ValueError("--model-path is required when using base_llm or trained policy")
        model, tokenizer, resolved_device = load_llm_model(args.model_path, device=args.device)

    if args.policy:
        if args.markdown or len(seeds) > 1:
            result = run_eval(
                args.task,
                args.policy,
                seeds,
                args.max_steps,
                model=model,
                tokenizer=tokenizer,
                device=resolved_device,
            )
            if args.markdown:
                _print_markdown_table(result)
            output = {
                "task": args.task,
                "policy": args.policy,
                "seeds": seeds,
                "max_steps": args.max_steps,
                "result": result,
            }
        else:
            result = run_policy(
                args.task,
                args.policy,
                args.seed,
                args.episodes,
                model=model,
                tokenizer=tokenizer,
                device=resolved_device,
            )
            output = {
                "task": args.task,
                "seed": args.seed,
                "episodes": args.episodes,
                "policy": result,
            }
    elif args.all_policies:
        policy_results = {
            name: run_policy(
                args.task,
                name,
                args.seed,
                args.episodes,
                model=model,
                tokenizer=tokenizer,
                device=resolved_device,
            )
            for name in POLICIES
        }
        output = {
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "policies": policy_results,
        }
    else:
        result_a = run_policy(
            args.task,
            args.policy_a,
            args.seed,
            args.episodes,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
        )
        result_b = run_policy(
            args.task,
            args.policy_b,
            args.seed,
            args.episodes,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
        )
        output = {
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "policy_a": result_a,
            "policy_b": result_b,
            "gap": round(result_b["mean"] - result_a["mean"], 4),
        }

    print(json.dumps(output, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
