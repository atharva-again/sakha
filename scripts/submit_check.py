import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(command: tuple[str, ...]) -> tuple[bool, str]:
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - started, 3)
    output = (result.stdout or "") + (result.stderr or "")
    line = f"$ {' '.join(command)}\n[exit={result.returncode} elapsed={elapsed}s]\n{output}".strip()
    return result.returncode == 0, line


def main() -> None:
    parser = argparse.ArgumentParser(description="Run submission readiness validation")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--output-json", default="artifacts/submit_check_report.json")
    args = parser.parse_args()

    checks: list[dict[str, str | bool]] = []

    commands = [
        ("uv", "run", "sakha", "ci"),
        ("uv", "run", "openenv", "validate"),
        (
            "uv",
            "run",
            "python",
            "inference.py",
            "--tasks",
            "easy,medium,hard",
            "--seed",
            "42",
            "--episodes",
            "3",
            "--deterministic-baseline",
            "--output-json",
            "artifacts/baseline_submit_check.json",
        ),
        (
            "uv",
            "run",
            "python",
            "scripts/benchmark_separation_report.py",
            "--task",
            "hard",
            "--seed",
            "42",
            "--episodes",
            "20",
            "--output-json",
            "artifacts/benchmark_separation_report.json",
        ),
        (
            "uv",
            "run",
            "python",
            "scripts/reproducibility_report.py",
            "--base-seed",
            "42",
            "--seed-count",
            "10",
            "--episodes",
            "5",
            "--output-json",
            "artifacts/reproducibility_report.json",
        ),
    ]

    if not args.skip_docker:
        commands.extend(
            [
                ("docker", "build", "-t", "sakha-submit-check", "."),
                (
                    "docker",
                    "run",
                    "--rm",
                    "-d",
                    "-p",
                    "7860:7860",
                    "--name",
                    "sakha-submit-check",
                    "sakha-submit-check",
                ),
                (
                    "uv",
                    "run",
                    "python",
                    "scripts/check_hf_endpoint.py",
                    "--url",
                    "http://127.0.0.1:7860",
                    "--mode",
                    "local",
                    "--max-attempts",
                    "40",
                ),
                ("docker", "stop", "sakha-submit-check"),
            ]
        )

    all_ok = True
    for cmd in commands:
        ok, output = _run(cmd)
        checks.append({"command": " ".join(cmd), "ok": ok, "output": output})
        all_ok = all_ok and ok
        print(output)
        print("-" * 80)
        if not ok and "docker run" in " ".join(cmd):
            _run(("docker", "stop", "sakha-submit-check"))

    payload = {
        "ok": all_ok,
        "generated_at": int(time.time()),
        "checks": checks,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
