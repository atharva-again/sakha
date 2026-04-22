"""
Modal app for running Sakha GRPO training smoke test.

Usage:
    modal run scripts/modal_smoke_test.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'transformers>=5.2.0' 'trl[quantization]' peft accelerate bitsandbytes jmespath",
        "uv pip install --system 'git+https://github.com/meta-pytorch/OpenEnv.git' pydantic fastapi uvicorn openai python-dotenv tenacity",
    )
    .add_local_dir("/home/verma/projects/sakha", remote_path="/sakha")
)

app = modal.App("sakha-grpo-smoke-test", image=image)


@app.function(gpu="T4", timeout=600)
def run_smoke_test(task: str = "hard", max_steps: int = 96) -> dict:
    """Run GRPO smoke test on Modal."""
    import os
    import subprocess
    import json
    import sys
    from pathlib import Path

    os.environ["PYTHONPATH"] = "/sakha/src:" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, "/sakha/src")

    install = subprocess.run(
        ["pip", "install", "--no-deps", "-e", "/sakha"],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        install = subprocess.run(
            ["pip", "install", "-e", "/sakha"],
            capture_output=True,
            text=True,
        )

    cmd = [
        "python",
        "/sakha/scripts/train_grpo.py",
        "--mode",
        "smoke",
        "--task",
        task,
        "--max-steps",
        str(max_steps),
        "--output-dir",
        "/tmp/grpo_smoke",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/sakha/src"
    env["TRL_EXPERIMENTAL_SILENCE"] = "1"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/sakha",
        env=env,
    )

    output = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }

    if install.returncode != 0:
        output["install_stdout"] = install.stdout
        output["install_stderr"] = install.stderr

    results_path = Path("/tmp/grpo_smoke") / "results.json"
    if results_path.exists():
        output["results"] = json.loads(results_path.read_text())

    return output


@app.local_entrypoint()
def main():
    """Run smoke test locally and print results."""
    print("Running Sakha GRPO smoke test on Modal...")
    result = run_smoke_test.remote()

    print(f"\nExit code: {result['returncode']}")
    print(f"Success: {result['success']}")
    print("\n--- STDOUT ---")
    print(result["stdout"])
    if result["stderr"]:
        print("\n--- STDERR ---")
        print(result["stderr"])
    if "results" in result:
        print("\n--- RESULTS ---")
        print(result["results"])

    if not result["success"]:
        raise RuntimeError("Smoke test failed!")

    print("\nSmoke test passed!")
