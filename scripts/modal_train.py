"""
Modal app for running Sakha GRPO training.

Usage:
    modal run scripts/modal_train.py --mode demo --task hard --episodes 50
    modal run scripts/modal_train.py --mode smoke
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'transformers>=5.2.0' 'trl[quantization]' peft accelerate bitsandbytes jmespath wandb",
        "uv pip install --system 'git+https://github.com/meta-pytorch/OpenEnv.git' pydantic fastapi uvicorn openai python-dotenv tenacity",
    )
    .add_local_dir(
        "/home/verma/projects/sakha",
        remote_path="/sakha",
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "artifacts",
            ".sisyphus",
            "*.pyc",
            "*.pyo",
        ],
    )
)

volume = modal.Volume.from_name("sakha-training", create_if_missing=True)

app = modal.App("sakha-grpo-training", image=image)


@app.function(
    gpu="T4",
    timeout=3600,
    volumes={"/artifacts": volume},
)
def run_training(
    mode: str = "demo",
    task: str = "hard",
    episodes: int | None = None,
    max_steps: int = 96,
    model: str = "Qwen/Qwen3-0.6B",
    seed: int = 42,
) -> dict:
    """Run GRPO training on Modal."""
    import os
    import subprocess
    import json
    import sys
    from pathlib import Path

    os.environ["PYTHONPATH"] = "/sakha/src"
    sys.path.insert(0, "/sakha/src")

    install = subprocess.run(
        ["pip", "install", "--no-deps", "-e", "/sakha"],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        subprocess.run(["pip", "install", "-e", "/sakha"], capture_output=True, text=True)

    output_dir = "/artifacts/grpo"
    cmd = [
        "python",
        "/sakha/scripts/train_grpo.py",
        "--mode",
        mode,
        "--task",
        task,
        "--model",
        model,
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
    ]
    if episodes is not None:
        cmd.extend(["--episodes", str(episodes)])

    env = os.environ.copy()
    env["PYTHONPATH"] = "/sakha/src"
    env["TRL_EXPERIMENTAL_SILENCE"] = "1"
    env["WANDB_MODE"] = "disabled"

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

    results_files = list(Path(output_dir).rglob("results.json"))
    if results_files:
        latest = max(results_files, key=lambda p: p.stat().st_mtime)
        output["results_file"] = str(latest)
        output["results"] = json.loads(latest.read_text())

    checkpoints = list(Path(output_dir).rglob("checkpoint-*"))
    if checkpoints:
        output["checkpoints"] = [str(c) for c in checkpoints]

    return output


@app.local_entrypoint()
def main(
    mode: str = "demo",
    task: str = "hard",
    episodes: int | None = None,
    max_steps: int = 96,
    model: str = "Qwen/Qwen3-0.6B",
    seed: int = 42,
):
    print(f"Running Sakha GRPO training: mode={mode}, task={task}, model={model}")
    result = run_training.remote(
        mode=mode,
        task=task,
        episodes=episodes,
        max_steps=max_steps,
        model=model,
        seed=seed,
    )

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
        raise RuntimeError("Training failed!")

    print("\nTraining completed!")
