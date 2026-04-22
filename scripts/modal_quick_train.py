"""
Quick GRPO training run for Sakha.
Runs small episode count and writes metrics to JSONL for plotting.

Usage:
    modal run scripts/modal_quick_train.py --episodes 10 --max-steps 12
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
            "output.md",
        ],
    )
)

vol = modal.Volume.from_name("sakha-training", create_if_missing=True)

app = modal.App("sakha-grpo-quick", image=image)


@app.function(gpu="T4", timeout=1800, volumes={"/artifacts": vol})
def train(episodes: int = 10, max_steps: int = 12, model: str = "Qwen/Qwen3-0.6B") -> str:
    import os
    import subprocess
    import json
    import sys
    from pathlib import Path

    os.environ["PYTHONPATH"] = "/sakha/src"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
    sys.path.insert(0, "/sakha/src")

    print("Installing sakha package...")
    subprocess.run(
        ["pip", "install", "--no-deps", "-e", "/sakha"], stdout=sys.stdout, stderr=sys.stderr
    )

    output_dir = "/artifacts/quick_train"
    cmd = [
        "python",
        "/sakha/scripts/train_grpo.py",
        "--mode",
        "smoke",
        "--task",
        "hard",
        "--model",
        model,
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        output_dir,
        "--report-to",
        "none",
    ]

    print(f"Running training command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, cwd="/sakha", env=os.environ)

    results_files = list(Path(output_dir).rglob("results.json"))
    results = {}
    if results_files:
        results = json.loads(max(results_files, key=lambda p: p.stat().st_mtime).read_text())

    return json.dumps({"success": result.returncode == 0, "results": results})


@app.local_entrypoint()
def main(episodes: int = 10, max_steps: int = 12, model: str = "Qwen/Qwen3-0.6B"):
    print(f"Starting quick training: {episodes} episodes, {max_steps} steps")
    outcome = train.remote(episodes=episodes, max_steps=max_steps, model=model)
    print(outcome)
