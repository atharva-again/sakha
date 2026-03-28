import subprocess
import sys


def run_inference() -> str:
    result = subprocess.run(
        [
            sys.executable,
            "inference.py",
            "--tasks",
            "easy,medium,hard",
            "--seed",
            "42",
            "--episodes",
            "3",
            "--deterministic-baseline",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_baseline_reproducibility():
    out1 = run_inference()
    out2 = run_inference()
    lines1 = [l for l in out1.splitlines() if l.startswith("[")]
    lines2 = [l for l in out2.splitlines() if l.startswith("[")]
    assert lines1 == lines2, "Baseline must produce identical outputs across runs"
