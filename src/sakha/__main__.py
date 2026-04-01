import subprocess
import sys


def _run(command: tuple[str, ...]) -> int:
    return subprocess.run(command).returncode


def _run_many(commands: list[tuple[str, ...]]) -> int:
    for command in commands:
        code = _run(command)
        if code != 0:
            return code
    return 0


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sakha <command>")
        print(
            "Commands: lint, format, typecheck, test, check, fix, ci, validate, "
            "repro-report, separation-report, submit-check"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    lint_targets = ("src/", "tests/", "server/", "scripts/", "inference.py")
    typecheck_targets = ("src/sakha",)

    if cmd == "lint":
        result = _run(("ruff", "check", *lint_targets))
    elif cmd == "format":
        result = _run(("ruff", "format", *lint_targets))
    elif cmd == "typecheck":
        result = _run(("ty", "check", *typecheck_targets))
    elif cmd == "test":
        result = _run(("python", "-m", "pytest", "tests/", "-v"))
    elif cmd == "check":
        result = _run_many(
            [
                ("ruff", "format", "--check", *lint_targets),
                ("ruff", "check", *lint_targets),
                ("ty", "check", *typecheck_targets),
            ]
        )
    elif cmd == "fix":
        result = _run_many(
            [
                ("ruff", "check", "--fix", *lint_targets),
                ("ruff", "format", *lint_targets),
            ]
        )
    elif cmd == "ci":
        result = _run_many(
            [
                ("ruff", "format", "--check", *lint_targets),
                ("ruff", "check", *lint_targets),
                ("ty", "check", *typecheck_targets),
                ("python", "-m", "pytest", "tests/", "-v"),
            ]
        )
    elif cmd == "validate":
        result = _run(("openenv", "validate"))
    elif cmd == "repro-report":
        result = _run(("python", "scripts/reproducibility_report.py"))
    elif cmd == "separation-report":
        result = _run(("python", "scripts/benchmark_separation_report.py"))
    elif cmd == "submit-check":
        result = _run(("python", "scripts/submit_check.py"))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    sys.exit(result)


if __name__ == "__main__":
    main()
