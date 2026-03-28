import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sakha <command>")
        print("Commands: lint, format, typecheck, check, all")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "lint":
        result = subprocess.run(("ruff", "check", "src/"))
    elif cmd == "format":
        result = subprocess.run(("ruff", "format", "src/"))
    elif cmd == "typecheck":
        result = subprocess.run(("ty", "check", "src/"))
    elif cmd == "check":
        r1 = subprocess.run(("ruff", "check", "src/"))
        if r1.returncode != 0:
            sys.exit(r1.returncode)
        result = subprocess.run(("ty", "check", "src/"))
    elif cmd == "all":
        r1 = subprocess.run(("ruff", "check", "src/"))
        if r1.returncode != 0:
            sys.exit(r1.returncode)
        r2 = subprocess.run(("ruff", "format", "src/"))
        if r2.returncode != 0:
            sys.exit(r2.returncode)
        result = subprocess.run(("ty", "check", "src/"))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
