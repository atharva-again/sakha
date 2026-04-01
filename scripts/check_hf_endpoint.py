import argparse
import json
import sys
import time

try:
    import httpx
except ImportError:
    import urllib.request


def check_endpoint(url: str, mode: str = "hf", max_attempts: int = 30) -> bool:
    reset_url = f"{url}/reset" if not url.endswith("/reset") else url
    timeout_s = 10.0
    backoff_s = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            try:
                resp = httpx.post(reset_url, json={"seed": 42}, timeout=timeout_s)
            except NameError:
                req = urllib.request.Request(
                    reset_url,
                    data=json.dumps({"seed": 42}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=int(timeout_s)) as r:
                    resp = type(
                        "Resp",
                        (),
                        {"status_code": r.status, "json": lambda: json.loads(r.read())},
                    )()

            if resp.status_code != 200:
                print(f"Attempt {attempt}: status {resp.status_code}")
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.25, 2.0)
                continue

            data = resp.json() if hasattr(resp, "json") else {}
            if "observation" not in data and "done" not in data:
                print(f"Attempt {attempt}: response missing observation/done keys")
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.25, 2.0)
                continue

            print(f"OK: {mode} endpoint returned valid reset response")
            return True

        except Exception as e:
            print(f"Attempt {attempt}: {e}")
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 1.25, 2.0)

    print("FAIL: endpoint did not become healthy in retry window")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", default="hf", choices=["local", "hf"])
    parser.add_argument("--max-attempts", type=int, default=30)
    args = parser.parse_args()

    ok = check_endpoint(args.url, args.mode, max_attempts=args.max_attempts)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
