import sys
import argparse
import json

try:
    import httpx
except ImportError:
    import urllib.request


def check_endpoint(url: str, mode: str = "hf") -> bool:
    reset_url = f"{url}/reset" if not url.endswith("/reset") else url
    try:
        try:
            resp = httpx.post(reset_url, json={"seed": 42}, timeout=10.0)
        except NameError:
            req = urllib.request.Request(
                reset_url,
                data=json.dumps({"seed": 42}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = type(
                    "Resp",
                    (),
                    {"status_code": r.status, "json": lambda: json.loads(r.read())},
                )()

        if resp.status_code != 200:
            print(f"FAIL: status {resp.status_code}")
            return False

        data = resp.json() if hasattr(resp, "json") else {}
        if "observation" not in data and "done" not in data:
            print(f"FAIL: response missing observation/done keys: {list(data.keys())}")
            return False

        print(f"OK: {mode} endpoint returned valid reset response")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", default="hf", choices=["local", "hf"])
    args = parser.parse_args()

    ok = check_endpoint(args.url, args.mode)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
