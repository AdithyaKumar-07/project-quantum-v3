from __future__ import annotations
"""Start the local API only when the Project Quantum API is not already running."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8000
BASE = f"http://{HOST}:{PORT}"
SERVICE_NAME = "Project Quantum V3 API"


def _json_get(url: str, timeout: float = 1.5) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, URLError, ValueError, TimeoutError):
        return None


def _port_is_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, PORT)) == 0


def _is_our_api() -> bool:
    payload = _json_get(f"{BASE}/api/health")
    return bool(payload and payload.get("service") == SERVICE_NAME and payload.get("status") == "ok")


def main() -> int:
    if _is_our_api():
        print(f"[OK] {SERVICE_NAME} is already running at {BASE}; reusing it.")
        dataset = _json_get(f"{BASE}/api/dataset/status")
        if dataset is None:
            print("[WARNING] API health is good, but dataset-status endpoint did not return JSON.")
        else:
            print(f"[OK] Dataset status endpoint responded (available={dataset.get('available', False)}).")
        return 0

    if _port_is_open():
        print(f"[ERROR] Port {PORT} is occupied by a different service, not {SERVICE_NAME}.")
        print("Stop the other process or configure this project and dashboard to use another port.")
        print(f"Find the owning PID with:  netstat -ano | findstr :{PORT}")
        print("Then inspect it with:      tasklist /FI \"PID eq <PID>\"")
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "uvicorn", "qmi.api.main:app", "--app-dir", "src",
               "--host", HOST, "--port", str(PORT)]
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
    try:
        subprocess.Popen(command, cwd=ROOT, env=env, creationflags=flags)
    except OSError as exc:
        print(f"[ERROR] Could not start the API process: {type(exc).__name__}: {exc}")
        return 3

    print(f"Starting {SERVICE_NAME} at {BASE} ...", flush=True)
    for _ in range(40):
        if _is_our_api():
            dataset = _json_get(f"{BASE}/api/dataset/status")
            print("[OK] API health endpoint verified.")
            if dataset is not None:
                print(f"[OK] Dataset-status endpoint verified (available={dataset.get('available', False)}).")
            else:
                print("[WARNING] API health responded, but dataset-status JSON is not available yet.")
            print(f"API documentation: {BASE}/docs")
            return 0
        time.sleep(1)
    print("[ERROR] API did not become healthy within 40 seconds.")
    print("Inspect the separate 'Project Quantum API' console for the startup traceback or port error.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
