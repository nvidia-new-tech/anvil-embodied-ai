#!/usr/bin/env python3
"""Local web tool for scoring grasp-and-flip benchmarks per model checkpoint.

Stdlib only — no extra dependencies. Serves a single page that auto-detects the
model checkpoint and inference config currently selected in .env, lets you tally
10 trials for each package type, and stores everything in a JSON file.

Usage:
    ./tools/benchmark_web/server.py            # http://127.0.0.1:8777
    ./tools/benchmark_web/server.py --port 9000 --host 0.0.0.0

Results file (override with --results):
    tools/benchmark_web/results/pack.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
INDEX_HTML = HERE / "index.html"

# A checkpoint directory is one that actually holds weights.
WEIGHT_FILES = ("model.safetensors", "model.pt", "pytorch_model.bin")
MAX_SCAN_DEPTH = 5


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def read_env_file(path: Path) -> dict[str, str]:
    """Parse the simple KEY=VALUE lines of .env; ignore comments and exports."""
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
    return env


def find_checkpoints(zoo: Path) -> list[str]:
    """Repo-relative paths of every directory under model_zoo holding weights."""
    if not zoo.is_dir():
        return []
    found: set[Path] = set()
    for root, dirs, files in os.walk(zoo):
        root_path = Path(root)
        if len(root_path.relative_to(zoo).parts) >= MAX_SCAN_DEPTH:
            dirs[:] = []
        # training_state holds optimizer shards, not a loadable checkpoint.
        dirs[:] = [d for d in dirs if d != "training_state"]
        if any(f in files for f in WEIGHT_FILES):
            found.add(root_path)
    return sorted(str(p.relative_to(REPO_ROOT)) for p in found)


def find_configs(config_dir: Path) -> list[str]:
    if not config_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(REPO_ROOT)) for p in config_dir.glob("*.yaml")
    )


def container_env(var: str) -> str | None:
    """Read a var from the running ros2 container, if docker is reachable."""
    try:
        names = subprocess.run(
            ["docker", "ps", "--filter", "name=ros2", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    for name in names:
        try:
            out = subprocess.run(
                ["docker", "exec", name, "printenv", var],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


def normalize(path_str: str) -> str:
    """Make an absolute or ./-prefixed path repo-relative when it lives here."""
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def match_known_checkpoint(model: str, checkpoints: list[str]) -> str:
    """Fold an out-of-tree MODEL_PATH onto the same checkpoint in this repo.

    .env / the container may reference the zoo through a different repo clone
    (e.g. /home/anvil/... vs /srv/anvil/...); the same suffix means same model.
    """
    if not model or model in checkpoints:
        return model
    parts = Path(model).parts
    if "model_zoo" not in parts:
        return model
    suffix = "/".join(parts[parts.index("model_zoo"):])
    return suffix if suffix in checkpoints else model


def checkpoint_meta(rel_path: str) -> dict:
    """Pull policy type and training step count out of a checkpoint's configs."""
    meta: dict = {}
    base = REPO_ROOT / rel_path
    for name in ("config.json", "train_config.json", "anvil_config.json"):
        candidate = base / name
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        policy = data.get("type") or (data.get("policy") or {}).get("type")
        if policy and "policy_type" not in meta:
            meta["policy_type"] = policy
        for key in ("steps", "step"):
            if isinstance(data.get(key), int) and "steps" not in meta:
                meta["steps"] = data[key]
    return meta


def build_context() -> dict:
    # The running container is ground truth; then our own env (run_inference.sh
    # exports what it mounted); then .env, which compose reads but bash does not.
    env = read_env_file(REPO_ROOT / ".env")

    def resolve(var: str) -> str:
        value = container_env(var) or os.environ.get(var) or env.get(var, "")
        return normalize(value)

    model = resolve("MODEL_PATH")
    config = resolve("CONFIG_FILE")
    checkpoints = find_checkpoints(REPO_ROOT / "model_zoo")
    model = match_known_checkpoint(model, checkpoints)
    # A checkpoint mounted from outside the repo still belongs in the dropdown.
    if model and model not in checkpoints:
        checkpoints.insert(0, model)
    configs = find_configs(REPO_ROOT / "configs" / "lerobot_control")
    if config and config not in configs:
        configs.insert(0, config)
    return {
        "repo_root": str(REPO_ROOT),
        "current_model": model,
        "current_config": config,
        "checkpoints": checkpoints,
        "configs": configs,
        "checkpoint_meta": {c: checkpoint_meta(c) for c in checkpoints},
        "git_branch": git_branch(),
    }


def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data.get("sessions", []) if isinstance(data, dict) else []

    def save(self, sessions: list[dict]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"sessions": sessions}, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def upsert(self, session: dict) -> dict:
        sessions = self.load()
        session.setdefault("id", uuid.uuid4().hex[:12])
        session["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for i, existing in enumerate(sessions):
            if existing.get("id") == session["id"]:
                session.setdefault("created_at", existing.get("created_at"))
                sessions[i] = session
                break
        else:
            session.setdefault("created_at", session["updated_at"])
            sessions.append(session)
        self.save(sessions)
        return session

    def delete(self, session_id: str) -> bool:
        sessions = self.load()
        kept = [s for s in sessions if s.get("id") != session_id]
        if len(kept) == len(sessions):
            return False
        self.save(kept)
        return True


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    store: Store  # injected below
    server_version = "AnvilBenchmark/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if self.command != "GET" or not self.path.startswith("/api/context"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return None

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                body = INDEX_HTML.read_bytes()
            except OSError:
                self._json(500, {"error": "index.html missing"})
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path.startswith("/api/context"):
            self._json(200, build_context())
        elif self.path.startswith("/api/sessions"):
            self._json(200, {"sessions": self.store.load()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/api/sessions"):
            self._json(404, {"error": "not found"})
            return
        payload = self._read_json()
        if not isinstance(payload, dict):
            self._json(400, {"error": "expected a JSON object"})
            return
        self._json(200, {"session": self.store.upsert(payload)})

    def do_DELETE(self):  # noqa: N802
        prefix = "/api/sessions/"
        if not self.path.startswith(prefix):
            self._json(404, {"error": "not found"})
            return
        session_id = self.path[len(prefix):]
        if self.store.delete(session_id):
            self._json(200, {"deleted": session_id})
        else:
            self._json(404, {"error": "no such session"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument(
        "--results",
        type=Path,
        default=HERE / "results" / "pack.json",
        help="JSON file that stores benchmark sessions",
    )
    args = parser.parse_args()

    Handler.store = Store(args.results.resolve())
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Benchmark tool:  http://{args.host}:{args.port}")
    print(f"Results file:    {Handler.store.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
