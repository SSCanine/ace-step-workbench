"""ACE-Step 1.5 integration: start/stop its REST API server and talk to it.

The Workbench never imports ACE-Step itself. ACE-Step runs in its own
environment (its .venv) as a separate process, and everything goes through its
documented REST API, so ACE-Step can be updated independently.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import config

IS_WIN = sys.platform == "win32"


class AceError(RuntimeError):
    pass


def venv_python(ace_dir: Path) -> Path:
    return ace_dir / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def is_installed(ace_dir: Path) -> bool:
    return (ace_dir / "acestep").is_dir() and venv_python(ace_dir).exists()


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30)
            proc.wait(timeout=15)  # make sure the port is free before anything restarts
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


def popen_logged(cmd: list[str], cwd: Path, env: dict, sink) -> subprocess.Popen:
    """Start a process whose combined output is fed line by line to sink(str)."""
    kw = {}
    if IS_WIN:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1, **kw,
    )

    def pump():
        for line in proc.stdout:
            sink(line.rstrip("\r\n"))

    threading.Thread(target=pump, daemon=True).start()
    return proc


def base_env() -> dict:
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", "TERM": "dumb"})
    # Don't leak the Workbench's own venv into ACE-Step's.
    env.pop("VIRTUAL_ENV", None)
    return env


class AceServer:
    """Owns the ACE-Step API server process when the Workbench started it."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.log = collections.deque(maxlen=3000)
        self.lock = threading.Lock()
        self.started_at = 0.0
        self.last_error = ""

    def _sink(self, line: str) -> None:
        self.log.append(line)

    @property
    def managed(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, settings: dict) -> None:
        with self.lock:
            if self.managed:
                return
            ace_dir = config.ace_dir(settings)
            if not is_installed(ace_dir):
                raise AceError(f"ACE-Step is not installed at {ace_dir}. Run install, or set its folder in Settings.")
            a = settings["ace"]
            u = urlparse(a["api_url"])
            host, port = u.hostname or "127.0.0.1", u.port or 8001
            env = base_env()
            env.update({
                "ACESTEP_NO_INIT": "false",  # load models at startup instead of on first request
                "ACESTEP_CONFIG_PATH": a["dit_model"],
                "ACESTEP_INIT_LLM": "true" if a["init_lm"] else "false",
                "ACESTEP_LM_MODEL_PATH": a["lm_model"],
                "ACESTEP_LM_BACKEND": a.get("lm_backend") or ("pt" if IS_WIN else "vllm"),
                "ACESTEP_LM_OFFLOAD_TO_CPU": "true" if a["lm_offload_cpu"] else "false",
            })
            if a.get("api_key"):
                env["ACESTEP_API_KEY"] = a["api_key"]
            env.update({k: str(v) for k, v in (a.get("extra_env") or {}).items()})
            cmd = [str(venv_python(ace_dir)), "-m", "acestep.api_server", "--host", host, "--port", str(port)]
            self.log.append(f"[workbench] starting ACE-Step: {' '.join(cmd)}")
            self.log.append(f"[workbench] DiT={a['dit_model']}  LM={a['lm_model'] if a['init_lm'] else 'off'}")
            self.proc = popen_logged(cmd, ace_dir, env, self._sink)
            self.started_at = time.time()
            self.last_error = ""

    def stop(self) -> None:
        with self.lock:
            if self.proc is not None:
                self.log.append("[workbench] stopping ACE-Step")
                kill_tree(self.proc)
                self.proc = None

    def restart(self, settings: dict) -> None:
        self.stop()
        time.sleep(1.0)
        self.start(settings)

    def tail(self, n: int = 200) -> list[str]:
        return list(self.log)[-n:]


class AceClient:
    def __init__(self, settings: dict) -> None:
        a = settings["ace"]
        self.base = a["api_url"].rstrip("/")
        self.headers = {"Authorization": f"Bearer {a['api_key']}"} if a.get("api_key") else {}

    def _unwrap(self, r: httpx.Response):
        try:
            body = r.json()
        except Exception:
            raise AceError(f"ACE-Step returned HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise AceError(body.get("detail") or body.get("error") or f"HTTP {r.status_code}")
        # ACE wraps everything as {data, code, error}; some errors are HTTP 200 with code 500.
        if isinstance(body, dict) and "code" in body and "data" in body:
            if body.get("code") not in (200, None):
                raise AceError(body.get("error") or f"ACE-Step error code {body.get('code')}")
            return body["data"]
        return body

    def get(self, path: str, timeout: float = 15, **params):
        try:
            return self._unwrap(httpx.get(self.base + path, headers=self.headers, params=params or None, timeout=timeout))
        except httpx.HTTPError as e:
            raise AceError(f"Cannot reach ACE-Step at {self.base} ({e.__class__.__name__}).") from e

    def post(self, path: str, body: dict | None = None, timeout: float = 60, files=None, data=None):
        try:
            if files:
                r = httpx.post(self.base + path, headers=self.headers, data=data, files=files, timeout=timeout)
            else:
                r = httpx.post(self.base + path, headers=self.headers, json=body or {}, timeout=timeout)
            return self._unwrap(r)
        except httpx.HTTPError as e:
            raise AceError(f"Cannot reach ACE-Step at {self.base} ({e.__class__.__name__}).") from e

    # ── high level ──
    def health(self, timeout: float = 3) -> dict:
        try:
            # Short connect timeout: Windows takes ~2 s to refuse a closed local port, which made /api/status
            # too slow for health checks (e.g. Lair's 1.5 s) whenever ACE-Step was stopped.
            r = httpx.get(self.base + "/health", timeout=httpx.Timeout(timeout, connect=0.3))
            return self._unwrap(r)
        except Exception:
            return {}

    def inventory(self) -> dict:
        return self.get("/v1/model_inventory")

    def submit(self, params: dict, files: dict | None = None) -> str:
        if files:
            # multipart: every scalar becomes a form field
            form = {k: ("" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))) for k, v in params.items() if v is not None}
            data = self.post("/release_task", files=files, data=form, timeout=120)
        else:
            data = self.post("/release_task", {k: v for k, v in params.items() if v is not None}, timeout=60)
        return data["task_id"]

    def query(self, task_id: str) -> dict:
        items = self.post("/query_result", {"task_id_list": [task_id]}, timeout=30)
        return items[0] if items else {}

    def download(self, file_url: str) -> bytes:
        url = file_url if file_url.startswith("http") else self.base + file_url
        r = httpx.get(url, headers=self.headers, timeout=120)
        if r.status_code >= 400:
            raise AceError(f"Could not download audio ({r.status_code})")
        return r.content

    def lora_status(self) -> dict:
        return self.get("/v1/lora/status")

    def lora_load(self, path: str) -> None:
        self.post("/v1/lora/load", {"lora_path": path}, timeout=300)

    def lora_toggle(self, on: bool) -> None:
        self.post("/v1/lora/toggle", {"use_lora": on})

    def lora_scale(self, scale: float) -> None:
        self.post("/v1/lora/scale", {"scale": max(0.0, min(1.0, float(scale)))})
