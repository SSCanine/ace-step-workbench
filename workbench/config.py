"""Settings, stored in data/settings.json (never committed)."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SETTINGS_FILE = DATA / "settings.json"

DEFAULTS = {
    "ace": {
        # Folder of the ACE-Step 1.5 checkout. Empty = ./ACE-Step-1.5 next to this app.
        "path": "",
        "api_url": "http://127.0.0.1:8001",
        "api_key": "",
        # Start ACE-Step's API server automatically when the app starts.
        "autostart": True,
        "dit_model": "acestep-v15-turbo",
        "lm_model": "acestep-5Hz-lm-1.7B",
        "init_lm": True,
        "lm_offload_cpu": False,
        # "" = automatic: "pt" on Windows (vllm needs Triton), "vllm" elsewhere.
        "lm_backend": "",
        "extra_env": {},
    },
    "llm": {
        "preset": "ollama",
        "kind": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "llama3.1:8b",
        "api_key": "",
        "temperature": 0.9,
        "max_tokens": 4096,
        "timeout": 300,
        "json_mode": True,
        "effort": "medium",
    },
    # Saved API keys per preset so switching providers does not lose them.
    "llm_keys": {},
    "writer": {"target": "ace", "soften": True},
    "training": {
        "datasets_dir": "",
        "outputs_dir": "",
        # More folders to scan for LoRAs (e.g. ones you downloaded).
        "extra_lora_dirs": [],
    },
    "ui": {"show_tips": True},
}

_lock = threading.Lock()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        # An empty dict clears the value (e.g. extra_env = {}); a non-empty one merges.
        if isinstance(v, dict) and v and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> dict:
    with _lock:
        if SETTINGS_FILE.exists():
            try:
                return _merge(DEFAULTS, json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
            except Exception:
                pass
        return copy.deepcopy(DEFAULTS)


def save(patch: dict) -> dict:
    cur = load()
    new = _merge(cur, patch)
    with _lock:
        DATA.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_FILE)
    return new


def public(settings: dict) -> dict:
    """Settings safe to send to the browser: API keys are masked."""
    s = copy.deepcopy(settings)
    for sect in ("ace", "llm"):
        if s[sect].get("api_key"):
            s[sect]["api_key"] = "••••" + s[sect]["api_key"][-4:]
    s["llm_keys"] = {k: ("••••" + v[-4:] if v else "") for k, v in s.get("llm_keys", {}).items()}
    return s


def ace_dir(settings: dict) -> Path:
    p = settings["ace"].get("path") or ""
    return Path(p).expanduser() if p else ROOT / "ACE-Step-1.5"


def datasets_dir(settings: dict) -> Path:
    p = settings["training"].get("datasets_dir") or ""
    d = Path(p).expanduser() if p else DATA / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir(settings: dict) -> Path:
    p = settings["training"].get("outputs_dir") or ""
    d = Path(p).expanduser() if p else DATA / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def songs_dir() -> Path:
    d = DATA / "songs"
    d.mkdir(parents=True, exist_ok=True)
    return d
