"""Pluggable LLM providers for the songwriter.

Every provider takes (system, user) text and returns the model's reply text.
Supported kinds:

  openai      Any OpenAI-compatible Chat Completions server: OpenAI, OpenRouter,
              Groq, DeepSeek, xAI, Together, plus local servers (Ollama, LM Studio,
              KoboldCpp, llama.cpp, vLLM, text-generation-webui, Jan).
  anthropic   Claude through the official Anthropic SDK (needs an API key).
  claude-cli  Claude Code CLI (`claude -p`), uses your Claude subscription login.
  codex-cli   OpenAI Codex CLI (`codex exec`), uses your ChatGPT login.

Nothing here adds a content filter. Hosted services apply their own policies;
local models (Ollama, LM Studio, KoboldCpp...) answer whatever you ask.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    pass


# Presets shown in Settings. base_url is what the OpenAI-compatible client calls.
PRESETS = {
    "ollama": {"kind": "openai", "label": "Ollama (local)", "base_url": "http://127.0.0.1:11434/v1", "model": "llama3.1:8b", "needs_key": False},
    "lmstudio": {"kind": "openai", "label": "LM Studio (local)", "base_url": "http://127.0.0.1:1234/v1", "model": "", "needs_key": False},
    "koboldcpp": {"kind": "openai", "label": "KoboldCpp (local)", "base_url": "http://127.0.0.1:5001/v1", "model": "koboldcpp", "needs_key": False},
    "llamacpp": {"kind": "openai", "label": "llama.cpp server (local)", "base_url": "http://127.0.0.1:8080/v1", "model": "local", "needs_key": False},
    "textgen": {"kind": "openai", "label": "text-generation-webui (local)", "base_url": "http://127.0.0.1:5000/v1", "model": "", "needs_key": False},
    "openai": {"kind": "openai", "label": "OpenAI (GPT)", "base_url": "https://api.openai.com/v1", "model": "gpt-5-mini", "needs_key": True},
    "openrouter": {"kind": "openai", "label": "OpenRouter (hundreds of models)", "base_url": "https://openrouter.ai/api/v1", "model": "", "needs_key": True},
    "groq": {"kind": "openai", "label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "", "needs_key": True},
    "deepseek": {"kind": "openai", "label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "needs_key": True},
    "anthropic": {"kind": "anthropic", "label": "Claude API (Anthropic key)", "base_url": "", "model": "claude-opus-5-5", "needs_key": True},
    "claude-cli": {"kind": "claude-cli", "label": "Claude Code CLI (your Claude login)", "base_url": "", "model": "", "needs_key": False},
    "codex-cli": {"kind": "codex-cli", "label": "Codex CLI (your ChatGPT login)", "base_url": "", "model": "", "needs_key": False},
}

# Claude models that accept the server-side refusal fallback.
_CLAUDE_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMConfig:
    preset: str = "ollama"
    kind: str = "openai"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "llama3.1:8b"
    api_key: str = ""
    temperature: float = 0.9
    max_tokens: int = 4096
    timeout: float = 300.0
    json_mode: bool = True
    effort: str = "medium"  # Claude only
    extra_headers: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models (Qwen3, DeepSeek R1) emit."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag was eaten by the server's template
        text = text.split("</think>", 1)[1]
    return text.strip()


def _key_for(cfg: LLMConfig) -> str:
    if cfg.api_key:
        return cfg.api_key
    env = {
        "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    }.get(cfg.preset)
    return os.getenv(env, "") if env else ""


# ── OpenAI-compatible ─────────────────────────────────────────────────

def _openai_chat(cfg: LLMConfig, system: str, user: str) -> str:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", **cfg.extra_headers}
    key = _key_for(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": cfg.model or "default",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.json_mode:
        body["response_format"] = {"type": "json_object"}

    # Servers disagree on optional fields. On a 400 that names one, drop or
    # rename it and try again rather than failing the whole request.
    for _ in range(4):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=cfg.timeout)
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach {cfg.base_url}: {e}. Is the server running?") from e
        if r.status_code == 400:
            msg = r.text.lower()
            if "response_format" in msg and "response_format" in body:
                body.pop("response_format")
                continue
            if "max_tokens" in msg and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
                continue
            if "temperature" in msg and "temperature" in body:
                body.pop("temperature")
                continue
        if r.status_code >= 400:
            raise LLMError(f"{cfg.base_url} returned HTTP {r.status_code}: {r.text[:400]}")
        break
    data = r.json()
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected response from {cfg.base_url}: {str(data)[:300]}") from e
    text = msg.get("content") or ""
    if not text.strip() and msg.get("refusal"):
        raise LLMError(f"The model refused: {msg['refusal']}. Try a local model for unrestricted writing.")
    return strip_thinking(text)


def list_openai_models(cfg: LLMConfig) -> list[str]:
    headers = {}
    key = _key_for(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = httpx.get(cfg.base_url.rstrip("/") + "/models", headers=headers, timeout=8)
        r.raise_for_status()
        return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))
    except Exception as e:
        raise LLMError(f"Could not list models at {cfg.base_url}: {e}") from e


# ── Anthropic (Claude API) ────────────────────────────────────────────

def _anthropic_chat(cfg: LLMConfig, system: str, user: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("The 'anthropic' package is missing. Run install again.") from e
    key = _key_for(cfg)
    if not key:
        raise LLMError("No Anthropic API key. Add one in Settings or set ANTHROPIC_API_KEY.")
    client = anthropic.Anthropic(api_key=key, timeout=cfg.timeout)
    model = cfg.model or "claude-opus-5-5"
    kwargs = dict(
        model=model,
        max_tokens=max(cfg.max_tokens, 4096),
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if cfg.effort:
        kwargs["output_config"] = {"effort": cfg.effort}
    try:
        if model.startswith(_CLAUDE_FALLBACK_MODELS):
            # If a safety classifier declines, the API retries on its recommended fallback model.
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = client.messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic rejected the API key.") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Anthropic rate limit hit. Wait a moment and retry.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Could not reach the Anthropic API: {e}") from e
    if resp.stop_reason == "refusal":
        raise LLMError("Claude declined this request. Rephrase it, or switch to a local model in Settings.")
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


# ── CLI providers ─────────────────────────────────────────────────────

def _which(name: str) -> str | None:
    # Prefer a real executable. On Windows npm installs a .cmd shim (cmd.exe mangles some
    # arguments, so prompts go through stdin/files) and a bash shim that won't run here.
    for cand in (f"{name}.exe", f"{name}.cmd", name):
        p = shutil.which(cand)
        if p:
            return p
    return None


def _run_cli(cmd: list[str], stdin: str, timeout: float, cwd: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"{os.path.basename(cmd[0])} timed out after {timeout:.0f}s") from e


def _claude_cli_chat(cfg: LLMConfig, system: str, user: str) -> str:
    exe = _which("claude")
    if not exe:
        raise LLMError("Claude Code CLI not found. Install it from https://claude.com/claude-code and log in once.")
    with tempfile.TemporaryDirectory() as tmp:
        # Multi-line system prompt goes through a file: cmd.exe shims cut arguments at newlines.
        sp = os.path.join(tmp, "system.txt")
        with open(sp, "w", encoding="utf-8") as f:
            f.write(system)
        cmd = [exe, "-p", "--output-format", "text", "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--system-prompt-file", sp]
        if cfg.model:
            cmd += ["--model", cfg.model]
        p = _run_cli(cmd, user, cfg.timeout, tmp)
    if p.returncode != 0:
        raise LLMError(f"claude CLI failed: {(p.stderr or p.stdout)[-400:]}")
    return p.stdout.strip()


def _codex_cli_chat(cfg: LLMConfig, system: str, user: str) -> str:
    exe = _which("codex")
    if not exe:
        raise LLMError("Codex CLI not found. Install it with `npm i -g @openai/codex` and run `codex login`.")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "last.txt")
        cmd = [exe, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "--color", "never", "-o", out, "-"]
        if cfg.model:
            cmd[2:2] = ["-m", cfg.model]
        prompt = f"{system}\n\n---\n\n{user}\n\nReply with the answer only. Do not run any commands."
        p = _run_cli(cmd, prompt, cfg.timeout, tmp)
        if p.returncode != 0:
            raise LLMError(f"codex CLI failed: {(p.stderr or p.stdout)[-400:]}")
        try:
            with open(out, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return p.stdout.strip()


# ── entry point ───────────────────────────────────────────────────────

def chat(cfg: LLMConfig, system: str, user: str) -> str:
    kind = cfg.kind
    if kind == "openai":
        text = _openai_chat(cfg, system, user)
    elif kind == "anthropic":
        text = _anthropic_chat(cfg, system, user)
    elif kind == "claude-cli":
        text = _claude_cli_chat(cfg, system, user)
    elif kind == "codex-cli":
        text = _codex_cli_chat(cfg, system, user)
    else:
        raise LLMError(f"Unknown provider kind: {kind}")
    if not text:
        raise LLMError("The model returned an empty reply.")
    return text


def cli_status() -> dict:
    return {"claude": bool(_which("claude")), "codex": bool(_which("codex"))}
