# AGENTS.md: setting up ACE-Step Workbench

This file is for an AI coding agent (Codex, Claude Code, Cursor, etc.) that has been asked to **install and set up ACE-Step Workbench** on the user's machine. Humans can read it too; the plain-English guide is `README.md` and the in-app Guide tab.

> Person using this: open this folder in your agent and say
> **"Read AGENTS.md and set up ACE-Step Workbench for me."**

---

## What this project is

A local web app (FastAPI backend + plain HTML/JS, no build step) that drives **ACE-Step 1.5**, an open-source AI music generator, through ACE-Step's own REST API.

```
ace-step-workbench/
  install.bat / install.sh   bootstrap: uv + app venv, then install.py
  install.py                 gets ACE-Step 1.5 (pinned commit), runs `uv sync` in it, downloads models
  start.bat / start.sh       starts the app on http://127.0.0.1:3470 (it starts ACE-Step on :8001 itself)
  workbench/                 backend (Python)
  web/                       frontend + in-app guide (web/guide.html)
  tests/                     pytest suite; uses a mock ACE-Step, no GPU needed
  ACE-Step-1.5/              created by the installer (git-ignored)
  data/                      user state: settings.json (API keys!), songs, datasets, LoRAs (git-ignored)
```

Two separate Python environments, on purpose:
- `./.venv`: the Workbench (small: fastapi, uvicorn, httpx, anthropic).
- `<ACE-Step folder>/.venv`: ACE-Step itself (PyTorch + CUDA, several GB). The Workbench never imports it; it runs it as a subprocess.

---

## Ground rules for the agent

1. **Tell the user before big downloads.** A full install downloads roughly 15 GB (ACE-Step environment about 5 GB + models about 10 GB) and takes 15-45 minutes. Confirm before starting.
2. **Ask before using the GPU** for a test song or training. Installing is fine; generating is the user's call.
3. **Never commit or upload `data/`.** It holds API keys and the user's music.
4. **Don't edit files inside the ACE-Step folder.** Configure it through the Workbench's Settings or `data/settings.json`. (The Workbench itself applies one small known fix there, see Facts.)
5. **Don't delete anything** the user didn't ask you to. The app itself moves deleted songs to `data/trash/`.
6. **Long commands:** `uv sync` and the model download print progress for many minutes. Use long timeouts (30+ minutes) and don't kill them for being quiet.
7. **No interactive prompts:** set `WB_NO_PAUSE=1` when running the `.bat` scripts so they never wait for a key press.
8. **Network access:** installing needs internet (github.com, astral.sh, pypi.org, download.pytorch.org, huggingface.co). If your sandbox blocks network, ask the user to allow it for this session.

---

## Step 1: check the machine (report back to the user)

| Check | Windows (PowerShell) | Linux / macOS | Need |
|---|---|---|---|
| GPU + VRAM | `nvidia-smi` | `nvidia-smi` | NVIDIA GPU, 8 GB+ VRAM (12 GB+ recommended, 16 GB+ for training) |
| Free disk | `Get-PSDrive -PSProvider FileSystem` | `df -h .` | about 20 GB on this drive |
| git | `git --version` | `git --version` | optional (installer falls back to a zip download) |
| uv | `uv --version` | `uv --version` | optional (installer installs it) |

- No NVIDIA GPU: tell the user. ACE-Step also has AMD ROCm, Intel XPU and Apple Silicon (MPS) support, but this Workbench is only tested on NVIDIA/CUDA. Proceed only if they want to try.
- Less than 8 GB VRAM: it can still work with the planner LM turned off (Settings > ACE-Step engine > untick "Load the LM").

## Step 2: ask one question

> "Do you already have ACE-Step 1.5 installed somewhere? If yes, which folder?"

- **No** → Step 3A (full install).
- **Yes** → Step 3B (reuse it; saves about 15 GB).

## Step 3A: full install

From the repo root:

```powershell
# Windows
$env:WB_NO_PAUSE = "1"; cmd /c install.bat
```
```bash
# Linux / macOS
./install.sh
```

What happens: installs `uv` if missing → creates `./.venv` → `install.py` clones ACE-Step 1.5 into `./ACE-Step-1.5` at the tested commit → `uv sync` there → downloads the main models (turbo DiT + 1.7B planner LM + VAE + text encoder) into `./ACE-Step-1.5/checkpoints`.

Useful flags (append to the install command): `--skip-models` (models download on first use instead), `--latest` (newest ACE-Step instead of the tested commit).

## Step 3B: reuse an existing ACE-Step 1.5

```powershell
$env:WB_NO_PAUSE = "1"; cmd /c install.bat --no-ace          # Windows
```
```bash
./install.sh --no-ace                                         # Linux / macOS
```

Then point the Workbench at it. Either:
- **UI (preferred):** start the app (Step 4), open **Settings → ACE-Step engine → ACE-Step folder**, paste the folder, press **check**, then **Save** and **Start**; or
- **API** (app running): `GET http://127.0.0.1:3470/api/ace/check?path=<folder>` to validate, then
  `POST http://127.0.0.1:3470/api/settings` with header `X-Workbench: 1` and body `{"ace": {"path": "<folder>"}}`.

`check` must report `ok: true`. If it says the environment is missing, run `uv sync` inside that ACE-Step folder (that is ACE-Step's own install step, not an edit).

## Step 4: start and verify

```powershell
$env:WB_NO_PAUSE = "1"; cmd /c start.bat --no-ace      # first check: app only, no GPU
```
```bash
./start.sh --no-ace
```

`start.*` runs in the foreground. Run it in the background, then verify:

- `GET http://127.0.0.1:3470/api/status` returns JSON with `"ace": {"installed": true, ...}`.
- Optional test suite (no GPU):
  ```
  uv pip install --python .venv/Scripts/python.exe pytest      # Linux/macOS: .venv/bin/python
  .venv/Scripts/python.exe -m pytest -q tests                   # expect all passed
  ```

Stop that instance, then start normally (without `--no-ace`). The app starts ACE-Step automatically. First start loads models into VRAM (1-5 minutes); `/api/status` shows `"ready": true` when done, and the green dot in the sidebar turns on.

Options for `start.*`: `--port 3500`, `--no-ace`, `--host 0.0.0.0` (LAN access; prints a token link, anyone with it can use the app).

## Step 5: AI writer (optional, ask the user)

The Write tab needs an LLM for lyrics and prompts (it falls back to rule-based drafts without one). Ask which they want, then set it in **Settings → AI writer** and press **Test**:

| They have | Provider to pick | Needs |
|---|---|---|
| Nothing, wants free + private | Ollama | install Ollama, `ollama pull llama3.1:8b` (or any model) |
| A ChatGPT plan + Codex CLI | Codex CLI | `codex login` done once |
| A Claude plan + Claude Code | Claude Code CLI | `claude` logged in once |
| An OpenAI / Anthropic / OpenRouter key | that provider | paste the key in Settings (stored only in `data/settings.json`) |
| Nothing at all | ACE-Step built-in LM | nothing (ACE-Step mode only) |

## Step 6: hand over

Tell the user:
- Open http://127.0.0.1:3470 (or run `start.bat` / `./start.sh` next time; it opens the browser).
- Read the **Guide** tab first; every control has a tooltip.
- Suggested first run: Write → idea → Write song → Send to Create → Generate (ask before you run this yourself; it uses the GPU).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv sync` fails on torch | No matching PyTorch/CUDA build | ACE-Step's `pyproject.toml` chooses the PyTorch CUDA build. Update the NVIDIA driver (an old driver is the usual cause) and run the install again. |
| Python version error | ACE-Step needs Python 3.11-3.12 | `uv` downloads the right Python itself; don't use a system Python. |
| Model download very slow / blocked | Hugging Face unreachable | Retry. To use the ModelScope mirror instead, add `"ace": {"extra_env": {"ACESTEP_DOWNLOAD_SOURCE": "modelscope"}}` to `data/settings.json` (or `POST /api/settings` with that body) and restart ACE-Step. |
| Status stays "loading" | First model load, or out of VRAM | Watch Settings > ACE-Step engine > Server log. Close other GPU apps; pick the 0.6B LM or untick "Load the LM". |
| `CUDA out of memory` | VRAM full | Same as above; also "Offload LM to CPU". |
| Port 3470 or 8001 in use | Another app | `start.bat --port 3500`; change the ACE-Step API address in Settings. |
| POST requests return 403 "Missing X-Workbench header" | Calling the API without the header | Send `X-Workbench: 1` on every POST/PUT/DELETE (cross-site protection). |
| AI writer "could not reach" | Its server isn't running / wrong URL | Start Ollama / LM Studio; press Test in Settings. |
| Windows path errors during install | Very long paths | Install closer to the drive root, e.g. `C:\ace-step-workbench`. |

## Facts an agent might need

- Workbench URL: `http://127.0.0.1:3470` (default). ACE-Step API: `http://127.0.0.1:8001`.
- Settings file: `data/settings.json` (merged over defaults in `workbench/config.py`; edit through the UI or `POST /api/settings`).
- ACE-Step pinned commit: see `PINNED` in `install.py`. Upstream: https://github.com/ACE-Step/ACE-Step-1.5
- Known ACE-Step bug (still in upstream main as of the pinned commit): `acestep/training_v2/cli/args.py` `VARIANT_DIR_MAP` lacks the XL variants, so `train.py fixed --model-variant xl_turbo` fails with "Model directory not found ... xl_turbo". `workbench/ace_patches.py` adds `xl_turbo` / `xl_base` / `xl_sft` (idempotent, a no-op once upstream fixes it). The installer runs it, and so does every XL training run, which covers installs reused with `--ace-path` / `--no-ace`.
- Health/status: `GET /api/status`. ACE-Step server log: `GET /api/ace/log`. Start/stop ACE-Step: `POST /api/ace/start` / `POST /api/ace/stop` (with `X-Workbench: 1`).
- On Windows the app forces ACE-Step's LM backend to `pt` (vllm needs Triton, not available there).
