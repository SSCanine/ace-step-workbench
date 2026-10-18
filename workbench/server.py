"""HTTP API + static UI for ACE-Step Workbench."""

from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import ace, config, gen, importer, library, llm, songmeta, suno, trainqueue, training, writer

WEB = config.ROOT / "web"
AUTOSTART_OVERRIDE: bool | None = None

server = ace.AceServer()
jobs = gen.Jobs(server)
trainer = training.Trainer(server, jobs.gpu, on_server_restart=jobs.reset_lora)
train_queue = trainqueue.TrainQueue(server, jobs, trainer)
autolabel_state: dict = {"status": "idle"}

DIT_MODELS = [
    {"id": "acestep-v15-turbo", "label": "Turbo (fast, 8 steps) - recommended", "vram": 8},
    {"id": "acestep-v15-xl-turbo", "label": "XL Turbo (best quality, needs ~16-24 GB)", "vram": 16},
    {"id": "acestep-v15-base", "label": "Base (slower, most controllable, all task types)", "vram": 8},
    {"id": "acestep-v15-sft", "label": "SFT (base fine-tune)", "vram": 8},
    {"id": "acestep-v15-xl-base", "label": "XL Base", "vram": 16},
]
LM_MODELS = [
    {"id": "acestep-5Hz-lm-0.6B", "label": "LM 0.6B (6-12 GB GPUs)", "vram": 6},
    {"id": "acestep-5Hz-lm-1.7B", "label": "LM 1.7B (12-16 GB) - recommended", "vram": 12},
    {"id": "acestep-5Hz-lm-4B", "label": "LM 4B (20 GB+)", "vram": 20},
]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    s = config.load()
    auto = s["ace"]["autostart"] if AUTOSTART_OVERRIDE is None else AUTOSTART_OVERRIDE
    if auto and ace.is_installed(config.ace_dir(s)) and not ace.AceClient(s).health():
        try:
            server.start(s)
        except Exception as e:
            server.last_error = str(e)
    train_queue.start_worker()
    yield
    trainer.stop()
    server.stop()


app = FastAPI(title="ACE-Step Workbench", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")

# ── request guard ─────────────────────────────────────────────────────
# The Workbench has no accounts. These checks stop other web pages (and DNS
# rebinding) from driving it through your browser, and require a token when it
# is opened up to the network with --host.
SECURITY = {"hosts": {"127.0.0.1", "localhost", "::1"}, "token": None}
TOKEN_COOKIE = "wb_token"


@app.middleware("http")
async def guard(request: Request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    if SECURITY["token"] is None and host not in SECURITY["hosts"]:
        return JSONResponse({"detail": f"Host '{host}' not allowed."}, status_code=403)
    if SECURITY["token"] is not None:
        given = request.query_params.get("token")
        if request.url.path == "/" and given == SECURITY["token"]:
            from fastapi.responses import RedirectResponse
            resp = RedirectResponse("/")
            resp.set_cookie(TOKEN_COOKIE, given, httponly=True, samesite="strict")
            return resp
        if request.cookies.get(TOKEN_COOKIE) != SECURITY["token"]:
            return JSONResponse({"detail": "Open the link with ?token=... printed in the Workbench console."}, status_code=401)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        # A custom header cannot be sent cross-site without a CORS preflight, which we never allow.
        if request.headers.get("x-workbench") != "1":
            return JSONResponse({"detail": "Missing X-Workbench header."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0].strip("[]").lower() != host:
            return JSONResponse({"detail": "Cross-origin request blocked."}, status_code=403)
    return await call_next(request)


async def _errors(request: Request, exc: Exception):
    """Expected failures (bad input, AI writer or ACE-Step problems) become readable 4xx messages."""
    status = 404 if isinstance(exc, FileNotFoundError) else 400
    return JSONResponse({"detail": str(exc) or exc.__class__.__name__}, status_code=status)


for _exc in (ValueError, KeyError, RuntimeError, FileNotFoundError, llm.LLMError, ace.AceError):
    app.add_exception_handler(_exc, _errors)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


# ── status + settings ─────────────────────────────────────────────────

@app.get("/api/status")
def status():
    s = config.load()
    h = ace.AceClient(s).health(timeout=1)  # status must stay fast even while ACE-Step is busy generating
    return {
        "ace": {
            "installed": ace.is_installed(config.ace_dir(s)), "path": str(config.ace_dir(s)),
            "managed": server.managed, "reachable": bool(h), "ready": bool(h.get("models_initialized")),
            "llm": bool(h.get("llm_initialized")), "model": h.get("loaded_model"), "lm": h.get("loaded_lm_model"),
            "error": server.last_error, "lora": jobs.loaded_lora if jobs.lora_on else None,
        },
        "writer": {"preset": s["llm"]["preset"], "label": llm.PRESETS.get(s["llm"]["preset"], {}).get("label", s["llm"]["preset"]),
                   "model": s["llm"]["model"]},
        "training": {"status": trainer.state.get("status"), "epoch": trainer.state.get("epoch"), "epochs": trainer.state.get("epochs")},
        "queue": train_queue.summary(),
    }


@app.get("/api/settings")
def get_settings():
    s = config.load()
    presets = {**llm.PRESETS,
               "ace-lm": {"kind": "ace-lm", "label": "ACE-Step built-in LM (no setup, ACE-Step mode only)", "base_url": "", "model": "", "needs_key": False},
               "custom": {"kind": "openai", "label": "Custom OpenAI-compatible server", "base_url": "http://127.0.0.1:8000/v1", "model": "", "needs_key": False},
               "none": {"kind": "none", "label": "None (rule-based drafts only)", "base_url": "", "model": "", "needs_key": False}}
    return {"settings": config.public(s), "presets": presets, "dit_models": DIT_MODELS, "lm_models": LM_MODELS,
            "cli": llm.cli_status(), "train_presets": training.PRESETS}


def _is_masked(v) -> bool:
    return isinstance(v, str) and v.startswith("••••")


@app.post("/api/settings")
async def post_settings(req: Request):
    patch = await req.json()
    cur = config.load()
    llm_patch = patch.get("llm") or {}
    preset = llm_patch.get("preset") or cur["llm"]["preset"]
    key = llm_patch.pop("api_key", None)
    if llm_patch.pop("clear_key", False):
        llm_patch["api_key"] = ""
        patch.setdefault("llm_keys", {})[preset] = ""
    elif key and not _is_masked(key):
        # A newly typed key: use it and remember it for this provider.
        llm_patch["api_key"] = key
        patch.setdefault("llm_keys", {})[preset] = key
    elif preset != cur["llm"]["preset"]:
        # Switching provider with an empty or masked box: bring back that provider's saved key.
        llm_patch["api_key"] = cur.get("llm_keys", {}).get(preset, "")
    if _is_masked((patch.get("ace") or {}).get("api_key")):
        patch["ace"].pop("api_key")
    if not patch.get("llm_keys"):
        patch.pop("llm_keys", None)
    return config.public(config.save(patch))


def _llm_cfg(s: dict) -> llm.LLMConfig | None:
    if s["llm"]["preset"] in ("none", "ace-lm"):
        return None
    return llm.LLMConfig.from_dict(s["llm"])


@app.post("/api/llm/test")
def llm_test():
    s = config.load()
    if s["llm"]["preset"] == "ace-lm":
        if not ace.AceClient(s).health().get("llm_initialized"):
            raise ValueError("ACE-Step's LM is not loaded (Settings > ACE-Step > Load the LM).")
        return {"ok": True, "reply": "ACE-Step LM is loaded."}
    cfg = _llm_cfg(s)
    if cfg is None:
        return {"ok": True, "reply": "No AI writer: rule-based drafts."}
    reply = llm.chat(cfg, "Reply with a JSON object only.", 'Return {"ok": true, "word": "<one random word>"}')
    return {"ok": True, "reply": reply[:300]}


@app.get("/api/llm/models")
def llm_models():
    s = config.load()
    cfg = llm.LLMConfig.from_dict(s["llm"])
    if cfg.kind != "openai":
        return {"models": []}
    return {"models": llm.list_openai_models(cfg)}


# ── ACE-Step server ───────────────────────────────────────────────────

@app.post("/api/ace/{action}")
def ace_action(action: str):
    s = config.load()
    if action == "start":
        server.start(s)
    elif action == "stop":
        server.stop()
        jobs.loaded_lora, jobs.lora_on = None, False
    elif action == "restart":
        server.restart(s)
        jobs.loaded_lora, jobs.lora_on = None, False
    else:
        raise ValueError("unknown action")
    return {"ok": True}


@app.get("/api/ace/check")
def ace_check(path: str = ""):
    """Validate a folder typed into Settings > ACE-Step folder (empty = the bundled location)."""
    s = config.load()
    d = config.ace_dir({"ace": {**s["ace"], "path": path}})
    if not d.is_dir():
        return {"ok": False, "path": str(d), "message": "Folder not found."}
    if not (d / "acestep").is_dir():
        return {"ok": False, "path": str(d), "message": "This is not an ACE-Step 1.5 folder (no 'acestep' inside)."}
    if not ace.venv_python(d).exists():
        return {"ok": False, "path": str(d), "message": "ACE-Step found, but its environment is not installed. Run 'uv sync' in that folder, or install.bat."}
    ck = d / "checkpoints"
    models = sorted(p.name for p in ck.iterdir() if p.is_dir() and not p.name.startswith(".")) if ck.is_dir() else []
    return {"ok": True, "path": str(d), "models": models,
            "message": f"ACE-Step 1.5 found. Models downloaded: {', '.join(models) or 'none yet (they download on first use)'}."}


@app.get("/api/ace/log")
def ace_log(n: int = 300):
    return {"lines": server.tail(n)}


# ── writer ────────────────────────────────────────────────────────────

def _ace_lm_write(s: dict, idea: str, target: str, opts: dict) -> dict:
    client = ace.AceClient(s)
    lang = (opts.get("language") or "").strip().lower()[:2] or "unknown"
    q = writer._brief(idea, opts).replace("\n", ". ")
    d = client.post("/v1/create_sample", {"query": q, "instrumental": opts.get("vocals") == "instrumental",
                                          "vocal_language": lang}, timeout=300)
    b = writer.normalize({**d, "title": d.get("title") or idea[:40]}, "ace")
    if target == "suno":
        b = writer.normalize(suno.to_suno_from_ace(b), "suno")
    return writer.finish(b, target, config.ROOT, opts, source="ace-step-lm")


@app.post("/api/write")
async def write(req: Request):
    body = await req.json()
    s = config.load()
    idea, target, opts = body.get("idea", ""), body.get("target", "ace"), body.get("opts") or {}
    if s["llm"]["preset"] == "ace-lm":
        return _ace_lm_write(s, idea, target, opts)
    return writer.write_song(_llm_cfg(s), idea, target, opts, config.ROOT)


def _need_chat(s: dict) -> llm.LLMConfig:
    cfg = _llm_cfg(s)
    if cfg is None:
        raise ValueError("This needs a chat AI writer. Pick one in Settings > AI writer (a local Ollama model works).")
    return cfg


@app.post("/api/write/idea")
def write_idea():
    return {"idea": writer.surprise_idea(_llm_cfg(config.load()))}


@app.post("/api/write/revise")
async def write_revise(req: Request):
    body = await req.json()
    return writer.revise(_need_chat(config.load()), body["bundle"], body["instruction"], config.ROOT, body.get("opts"))


@app.post("/api/write/convert")
async def write_convert(req: Request):
    body = await req.json()
    return writer.convert(_llm_cfg(config.load()), body["bundle"], body["to"], config.ROOT, body.get("opts"))


@app.post("/api/write/phrase")
async def write_phrase(req: Request):
    body = await req.json()
    return {"options": writer.phrase_options(_need_chat(config.load()), body["line"], body.get("wish", ""), body.get("context", ""))}


# ── generation ────────────────────────────────────────────────────────

def _check_ready(s: dict) -> None:
    if trainer.state.get("status") in ("preparing", "preprocessing", "training"):
        raise RuntimeError("Training is running and has the GPU. Wait for it to finish or stop it.")
    if not ace.AceClient(s).health() and not server.managed:
        raise RuntimeError("ACE-Step is not running. Start it in Settings > ACE-Step.")


@app.post("/api/generate")
async def generate(req: Request):
    form = await req.json()
    s = config.load()
    _check_ready(s)
    return {"job": jobs.start(s, form, "new")}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    return jobs.get(job_id)


@app.post("/api/songs/{song_id}/remix")
async def remix(song_id: str, req: Request):
    """mode=same_seed: reuse the song's seed with (edited) settings. mode=variation: new seeds."""
    body = await req.json()
    s = config.load()
    _check_ready(s)
    parent = library.get(song_id)
    form = {**parent.get("form", {}), **(body.get("form") or {})}
    mode = body.get("mode", "same_seed")
    if mode == "same_seed":
        form["seed"], form["batch_size"] = parent.get("seed"), 1
    else:
        form["seed"], form["batch_size"] = -1, int(body.get("count") or 2)
    form.setdefault("title", parent.get("title"))
    return {"job": jobs.start(s, form, "remix" if mode == "same_seed" else "variation", parent=parent)}


@app.post("/api/songs/{song_id}/repaint")
async def repaint(song_id: str, req: Request):
    body = await req.json()
    s = config.load()
    _check_ready(s)
    parent = library.get(song_id)
    form = {**parent.get("form", {}), **(body.get("form") or {})}
    start, end = float(form.get("repaint_start") or 0), float(form.get("repaint_end") or -1)
    if end != -1 and end - start < 0.5:
        raise ValueError("Select at least half a second to re-sing.")
    form["batch_size"] = int(body.get("count") or 1)
    if body.get("keep_seed", True):
        form["seed"] = parent.get("seed")
    form.setdefault("title", parent.get("title"))
    return {"job": jobs.start(s, form, "repaint", parent=parent, src_audio=library.audio_path(song_id))}


# ── library ───────────────────────────────────────────────────────────

@app.get("/api/songs")
def songs(q: str = ""):
    return {"songs": library.list_songs(q)}


@app.get("/api/songs/{song_id}")
def song(song_id: str):
    return library.get(song_id)


@app.patch("/api/songs/{song_id}")
async def song_patch(song_id: str, req: Request):
    return library.update(song_id, await req.json())


@app.delete("/api/songs/{song_id}")
def song_delete(song_id: str):
    library.trash(song_id)
    return {"ok": True}


@app.get("/api/songs/{song_id}/audio")
def song_audio(song_id: str, download: int = 0):
    p = library.audio_path(song_id)
    if not download:
        return FileResponse(p)
    meta = library.get(song_id)
    safe = "".join(c for c in (meta.get("title") or "song") if c.isalnum() or c in " -_").strip() or "song"
    name = f"{safe} ({meta.get('seed')}){p.suffix}"
    # The download carries its settings, so dropping it back in (Create or Studio) restores them.
    data = songmeta.embed(p.read_bytes(), importer.export_meta(config.load(), meta))
    return Response(data, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename*=utf-8''{quote(name)}"})


@app.post("/api/import")
async def import_song(files: list[UploadFile] = File(...)):
    """Dropped song file(s) -> the settings that made them (+ a Library entry so Studio can use it)."""
    items, total = [], 0
    for f in files[:8]:
        data = await f.read()
        total += len(data)
        if total > 400 * 2**20:
            raise ValueError("Too large to import (400 MB max).")
        items.append((Path(f.filename or "song").name, data))
    return importer.import_files(config.load(), items)


# ── autosave ──────────────────────────────────────────────────────────
# Everything typed in the app is saved here as you go, so a reload never loses work.

@app.get("/api/draft")
def draft_get():
    try:
        return {"draft": json.loads((config.DATA / "draft.json").read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"draft": None}


@app.put("/api/draft")
async def draft_put(req: Request):
    raw = await req.body()
    if len(raw) > 4 * 2**20:
        raise ValueError("Draft too large to save.")
    draft = json.loads(raw)
    if not isinstance(draft, dict):
        raise ValueError("Draft must be an object.")
    config.DATA.mkdir(parents=True, exist_ok=True)
    tmp = config.DATA / "draft.tmp"
    tmp.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, config.DATA / "draft.json")
    return {"ok": True}


@app.get("/api/loras")
def loras():
    return {"loras": training.find_adapters(config.load())}


# ── datasets + training ───────────────────────────────────────────────

@app.get("/api/datasets")
def datasets():
    return {"datasets": training.list_datasets(config.load())}


@app.post("/api/datasets")
async def dataset_create(req: Request):
    body = await req.json()
    return training.create_dataset(config.load(), body.get("name", ""), body.get("trigger", ""))


@app.get("/api/datasets/{name}")
def dataset_get(name: str):
    return training.load_dataset(config.load(), name)


@app.put("/api/datasets/{name}")
async def dataset_put(name: str, req: Request):
    body = await req.json()
    return training.update_samples(config.load(), name, body.get("metadata"), body.get("samples"))


@app.post("/api/datasets/{name}/files")
async def dataset_files(name: str, files: list[UploadFile] = File(...)):
    uploads = [(f.filename or "file", await f.read()) for f in files]
    return {"added": training.add_files(config.load(), name, uploads)}


@app.post("/api/datasets/{name}/import")
async def dataset_import(name: str, req: Request):
    body = await req.json()
    return {"added": training.import_folder(config.load(), name, body.get("folder", ""))}


@app.delete("/api/datasets/{name}/files/{filename}")
def dataset_file_delete(name: str, filename: str):
    training.remove_file(config.load(), name, filename)
    return {"ok": True}


@app.get("/api/datasets/{name}/audio/{filename}")
def dataset_audio(name: str, filename: str):
    p = training.ds_dir(config.load(), name) / "audio" / Path(filename).name
    if not p.is_file():
        raise FileNotFoundError(filename)
    return FileResponse(p)


@app.post("/api/datasets/{name}/autolabel")
async def dataset_autolabel(name: str, req: Request):
    body = await req.json()
    s = config.load()
    if autolabel_state.get("status") == "running":
        raise RuntimeError("Auto-label is already running.")
    h = ace.AceClient(s).health()
    if not h.get("models_initialized") or not h.get("llm_initialized"):
        raise RuntimeError("Auto-label needs ACE-Step running with its LM loaded (Settings > ACE-Step).")

    def run():
        autolabel_state.update(status="running", dataset=name, progress="Starting", error="", labeled=0)
        try:
            with jobs.gpu:
                n = training.auto_label(s, name, bool(body.get("transcribe")), bool(body.get("overwrite")),
                                        lambda p: autolabel_state.update(progress=p))
            autolabel_state.update(status="done", progress=f"Labeled {n} file(s)", labeled=n)
        except Exception as e:
            autolabel_state.update(status="error", error=str(e), progress="Failed")

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.get("/api/autolabel")
def autolabel_status():
    return autolabel_state


@app.post("/api/train/start")
async def train_start(req: Request):
    body = await req.json()
    trainer.start(config.load(), body["dataset"], body.get("opts") or {})
    return {"ok": True}


@app.post("/api/train/stop")
def train_stop():
    trainer.stop()
    return {"ok": True}


@app.get("/api/train/status")
def train_status():
    return trainer.status()


# ── training queue ────────────────────────────────────────────────────

@app.get("/api/train/queue")
def queue_list():
    s = config.load()
    return {"jobs": train_queue.list(), "models": training.model_variants(s), "gate": s["training"]["gate"],
            "gpu": trainqueue.gpu_stats(), "current_model": server.model if server.managed else s["ace"]["dit_model"]}


@app.post("/api/train/queue")
async def queue_add(req: Request):
    b = await req.json()
    added = train_queue.add(config.load(), b.get("dataset", ""), b.get("variants") or [], b.get("preset") or "balanced",
                            b.get("opts") or {}, source_folder=(b.get("source_folder") or "").strip(),
                            run_prefix=(b.get("run_prefix") or "").strip(), label_missing=b.get("label_missing", True),
                            transcribe=bool(b.get("transcribe")), test_clip=b.get("test_clip", True))
    return {"jobs": added}


@app.post("/api/train/queue/clear")
def queue_clear():
    train_queue.clear_finished()
    return {"ok": True}


@app.post("/api/train/queue/{job_id}/{what}")
def queue_action(job_id: str, what: str):
    train_queue.action(job_id, what)
    return {"ok": True}


@app.post("/api/dataset-import")
async def dataset_import_existing(req: Request):
    """A dataset labeled elsewhere (folder with dataset.json + songs) becomes a Workbench dataset."""
    b = await req.json()
    return training.import_dataset(config.load(), (b.get("folder") or "").strip(), (b.get("name") or "").strip())
