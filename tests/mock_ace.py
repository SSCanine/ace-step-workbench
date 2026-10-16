"""A tiny fake ACE-Step API for tests: same envelope, task queue and status codes."""

import json
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()
TASKS: dict = {}
CALLS: list = []
LORA = {"loaded": None, "on": False, "scale": 1.0, "loads": 0}


def wrap(data, code=200, error=None):
    return {"data": data, "code": code, "error": error, "timestamp": int(time.time() * 1000), "extra": None}


@app.get("/health")
def health():
    return wrap({"status": "ok", "models_initialized": True, "llm_initialized": True,
                 "loaded_model": "acestep-v15-turbo", "loaded_lm_model": "acestep-5Hz-lm-1.7B"})


@app.post("/release_task")
async def release(req: Request):
    ctype = req.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        form = await req.form()
        params = {k: v for k, v in form.items() if isinstance(v, str)}
        params["_src_audio_bytes"] = len(await form["src_audio"].read()) if "src_audio" in form else 0
    else:
        params = await req.json()
    CALLS.append(params)
    tid = uuid.uuid4().hex
    TASKS[tid] = {"t0": time.time(), "params": params}
    return wrap({"task_id": tid, "status": "queued", "queue_position": 1})


@app.post("/query_result")
async def query(req: Request):
    body = await req.json()
    out = []
    for tid in body["task_id_list"]:
        t = TASKS[tid]
        if time.time() - t["t0"] < 1.5:
            out.append({"task_id": tid, "status": 0, "progress_text": "denoising", "result": json.dumps({"progress": 0.5, "stage": "Denoising"})})
            continue
        p = t["params"]
        seeds = str(p.get("seed", "")).split(",")
        n = int(p.get("batch_size") or 1)
        res = [{"file": f"/v1/audio?path=x{i}.mp3", "status": 1, "seed_value": ",".join(seeds),
                "metas": {"bpm": p.get("bpm") or 120, "duration": p.get("audio_duration") or 30,
                          "keyscale": p.get("key_scale") or "N/A", "timesignature": "4"},
                "dit_model": "acestep-v15-turbo", "lm_model": "acestep-5Hz-lm-1.7B"} for i in range(n)]
        out.append({"task_id": tid, "status": 1, "progress_text": "", "result": json.dumps(res)})
    return wrap(out)


@app.get("/v1/audio")
def audio(path: str):
    return Response(b"ID3fake-mp3-" + path.encode(), media_type="audio/mpeg")


@app.get("/v1/lora/status")
def lora_status():
    return wrap({"lora_loaded": LORA["loaded"] is not None, "use_lora": LORA["on"], "lora_scale": LORA["scale"]})


@app.post("/v1/lora/load")
async def lora_load(req: Request):
    LORA["loaded"] = (await req.json())["lora_path"]
    LORA["loads"] += 1
    return wrap({"message": "ok"})


@app.post("/v1/lora/toggle")
async def lora_toggle(req: Request):
    LORA["on"] = (await req.json())["use_lora"]
    return wrap({"message": "ok"})


@app.post("/v1/lora/scale")
async def lora_scale(req: Request):
    LORA["scale"] = (await req.json())["scale"]
    return wrap({"message": "ok"})


def serve(port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server
