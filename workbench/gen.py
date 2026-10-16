"""Generation jobs: submit to ACE-Step, poll, save every result to the library.

Seeds are always chosen here (never left to ACE-Step) so each saved song knows
its exact seed. That is what makes "regenerate with the same seed" and
"re-sing a section" reproducible.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from pathlib import Path

from . import ace, library

MAX_SEED = 2**31 - 1

# Request fields the Create form may send, with ACE-Step's names.
FORM_FIELDS = {
    "caption": "prompt", "lyrics": "lyrics", "vocal_language": "vocal_language", "bpm": "bpm",
    "keyscale": "key_scale", "timesignature": "time_signature", "duration": "audio_duration",
    "inference_steps": "inference_steps", "guidance_scale": "guidance_scale", "shift": "shift",
    "infer_method": "infer_method", "audio_format": "audio_format", "thinking": "thinking",
    "use_cot_caption": "use_cot_caption", "use_cot_language": "use_cot_language",
    "lm_temperature": "lm_temperature", "lm_cfg_scale": "lm_cfg_scale", "lm_top_p": "lm_top_p",
    "lm_negative_prompt": "lm_negative_prompt",
    "repaint_strength": "repaint_strength",
}


def build_params(form: dict, settings: dict) -> dict:
    p: dict = {}
    for k, ace_k in FORM_FIELDS.items():
        v = form.get(k)
        if v is None or v == "" or v == "auto":
            continue
        p[ace_k] = v
    if form.get("instrumental"):
        p["lyrics"] = "[Instrumental]"
    p["task_type"] = "text2music"
    p["lm_backend"] = settings["ace"].get("lm_backend") or ("pt" if ace.IS_WIN else "vllm")
    p.setdefault("audio_format", "mp3")
    # Our seeds, always explicit.
    n = max(1, min(8, int(form.get("batch_size") or 1)))
    seed = form.get("seed")
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = -1
    seeds = [seed + i if seed >= 0 else random.randint(0, MAX_SEED) for i in range(n)]
    p["use_random_seed"] = False
    p["seed"] = ",".join(str(s) for s in seeds)
    p["batch_size"] = n
    return p


class Jobs:
    def __init__(self, server: ace.AceServer) -> None:
        self.server = server
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.gpu = threading.Lock()  # one generation at a time; training takes it too
        self.loaded_lora: str | None = None  # path ACE-Step currently has loaded
        self.lora_on = False

    def get(self, job_id: str) -> dict:
        with self.lock:
            j = self.jobs.get(job_id)
            if not j:
                raise KeyError(job_id)
            return {k: v for k, v in j.items() if not k.startswith("_")}

    def _set(self, job_id: str, **kw) -> None:
        with self.lock:
            self.jobs[job_id].update(kw)

    def start(self, settings: dict, form: dict, kind: str = "new", parent: dict | None = None,
              src_audio: Path | None = None) -> str:
        job_id = uuid.uuid4().hex[:10]
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "status": "queued", "stage": "Waiting", "progress": 0.0,
                                 "songs": [], "error": "", "kind": kind, "created": time.time()}
        threading.Thread(target=self._run, args=(job_id, settings, form, kind, parent, src_audio), daemon=True).start()
        return job_id

    def reset_lora(self) -> None:
        """Forget what we think is loaded (after ACE-Step restarted)."""
        self.loaded_lora, self.lora_on = None, False

    # ── LoRA handling ──
    def _apply_lora(self, client: ace.AceClient, settings: dict, form: dict, job_id: str) -> None:
        path = (form.get("lora") or "").strip()
        scale = float(form.get("lora_scale") or 1.0)
        # Trust the server over our memory: it may have restarted since the last job.
        try:
            st = client.lora_status()
            if not st.get("lora_loaded"):
                self.reset_lora()
            else:
                self.lora_on = bool(st.get("use_lora"))
        except ace.AceError:
            pass
        if not path:
            if self.lora_on:
                client.lora_toggle(False)
                self.lora_on = False
            return
        if self.loaded_lora and self.loaded_lora != path:
            # Upstream unload leaves adapter layers behind (ACE-Step issue #705),
            # so switching adapters means a clean server restart.
            if not self.server.managed:
                raise ace.AceError("Switching LoRAs needs a fresh ACE-Step server. Restart it in Settings, then generate again.")
            self._set(job_id, stage="Restarting ACE-Step to switch LoRA (clean slate)")
            self.server.restart(settings)
            self.loaded_lora, self.lora_on = None, False
            wait_ready(client, timeout=900, on_wait=lambda s: self._set(job_id, stage=s))
        if self.loaded_lora != path:
            self._set(job_id, stage="Loading LoRA")
            client.lora_load(path)
            self.loaded_lora = path
        if not self.lora_on:
            client.lora_toggle(True)
            self.lora_on = True
        client.lora_scale(scale)

    def _run(self, job_id, settings, form, kind, parent, src_audio) -> None:
        client = ace.AceClient(settings)
        try:
            with self.gpu:
                if not client.health().get("models_initialized"):
                    if not self.server.managed:
                        # An ACE-Step server started elsewhere may be lazy-loading: ask it to load now.
                        self._set(job_id, stage="Asking ACE-Step to load its models")
                        client.post("/v1/init", {"model": settings["ace"]["dit_model"],
                                                 "init_llm": bool(settings["ace"]["init_lm"]),
                                                 "lm_model_path": settings["ace"]["lm_model"]}, timeout=900)
                    self._set(job_id, stage="Waiting for ACE-Step to load models")
                    wait_ready(client, timeout=900, on_wait=lambda s: self._set(job_id, stage=s))
                self._apply_lora(client, settings, form, job_id)
                params = build_params(form, settings)
                files = None
                if kind == "repaint":
                    params["task_type"] = "repaint"
                    params["repainting_start"] = float(form.get("repaint_start") or 0)
                    params["repainting_end"] = float(form.get("repaint_end") or -1)
                    params["thinking"] = False
                    files = {"src_audio": (src_audio.name, src_audio.read_bytes(), "application/octet-stream")}
                self._set(job_id, status="running", stage="Submitting")
                task_id = client.submit(params, files)
                self._set(job_id, stage="Generating", ace_task=task_id)
                items = self._poll(client, job_id, task_id)
                self._set(job_id, stage="Saving")
                self._save(job_id, client, settings, form, params, items, kind, parent)
            self._set(job_id, status="done", stage="Done", progress=1.0)
        except Exception as e:  # report every failure to the UI
            self._set(job_id, status="error", stage="Failed", error=str(e))

    def _poll(self, client: ace.AceClient, job_id: str, task_id: str) -> list:
        t0 = time.time()
        while True:
            time.sleep(1.2)
            item = client.query(task_id)
            status = item.get("status", 0)
            raw = item.get("result")
            try:
                result = json.loads(raw) if isinstance(raw, str) and raw else raw
            except Exception:
                result = None
            if status == 1:
                if not isinstance(result, list) or not result:
                    raise ace.AceError("ACE-Step finished but returned no audio.")
                return result
            if status == 2:
                err = (result or {}).get("error") if isinstance(result, dict) else None
                raise ace.AceError(err or item.get("progress_text") or "Generation failed (see ACE-Step log in Settings).")
            prog = result.get("progress") if isinstance(result, dict) else None
            stage = (result.get("stage") if isinstance(result, dict) else None) or item.get("progress_text") or "Generating"
            self._set(job_id, stage=str(stage)[:120], progress=float(prog) if isinstance(prog, (int, float)) else 0.0,
                      elapsed=round(time.time() - t0, 1))
            if time.time() - t0 > 1800:
                raise ace.AceError("Timed out after 30 minutes.")

    def _save(self, job_id, client, settings, form, params, items, kind, parent) -> None:
        seeds = [int(s) for s in str(params.get("seed", "")).split(",") if s.strip().lstrip("-").isdigit()]
        saved = []
        for i, it in enumerate(items):
            f = it.get("file")
            if not f:
                continue
            audio = client.download(f)
            ext = "." + (params.get("audio_format") or "mp3").replace("wav32", "wav")
            metas = it.get("metas") or {}
            reported = [s for s in str(it.get("seed_value", "")).split(",") if s.strip()]
            seed = seeds[i] if i < len(seeds) else (int(reported[i]) if i < len(reported) and reported[i].isdigit() else None)

            def meta_or(key, fallback):
                v = metas.get(key)
                return fallback if v in (None, "", "N/A") else v

            song_form = {**form, "seed": seed, "batch_size": 1}
            meta = {
                "title": form.get("title") or (parent or {}).get("title") or "Untitled",
                "kind": kind,
                "parent": (parent or {}).get("id"),
                "caption": params.get("prompt", ""),
                "lyrics": params.get("lyrics", ""),
                "seed": seed,
                "bpm": meta_or("bpm", params.get("bpm")),
                "keyscale": meta_or("keyscale", params.get("key_scale")),
                "timesignature": meta_or("timesignature", params.get("time_signature")),
                "duration": meta_or("duration", params.get("audio_duration")),
                "lora": form.get("lora") or "",
                "lora_scale": form.get("lora_scale"),
                "dit_model": it.get("dit_model") or settings["ace"]["dit_model"],
                "lm_model": it.get("lm_model") or "",
                "form": song_form,  # exactly what to send to make this song again
                "ace_params": params,
                "repaint": {"start": form.get("repaint_start"), "end": form.get("repaint_end")} if kind == "repaint" else None,
            }
            saved.append(library.save(meta, audio_bytes=audio, ext=ext))
        with self.lock:
            self.jobs[job_id]["songs"] = [{"id": m["id"], "seed": m["seed"], "title": m["title"]} for m in saved]


def wait_ready(client: ace.AceClient, timeout: float = 900, on_wait=None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = client.health()
        if h.get("models_initialized"):
            return
        if on_wait:
            on_wait(f"Waiting for ACE-Step to load models ({int(time.time() - t0)}s)")
        time.sleep(2)
    raise ace.AceError("ACE-Step did not finish loading. Check the server log in Settings.")
