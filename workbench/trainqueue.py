"""Training queue: LoRAs for one or more models, made one after another when the GPU is free.

Jobs are kept in data/train_queue.json, so the queue survives a restart (an interrupted job starts again).
Each job is one dataset for one model, and runs these steps:

  wait for an idle GPU  ->  copy new songs from a folder  ->  auto-label songs without a caption
  ->  train  ->  45-second test clip on the model the LoRA was made for

The GPU gate waits until ComfyUI's queue is empty, the GPU is not busy and no song is generating, for N
minutes in a row; then it asks ComfyUI to unload its models and waits for enough free VRAM. "Start now"
skips the gate for one job. ACE-Step is put back the way it was once the queue is empty.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

from . import ace, config, gen, training

ACTIVE = ("waiting", "importing", "labeling", "training", "testing")
FINISHED = ("done", "error", "cancelled")
# Original words for vocal test clips (never lyrics from the training songs).
TEST_LYRICS = "[Verse]\nStreetlights humming, I keep walking on\nEvery window dark, every echo gone\n\n[Chorus]\nHold on, hold on\nWe are almost home"
_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def gpu_stats() -> dict | None:
    """Free / total VRAM in GB and utilisation in % of the first NVIDIA GPU, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, **_NOWIN).stdout
        free, total, util = (float(x) for x in out.strip().splitlines()[0].split(","))
        return {"free_gb": round(free / 1024, 1), "total_gb": round(total / 1024, 1), "util": util}
    except Exception:
        return None


def comfy_busy(urls: list[str]) -> str:
    for u in urls:
        try:
            q = httpx.get(u.rstrip("/") + "/queue", timeout=httpx.Timeout(2, connect=0.3)).json()  # see ace.health
        except Exception:
            continue  # not running: not busy
        if q.get("queue_running") or q.get("queue_pending"):
            return f"ComfyUI at {u} is rendering"
    return ""


def comfy_free(urls: list[str]) -> None:
    for u in urls:
        try:
            httpx.post(u.rstrip("/") + "/free", json={"unload_models": True, "free_memory": True},
                       timeout=httpx.Timeout(5, connect=0.3))
        except Exception:
            pass


def test_clip_form(ds: dict, variant: str, lora: Path, title: str) -> dict:
    """A short render that shows what a new LoRA does: the dataset's own trigger + a caption from it."""
    samples = ds.get("samples", [])
    caption = next((s["caption"] for s in samples if s.get("caption")), "")
    trigger = ds.get("metadata", {}).get("custom_tag", "")
    if trigger and trigger.lower() not in caption.lower():
        caption = f"{trigger}, {caption}"
    vocal = sum(1 for s in samples if not s.get("is_instrumental"))
    instrumental = bool(ds.get("metadata", {}).get("all_instrumental")) or vocal * 2 < len(samples)
    turbo = variant.endswith("turbo")
    return {"title": title, "caption": caption, "lyrics": "[Instrumental]" if instrumental else TEST_LYRICS,
            "instrumental": instrumental, "duration": 45, "seed": 42, "batch_size": 1, "lora": str(lora.resolve()),
            "lora_scale": 0.8, "inference_steps": 8 if turbo else 50, "shift": 3.0 if turbo else 1.0,
            "thinking": False, "audio_format": "mp3"}


class Cancelled(RuntimeError):
    pass


class TrainQueue:
    def __init__(self, server: ace.AceServer, jobs: gen.Jobs, trainer: training.Trainer) -> None:
        self.server, self.jobs, self.trainer = server, jobs, trainer
        self.lock = threading.RLock()
        self.items: list[dict] = []
        self.current: str | None = None
        self.wake = threading.Event()
        self._cancel = False
        self._restore: bool | None = None  # was ACE-Step running when this batch of jobs began?
        self._thread: threading.Thread | None = None

    # ── persistence ──
    @staticmethod
    def _file() -> Path:
        return config.DATA / "train_queue.json"

    def load(self) -> None:
        try:
            items = json.loads(self._file().read_text(encoding="utf-8")).get("jobs", [])
        except (OSError, ValueError):
            items = []
        for j in items:
            if j.get("status") in ACTIVE:
                j.update(status="queued", stage="Interrupted by a restart; it starts again")
        with self.lock:
            self.items = items

    def _save(self) -> None:
        with self.lock:
            config.DATA.mkdir(parents=True, exist_ok=True)
            tmp = self._file().with_suffix(".tmp")
            tmp.write_text(json.dumps({"jobs": self.items}, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._file())

    def _set(self, job: dict, **kw) -> None:
        with self.lock:
            if any(job.get(k) != v for k, v in kw.items()):
                job.update(kw)
                self._save()

    # ── public API ──
    def start_worker(self) -> None:
        self.load()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def list(self) -> list[dict]:
        with self.lock:
            out = copy.deepcopy(self.items)
        for j in out:
            if j["id"] == self.current and j["status"] == "training":
                st = self.trainer.state
                j.update({k: st.get(k) for k in ("epoch", "epochs", "loss", "eta_seconds")})
                j["stage"] = st.get("stage") or j.get("stage")
        return out

    def add(self, settings: dict, dataset: str, variants: list[str], preset: str = "balanced", opts: dict | None = None,
            source_folder: str = "", run_prefix: str = "", label_missing: bool = True, transcribe: bool = False,
            test_clip: bool = True) -> list[dict]:
        training.load_dataset(settings, dataset)  # must exist
        variants = [v for v in dict.fromkeys(training.norm_variant(v) for v in variants) if v]
        if not variants:
            raise ValueError("Pick at least one model to train for.")
        installed = {m["variant"] for m in training.model_variants(settings) if m["installed"]}
        missing = [training.VARIANT_LABEL[v] for v in variants if v not in installed]
        if missing:
            raise ValueError(f"Not downloaded yet: {', '.join(missing)}. Download it with ACE-Step first.")
        if source_folder and not Path(source_folder).expanduser().is_dir():
            raise ValueError(f"Folder not found: {source_folder}")
        if preset not in training.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        added = []
        with self.lock:
            taken = {j["run_name"] for j in self.items if j["status"] not in FINISHED}
            for v in variants:
                run = self._run_name(settings, run_prefix or dataset, v, taken)
                taken.add(run)
                job = {"id": uuid.uuid4().hex[:8], "dataset": dataset, "variant": v, "model_label": training.VARIANT_LABEL[v],
                       "preset": preset, "opts": dict(opts or {}), "run_name": run, "source_folder": source_folder,
                       "label_missing": label_missing, "transcribe": transcribe, "test_clip": test_clip,
                       "skip_gate": False, "status": "queued", "stage": "Queued", "error": "", "notes": [],
                       "created": time.time(), "started": None, "finished": None, "output": "", "test_song": None}
                self.items.append(job)
                added.append(job)
            self._save()
        self.wake.set()
        return added

    @staticmethod
    def _run_name(settings: dict, prefix: str, variant: str, taken: set) -> str:
        base = training._check_name(f"{prefix}-{variant.replace('_', '-')}"[:56])
        name, k = base, 2
        while name in taken or (config.outputs_dir(settings) / name).exists():
            name, k = f"{base}-v{k}", k + 1
        return name

    def _job(self, job_id: str) -> dict:
        for j in self.items:
            if j["id"] == job_id:
                return j
        raise KeyError(f"No queued job {job_id}")

    def action(self, job_id: str, what: str) -> None:
        with self.lock:
            job = self._job(job_id)
            if what == "cancel":
                if job["id"] == self.current:
                    self._cancel = True
                    self.trainer.stop()
                elif job["status"] == "queued":
                    job.update(status="cancelled", stage="Cancelled", finished=time.time())
            elif what == "now":
                job["skip_gate"] = True
            elif what in ("up", "down"):
                i = self.items.index(job)
                k = i - 1 if what == "up" else i + 1
                if 0 <= k < len(self.items) and self.items[k]["id"] != self.current:
                    self.items[i], self.items[k] = self.items[k], self.items[i]
            elif what == "retry":
                if job["status"] in ("error", "cancelled"):
                    job.update(status="queued", stage="Queued", error="", skip_gate=False, finished=None)
            elif what == "remove":
                if job["id"] == self.current:
                    raise ValueError("Cancel it first.")
                self.items.remove(job)
            else:
                raise ValueError(f"Unknown action: {what}")
            self._save()
        self.wake.set()

    def clear_finished(self) -> None:
        with self.lock:
            self.items = [j for j in self.items if j["status"] not in FINISHED]
            self._save()

    def summary(self) -> dict:
        with self.lock:
            cur = next((j for j in self.items if j["id"] == self.current), None)
            return {"active": cur is not None, "stage": (cur or {}).get("stage", ""), "status": (cur or {}).get("status", ""),
                    "waiting": sum(1 for j in self.items if j["status"] == "queued")}

    # ── worker ──
    def _next(self) -> dict | None:
        with self.lock:
            return next((j for j in self.items if j["status"] == "queued"), None)

    def _loop(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self.wake.wait(5)
                self.wake.clear()
                continue
            self._cancel = False
            self.current = job["id"]
            self._set(job, started=time.time(), error="")
            try:
                self._run(job)
                self._set(job, status="done", stage="Done", finished=time.time())
            except Exception as e:  # every failure ends up on the job's row
                cancelled = self._cancel or isinstance(e, Cancelled)
                self._set(job, status="cancelled" if cancelled else "error", stage="Cancelled" if cancelled else "Failed",
                          error="" if cancelled else str(e), finished=time.time())
            finally:
                self.current = None
                if self._next() is None:
                    self._restore_server()

    def _check(self) -> None:
        if self._cancel:
            raise Cancelled("Cancelled")

    def _sleep(self, seconds: float) -> None:
        self.wake.wait(seconds)
        self.wake.clear()
        self._check()

    def _run(self, job: dict) -> None:
        settings = config.load()
        if self._restore is None:
            self._restore = self.server.managed
        gate = settings["training"].get("gate") or {}
        self._wait_for_gpu(job, gate)
        if job.get("source_folder"):
            self._set(job, status="importing", stage=f"Copying new songs from {job['source_folder']}")
            n = training.import_folder(settings, job["dataset"], job["source_folder"])
            job["notes"].append(f"Copied {n} new song(s) from the folder.")
        self._check()
        ds = training.load_dataset(settings, job["dataset"])
        missing = sum(1 for s in ds["samples"] if not s.get("caption"))
        if missing and job.get("label_missing"):
            self._label(job, settings, missing)
        self._check()
        self._train(job, settings, gate)
        if job.get("test_clip"):
            self._test(job, settings)

    def _busy(self, gate: dict) -> str:
        if self.jobs.gpu.locked():
            return "a song is generating"
        reason = comfy_busy(gate.get("comfy_urls") or [])
        if reason:
            return reason
        st = gpu_stats()
        if st and st["util"] > float(gate.get("max_util", 40)):
            return f"the GPU is {st['util']:.0f}% busy"
        return ""

    def _wait_for_gpu(self, job: dict, gate: dict) -> None:
        if gate.get("enabled", True) and not job.get("skip_gate"):
            need = float(gate.get("idle_minutes", 10)) * 60
            idle_since = None
            while not job.get("skip_gate"):
                busy, now = self._busy(gate), time.time()
                if busy:
                    idle_since = None
                    self._set(job, status="waiting", stage=f"Waiting: {busy}")
                else:
                    idle_since = idle_since or now
                    if now - idle_since >= need:
                        break
                    self._set(job, status="waiting", stage=f"Waiting for the GPU to stay idle "
                                                            f"({int((now - idle_since) // 60)} of {int(need // 60)} min)")
                self._sleep(15)
        comfy_free(gate.get("comfy_urls") or [])

    def _vram_hook(self, job: dict, gate: dict):
        def wait_for_vram(variant: str) -> None:
            need = training.VRAM_NEED_GB.get(variant, 10)
            while not job.get("skip_gate"):
                st = gpu_stats()
                if st is None or st["free_gb"] >= need:
                    return
                comfy_free(gate.get("comfy_urls") or [])
                msg = (f"Waiting for {need} GB of free GPU memory ({st['free_gb']} GB free). "
                       "Close other GPU apps, or press Start now.")
                self.trainer.state["stage"] = msg
                self._set(job, stage=msg)
                self._sleep(15)
        return wait_for_vram

    def _ensure_ace(self, job: dict, settings: dict, model: str, why: str) -> ace.AceClient:
        """ACE-Step running with `model` (restarting the Workbench's own server if it runs another one)."""
        client = ace.AceClient(settings)
        if self.server.managed and self.server.model == model:
            pass
        elif self.server.managed:
            self._set(job, stage=f"Restarting ACE-Step with {training.VARIANT_LABEL.get(training.VARIANT_FOR_MODEL.get(model), model)} {why}")
            s = copy.deepcopy(settings)
            s["ace"]["dit_model"] = model
            self.server.restart(s)
            self.jobs.reset_lora()
        elif client.health():
            loaded = client.health().get("loaded_model")
            if loaded and loaded != model:
                raise RuntimeError(f"An ACE-Step started outside the Workbench runs {loaded}; {why} needs {model}.")
        else:
            self._set(job, stage=f"Starting ACE-Step {why}")
            s = copy.deepcopy(settings)
            s["ace"]["dit_model"] = model
            self.server.start(s)
            self.jobs.reset_lora()
        gen.wait_ready(client, timeout=900, on_wait=lambda m: self._set(job, stage=m))
        return client

    def _label(self, job: dict, settings: dict, missing: int) -> None:
        self._set(job, status="labeling", stage=f"Describing {missing} song(s) with ACE-Step")
        if not settings["ace"].get("init_lm"):
            raise RuntimeError(f"{missing} song(s) have no caption, and auto-label needs ACE-Step's LM "
                               "(Settings > Load the LM). Or describe them in step 2 and retry.")
        model = self.server.model if self.server.managed else settings["ace"]["dit_model"]
        client = self._ensure_ace(job, settings, model, "to describe the songs")
        if not client.health().get("llm_initialized"):
            raise RuntimeError("ACE-Step is running without its LM, which auto-label needs (Settings > Load the LM).")
        with self.jobs.gpu:
            training.auto_label(settings, job["dataset"], bool(job.get("transcribe")), False,
                                lambda p: self._set(job, stage=f"Describing songs {p}"))
        ds = training.load_dataset(settings, job["dataset"])
        still = sum(1 for s in ds["samples"] if not s.get("caption"))
        if still:
            raise RuntimeError(f"{still} song(s) still have no caption. Describe them in step 2, then press Retry.")

    def _train(self, job: dict, settings: dict, gate: dict) -> None:
        opts = {**training.PRESETS.get(job["preset"], training.PRESETS["balanced"]), **job.get("opts", {})}
        opts = {k: v for k, v in opts.items() if k not in ("label", "about")}
        opts.update(model_variant=job["variant"], run_name=job["run_name"], restart_after=False)
        self._set(job, status="training", stage="Training")
        self.trainer.before_train = self._vram_hook(job, gate)
        try:
            st = self.trainer.run(settings, job["dataset"], opts)
        finally:
            self.trainer.before_train = None
        self._check()
        if st.get("status") != "done":
            raise RuntimeError(st.get("error") or "Training failed. See the log.")
        self._set(job, output=st.get("output", ""))

    def _test(self, job: dict, settings: dict) -> None:
        out = Path(job["output"])
        lora = out / "final" if (out / "final").is_dir() else out
        if not ((lora / "adapter_config.json").is_file() or (lora / "lokr_weights.safetensors").is_file()):
            job["notes"].append("No final adapter found, so no test clip was made.")
            return
        self._set(job, status="testing", stage="Making a 45-second test clip")
        model = training.MODEL_FOR_VARIANT[job["variant"]]
        try:
            self._ensure_ace(job, settings, model, "for the test clip")
            ts = copy.deepcopy(settings)
            ts["ace"]["dit_model"] = model
            form = test_clip_form(training.load_dataset(settings, job["dataset"]), job["variant"], lora, f"{job['run_name']} test")
            gid = self.jobs.start(ts, form, kind="test")
            while True:
                g = self.jobs.get(gid)
                if g["status"] in ("done", "error"):
                    break
                self._sleep(2)
            if g["status"] == "error":
                job["notes"].append(f"Test clip failed: {g['error']}")
            elif g["songs"]:
                self._set(job, test_song=g["songs"][0]["id"])
        except Cancelled:
            raise
        except Exception as e:  # a failed test clip doesn't undo a finished LoRA
            job["notes"].append(f"Test clip skipped: {e}")
        self._save()

    def _restore_server(self) -> None:
        """Queue empty: leave ACE-Step as the user had it (running their model, or stopped)."""
        was_running, self._restore = self._restore, None
        if was_running is None:
            return
        settings = config.load()
        try:
            if was_running and (not self.server.managed or self.server.model != settings["ace"]["dit_model"]):
                self.server.restart(settings)
                self.jobs.reset_lora()
            elif not was_running and self.server.managed:
                self.server.stop()
                self.jobs.reset_lora()
        except Exception as e:
            self.server.last_error = f"Could not restore ACE-Step after training: {e}"
