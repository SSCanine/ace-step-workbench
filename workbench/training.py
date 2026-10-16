"""LoRA / LoKr training on top of ACE-Step 1.5's built-in trainer (train.py fixed).

A dataset is a folder:  <datasets>/<name>/audio/*.wav|mp3|flac...  +  dataset.json
dataset.json uses ACE-Step's own format, so it also opens in ACE-Step's Gradio UI.

A training run:  stop the ACE-Step server (frees VRAM)  ->  preprocess to tensors
->  train  ->  restart the server. Progress lines are parsed into a live loss chart.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from pathlib import Path

from . import ace, config

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,60}$")

PRESETS = {
    "quick": {
        "label": "Quick (LoKr)", "adapter_type": "lokr", "rank": 64, "alpha": 128, "lr": 1e-3,
        "epochs": 150, "save_every": 25, "batch_size": 1, "grad_accum": 4,
        "about": "Fastest. Good for a first test of a dataset. Smaller file.",
    },
    "balanced": {
        "label": "Balanced (LoRA)", "adapter_type": "lora", "rank": 64, "alpha": 128, "lr": 1e-4,
        "epochs": 200, "save_every": 25, "batch_size": 1, "grad_accum": 4,
        "about": "Recommended. Solid style capture for 10-30 songs.",
    },
    "strong": {
        "label": "Strong (LoRA, more epochs)", "adapter_type": "lora", "rank": 128, "alpha": 256, "lr": 1e-4,
        "epochs": 350, "save_every": 25, "batch_size": 1, "grad_accum": 4,
        "about": "More capacity and time. Watch for overfitting (copying songs, garbled vocals).",
    },
}

VARIANT_FOR_MODEL = {
    "acestep-v15-turbo": "turbo", "acestep-v15-base": "base", "acestep-v15-sft": "sft",
    "acestep-v15-xl-turbo": "xl_turbo", "acestep-v15-xl-base": "xl_base", "acestep-v15-xl-sft": "xl_sft",
}

STEP_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+),\s*Step\s+(\d+),\s*Loss:\s*([\d.eE+-]+)")
EPOCH_RE = re.compile(r"\[OK\]\s*Epoch\s+(\d+)\s*/\s*(\d+)\s+in\s+([\d.]+)s,\s*Loss:\s*([\d.eE+-]+)")


# ── datasets ──────────────────────────────────────────────────────────

def _check_name(name: str) -> str:
    name = (name or "").strip()
    if not NAME_RE.match(name) or ".." in name:
        raise ValueError("Use letters, numbers, spaces, - _ . (max 60 chars).")
    return name


def ds_dir(settings: dict, name: str) -> Path:
    return config.datasets_dir(settings) / _check_name(name)


def list_datasets(settings: dict) -> list[dict]:
    out = []
    for d in sorted(config.datasets_dir(settings).iterdir()):
        j = d / "dataset.json"
        if j.is_file():
            try:
                ds = json.loads(j.read_text(encoding="utf-8"))
                s = ds.get("samples", [])
                out.append({"name": d.name, "trigger": ds["metadata"].get("custom_tag", ""), "count": len(s),
                            "labeled": sum(1 for x in s if x.get("caption"))})
            except Exception:
                continue
    return out


def create_dataset(settings: dict, name: str, trigger: str) -> dict:
    d = ds_dir(settings, name)
    if (d / "dataset.json").exists():
        raise ValueError("A dataset with that name already exists.")
    (d / "audio").mkdir(parents=True, exist_ok=True)
    ds = {"metadata": {"name": name, "custom_tag": trigger.strip(), "tag_position": "prepend", "genre_ratio": 0,
                       "all_instrumental": False, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "samples": []}
    save_dataset(settings, name, ds)
    return ds


def load_dataset(settings: dict, name: str) -> dict:
    d = ds_dir(settings, name)
    ds = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    sync_files(d, ds)
    return ds


def save_dataset(settings: dict, name: str, ds: dict) -> None:
    d = ds_dir(settings, name)
    meta = ds.setdefault("metadata", {})
    tag = meta.get("custom_tag", "")
    for i, s in enumerate(ds.get("samples", [])):
        s["custom_tag"] = tag
        s.setdefault("id", f"s{i:04d}")
        s["labeled"] = bool(s.get("caption"))
    meta["num_samples"] = len(ds.get("samples", []))
    tmp = d / "dataset.json.tmp"
    tmp.write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / "dataset.json")


def _blank_sample(path: Path) -> dict:
    return {"id": "", "audio_path": str(path.resolve()), "filename": path.name, "caption": "", "genre": "",
            "lyrics": "[Instrumental]", "raw_lyrics": "", "formatted_lyrics": "", "bpm": None, "keyscale": "",
            "timesignature": "", "duration": 0, "language": "unknown", "is_instrumental": True,
            "custom_tag": "", "labeled": False, "prompt_override": None}


def sync_files(d: Path, ds: dict) -> None:
    """Make samples match the files in audio/ (new files added, removed files dropped)."""
    files = {p.name: p for p in sorted((d / "audio").iterdir()) if p.suffix.lower() in AUDIO_EXT}
    samples = [s for s in ds.get("samples", []) if s.get("filename") in files]
    have = {s["filename"] for s in samples}
    for name, p in files.items():
        if name not in have:
            samples.append(_blank_sample(p))
    for s in samples:
        s["audio_path"] = str(files[s["filename"]].resolve())
    ds["samples"] = samples


def add_files(settings: dict, name: str, uploads: list[tuple[str, bytes]]) -> int:
    d = ds_dir(settings, name)
    n = 0
    for fname, data in uploads:
        safe = Path(fname).name
        if Path(safe).suffix.lower() not in AUDIO_EXT:
            continue
        (d / "audio" / safe).write_bytes(data)
        n += 1
    ds = load_dataset(settings, name)
    save_dataset(settings, name, ds)
    return n


def import_folder(settings: dict, name: str, folder: str) -> int:
    src = Path(folder).expanduser()
    if not src.is_dir():
        raise ValueError(f"Folder not found: {src}")
    d = ds_dir(settings, name)
    dest = (d / "audio").resolve()
    if src.resolve() == dest or dest in src.resolve().parents:
        raise ValueError("That is the dataset's own folder. Pick the folder your songs are in.")
    n = 0
    for p in sorted(src.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in AUDIO_EXT):
            continue
        # Same file name in two subfolders (e.g. "01 Intro.mp3" from two albums): keep both.
        stem, k, dup = p.stem, 2, False
        while (dest / f"{stem}{p.suffix}").exists():
            if (dest / f"{stem}{p.suffix}").stat().st_size == p.stat().st_size:
                dup = True  # already imported (re-running Import on the same folder)
                break
            stem, k = f"{p.stem}-{k}", k + 1
        if dup:
            continue
        shutil.copy2(p, dest / f"{stem}{p.suffix}")
        n += 1
        for side in (".lyrics.txt", ".caption.txt"):  # bring sidecars along if present
            sp = p.with_name(p.stem + side)
            if sp.exists():
                shutil.copy2(sp, dest / f"{stem}{side}")
    ds = load_dataset(settings, name)
    for s in ds["samples"]:
        stem = Path(s["filename"]).stem
        lp, cp = d / "audio" / f"{stem}.lyrics.txt", d / "audio" / f"{stem}.caption.txt"
        if lp.exists() and s.get("lyrics") in ("", "[Instrumental]"):
            s["lyrics"] = lp.read_text(encoding="utf-8").strip()
            s["is_instrumental"] = False
        if cp.exists() and not s.get("caption"):
            s["caption"] = cp.read_text(encoding="utf-8").strip()
    save_dataset(settings, name, ds)
    return n


def remove_file(settings: dict, name: str, filename: str) -> None:
    d = ds_dir(settings, name)
    p = d / "audio" / Path(filename).name
    if p.exists():
        trash = config.DATA / "trash" / "datasets" / name
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(trash / p.name))
    ds = load_dataset(settings, name)
    save_dataset(settings, name, ds)


def update_samples(settings: dict, name: str, metadata: dict | None, samples: list[dict] | None) -> dict:
    ds = load_dataset(settings, name)
    if metadata:
        for k in ("custom_tag", "all_instrumental", "genre_ratio"):
            if k in metadata:
                ds["metadata"][k] = metadata[k]
    if samples is not None:
        by_name = {s["filename"]: s for s in ds["samples"]}
        editable = ("caption", "genre", "lyrics", "bpm", "keyscale", "timesignature", "language", "is_instrumental")
        for upd in samples:
            s = by_name.get(upd.get("filename"))
            if s:
                for k in editable:
                    if k in upd:
                        s[k] = upd[k]
                if s.get("is_instrumental") and not (s.get("lyrics") or "").strip():
                    s["lyrics"] = "[Instrumental]"
    save_dataset(settings, name, ds)
    return ds


def merge_labels(ds: dict, labeled: list[dict], overwrite: bool) -> int:
    by_name = {s["filename"]: s for s in ds["samples"]}
    n = 0
    for lab in labeled:
        s = by_name.get(lab.get("filename"))
        if not s:
            continue
        for k in ("caption", "genre", "bpm", "keyscale", "timesignature", "duration", "language"):
            v = lab.get(k)
            if v in (None, "", 0, "unknown"):
                continue
            if overwrite or s.get(k) in (None, "", 0, "unknown"):
                s[k] = v
        lyr = lab.get("lyrics") or ""
        if lyr and lyr != "[Instrumental]" and (overwrite or s.get("lyrics") in ("", "[Instrumental]")):
            s["lyrics"] = lyr
            s["is_instrumental"] = False
        n += 1
    return n


def _ds_hash(ds: dict) -> str:
    return hashlib.sha1(json.dumps(ds["samples"], sort_keys=True).encode()).hexdigest()[:12]


# ── LoRAs on disk ─────────────────────────────────────────────────────

def find_adapters(settings: dict) -> list[dict]:
    roots = [config.outputs_dir(settings)] + [Path(p) for p in settings["training"].get("extra_lora_dirs", []) if p]
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for cfg in root.rglob("adapter_config.json"):
            out.append(_adapter_entry(root, cfg.parent))
        for st in root.rglob("lokr_weights.safetensors"):
            out.append(_adapter_entry(root, st.parent))
    seen, uniq = set(), []
    for a in sorted(out, key=lambda a: a["label"]):
        if a["path"] not in seen:
            seen.add(a["path"])
            uniq.append(a)
    return uniq


def _adapter_entry(root: Path, folder: Path) -> dict:
    run = None
    for p in [folder, *folder.parents]:
        if (p / "run.json").is_file():
            try:
                run = json.loads((p / "run.json").read_text(encoding="utf-8"))
            except Exception:
                run = None
            break
        if p == root:
            break
    try:
        label = str(folder.relative_to(root)).replace("\\", "/")
    except ValueError:
        label = folder.name
    return {"path": str(folder.resolve()), "label": label, "trigger": (run or {}).get("trigger", ""),
            "model": (run or {}).get("model_variant", "")}


# ── training job ──────────────────────────────────────────────────────

class Trainer:
    def __init__(self, server: ace.AceServer, gpu_lock: threading.Lock, on_server_restart=None) -> None:
        self.server = server
        self.gpu = gpu_lock
        self.on_server_restart = on_server_restart or (lambda: None)
        self.proc = None
        self.state: dict = {"status": "idle"}
        self.log: list[str] = []
        self._stop = False

    def _sink(self, line: str) -> None:
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        self.log.append(line)
        if len(self.log) > 4000:
            del self.log[:1000]
        m = EPOCH_RE.search(line)
        if m:
            ep, tot, secs, loss = int(m[1]), int(m[2]), float(m[3]), float(m[4])
            self.state["epoch_losses"].append([ep, loss])
            self.state.update(epoch=ep, epochs=tot, loss=loss, epoch_seconds=secs,
                              eta_seconds=round(secs * (tot - ep)))
            return
        m = STEP_RE.search(line)
        if m:
            self.state.update(epoch=int(m[1]), epochs=int(m[2]), step=int(m[3]), loss=float(m[4]))

    def status(self, tail: int = 150) -> dict:
        return {**self.state, "log": self.log[-tail:]}

    def start(self, settings: dict, name: str, opts: dict) -> None:
        if self.state.get("status") in ("preparing", "preprocessing", "training"):
            raise RuntimeError("A training run is already in progress.")
        ace_dir = config.ace_dir(settings)
        if not ace.is_installed(ace_dir):
            raise RuntimeError("ACE-Step is not installed (Settings > ACE-Step).")
        ds = load_dataset(settings, name)
        if len(ds["samples"]) < 2:
            raise RuntimeError("Add at least 2 audio files (8-30 recommended).")
        unlabeled = [s["filename"] for s in ds["samples"] if not s.get("caption")]
        if unlabeled and not opts.get("allow_unlabeled"):
            raise RuntimeError(f"{len(unlabeled)} file(s) have no caption. Run Auto-label or write captions first.")
        if not self.server.managed and ace.AceClient(settings).health():
            raise RuntimeError("An ACE-Step server started outside the Workbench is running and holding the GPU. "
                               "Close it first (or let the Workbench start ACE-Step), then train.")
        run_name = _check_name(opts.get("run_name") or f"{name}-{time.strftime('%Y%m%d-%H%M')}")
        self._stop = False
        self.log = []
        self.state = {"status": "preparing", "stage": "Preparing", "dataset": name, "run": run_name, "epoch": 0,
                      "epochs": int(opts.get("epochs") or 0), "loss": None, "epoch_losses": [], "error": "",
                      "started": time.time()}
        threading.Thread(target=self._run, args=(settings, name, ds, run_name, opts), daemon=True).start()

    def stop(self) -> None:
        self._stop = True
        proc = self.proc  # may be cleared by the worker thread at any moment
        if proc is not None:
            ace.kill_tree(proc)

    def _exec(self, cmd: list[str], cwd: Path) -> int:
        self._sink("[workbench] $ " + " ".join(cmd))
        self.proc = ace.popen_logged(cmd, cwd, ace.base_env(), self._sink)
        code = self.proc.wait()
        self.proc = None
        return code

    def _run(self, settings, name, ds, run_name, opts) -> None:
        ace_dir = config.ace_dir(settings)
        py = str(ace.venv_python(ace_dir))
        ckpt = str(ace_dir / "checkpoints")
        d = ds_dir(settings, name)
        tensors = d / "tensors"
        out = config.outputs_dir(settings) / run_name
        variant = opts.get("model_variant") or VARIANT_FOR_MODEL.get(settings["ace"]["dit_model"], "turbo")
        was_running = self.server.managed
        try:
            with self.gpu:
                if self._stop:  # Stop pressed while waiting for a generation to finish
                    raise RuntimeError("Stopped.")
                if was_running:
                    self.state.update(stage="Stopping ACE-Step to free GPU memory")
                    self.server.stop()
                    self.on_server_restart()
                    time.sleep(2)
                if self._stop:
                    raise RuntimeError("Stopped.")
                # preprocess (skipped when the dataset has not changed since last time)
                h = _ds_hash(ds) + variant
                stamp = tensors / ".workbench_hash"
                if not (stamp.exists() and stamp.read_text() == h and any(tensors.glob("*.pt"))):
                    self.state.update(status="preprocessing", stage="Preprocessing audio into tensors")
                    if tensors.exists():
                        shutil.rmtree(tensors)
                    tensors.mkdir(parents=True)
                    code = self._exec([py, "train.py", "--plain", "--yes", "fixed", "--preprocess",
                                       "--checkpoint-dir", ckpt, "--model-variant", variant,
                                       "--dataset-dir", str(tensors), "--output-dir", str(out),
                                       "--dataset-json", str(d / "dataset.json"), "--tensor-output", str(tensors),
                                       "--max-duration", str(opts.get("max_duration") or 240)], ace_dir)
                    if self._stop:
                        raise RuntimeError("Stopped.")
                    if code != 0 or not any(tensors.glob("*.pt")):
                        raise RuntimeError(f"Preprocessing failed (exit {code}). See the log.")
                    stamp.write_text(h)
                if self._stop:
                    raise RuntimeError("Stopped.")
                out.mkdir(parents=True, exist_ok=True)
                run = {"dataset": name, "trigger": ds["metadata"].get("custom_tag", ""), "model_variant": variant,
                       "opts": opts, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": len(ds["samples"])}
                (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
                self.state.update(status="training", stage="Training")
                base = variant.endswith("base") or variant.endswith("sft")
                cmd = [py, "train.py", "--plain", "--yes", "fixed", "--checkpoint-dir", ckpt,
                       "--model-variant", variant, "--dataset-dir", str(tensors), "--output-dir", str(out),
                       "--adapter-type", opts.get("adapter_type", "lora"),
                       "--lr", str(opts.get("lr", 1e-4)), "--epochs", str(int(opts.get("epochs", 200))),
                       "--batch-size", str(int(opts.get("batch_size", 1))),
                       "--gradient-accumulation", str(int(opts.get("grad_accum", 4))),
                       "--save-every", str(int(opts.get("save_every", 25))),
                       "--shift", "1.0" if base else "3.0", "--num-inference-steps", "50" if base else "8",
                       "--seed", str(int(opts.get("seed", 42)))]
                if opts.get("adapter_type") == "lokr":
                    cmd += ["--lokr-linear-dim", str(int(opts.get("rank", 64))), "--lokr-linear-alpha", str(int(opts.get("alpha", 128)))]
                else:
                    cmd += ["--rank", str(int(opts.get("rank", 64))), "--alpha", str(int(opts.get("alpha", 128)))]
                if opts.get("offload_encoder"):
                    cmd.append("--offload-encoder")
                code = self._exec(cmd, ace_dir)
                if self._stop:
                    raise RuntimeError("Stopped. Checkpoints saved so far are kept.")
                if code != 0:
                    raise RuntimeError(f"Training failed (exit {code}). See the log.")
            self.state.update(status="done", stage="Done", output=str(out))
        except Exception as e:
            self.state.update(status="stopped" if self._stop else "error", stage="Stopped" if self._stop else "Failed", error=str(e))
        finally:
            self.state["finished"] = time.time()
            if was_running and not self.server.managed:
                try:
                    self.on_server_restart()
                    self.server.start(settings)
                except Exception as e:
                    self._sink(f"[workbench] could not restart ACE-Step: {e}")


# ── auto-label through the ACE-Step API ───────────────────────────────

def auto_label(settings: dict, name: str, transcribe: bool, overwrite: bool, progress) -> int:
    client = ace.AceClient(settings)
    d = ds_dir(settings, name)
    ds = load_dataset(settings, name)
    client.post("/v1/dataset/scan", {"audio_dir": str((d / "audio").resolve()), "dataset_name": name,
                                     "custom_tag": "", "tag_position": "prepend", "all_instrumental": False}, timeout=120)
    started = client.post("/v1/dataset/auto_label_async", {"transcribe_lyrics": transcribe, "format_lyrics": False,
                                                          "save_path": str((d / "autolabel.json").resolve())}, timeout=120)
    task = started.get("task_id")
    if not started.get("total"):
        return 0
    while True:
        time.sleep(2)
        st = client.get(f"/v1/dataset/auto_label_status/{task}")
        progress(f"{st.get('current', 0)}/{st.get('total', 0)}: {st.get('progress', '')}")
        if st.get("status") == "completed":
            labeled = (st.get("result") or {}).get("samples") or []
            if not labeled and (d / "autolabel.json").exists():
                labeled = json.loads((d / "autolabel.json").read_text(encoding="utf-8")).get("samples", [])
            ds = load_dataset(settings, name)
            n = merge_labels(ds, labeled, overwrite)
            save_dataset(settings, name, ds)
            return n
        if st.get("status") == "failed":
            raise ace.AceError(st.get("error") or "Auto-label failed")
