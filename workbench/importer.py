"""Drop a song back in: find the settings that made an audio file.

Sources, first match wins:
  1. a .json dropped with it (ACE-Step's own UI writes one next to every song; a Workbench song.json works too)
  2. settings embedded in a Workbench download (see songmeta.py)
  3. a Library song with the same audio bytes
  4. an export of ACE-Step's own UI in <ACE-Step folder>/gradio_outputs (same file name, or same bytes)

Imported audio is added to the Library (once) so Studio can regenerate and re-sing it.
LoRAs are matched the way ACE-Step fingerprints them: SHA-256 of the adapter weight file(s).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

from . import config, library, songmeta, training

APP_ID = "ace-step-workbench"
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin", "lokr_weights.safetensors")
_hash_lock = threading.Lock()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── LoRA fingerprints ──

def lora_hash(adapter_dir: str | Path) -> str:
    """ACE-Step's lora_weights_hash for one adapter folder, cached in data/lora_hashes.json."""
    d = Path(adapter_dir)
    files = [d / n for n in WEIGHT_FILES if (d / n).is_file()] if d.is_dir() else [d] if d.is_file() else []
    if not files:
        return ""
    cache_file = config.DATA / "lora_hashes.json"
    with _hash_lock:
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        key = "|".join(f"{f}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
        if key not in cache:
            h = hashlib.sha256()
            for f in files:
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            cache[key] = h.hexdigest()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        return cache[key]


def find_lora(settings: dict, sha256: str = "", label: str = "") -> dict | None:
    adapters = training.find_adapters(settings)
    if sha256:
        for a in adapters:
            if lora_hash(a["path"]) == sha256:
                return a
    if label:
        for a in adapters:
            if a["label"] == label:
                return a
    return None


def lora_label(settings: dict, path: str) -> str:
    for a in training.find_adapters(settings):
        if os.path.normcase(a["path"]) == os.path.normcase(str(Path(path).resolve())):
            return a["label"]
    return "/".join(Path(path).parts[-2:])


# ── exported metadata ──

def export_meta(settings: dict, song: dict) -> dict:
    """What a Workbench download carries: the full form, minus local paths."""
    form = {k: v for k, v in (song.get("form") or {}).items() if k not in ("lora", "lora_scale")}
    lora = None
    if song.get("lora"):
        lora = {"label": lora_label(settings, song["lora"]), "sha256": lora_hash(song["lora"]),
                "scale": song.get("lora_scale")}
    keep = ("title", "seed", "caption", "lyrics", "bpm", "keyscale", "timesignature", "duration", "dit_model")
    return {"app": APP_ID, "v": 1, **{k: song.get(k) for k in keep}, "form": form, "lora": lora}


# ── mapping each source to a Create form ──

def _num(v, positive=True):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if positive and n <= 0:
        return None
    return int(n) if n == int(n) else n


def _timesig(v) -> str:
    s = str(v or "").split("/")[0].strip()
    return s if s in ("2", "3", "4", "6") else ""


def from_ace_ui(settings: dict, d: dict) -> tuple[dict, dict, list[str]]:
    """ACE-Step UI sidecar JSON -> (form, resolved metas, notes). Keeps 'auto' inputs as auto so the seed reproduces."""
    lang = str(d.get("vocal_language") or "")
    form = {
        "caption": d.get("caption") or "", "lyrics": d.get("lyrics") or "", "instrumental": bool(d.get("instrumental")),
        "vocal_language": "" if lang in ("", "unknown") else lang,
        "bpm": _num(d.get("bpm")), "keyscale": d.get("keyscale") or "", "timesignature": _timesig(d.get("timesignature")),
        "duration": _num(d.get("duration")), "seed": _num(d.get("seed"), positive=False), "batch_size": 1,
        "audio_format": d.get("audio_format") or "mp3",
    }
    for k in ("inference_steps", "guidance_scale", "shift", "lm_temperature", "lm_cfg_scale", "lm_top_p"):
        if _num(d.get(k)) is not None:
            form[k] = _num(d.get(k))
    for k in ("thinking", "use_cot_caption", "use_cot_language"):
        if k in d:
            form[k] = bool(d[k])
    if d.get("infer_method") in ("ode", "sde"):
        form["infer_method"] = d["infer_method"]
    resolved = {
        "bpm": form["bpm"] or _num(d.get("cot_bpm")), "keyscale": form["keyscale"] or d.get("cot_keyscale") or "",
        "timesignature": form["timesignature"] or _timesig(d.get("cot_timesignature")),
        "duration": form["duration"] or _num(d.get("cot_duration")),
    }
    notes = []
    if d.get("lora_loaded") and d.get("use_lora") and d.get("lora_weights_hash"):
        a = find_lora(settings, sha256=d["lora_weights_hash"])
        if a:
            form["lora"], form["lora_scale"] = a["path"], _num(d.get("lora_scale")) or 1.0
            notes.append(f"LoRA: {a['label']} at {form['lora_scale']}")
        else:
            notes.append("It was made with a LoRA that isn't in your LoRA folders (or a mix of LoRAs). Pick one in the LoRA box.")
    return form, resolved, notes


def from_workbench(settings: dict, meta: dict) -> tuple[dict, dict, list[str]]:
    """Workbench song.json or embedded download metadata -> (form, resolved metas, notes)."""
    form = dict(meta.get("form") or {})
    form.setdefault("seed", meta.get("seed"))
    form["batch_size"] = 1
    for k in ("caption", "lyrics"):
        if not form.get(k) and meta.get(k):
            form[k] = meta[k]
    notes = []
    lora = meta.get("lora")
    if isinstance(lora, dict):  # embedded download: matched by fingerprint, then by name
        a = find_lora(settings, sha256=lora.get("sha256") or "", label=lora.get("label") or "")
        if a:
            form["lora"], form["lora_scale"] = a["path"], lora.get("scale") or 0.8
            notes.append(f"LoRA: {a['label']} at {form['lora_scale']}")
        else:
            notes.append(f"It was made with the LoRA \"{lora.get('label')}\", which isn't in your LoRA folders.")
    elif lora:  # Library song.json: a local path
        if Path(lora).exists():
            form["lora"], form["lora_scale"] = lora, meta.get("lora_scale") or form.get("lora_scale") or 0.8
        else:
            form.pop("lora", None)
            notes.append("Its LoRA folder no longer exists. Pick one in the LoRA box.")
    if meta.get("dit_model") and meta["dit_model"] != settings["ace"]["dit_model"]:
        notes.append(f"It was made with {meta['dit_model']}; you're on {settings['ace']['dit_model']} (Settings).")
    resolved = {k: meta.get(k) for k in ("bpm", "keyscale", "timesignature", "duration")}
    return form, resolved, notes


# ── finding the source ──

AUDIO_MAGIC = (b"ID3", b"RIFF", b"fLaC", b"OggS")


def _is_audio(name: str, data: bytes) -> bool:
    return Path(name).suffix.lower() in library.AUDIO_EXT or data[:4].startswith(AUDIO_MAGIC) or songmeta._is_mpeg(data)


def _library_match(size: int, digest: str) -> dict | None:
    root = config.songs_dir()
    for d in root.iterdir():
        try:
            meta = json.loads((d / "song.json").read_text(encoding="utf-8"))
            audio = d / meta["audio"]
            if audio.stat().st_size == size and _sha256_file(audio) == digest:
                return meta
        except (OSError, ValueError, KeyError):
            continue
    return None


def _library_by_seed(seed, caption) -> dict | None:
    """The Library song a Workbench download came from (its bytes differ: the download carries settings)."""
    if seed is None:
        return None
    for s in library.list_songs():
        if s.get("seed") == seed and (s.get("caption") or "") == (caption or ""):
            try:
                return library.get(s["id"])
            except (OSError, ValueError):
                return None
    return None


def _ace_ui_match(settings: dict, name: str, size: int, digest: str) -> dict | None:
    root = config.ace_dir(settings) / "gradio_outputs"
    if not root.is_dir():
        return None
    stem = Path(name).stem
    same_size = []
    for batch in os.scandir(root):
        if not batch.is_dir():
            continue
        side = Path(batch.path) / f"{stem}.json"
        if side.is_file() and any((Path(batch.path) / f"{stem}{ext}").is_file() for ext in library.AUDIO_EXT):
            try:
                return json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        for f in os.scandir(batch.path):
            if f.is_file() and Path(f.name).suffix.lower() in library.AUDIO_EXT and f.stat().st_size == size:
                same_size.append(Path(f.path))
    for p in same_size:
        side = p.with_suffix(".json")
        if side.is_file() and _sha256_file(p) == digest:
            try:
                return json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    return None


def _title_from(name: str) -> str:
    stem = Path(name).stem
    if re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", stem, re.I):
        return "Imported song"
    return re.sub(r"[_]+", " ", stem).strip() or "Imported song"


def import_files(settings: dict, files: list[tuple[str, bytes]]) -> dict:
    """Identify dropped files. Returns {form, title, source, notes, resolved, song_id}."""
    audio = next(((n, b) for n, b in files if _is_audio(n, b)), None)
    side = None
    for n, b in files:
        if n.lower().endswith(".json"):
            try:
                side = json.loads(b.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise ValueError(f"{n} is not valid JSON.")
    if audio is None and side is None:
        raise ValueError("Drop an audio file (MP3, WAV, FLAC...) or the .json that ACE-Step saved next to it.")

    song, source, form = None, "", None
    digest = _sha256(audio[1]) if audio else ""
    if isinstance(side, dict) and "form" in side and "seed" in side:
        form, resolved, notes = from_workbench(settings, side)
        source, title = "Workbench song file", side.get("title") or "Imported song"
    elif isinstance(side, dict) and "caption" in side and "inference_steps" in side:
        form, resolved, notes = from_ace_ui(settings, side)
        source, title = "ACE-Step UI export", _title_from(audio[0] if audio else "")
    elif side is not None:
        raise ValueError("That .json isn't an ACE-Step or Workbench song file.")
    if form is None and audio:
        emb = songmeta.read(audio[1])
        song = _library_match(len(audio[1]), digest)
        if emb and emb.get("app") == APP_ID:
            form, resolved, notes = from_workbench(settings, emb)
            source, title = "Workbench download", emb.get("title") or _title_from(audio[0])
            song = song or _library_by_seed(emb.get("seed"), emb.get("caption"))
        elif song:
            form, resolved, notes = from_workbench(settings, song)
            source, title = "Library", song.get("title") or "Untitled"
        else:
            ui = _ace_ui_match(settings, audio[0], len(audio[1]), digest)
            if ui:
                form, resolved, notes = from_ace_ui(settings, ui)
                source, title = "ACE-Step UI export", _title_from(audio[0])
    if form is None:
        raise ValueError(f"No saved settings found for {audio[0]}. Songs made in the Workbench or in ACE-Step's own UI "
                         "work; if the song was moved, drop the .json that was saved next to it as well.")

    form["title"] = title
    song_id = None
    if audio:
        song = song or _library_match(len(audio[1]), digest)
        if song:
            song_id = song["id"]
        else:
            ext = Path(audio[0]).suffix.lower() if Path(audio[0]).suffix.lower() in library.AUDIO_EXT else ".mp3"
            meta = library.save({
                "title": title, "kind": "import", "parent": None, "caption": form.get("caption", ""),
                "lyrics": form.get("lyrics", ""), "seed": form.get("seed"), **{k: v for k, v in resolved.items()},
                "lora": form.get("lora", ""), "lora_scale": form.get("lora_scale"),
                "dit_model": settings["ace"]["dit_model"], "lm_model": "", "form": form, "ace_params": None,
                "source": source, "imported_from": Path(audio[0]).name,
            }, audio_bytes=audio[1], ext=ext)
            song_id = meta["id"]
    return {"form": form, "title": title, "source": source, "notes": notes, "resolved": resolved, "song_id": song_id}
