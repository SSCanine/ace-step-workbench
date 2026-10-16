"""Song library: every generated song keeps its audio and the exact settings that made it.

Layout:  data/songs/<id>/song.json  +  the audio file(s)
Deleting moves the folder to data/trash/ (nothing is hard-deleted).
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from . import config

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac")


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def song_dir(song_id: str) -> Path:
    if not song_id or any(c in song_id for c in "/\\:") or song_id.startswith("."):
        raise ValueError("bad song id")
    return config.songs_dir() / song_id


def save(meta: dict, audio_src: Path | None = None, audio_bytes: bytes | None = None, ext: str = ".mp3") -> dict:
    sid = meta.get("id") or new_id()
    d = song_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    name = "audio" + (audio_src.suffix if audio_src else ext)
    if audio_src is not None:
        shutil.copy2(audio_src, d / name)
    elif audio_bytes is not None:
        (d / name).write_bytes(audio_bytes)
    meta = {**meta, "id": sid, "audio": name, "created": meta.get("created") or time.time()}
    (d / "song.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def get(song_id: str) -> dict:
    p = song_dir(song_id) / "song.json"
    if not p.exists():
        raise FileNotFoundError(song_id)
    return json.loads(p.read_text(encoding="utf-8"))


def update(song_id: str, patch: dict) -> dict:
    meta = get(song_id)
    allowed = {"title", "favorite", "notes", "tags"}
    meta.update({k: v for k, v in patch.items() if k in allowed})
    (song_dir(song_id) / "song.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def audio_path(song_id: str) -> Path:
    meta = get(song_id)
    return song_dir(song_id) / meta["audio"]


def list_songs(query: str = "") -> list[dict]:
    out = []
    q = query.lower().strip()
    for d in config.songs_dir().iterdir():
        p = d / "song.json"
        if not p.is_file():
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if q:
            hay = " ".join(str(m.get(k, "")) for k in ("title", "caption", "lyrics", "tags", "kind")).lower()
            if q not in hay:
                continue
        out.append({k: m.get(k) for k in ("id", "title", "caption", "created", "duration", "seed", "kind", "parent", "favorite", "lora", "bpm", "keyscale")})
    out.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return out


def trash(song_id: str) -> None:
    d = song_dir(song_id)
    if not d.exists():
        raise FileNotFoundError(song_id)
    dest = config.DATA / "trash"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest / f"{song_id}-{int(time.time())}"))
