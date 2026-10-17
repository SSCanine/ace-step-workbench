"""Settings embedded in exports, dropping songs back in, and autosave drafts."""

import io
import json
import struct
import wave
from pathlib import Path

from tests.test_e2e import env, wait_job  # noqa: F401  (fixture)
from workbench import importer, songmeta

MPEG = b"\xff\xfb\x90\x64" + b"\x00" * 400  # a fake MPEG frame is enough: we never touch audio data


def id3_tag(frames: bytes) -> bytes:
    return b"ID3\x04\x00\x00" + songmeta._syncsafe(len(frames) + 64) + frames + b"\x00" * 64  # with padding


def tit2(text: str) -> bytes:
    body = b"\x03" + text.encode()
    return b"TIT2" + songmeta._syncsafe(len(body)) + b"\x00\x00" + body


def test_mp3_embed_keeps_frames_and_audio():
    mp3 = id3_tag(tit2("Song")) + MPEG
    out = songmeta.embed(mp3, {"app": "x", "n": 1})
    assert songmeta.read(out) == {"app": "x", "n": 1}
    assert out.endswith(MPEG) and b"TIT2" in out
    again = songmeta.embed(out, {"app": "x", "n": 2})  # re-embedding replaces, never stacks
    assert songmeta.read(again) == {"app": "x", "n": 2} and again.count(b"TXXX") == 1
    bare = songmeta.embed(MPEG, {"a": 1})  # no tag yet: one is created
    assert bare.startswith(b"ID3\x04") and songmeta.read(bare) == {"a": 1} and bare.endswith(MPEG)


def test_wav_and_flac_embed():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000), w.writeframes(b"\x01\x02" * 100)
    wav = songmeta.embed(buf.getvalue(), {"k": "v" * 3})  # odd length -> padded chunk
    assert songmeta.read(wav) == {"k": "vvv"} and struct.unpack("<I", wav[4:8])[0] == len(wav) - 8
    with wave.open(io.BytesIO(wav)) as w:  # still a valid WAV with the same audio
        assert w.readframes(100) == b"\x01\x02" * 100

    streaminfo = bytes((0x80,)) + (34).to_bytes(3, "big") + b"\x00" * 34
    flac = b"fLaC" + streaminfo + b"\xff\xf8FRAMES"
    out = songmeta.embed(flac, {"f": 1})
    assert songmeta.read(out) == {"f": 1} and out.endswith(b"\xff\xf8FRAMES")
    assert out[4] == 0x00  # STREAMINFO is no longer the last block
    assert songmeta.embed(b"OggS....", {"x": 1}) == b"OggS...."  # unsupported: unchanged


def _make_lora(env_data: Path) -> tuple[Path, str]:
    d = env_data / "loras" / "my-style" / "final"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"weights" * 1000)
    import hashlib
    return d, hashlib.sha256(b"weights" * 1000).hexdigest()


def test_download_carries_settings_and_drops_back(env):
    c, _ = env
    from workbench import config, library
    lora_dir, _ = _make_lora(config.DATA)
    form = {"title": "Paper Lungs", "caption": "nu-metal", "lyrics": "[Verse]\nhi", "bpm": 98, "seed": 77,
            "batch_size": 1, "inference_steps": 8, "thinking": False, "lora": str(lora_dir.resolve()), "lora_scale": 0.6}
    j = wait_job(c, c.post("/api/generate", json=form).json()["job"])
    assert j["status"] == "done", j
    sid = j["songs"][0]["id"]
    library.audio_path(sid).write_bytes(id3_tag(tit2("x")) + MPEG)  # mock audio isn't a real MP3

    dl = c.get(f"/api/songs/{sid}/audio?download=1")
    assert "Paper%20Lungs%20%2877%29.mp3" in dl.headers["content-disposition"]
    meta = songmeta.read(dl.content)
    assert meta["app"] == "ace-step-workbench" and meta["form"]["bpm"] == 98 and "lora" not in meta["form"]
    assert meta["lora"]["label"] == "my-style/final" and "\\\\" not in json.dumps(meta)  # no local paths

    r = c.post("/api/import", files=[("files", ("Paper Lungs (77).mp3", dl.content, "audio/mpeg"))]).json()
    assert r["source"] == "Workbench download" and r["song_id"] == sid  # same song, not a duplicate
    assert r["form"]["seed"] == 77 and r["form"]["caption"] == "nu-metal" and r["form"]["lora_scale"] == 0.6
    assert Path(r["form"]["lora"]) == lora_dir.resolve()


def test_ace_ui_export_by_name_and_by_bytes(env):
    c, fake_ace = env
    from workbench import config
    lora_dir, lora_sha = _make_lora(config.DATA)
    batch = fake_ace / "gradio_outputs" / "batch_1"
    batch.mkdir(parents=True)
    audio = id3_tag(tit2("y")) + MPEG + b"unique"
    uid = "c43f01a8-5b74-c6b8-70f4-78338182a555"
    (batch / f"{uid}.mp3").write_bytes(audio)
    (batch / f"{uid}.json").write_text(json.dumps({
        "caption": "lpnumetal, anthem", "lyrics": "[Chorus]\nla", "instrumental": False, "vocal_language": "unknown",
        "bpm": 0, "keyscale": "", "timesignature": "", "duration": -1, "inference_steps": 8, "seed": 1286263321,
        "guidance_scale": 7, "shift": 3, "infer_method": "ode", "thinking": True, "lm_temperature": 0.85,
        "lm_cfg_scale": 2, "lm_top_p": 0.9, "use_cot_caption": False, "use_cot_language": True,
        "cot_bpm": 105, "cot_keyscale": "D minor", "cot_timesignature": "4", "cot_duration": 192.0,
        "lora_loaded": True, "use_lora": True, "lora_scale": 1.0, "lora_weights_hash": lora_sha, "audio_format": "mp3"}))

    # renamed copy (e.g. moved out of gradio_outputs): matched by its bytes
    r = c.post("/api/import", files=[("files", ("shirtless-take1.mp3", audio, "audio/mpeg"))]).json()
    f = r["form"]
    assert r["source"] == "ACE-Step UI export" and r["title"] == "shirtless-take1"
    assert f["seed"] == 1286263321 and f["thinking"] is True and f["bpm"] is None and f["vocal_language"] == ""
    assert Path(f["lora"]) == lora_dir.resolve() and f["lora_scale"] == 1.0
    assert r["resolved"] == {"bpm": 105, "keyscale": "D minor", "timesignature": "4", "duration": 192}
    song = c.get(f"/api/songs/{r['song_id']}").json()
    assert song["kind"] == "import" and song["seed"] == 1286263321 and song["bpm"] == 105

    # the original file name finds its .json too; the Library copy is reused, not duplicated
    r2 = c.post("/api/import", files=[("files", (f"{uid}.mp3", audio, "audio/mpeg"))]).json()
    assert r2["song_id"] == r["song_id"] and r2["source"] == "Library"
    assert len(c.get("/api/songs").json()["songs"]) == 1

    # only the .json: settings, no Library entry
    r3 = c.post("/api/import", files=[("files", (f"{uid}.json", (batch / f"{uid}.json").read_bytes(), "application/json"))]).json()
    assert r3["form"]["seed"] == 1286263321 and r3["song_id"] is None


def test_import_unknown_file_explains(env):
    c, _ = env
    r = c.post("/api/import", files=[("files", ("mastered.wav", b"RIFF\x04\x00\x00\x00WAVE", "audio/wav"))])
    assert r.status_code == 400 and "No saved settings found for mastered.wav" in r.json()["detail"]


def test_lora_hash_is_cached(env):
    from workbench import config
    lora_dir, sha = _make_lora(config.DATA)
    assert importer.lora_hash(lora_dir) == sha
    cache_file = config.DATA / "lora_hashes.json"
    cache = json.loads(cache_file.read_text())
    assert list(cache.values()) == [sha]
    cache_file.write_text(json.dumps({k: "from-cache" for k in cache}))
    assert importer.lora_hash(lora_dir) == "from-cache"  # unchanged file: not hashed again
    (lora_dir / "adapter_model.safetensors").write_bytes(b"retrained")
    assert importer.lora_hash(lora_dir) != "from-cache"  # changed file: hashed again


def test_draft_roundtrip(env):
    c, _ = env
    assert c.get("/api/draft").json() == {"draft": None}
    d = {"v": 1, "tab": "studio", "create": {"caption": "dark", "lyrics": "[Verse]\nä ö"}}
    assert c.put("/api/draft", json=d).json() == {"ok": True}
    assert c.get("/api/draft").json()["draft"] == d
    assert c.put("/api/draft", content=b"[1]", headers={"Content-Type": "application/json"}).status_code == 400
