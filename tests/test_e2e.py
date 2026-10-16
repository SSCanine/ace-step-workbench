"""End-to-end without a GPU: Workbench API -> mock ACE-Step -> library; fake trainer."""

import json
import socket
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests import mock_ace


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    from workbench import ace, config, server

    port = free_port()
    mock = mock_ace.serve(port)
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "data" / "settings.json")
    fake_ace = tmp_path / "ace"
    (fake_ace / "acestep").mkdir(parents=True)
    (fake_ace / "checkpoints").mkdir()
    monkeypatch.setattr(ace, "venv_python", lambda d: Path(sys.executable))
    config.save({"ace": {"path": str(fake_ace), "api_url": f"http://127.0.0.1:{port}", "autostart": False},
                 "llm": {"preset": "none", "kind": "none"}})
    server.jobs.loaded_lora, server.jobs.lora_on = None, False
    mock_ace.CALLS.clear()
    server.SECURITY["token"] = None
    with TestClient(server.app, base_url="http://127.0.0.1:3470", headers={"X-Workbench": "1"}) as c:
        yield c, fake_ace
    mock.should_exit = True


def wait_job(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise AssertionError("job timed out")


def test_generate_remix_variation_repaint(env):
    c, _ = env
    form = {"title": "T", "caption": "dark techno", "lyrics": "[Verse 1]\nhello", "bpm": 128, "keyscale": "A minor",
            "duration": 30, "batch_size": 2, "seed": 1000, "inference_steps": 8}
    j = wait_job(c, c.post("/api/generate", json=form).json()["job"])
    assert j["status"] == "done", j
    assert [s["seed"] for s in j["songs"]] == [1000, 1001]
    sent = mock_ace.CALLS[-1]
    assert sent["use_random_seed"] is False and sent["seed"] == "1000,1001"
    assert sent["prompt"] == "dark techno" and sent["key_scale"] == "A minor" and sent["audio_duration"] == 30

    sid = j["songs"][1]["id"]
    song = c.get(f"/api/songs/{sid}").json()
    assert song["seed"] == 1001 and song["form"]["seed"] == 1001 and song["keyscale"] == "A minor"
    assert c.get(f"/api/songs/{sid}/audio").content.startswith(b"ID3")

    # same seed, edited lyrics
    j2 = wait_job(c, c.post(f"/api/songs/{sid}/remix", json={"mode": "same_seed", "form": {"lyrics": "[Verse 1]\nhey"}}).json()["job"])
    assert j2["status"] == "done"
    sent = mock_ace.CALLS[-1]
    assert sent["seed"] == "1001" and sent["batch_size"] == 1 and sent["lyrics"] == "[Verse 1]\nhey"
    child = c.get(f"/api/songs/{j2['songs'][0]['id']}").json()
    assert child["parent"] == sid and child["kind"] == "remix" and child["seed"] == 1001

    # variations get new seeds
    j3 = wait_job(c, c.post(f"/api/songs/{sid}/remix", json={"mode": "variation", "count": 2}).json()["job"])
    assert len(j3["songs"]) == 2 and all(s["seed"] != 1001 for s in j3["songs"])

    # repaint uploads the parent audio as multipart with the span
    j4 = wait_job(c, c.post(f"/api/songs/{sid}/repaint", json={"form": {"repaint_start": 4.5, "repaint_end": 7.25, "repaint_strength": 0.4}}).json()["job"])
    assert j4["status"] == "done", j4
    sent = mock_ace.CALLS[-1]
    assert sent["task_type"] == "repaint" and sent["repainting_start"] == "4.5" and sent["repainting_end"] == "7.25"
    assert sent["_src_audio_bytes"] > 0 and sent["seed"] == "1001" and sent["thinking"] == "false"

    lib = c.get("/api/songs").json()["songs"]
    assert len(lib) == 6
    assert c.delete(f"/api/songs/{sid}").status_code == 200
    assert len(c.get("/api/songs").json()["songs"]) == 5


def test_lora_toggle_and_trigger(env):
    c, _ = env
    form = {"caption": "x", "lora": "/loras/a", "lora_scale": 0.7, "batch_size": 1}
    assert wait_job(c, c.post("/api/generate", json=form).json()["job"])["status"] == "done"
    assert mock_ace.LORA["loaded"] == "/loras/a" and mock_ace.LORA["on"] and mock_ace.LORA["scale"] == 0.7
    assert wait_job(c, c.post("/api/generate", json={"caption": "x"}).json()["job"])["status"] == "done"
    assert mock_ace.LORA["on"] is False  # no LoRA requested -> toggled off, not unloaded
    # switching to a different LoRA without a managed server must refuse (needs a clean restart)
    j = wait_job(c, c.post("/api/generate", json={"caption": "x", "lora": "/loras/b"}).json()["job"])
    assert j["status"] == "error" and "fresh ACE-Step server" in j["error"]


def test_dataset_and_training_flow(env, monkeypatch):
    c, fake_ace = env
    (fake_ace / "train.py").write_text(textwrap.dedent("""
        import sys, pathlib, json
        a = sys.argv
        def arg(k): return a[a.index(k) + 1]
        if "--preprocess" in a:
            ds = json.load(open(arg("--dataset-json"), encoding="utf-8"))
            out = pathlib.Path(arg("--tensor-output"))
            for s in ds["samples"]:
                (out / (s["filename"] + ".pt")).write_bytes(b"t")
            print("[OK] preprocessed", len(ds["samples"]))
        else:
            assert "--rank" in a and arg("--model-variant") == "turbo"
            ep = int(arg("--epochs"))
            for e in range(1, ep + 1):
                print(f"Epoch {e}/{ep}, Step {e*2}, Loss: {1.0/e:.4f}", flush=True)
                print(f"[OK] Epoch {e}/{ep} in 0.01s, Loss: {1.0/e:.4f}", flush=True)
            final = pathlib.Path(arg("--output-dir")) / "final"
            final.mkdir(parents=True)
            (final / "adapter_config.json").write_text("{}")
    """), encoding="utf-8")

    r = c.post("/api/datasets", json={"name": "my set", "trigger": "zxsynth"})
    assert r.status_code == 200
    files = [("files", ("a.mp3", b"aaa", "audio/mpeg")), ("files", ("b.wav", b"bbb", "audio/wav")), ("files", ("notes.txt", b"x", "text/plain"))]
    assert c.post("/api/datasets/my set/files", files=files).json()["added"] == 2
    ds = c.get("/api/datasets/my set").json()
    assert [s["filename"] for s in ds["samples"]] == ["a.mp3", "b.wav"]

    r = c.post("/api/train/start", json={"dataset": "my set", "opts": {"epochs": 3}})
    assert r.status_code == 400 and "no caption" in r.json()["detail"]

    c.put("/api/datasets/my set", json={"samples": [
        {"filename": "a.mp3", "caption": "warm synthwave", "bpm": 110, "is_instrumental": True},
        {"filename": "b.wav", "caption": "dark synth", "lyrics": "[Verse]\nla", "is_instrumental": False}]})
    saved = json.loads((Path(c.get("/api/settings").json()["settings"]["training"]["datasets_dir"] or
                             (Path(fake_ace).parent / "data" / "datasets")) / "my set" / "dataset.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["custom_tag"] == "zxsynth" and saved["metadata"]["tag_position"] == "prepend"
    assert all(s["custom_tag"] == "zxsynth" for s in saved["samples"])

    # an ACE-Step server running outside the Workbench (the mock) blocks training
    r = c.post("/api/train/start", json={"dataset": "my set", "opts": {"epochs": 3}})
    assert r.status_code == 400 and "outside the Workbench" in r.json()["detail"]
    c.post("/api/settings", json={"ace": {"api_url": f"http://127.0.0.1:{free_port()}"}})
    assert c.post("/api/train/start", json={"dataset": "my set", "opts": {"epochs": 3, "run_name": "run1"}}).status_code == 200
    t0 = time.time()
    while time.time() - t0 < 30:
        st = c.get("/api/train/status").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert st["status"] == "done", st
    assert st["epoch"] == 3 and abs(st["loss"] - 0.3333) < 1e-3 and len(st["epoch_losses"]) == 3
    loras = c.get("/api/loras").json()["loras"]
    assert len(loras) == 1 and loras[0]["trigger"] == "zxsynth" and loras[0]["label"] == "run1/final"


def test_ace_folder_check(env, tmp_path):
    c, fake_ace = env
    assert c.get("/api/ace/check", params={"path": str(fake_ace)}).json()["ok"] is True
    r = c.get("/api/ace/check", params={"path": str(tmp_path / "nope")}).json()
    assert r["ok"] is False and "not found" in r["message"]
    (tmp_path / "other").mkdir()
    assert "not an ACE-Step" in c.get("/api/ace/check", params={"path": str(tmp_path / "other")}).json()["message"]


def test_settings_masks_and_remembers_keys(env):
    c, _ = env
    c.post("/api/settings", json={"llm": {"preset": "openai", "kind": "openai", "api_key": "sk-secret-1234"}})
    s = c.get("/api/settings").json()["settings"]
    assert s["llm"]["api_key"] == "••••1234" and "secret" not in json.dumps(s)
    c.post("/api/settings", json={"llm": {"preset": "openai", "api_key": "••••1234", "model": "gpt-x"}})
    from workbench import config
    assert config.load()["llm"]["api_key"] == "sk-secret-1234"
    # the UI sends an empty key box when switching providers: saved keys must survive
    c.post("/api/settings", json={"llm": {"preset": "ollama", "kind": "openai", "api_key": ""}})
    assert config.load()["llm"]["api_key"] == ""
    c.post("/api/settings", json={"llm": {"preset": "openai", "kind": "openai", "api_key": ""}})
    assert config.load()["llm"]["api_key"] == "sk-secret-1234"
    c.post("/api/settings", json={"llm": {"preset": "openai", "api_key": ""}})  # save again, same provider
    assert config.load()["llm"]["api_key"] == "sk-secret-1234"
    c.post("/api/settings", json={"llm": {"preset": "openai", "clear_key": True}})
    assert config.load()["llm"]["api_key"] == "" and config.load()["llm_keys"]["openai"] == ""


def test_request_guard(env):
    from workbench import server
    c, _ = env
    assert c.post("/api/ace/stop", headers={"X-Workbench": ""}).status_code == 403  # no custom header (cross-site form/fetch)
    assert c.post("/api/ace/stop", headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.get("/api/status", headers={"Host": "evil.example"}).status_code == 403  # DNS rebinding
    assert c.get("/api/status").status_code == 200
    server.SECURITY["token"] = "tok"
    try:
        assert c.get("/api/status").status_code == 401
        r = c.get("/?token=tok", follow_redirects=False)
        assert r.status_code in (302, 307) and "wb_token" in r.headers["set-cookie"]
        c.cookies.set("wb_token", "tok")
        assert c.get("/api/status", headers={"Host": "192.168.1.5:3470"}).status_code == 200
    finally:
        server.SECURITY["token"] = None
        c.cookies.clear()


def test_lora_state_resyncs_after_restart(env):
    c, _ = env
    form = {"caption": "x", "lora": "/loras/a", "batch_size": 1}
    assert wait_job(c, c.post("/api/generate", json=form).json()["job"])["status"] == "done"
    loads = mock_ace.LORA["loads"]
    mock_ace.LORA.update(loaded=None, on=False)  # ACE-Step restarted behind our back
    assert wait_job(c, c.post("/api/generate", json=form).json()["job"])["status"] == "done"
    assert mock_ace.LORA["loads"] == loads + 1 and mock_ace.LORA["on"]


def test_import_folder_keeps_same_named_files(env, tmp_path):
    c, _ = env
    src = tmp_path / "music"
    (src / "album1").mkdir(parents=True)
    (src / "album2").mkdir()
    (src / "album1" / "01 Intro.mp3").write_bytes(b"one")
    (src / "album2" / "01 Intro.mp3").write_bytes(b"second")
    c.post("/api/datasets", json={"name": "imp", "trigger": "t"})
    assert c.post("/api/datasets/imp/import", json={"folder": str(src)}).json()["added"] == 2
    assert c.post("/api/datasets/imp/import", json={"folder": str(src)}).json()["added"] == 0  # re-import: no dupes
    names = sorted(s["filename"] for s in c.get("/api/datasets/imp").json()["samples"])
    assert names == ["01 Intro-2.mp3", "01 Intro.mp3"]
