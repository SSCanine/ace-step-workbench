"""Training queue: several models per dataset, GPU gate, cancel/retry/order, restart, dataset import, model-aware LoRAs."""

import json
import textwrap
import time
from pathlib import Path

import pytest

from tests.test_e2e import env, free_port, wait_job  # noqa: F401  (fixture)
from workbench import config, trainqueue

FAKE_TRAIN = textwrap.dedent("""
    import sys, pathlib, json
    a = sys.argv
    def arg(k): return a[a.index(k) + 1]
    if "--preprocess" in a:
        ds = json.load(open(arg("--dataset-json"), encoding="utf-8"))
        out = pathlib.Path(arg("--tensor-output"))
        for s in ds["samples"]:
            (out / (s["filename"] + ".pt")).write_bytes(b"t")
    else:
        ep = int(arg("--epochs"))
        for e in range(1, ep + 1):
            print(f"[OK] Epoch {e}/{ep} in 0.01s, Loss: {1.0/e:.4f}", flush=True)
        final = pathlib.Path(arg("--output-dir")) / "final"
        final.mkdir(parents=True)
        (final / "adapter_config.json").write_text(json.dumps({"variant": arg("--model-variant")}))
""")


@pytest.fixture
def q(env, monkeypatch):
    """A labeled dataset, turbo + XL turbo installed, no ACE-Step server in the way, gate off, no real GPU."""
    c, fake_ace = env
    (fake_ace / "train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    for m in ("acestep-v15-turbo", "acestep-v15-xl-turbo"):
        (fake_ace / "checkpoints" / m).mkdir(parents=True)
    c.post("/api/settings", json={"ace": {"api_url": f"http://127.0.0.1:{free_port()}"},
                                  "training": {"gate": {"enabled": False}}})
    monkeypatch.setattr(trainqueue, "gpu_stats", lambda: None)
    c.post("/api/datasets", json={"name": "my set", "trigger": "zxsynth"})
    c.post("/api/datasets/my set/files", files=[("files", ("a.mp3", b"aaa", "audio/mpeg")), ("files", ("b.wav", b"bbb", "audio/wav"))])
    c.put("/api/datasets/my set", json={"samples": [{"filename": "a.mp3", "caption": "warm synthwave", "is_instrumental": True},
                                                    {"filename": "b.wav", "caption": "dark synth", "is_instrumental": True}]})
    return c


def wait_queue(c, until, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = c.get("/api/train/queue").json()["jobs"]
        if until(jobs):
            return jobs
        time.sleep(0.2)
    raise AssertionError(f"queue never settled: {jobs}")


def test_queue_trains_each_model_in_turn(q):
    c = q
    r = c.post("/api/train/queue", json={"dataset": "my set", "variants": ["turbo", "xl-turbo"], "preset": "quick",
                                         "opts": {"epochs": 2}, "test_clip": False})
    assert r.status_code == 200, r.text
    assert [(j["run_name"], j["model_label"]) for j in r.json()["jobs"]] == [("my set-turbo", "Turbo"), ("my set-xl-turbo", "XL Turbo")]
    jobs = wait_queue(c, lambda js: all(j["status"] in trainqueue.FINISHED for j in js))
    assert [j["status"] for j in jobs] == ["done", "done"], jobs
    assert jobs[0]["finished"] <= jobs[1]["started"]  # one after another
    loras = {lo["label"]: lo for lo in c.get("/api/loras").json()["loras"]}
    assert loras["my set-turbo/final"]["model"] == "turbo" and loras["my set-xl-turbo/final"]["model_label"] == "XL Turbo"
    assert loras["my set-xl-turbo/final"]["trigger"] == "zxsynth" and loras["my set-xl-turbo/final"]["model_id"] == "acestep-v15-xl-turbo"
    saved = json.loads((config.DATA / "train_queue.json").read_text(encoding="utf-8"))["jobs"]
    assert [j["status"] for j in saved] == ["done", "done"]
    # same dataset + model again: a new run name, the first LoRA is never overwritten
    again = c.post("/api/train/queue", json={"dataset": "my set", "variants": ["turbo"], "test_clip": False}).json()["jobs"]
    assert again[0]["run_name"] == "my set-turbo-v2"
    c.post(f"/api/train/queue/{again[0]['id']}/cancel")


def test_queue_refuses_models_not_downloaded(q):
    r = q.post("/api/train/queue", json={"dataset": "my set", "variants": ["xl_base"]})
    assert r.status_code == 400 and "Not downloaded yet: XL Base" in r.json()["detail"]
    r = q.post("/api/train/queue", json={"dataset": "nope", "variants": ["turbo"]})
    assert r.status_code >= 400


def test_gate_waits_then_start_now_cancel_retry_order(q, monkeypatch):
    c = q
    c.post("/api/settings", json={"training": {"gate": {"enabled": True, "idle_minutes": 10}}})
    monkeypatch.setattr(trainqueue.TrainQueue, "_busy", lambda self, gate: "ComfyUI at http://x is rendering")
    a, b = (c.post("/api/train/queue", json={"dataset": "my set", "variants": [v], "opts": {"epochs": 1},
                                             "test_clip": False}).json()["jobs"][0] for v in ("turbo", "xl_turbo"))
    jobs = wait_queue(c, lambda js: js[0]["status"] == "waiting")
    assert jobs[0]["stage"] == "Waiting: ComfyUI at http://x is rendering"
    assert c.get("/api/status").json()["queue"]["active"] is True

    c.post(f"/api/train/queue/{b['id']}/up")  # can't jump ahead of the running job
    assert [j["id"] for j in c.get("/api/train/queue").json()["jobs"]] == [a["id"], b["id"]]
    c.post(f"/api/train/queue/{b['id']}/cancel")
    c.post(f"/api/train/queue/{a['id']}/now")  # skip the gate for this one
    jobs = wait_queue(c, lambda js: js[0]["status"] == "done")
    assert jobs[1]["status"] == "cancelled"

    c.post(f"/api/train/queue/{b['id']}/retry")
    wait_queue(c, lambda js: js[1]["status"] == "waiting")
    c.post(f"/api/train/queue/{b['id']}/cancel")  # cancelling the job that is waiting
    jobs = wait_queue(c, lambda js: js[1]["status"] == "cancelled")
    c.post("/api/train/queue/clear")
    assert c.get("/api/train/queue").json()["jobs"] == []


def test_interrupted_job_starts_again_after_restart(env):
    config.DATA.mkdir(parents=True, exist_ok=True)
    (config.DATA / "train_queue.json").write_text(json.dumps({"jobs": [
        {"id": "x1", "status": "training", "stage": "Training"}, {"id": "x2", "status": "done", "stage": "Done"}]}))
    tq = trainqueue.TrainQueue(None, None, None)
    tq.load()
    assert [(j["status"], j["stage"]) for j in tq.items] == [
        ("queued", "Interrupted by a restart; it starts again"), ("done", "Done")]


def test_import_labeled_dataset(env, tmp_path):
    c, _ = env
    src = tmp_path / "rezz-v1"
    src.mkdir()
    (src / "one.mp3").write_bytes(b"1")
    elsewhere = tmp_path / "elsewhere.flac"
    elsewhere.write_bytes(b"2")
    (src / "dataset.json").write_text(json.dumps({
        "metadata": {"name": "rezz-v1", "custom_tag": "rezzstyle", "all_instrumental": True},
        "samples": [  # values as strings, the way some tools write them
            {"filename": "one.mp3", "audio_path": str(tmp_path / "moved" / "one.mp3"), "caption": "dark mid-tempo",
             "bpm": "85", "keyscale": "Ab major", "timesignature": "4", "duration": "213", "is_instrumental": "True",
             "lyrics": "[Instrumental]"},
            {"filename": "elsewhere.flac", "audio_path": str(elsewhere), "caption": "witchy", "bpm": 90.0,
             "is_instrumental": "False", "lyrics": "[Verse]\nla"}]}))
    r = c.post("/api/dataset-import", json={"folder": str(src)})
    assert r.json() == {"name": "rezz-v1", "added": 2}
    ds = c.get("/api/datasets/rezz-v1").json()
    s = {x["filename"]: x for x in ds["samples"]}
    assert ds["metadata"]["custom_tag"] == "rezzstyle" and ds["metadata"]["all_instrumental"] is True
    assert s["one.mp3"]["bpm"] == 85 and s["one.mp3"]["is_instrumental"] is True and s["one.mp3"]["duration"] == 213
    assert s["elsewhere.flac"]["caption"] == "witchy" and s["elsewhere.flac"]["is_instrumental"] is False
    assert c.post("/api/dataset-import", json={"folder": str(src)}).status_code == 400  # already exists
    assert "No dataset.json" in c.post("/api/dataset-import", json={"folder": str(tmp_path)}).json()["detail"]


def _xl_lora() -> Path:
    """A LoRA trained elsewhere (Side-Step layout): variant in training_config.json, trigger in style_card.json."""
    run = config.DATA / "loras" / "lp-v1"
    ckpt = run / "checkpoints" / "epoch_250"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (run / "training_config.json").write_text(json.dumps({"model_variant": "xl-turbo"}))
    (run / "style_card.json").write_text(json.dumps({"trigger": "lpnumetal"}))
    return ckpt


def test_lora_knows_its_model_and_trigger(env):
    c, _ = env
    _xl_lora()
    lo = next(x for x in c.get("/api/loras").json()["loras"] if x["label"] == "lp-v1/checkpoints/epoch_250")
    assert (lo["model"], lo["model_label"], lo["model_id"], lo["trigger"]) == ("xl_turbo", "XL Turbo", "acestep-v15-xl-turbo", "lpnumetal")


def test_xl_lora_on_a_turbo_server_explains(env):
    c, _ = env
    path = _xl_lora()
    j = wait_job(c, c.post("/api/generate", json={"caption": "x", "lora": str(path.resolve())}).json()["job"])
    assert j["status"] == "error" and "made for XL Turbo" in j["error"] and "acestep-v15-turbo" in j["error"]


def test_test_clip_form():
    vocal = {"metadata": {"custom_tag": "lpnumetal"}, "samples": [
        {"caption": "nu-metal anthem", "is_instrumental": False}, {"caption": "ballad", "is_instrumental": False}]}
    f = trainqueue.test_clip_form(vocal, "xl_turbo", Path("x"), "t")
    assert f["caption"] == "lpnumetal, nu-metal anthem" and f["lyrics"] == trainqueue.TEST_LYRICS and f["inference_steps"] == 8
    inst = {"metadata": {"custom_tag": "", "all_instrumental": True}, "samples": [{"caption": "dark", "is_instrumental": True}]}
    f = trainqueue.test_clip_form(inst, "base", Path("x"), "t")
    assert f["lyrics"] == "[Instrumental]" and f["instrumental"] and f["inference_steps"] == 50 and f["shift"] == 1.0
