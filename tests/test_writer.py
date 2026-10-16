import json
from pathlib import Path

import pytest

from workbench import llm, suno, writer


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path


def test_parse_json_handles_think_fences_and_raw_newlines():
    text = '<think>planning...</think>\n```json\n{"title": "X", "lyrics": "[Verse 1]\nline one\nline two"}\n```'
    out = writer.parse_json(text)
    assert out["title"] == "X"
    assert out["lyrics"].startswith("[Verse 1]\nline one")


def test_parse_json_raises_on_garbage():
    with pytest.raises(llm.LLMError):
        writer.parse_json("no json here")


def test_normalize_ace_coerces_fields():
    b = writer.normalize({"caption": "dark techno", "bpm": "128.4", "timesignature": "4/4",
                          "duration": 9999, "vocal_language": "EN", "instrumental": "true"}, "ace")
    assert b["bpm"] == 128 and b["timesignature"] == "4" and b["duration"] == 600
    assert b["vocal_language"] == "en" and b["instrumental"] is True and b["lyrics"] == "[Instrumental]"


def test_sanitize_strips_artist_refs_and_converts_tags(base):
    (base / "data").mkdir()
    (base / "data" / "suno_blocklist.txt").write_text("Daft Punk\n", encoding="utf-8")
    bundle = {"style": "french house, in the style of Daft Punk, vocoder vocals, Drake type beat, warm analog",
              "lyrics": "[Chorus - anthemic]\nThis shit is fire, Daft Punk on the radio", "title": "T", "exclude": ""}
    out, changes = suno.sanitize(bundle, base)
    assert "Daft Punk" not in out["style"] and "Drake" not in out["style"]
    assert out["style"].startswith("french house") and "warm analog" in out["style"]
    assert out["lyrics"].startswith("[Chorus: anthemic]")
    assert "stuff" in out["lyrics"]
    assert any("blocklist" in c for c in changes)
    assert any("Softened" in c for c in changes)


def test_sanitize_keeps_ordinary_style_words(base):
    out, changes = suno.sanitize({"style": "retro style synthwave, Lo-Fi texture, dreamy", "lyrics": ""}, base)
    assert out["style"] == "retro style synthwave, Lo-Fi texture, dreamy"
    assert changes == []


def test_soften_can_be_disabled(base):
    out, _ = suno.sanitize({"style": "punk", "lyrics": "shit"}, base, soften_words=False)
    assert out["lyrics"] == "shit"


def test_tag_roundtrip():
    assert suno.suno_tags_to_ace(suno.ace_tags_to_suno("[Bridge - whispered]")) == "[Bridge - whispered]"


def test_rules_fallback_both_targets(base):
    ace = writer.write_song(None, "a techno track about rain", "ace", {}, base)
    assert ace["bpm"] == 128 and "techno" in ace["caption"] and ace["source"] == "rules"
    s = writer.write_song(None, "a techno track about rain", "suno", {"vocals": "female"}, base)
    assert "techno" in s["style"] and "female" in s["style"]


def test_write_song_with_fake_llm(monkeypatch, base):
    reply = {"title": "Neon Rain", "caption": "melodic techno", "lyrics": "[Verse 1]\nhi", "bpm": 126,
             "keyscale": "F minor", "timesignature": "4", "duration": 200, "vocal_language": "en", "instrumental": False}
    monkeypatch.setattr(llm, "chat", lambda cfg, system, user: json.dumps(reply))
    cfg = llm.LLMConfig()
    b = writer.write_song(cfg, "rain", "ace", {"explicit": True}, base)
    assert b["title"] == "Neon Rain" and b["bpm"] == 126 and b["source"].startswith("ollama")


def test_strip_thinking_without_opening_tag():
    assert llm.strip_thinking("reasoning here</think>{\"a\":1}") == '{"a":1}'
