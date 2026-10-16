"""Suno-safe post-processing.

Suno rejects or rewrites prompts that name real artists, and its lyric
moderation flags some explicit words. The LLM rephrase pass in writer.py does
most of the work; this module is the deterministic safety net that runs after
it and reports every change it makes.

Lists live in data/suno_blocklist.txt (names to strip) and
data/suno_soften.json (word -> replacement) so you can edit them without code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STYLE_MAX = 1000
LYRICS_MAX = 5000
TITLE_MAX = 80
EXCLUDE_MAX = 1000

DEFAULT_SOFTEN = {
    # Starter list. Edit data/suno_soften.json to add your own.
    # Keys are matched as whole words, case-insensitive.
    "fuck": "funk",
    "fucking": "freaking",
    "motherfucker": "mother-trucker",
    "shit": "stuff",
    "bitch": "witch",
    "cunt": "punk",
    "cock": "rock",
    "dick": "trick",
    "pussy": "kitty",
}

# Phrases that introduce an artist reference in a style prompt.
_REF_PATTERNS = [
    r"\b(?:in the style of|inspired by|reminiscent of|sounds? like|similar to|a la|à la|influenced by|channeling|in the vein of)\s+[^,;.\n]+",
    r"\b(?:like)\s+(?:[A-Z][\w'&.-]*\s?){1,4}(?=[,;.\n]|$)",
    r"\b(?:[A-Z][\w'&.]*\s){0,3}[A-Z][\w'&.]*(?:-esque|\s+type beat)\b",
]


def _data_dir(base: Path) -> Path:
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_lists(base: Path) -> tuple[list[str], dict]:
    d = _data_dir(base)
    bl = d / "suno_blocklist.txt"
    if not bl.exists():
        bl.write_text(
            "# One artist, band or person name per line. They are removed from Suno style prompts\n"
            "# and flagged in lyrics. Lines starting with # are ignored.\n",
            encoding="utf-8",
        )
    names = [ln.strip() for ln in bl.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    sf = d / "suno_soften.json"
    if not sf.exists():
        sf.write_text(json.dumps(DEFAULT_SOFTEN, indent=2), encoding="utf-8")
    try:
        soften = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        soften = dict(DEFAULT_SOFTEN)
    return names, soften


def _tidy_list(s: str) -> str:
    parts = [p.strip(" -") for p in re.split(r",", s)]
    return ", ".join(p for p in parts if p)


def ace_tags_to_suno(lyrics: str) -> str:
    """[Chorus - anthemic] -> [Chorus: anthemic] (Suno's parameterized tag style)."""
    return re.sub(r"\[([A-Za-z][\w ]*?)\s+-\s+([^\]]+)\]", r"[\1: \2]", lyrics)


def suno_tags_to_ace(lyrics: str) -> str:
    """[Chorus: anthemic] -> [Chorus - anthemic] (ACE-Step's tag style)."""
    return re.sub(r"\[([A-Za-z][\w ]*?):\s*([^\]]+)\]", r"[\1 - \2]", lyrics)


def sanitize(bundle: dict, base: Path, soften_words: bool = True) -> tuple[dict, list[str]]:
    """Make a song bundle Suno-safe. Returns (new_bundle, list_of_changes)."""
    names, soften = load_lists(base)
    out = dict(bundle)
    changes: list[str] = []

    style = out.get("style", "") or ""
    before = style
    for pat in _REF_PATTERNS:
        style = re.sub(pat, "", style)
    for n in names:
        style = re.sub(rf"\b{re.escape(n)}\b", "", style, flags=re.IGNORECASE)
    style = _tidy_list(style)
    if style != _tidy_list(before):
        changes.append("Removed artist references from the style prompt.")
    if len(style) > STYLE_MAX:
        style = style[:STYLE_MAX].rsplit(",", 1)[0]
        changes.append(f"Trimmed style to Suno's {STYLE_MAX}-character limit.")
    out["style"] = style

    lyrics = ace_tags_to_suno(out.get("lyrics", "") or "")
    if lyrics != (out.get("lyrics", "") or ""):
        changes.append("Converted section tags to Suno format ([Chorus: ...]).")
    for n in names:
        if re.search(rf"\b{re.escape(n)}\b", lyrics, flags=re.IGNORECASE):
            changes.append(f"Lyrics mention '{n}' (on your blocklist). Suno may reject this; consider rewording.")
    if soften_words:
        hits = []
        for word, repl in soften.items():
            rx = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            if rx.search(lyrics):
                hits.append(word)
                lyrics = rx.sub(lambda m, r=repl: r.upper() if m.group(0).isupper() else r, lyrics)
        if hits:
            changes.append(f"Softened {len(hits)} word(s) Suno tends to flag: {', '.join(sorted(hits))}.")
    if len(lyrics) > LYRICS_MAX:
        lyrics = lyrics[:LYRICS_MAX].rsplit("\n", 1)[0]
        changes.append(f"Trimmed lyrics to Suno's {LYRICS_MAX}-character limit.")
    out["lyrics"] = lyrics

    out["title"] = (out.get("title", "") or "")[:TITLE_MAX]
    out["exclude"] = _tidy_list(out.get("exclude", "") or "")[:EXCLUDE_MAX]
    return out, changes


def to_suno_from_ace(bundle: dict) -> dict:
    """Map an ACE-Step bundle (caption/bpm/key) to Suno fields without an LLM."""
    style = bundle.get("style") or bundle.get("caption", "")
    extras = []
    if bundle.get("bpm"):
        extras.append(f"{bundle['bpm']} BPM")
    if bundle.get("keyscale"):
        extras.append(str(bundle["keyscale"]))
    if extras:
        style = f"{style}, {', '.join(extras)}" if style else ", ".join(extras)
    return {**bundle, "style": style, "exclude": bundle.get("exclude", ""), "target": "suno"}
