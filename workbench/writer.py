"""Songwriter: turns an idea into a ready-to-generate song bundle.

Two targets:
  ace   Unfiltered. Writes exactly what you ask for and fills the ACE-Step form
        (caption, lyrics, BPM, key, time signature, duration, language).
  suno  Suno-safe. Writes a v5-style prompt (style / exclude / title / lyrics),
        rephrases anything Suno is likely to reject, then runs suno.sanitize().

Works with any provider in llm.py. With no LLM configured it falls back to a
rule-based draft so the button never just fails.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import llm, suno

ACE_SYSTEM = """You are a professional songwriter and producer writing for ACE-Step, a private, local AI music model.
There is no content filter here. Write exactly what the user asks for, including explicit language, adult, dark or controversial themes when the request calls for it. Never add disclaimers, warnings or moral commentary, and never water a request down.

Return ONE JSON object with these keys:
- "title": short evocative title (2-6 words).
- "caption": the sound description, the most important field. Comma-separated or short natural language covering genre/subgenre, mood, key instruments, timbre/texture words (warm, crisp, punchy, airy), production style, era, and vocal character (e.g. "breathy female vocal", "raspy male vocal"). 15-60 words. Do NOT put BPM or key in the caption. Avoid conflicting styles unless the user wants an evolution ("starts as soft piano, builds into metal").
- "lyrics": full lyrics with structure tags on their own lines: [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro], and dynamic tags like [Build], [Drop], [Breakdown], [Guitar Solo], [Instrumental]. Optional short modifiers: [Chorus - anthemic]. Keep 6-10 syllables per line, similar counts for matching lines, blank line between sections. UPPERCASE = shouted/intense, (parentheses) = backing vocals. One core metaphor; concrete images over vague filler. For an instrumental return "[Instrumental]".
- "bpm": integer 40-220 that suits the genre.
- "keyscale": e.g. "A minor", "F# major".
- "timesignature": "4" for 4/4, "3" for 3/4, "6" for 6/8, "2" for 2/4.
- "duration": song length in seconds (30-300) that fits the lyric length and tempo; a full song with intro/outro is usually 180-240.
- "vocal_language": ISO code like "en", "es", "ja", "zh", "fr", "de", "ko", "pt", "ru", "it".
- "instrumental": true or false.
- "notes": one sentence of generation advice.
Respond with the JSON object only."""

SUNO_SYSTEM = """You are a Suno AI prompt engineer (Suno v4.5 / v5 / v5.5). Write songs that Suno will accept without moderation errors.

Suno rules you must follow:
- NEVER name real artists, bands, producers or songs anywhere. Describe the sound instead (era, instruments, vocal timbre, production).
- No real people's names in lyrics.
- Suno's lyric moderation flags explicit sexual words, slurs and graphic violence. Keep the meaning and heat of the request but express it with innuendo, metaphor and suggestion instead of explicit words.
- Never copy existing lyrics.

Return ONE JSON object with these keys:
- "title": short evocative title (2-6 words).
- "style": 4-8 comma-separated descriptors in this order: genre/subgenre, tempo or energy feel, 1-2 key instruments, vocal style, production quality, mood. 15-40 words. No negatives here ("no drums" goes in exclude).
- "exclude": comma-separated elements to avoid, or "".
- "lyrics": full lyrics with Suno structure tags: [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro], [Instrumental Break]. Use parameterized tags for per-section control, e.g. [Chorus: full band, soaring vocals]. Singable, concrete, not generic. "" for an instrumental.
- "notes": one sentence of advice (model choice, Weirdness / Style Influence sliders).
Respond with the JSON object only."""

REVISE_TAIL = """
You are revising an existing song. Apply the user's instruction, keep everything else that works, and return the complete updated JSON object with the same keys."""

SUNO_REPHRASE_SYSTEM = SUNO_SYSTEM + """
You are converting an existing song to be Suno-safe. Keep its structure, story and feel. Replace artist names with sound descriptions, rephrase anything Suno would flag, and convert section tags to Suno format. Return the complete JSON object."""

ACE_FROM_SUNO_SYSTEM = ACE_SYSTEM + """
You are converting a Suno prompt bundle into ACE-Step fields. Turn the style into a rich caption, keep the lyrics (convert [Chorus: x] tags to [Chorus - x]), and choose bpm, keyscale, timesignature, duration and vocal_language. Return the complete JSON object."""

IDEA_SYSTEM = """You invent song concepts. Reply with ONE JSON object: {"idea": "<one or two sentences: a specific, vivid song concept with a genre, a mood, and a story or image>"}. Be surprising and specific. No disclaimers."""


def parse_json(text: str) -> dict:
    text = llm.strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    for candidate in (text, (re.search(r"\{.*\}", text, re.DOTALL) or [None])[0]):
        if not candidate:
            continue
        try:
            return json.loads(candidate, strict=False)  # strict=False allows raw newlines in lyrics
        except Exception:
            continue
    raise llm.LLMError(f"The model did not return valid JSON. First 300 chars: {text[:300]}")


def _int(v, lo, hi, default=None):
    try:
        n = int(round(float(v)))
        return max(lo, min(hi, n))
    except (TypeError, ValueError):
        return default


def normalize(raw: dict, target: str) -> dict:
    """Coerce whatever the model returned into the bundle shape the UI expects."""
    b = {
        "target": target,
        "title": str(raw.get("title", "")).strip(),
        "lyrics": str(raw.get("lyrics", "")).strip(),
        "notes": str(raw.get("notes", "")).strip(),
    }
    if target == "suno":
        b["style"] = str(raw.get("style") or raw.get("caption") or "").strip()
        b["exclude"] = str(raw.get("exclude", "")).strip()
    else:
        b["caption"] = str(raw.get("caption") or raw.get("style") or "").strip()
        b["bpm"] = _int(raw.get("bpm"), 30, 300)
        b["keyscale"] = str(raw.get("keyscale") or raw.get("key") or "").strip()
        ts = str(raw.get("timesignature") or raw.get("time_signature") or "").strip()
        b["timesignature"] = ts.split("/")[0] if ts else ""
        b["duration"] = _int(raw.get("duration"), 10, 600)
        b["vocal_language"] = str(raw.get("vocal_language") or raw.get("language") or "").strip().lower()[:5]
        instr = raw.get("instrumental")
        b["instrumental"] = bool(instr) if isinstance(instr, bool) else str(instr).lower() == "true"
        if b["instrumental"] and not b["lyrics"]:
            b["lyrics"] = "[Instrumental]"
    return b


def _brief(idea: str, opts: dict) -> str:
    parts = [f"Song idea: {idea.strip() or 'surprise me'}"]
    labels = {
        "genre": "Genre direction", "mood": "Mood", "vocals": "Vocals", "language": "Lyric language",
        "length": "Length", "structure": "Structure", "theme_notes": "Extra notes",
    }
    for k, label in labels.items():
        v = str(opts.get(k, "") or "").strip()
        if v and v != "auto":
            parts.append(f"{label}: {v}")
    if opts.get("vocals") == "instrumental":
        parts.append("This is an INSTRUMENTAL track: no sung lyrics.")
    if opts.get("explicit"):
        parts.append("Explicit language is welcome.")
    return "\n".join(parts)


def write_song(cfg: llm.LLMConfig | None, idea: str, target: str, opts: dict, base: Path) -> dict:
    target = "suno" if target == "suno" else "ace"
    if cfg is None:
        return finish(rules_draft(idea, target, opts), target, base, opts)
    system = SUNO_SYSTEM if target == "suno" else ACE_SYSTEM
    raw = parse_json(llm.chat(cfg, system, _brief(idea, opts)))
    return finish(normalize(raw, target), target, base, opts, source=f"{cfg.preset}:{cfg.model or cfg.kind}")


def revise(cfg: llm.LLMConfig, bundle: dict, instruction: str, base: Path, opts: dict | None = None) -> dict:
    target = bundle.get("target", "ace")
    system = (SUNO_SYSTEM if target == "suno" else ACE_SYSTEM) + REVISE_TAIL
    user = f"Current song JSON:\n{json.dumps(_strip_meta(bundle), ensure_ascii=False, indent=2)}\n\nInstruction: {instruction}"
    raw = parse_json(llm.chat(cfg, system, user))
    return finish(normalize(raw, target), target, base, opts or {}, source=f"{cfg.preset}:{cfg.model or cfg.kind}")


def convert(cfg: llm.LLMConfig | None, bundle: dict, to_target: str, base: Path, opts: dict | None = None) -> dict:
    """Move a song between targets. Suno conversion rephrases; ACE conversion restores detail."""
    opts = opts or {}
    if to_target == "suno":
        if cfg is None:
            return finish(normalize(suno.to_suno_from_ace(bundle), "suno"), "suno", base, opts)
        user = f"Song JSON to make Suno-safe:\n{json.dumps(_strip_meta(bundle), ensure_ascii=False, indent=2)}"
        raw = parse_json(llm.chat(cfg, SUNO_REPHRASE_SYSTEM, user))
        return finish(normalize(raw, "suno"), "suno", base, opts, source=f"{cfg.preset}:{cfg.model or cfg.kind}")
    if cfg is None:
        b = normalize({**bundle, "caption": bundle.get("style", ""), "lyrics": suno.suno_tags_to_ace(bundle.get("lyrics", ""))}, "ace")
        return finish(b, "ace", base, opts)
    user = f"Suno bundle to convert:\n{json.dumps(_strip_meta(bundle), ensure_ascii=False, indent=2)}"
    raw = parse_json(llm.chat(cfg, ACE_FROM_SUNO_SYSTEM, user))
    return finish(normalize(raw, "ace"), "ace", base, opts, source=f"{cfg.preset}:{cfg.model or cfg.kind}")


PHRASE_SYSTEM = """You are a vocal coach for ACE-Step, an AI singer that reads lyrics as plain text.
The singer can only be steered through the text itself:
- phonetic respelling for words it mispronounces ("Siobhan" -> "Shiv-awn", "live" (verb) -> "liv", "read" (past) -> "red")
- hyphenating syllables to slow or separate them ("beau-ti-ful")
- stretching vowels to hold a note ("lo-o-ove", "feeeel") - use sparingly
- UPPERCASE for shouted or intense delivery, (parentheses) for backing vocals or echoes
- short inline tags on their own line before the line: [whispered], [falsetto], [spoken word], [powerful belting], [raspy vocal], [harmonies], [ad-lib]
- punctuation for phrasing: commas and "..." add breaths and pauses, "!" adds punch
Given a lyric line and what the user wants, reply with ONE JSON object:
{"options": [{"line": "<rewritten line, may start with a tag line followed by \\n>", "why": "<a few words>"}, ... 3 options]}
Keep the meaning and syllable count close to the original unless asked otherwise. No disclaimers."""


def phrase_options(cfg: llm.LLMConfig, line: str, wish: str, context: str = "") -> list[dict]:
    user = f"Lyric line: {line}\nWhat I want: {wish or 'make it sing more clearly'}"
    if context:
        user += f"\nSurrounding lyrics for context:\n{context[:1500]}"
    raw = parse_json(llm.chat(cfg, PHRASE_SYSTEM, user))
    opts = raw.get("options") or []
    return [{"line": str(o.get("line", "")).strip(), "why": str(o.get("why", "")).strip()} for o in opts if isinstance(o, dict) and o.get("line")]


def surprise_idea(cfg: llm.LLMConfig | None) -> str:
    if cfg is not None:
        try:
            return str(parse_json(llm.chat(cfg, IDEA_SYSTEM, f"Give me a fresh idea. Random seed {random.randint(1, 10**6)}.")).get("idea", "")).strip()
        except llm.LLMError:
            pass
    return random.choice(_IDEAS)


def finish(b: dict, target: str, base: Path, opts: dict, source: str = "rules") -> dict:
    b["source"] = source
    b["changes"] = []
    if target == "suno":
        b, changes = suno.sanitize(b, base, soften_words=opts.get("soften", True))
        b["changes"] = changes
    return b


def _strip_meta(b: dict) -> dict:
    return {k: v for k, v in b.items() if k not in ("source", "changes", "target")}


# ── rule-based fallback (no LLM configured) ───────────────────────────

_GENRES = {
    "house": ("deep house, four-on-the-floor groove, punchy kick, warm analog bass, club-ready polished mix", 124, "A minor"),
    "techno": ("melodic techno, driving hypnotic energy, rolling synth arpeggios, rumbling sub bass, dark warehouse production", 128, "F minor"),
    "trance": ("uplifting trance, euphoric build, supersaw leads, rolling bassline, festival mainstage production", 138, "D minor"),
    "drum and bass": ("liquid drum and bass, fast rolling breakbeats, deep sub bass, airy pads, crisp modern production", 174, "E minor"),
    "dnb": ("liquid drum and bass, fast rolling breakbeats, deep sub bass, airy pads, crisp modern production", 174, "E minor"),
    "dubstep": ("melodic dubstep, heavy half-time drops, growling bass, cinematic synths, loud modern mix", 140, "F minor"),
    "rock": ("alternative rock, driving rhythm, distorted electric guitars, live drums, big room production", 120, "E minor"),
    "metal": ("melodic metal, aggressive energy, chugging guitars, double kick drums, heavy modern production", 150, "D minor"),
    "hip hop": ("boom bap hip hop, head-nodding groove, dusty drum breaks, deep 808 bass, gritty analog production", 90, "C minor"),
    "trap": ("modern trap, hard-hitting bounce, 808 bass, crisp rolling hi-hats, dark atmospheric pads", 140, "G minor"),
    "folk": ("indie folk, gentle sway, fingerpicked acoustic guitar, soft strings, warm intimate production", 96, "G major"),
    "country": ("modern country, easygoing groove, twangy telecaster, brushed drums, polished Nashville production", 100, "D major"),
    "jazz": ("smooth jazz, laid-back swing, upright bass, brushed drums, warm saxophone, analog warmth", 92, "Bb major"),
    "lofi": ("lo-fi hip hop, relaxed downtempo groove, dusty piano, vinyl crackle, hazy bedroom production", 80, "F major"),
    "ambient": ("ambient electronic, slow drifting textures, evolving pads, field recordings, spacious cinematic mix", 70, "C major"),
    "synthwave": ("synthwave, steady retro pulse, analog synth arpeggios, gated reverb drums, neon 80s production", 108, "A minor"),
    "orchestral": ("cinematic orchestral, sweeping dynamic swells, full string section, brass and timpani, epic film-score production", 90, "D minor"),
    "r&b": ("contemporary R&B, slow sensual groove, silky electric piano, sub bass, smooth modern production", 85, "Eb major"),
    "pop": ("modern pop, catchy mid-tempo bounce, bright synths, tight drums, radio-ready polished production", 118, "C major"),
    "funk": ("funk, tight syncopated groove, slap bass, wah guitar and horns, punchy vintage production", 110, "E minor"),
}
_VOCALS = {
    "female": "expressive female vocal", "male": "powerful male vocal", "duet": "male and female duet vocals",
    "choir": "layered choir vocals", "rap": "confident rap vocal", "instrumental": "", "auto": "expressive lead vocal",
}
_IDEAS = [
    "A synthwave night drive where the city lights are the only ones who remember your name.",
    "A sea shanty sung by lighthouse keepers who have never seen a ship.",
    "Lo-fi hip hop about the last coffee shop open at 4am and the stranger who always orders tea.",
    "An outlaw country ballad about a robot who learned to lie.",
    "Melodic techno for a rave inside a cathedral, slow-building to a choir drop.",
    "A breakup song told entirely through the houseplants left behind.",
    "Uplifting trance about the first sunrise seen from a space station.",
    "A swaggering funk track about a cat who runs the neighborhood.",
]


def rules_draft(idea: str, target: str, opts: dict) -> dict:
    text = f"{opts.get('genre', '')} {idea}".lower()
    caption, bpm, key = _GENRES["pop"]
    for kw, spec in _GENRES.items():
        if kw in text:
            caption, bpm, key = spec
            break
    vocals = opts.get("vocals", "auto") or "auto"
    vocal = _VOCALS.get(vocals, _VOCALS["auto"])
    mood = (opts.get("mood") or "").strip()
    desc = ", ".join(p for p in (caption, vocal, mood) if p)
    title = " ".join(w.capitalize() for w in re.findall(r"[A-Za-z']+", idea)[:4]) or "Untitled"
    if vocals == "instrumental":
        lyrics = "[Intro]\n\n[Build]\n\n[Drop]\n\n[Breakdown]\n\n[Drop]\n\n[Outro]" if target == "ace" else ""
    else:
        lyrics = (
            "[Intro]\n\n[Verse 1]\n(write about: " + (idea or "your idea") + ")\n\n"
            "[Pre-Chorus]\n\n[Chorus]\n\n[Verse 2]\n\n[Chorus]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]"
        )
    notes = "Rule-based draft: no AI writer is set up. Add one in Settings for real lyrics."
    if target == "suno":
        return {"target": "suno", "title": title, "style": desc, "exclude": "", "lyrics": suno.ace_tags_to_suno(lyrics), "notes": notes}
    return {
        "target": "ace", "title": title, "caption": desc, "lyrics": lyrics, "bpm": bpm, "keyscale": key,
        "timesignature": "4", "duration": 180, "vocal_language": "en", "instrumental": vocals == "instrumental", "notes": notes,
    }
