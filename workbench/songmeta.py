"""Workbench settings embedded in exported audio, so a downloaded song can be dropped back in later.

MP3: an ID3v2 TXXX frame. WAV: an "acwb" RIFF chunk. FLAC: an APPLICATION block with id "ACWB".
Other formats are exported unchanged. The payload is ASCII JSON. Embedding never touches the
audio data itself, and anything we cannot parse safely is left as it was.
"""

from __future__ import annotations

import json
import struct

DESC = "ACE-Step Workbench"
WAV_CHUNK = b"acwb"
FLAC_APP = b"ACWB"


def _payload(meta: dict) -> bytes:
    return json.dumps(meta, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def _decode(raw: bytes) -> dict | None:
    try:
        meta = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def embed(data: bytes, meta: dict) -> bytes:
    """Return `data` with `meta` embedded, or `data` unchanged if the format is not supported."""
    try:
        if data[:3] == b"ID3" or _is_mpeg(data):
            return _mp3_embed(data, _payload(meta))
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return _wav_embed(data, _payload(meta))
        if data[:4] == b"fLaC":
            return _flac_embed(data, _payload(meta))
    except (struct.error, IndexError, ValueError):
        pass
    return data


def read(data: bytes) -> dict | None:
    """Embedded Workbench settings, or None."""
    try:
        if data[:3] == b"ID3":
            raw = _mp3_read(data)
        elif data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            raw = _wav_read(data)
        elif data[:4] == b"fLaC":
            raw = _flac_read(data)
        else:
            raw = None
    except (struct.error, IndexError, ValueError):
        raw = None
    return _decode(raw) if raw else None


# ── MP3 (ID3v2.3 / 2.4) ──

def _is_mpeg(data: bytes) -> bool:
    return len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _syncsafe(n: int) -> bytes:
    return bytes(((n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F))


def _unsyncsafe(b: bytes) -> int:
    return (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]


def _id3_frames(data: bytes):
    """(major version, list of (frame id, raw frame bytes), end of tag) or None if the tag is unusual."""
    major, flags = data[3], data[5]
    if major not in (3, 4) or flags & 0xF0:  # unsynchronisation / extended header / footer: leave alone
        return None
    end = 10 + _unsyncsafe(data[6:10])
    frames, pos = [], 10
    while pos + 10 <= end and data[pos] != 0:
        fid = data[pos:pos + 4]
        size = _unsyncsafe(data[pos + 4:pos + 8]) if major == 4 else struct.unpack(">I", data[pos + 4:pos + 8])[0]
        frames.append((fid, data[pos:pos + 10 + size]))
        pos += 10 + size
    return major, frames, end


def _is_ours(frame: bytes) -> bool:
    return frame[:4] == b"TXXX" and frame[11:11 + len(DESC) + 1] == DESC.encode("ascii") + b"\x00"


def _mp3_embed(data: bytes, payload: bytes) -> bytes:
    if data[:3] == b"ID3":
        parsed = _id3_frames(data)
        if parsed is None:
            return data
        major, frames, end = parsed
        audio = data[end:]
    else:
        major, frames, audio = 4, [], data
    body = b"\x00" + DESC.encode("ascii") + b"\x00" + payload  # encoding 0 (ISO-8859-1): payload is ASCII
    size = _syncsafe(len(body)) if major == 4 else struct.pack(">I", len(body))
    kept = b"".join(f for _, f in frames if not _is_ours(f))
    tag = kept + b"TXXX" + size + b"\x00\x00" + body
    return b"ID3" + bytes((major, 0, 0)) + _syncsafe(len(tag)) + tag + audio


def _mp3_read(data: bytes) -> bytes | None:
    parsed = _id3_frames(data)
    if parsed is None:
        return None
    for _, f in parsed[1]:
        if _is_ours(f):
            return f[11 + len(DESC) + 1:]
    return None


# ── WAV ──

def _wav_chunks(data: bytes):
    pos = 12
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        yield cid, pos, pos + 8 + size + (size & 1)
        pos += 8 + size + (size & 1)


def _wav_embed(data: bytes, payload: bytes) -> bytes:
    kept = b"".join(data[s:e] for cid, s, e in _wav_chunks(data) if cid != WAV_CHUNK)
    chunk = WAV_CHUNK + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")
    body = b"WAVE" + kept + chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav_read(data: bytes) -> bytes | None:
    for cid, s, _ in _wav_chunks(data):
        if cid == WAV_CHUNK:
            size = struct.unpack("<I", data[s + 4:s + 8])[0]
            return data[s + 8:s + 8 + size]
    return None


# ── FLAC ──

def _flac_blocks(data: bytes):
    """[(type, block body)] and the offset where audio frames start."""
    blocks, pos = [], 4
    while True:
        head = data[pos]
        size = int.from_bytes(data[pos + 1:pos + 4], "big")
        blocks.append((head & 0x7F, data[pos + 4:pos + 4 + size]))
        pos += 4 + size
        if head & 0x80:
            return blocks, pos


def _flac_embed(data: bytes, payload: bytes) -> bytes:
    blocks, audio = _flac_blocks(data)
    blocks = [b for b in blocks if not (b[0] == 2 and b[1][:4] == FLAC_APP)]
    blocks.append((2, FLAC_APP + payload))
    out = bytearray(b"fLaC")
    for i, (btype, body) in enumerate(blocks):
        out += bytes(((0x80 if i == len(blocks) - 1 else 0) | btype,)) + len(body).to_bytes(3, "big") + body
    return bytes(out) + data[audio:]


def _flac_read(data: bytes) -> bytes | None:
    blocks, _ = _flac_blocks(data)
    for btype, body in blocks:
        if btype == 2 and body[:4] == FLAC_APP:
            return body[4:]
    return None
