"""Small, idempotent fixes for known bugs in the pinned ACE-Step 1.5 commit.

Each patch checks the file first and only edits it when the bug is still there,
so running it on a fixed (newer) ACE-Step is a no-op.
"""

from __future__ import annotations

import re
from pathlib import Path

ARGS_PY = Path("acestep") / "training_v2" / "cli" / "args.py"

# ACE-Step's trainer CLI (training_v2/cli/args.py) maps --model-variant to a
# checkpoint folder. Its VARIANT_DIR_MAP lists only turbo/base/sft, while the
# model loader also knows the XL (4B) variants, so `train.py fixed
# --model-variant xl_turbo` fails with "Model directory not found ... xl_turbo".
XL_ENTRIES = {
    "xl_turbo": "acestep-v15-xl-turbo",
    "xl_base": "acestep-v15-xl-base",
    "xl_sft": "acestep-v15-xl-sft",
}

_MAP_RE = re.compile(r"VARIANT_DIR_MAP\s*=\s*\{(?P<body>[^}]*)\}", re.S)


def patch_xl_variant_map(ace_dir: Path) -> str:
    """Add the XL variants to ACE-Step's trainer variant map if missing.

    Returns "patched", "ok" (nothing to do) or "skipped" (file not found or
    layout not recognised; training may still work on a newer ACE-Step).
    """
    path = Path(ace_dir) / ARGS_PY
    if not path.is_file():
        return "skipped"
    text = path.read_text(encoding="utf-8")
    m = _MAP_RE.search(text)
    if not m:
        return "skipped"
    body = m.group("body")
    missing = {k: v for k, v in XL_ENTRIES.items() if f'"{k}"' not in body and f"'{k}'" not in body}
    if not missing:
        return "ok"
    indent = "    "
    for line in body.splitlines():
        if line.strip():
            indent = line[: len(line) - len(line.lstrip())]
            break
    new_body = body.rstrip()
    if new_body and not new_body.endswith(","):
        new_body += ","
    new_body += f"\n{indent}# XL (4B DiT) variants, added by ACE-Step Workbench (same as model_loader._VARIANT_DIR)\n"
    new_body += "".join(f'{indent}"{k}": "{v}",\n' for k, v in missing.items())
    path.write_text(text[: m.start("body")] + new_body + text[m.end("body"):], encoding="utf-8")
    return "patched"


def apply_all(ace_dir: Path) -> dict[str, str]:
    return {"xl_variant_map": patch_xl_variant_map(ace_dir)}
