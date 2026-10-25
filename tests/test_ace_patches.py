"""The XL variant-map fix for ACE-Step's trainer CLI (no GPU, no ACE-Step install)."""

from __future__ import annotations

import ast
import textwrap

from workbench import ace_patches

PINNED_ARGS = textwrap.dedent('''\
    """Argparse construction for ACE-Step Training V2 CLI."""

    from __future__ import annotations

    import argparse
    import sys

    _DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 4

    # Model variant -> checkpoint subdirectory mapping
    VARIANT_DIR_MAP = {
        "turbo": "acestep-v15-turbo",
        "base": "acestep-v15-base",
        "sft": "acestep-v15-sft",
    }


    def build_root_parser() -> argparse.ArgumentParser:
        return argparse.ArgumentParser()
''')


def _write(tmp_path, text):
    p = tmp_path / ace_patches.ARGS_PY
    p.parent.mkdir(parents=True)
    p.write_text(text, encoding="utf-8")
    return p


def _variant_map(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "VARIANT_DIR_MAP":
            return ast.literal_eval(node.value)
    raise AssertionError("VARIANT_DIR_MAP not found")


def test_adds_xl_entries_once(tmp_path):
    p = _write(tmp_path, PINNED_ARGS)
    assert ace_patches.patch_xl_variant_map(tmp_path) == "patched"
    m = _variant_map(p)
    assert m["turbo"] == "acestep-v15-turbo"
    assert m["xl_turbo"] == "acestep-v15-xl-turbo"
    assert m["xl_base"] == "acestep-v15-xl-base"
    assert m["xl_sft"] == "acestep-v15-xl-sft"
    once = p.read_text(encoding="utf-8")
    assert ace_patches.patch_xl_variant_map(tmp_path) == "ok"
    assert p.read_text(encoding="utf-8") == once
    assert "def build_root_parser" in once


def test_already_fixed_upstream_is_left_alone(tmp_path):
    fixed = PINNED_ARGS.replace('    "sft": "acestep-v15-sft",\n',
                                '    "sft": "acestep-v15-sft",\n    "xl_turbo": "acestep-v15-xl-turbo",\n'
                                '    "xl_base": "acestep-v15-xl-base",\n    "xl_sft": "acestep-v15-xl-sft",\n')
    p = _write(tmp_path, fixed)
    assert ace_patches.patch_xl_variant_map(tmp_path) == "ok"
    assert p.read_text(encoding="utf-8") == fixed


def test_missing_or_unknown_layout_is_skipped(tmp_path):
    assert ace_patches.patch_xl_variant_map(tmp_path) == "skipped"
    _write(tmp_path, "VARIANTS = dict(turbo='x')\n")
    assert ace_patches.patch_xl_variant_map(tmp_path) == "skipped"
