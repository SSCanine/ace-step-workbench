"""Workbench installer: gets ACE-Step 1.5, installs its environment, downloads models.

Run through install.bat / install.sh (they make sure uv and the Workbench venv exist).

  python install.py                     # fresh install: ACE-Step goes next to this app
  python install.py --no-ace            # Workbench only; set your existing ACE-Step folder in Settings
  python install.py --ace-path <folder> # same, but save that folder right away
  python install.py --skip-models       # download models later (first generation does it)
  python install.py --latest            # newest ACE-Step instead of the tested version
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from workbench import ace, ace_patches, config  # noqa: E402

REPO = "https://github.com/ACE-Step/ACE-Step-1.5"
PINNED = "ca1e85fe9430179831e6bc6be790c332190a3866"  # tested with this Workbench release


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run(cmd: list[str], cwd: Path) -> None:
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"\n  Command failed (exit {r.returncode}). Fix the error above and run install again.")


def fetch_ace(dest: Path, latest: bool) -> None:
    ref = "main" if latest else PINNED
    if shutil.which("git"):
        say(f"Downloading ACE-Step 1.5 ({'latest' if latest else 'tested version ' + PINNED[:7]}) with git...")
        run(["git", "clone", REPO + ".git", str(dest)], ROOT)
        if not latest:
            run(["git", "checkout", "-q", PINNED], dest)
        return
    say("git not found, downloading a zip instead...")
    url = f"{REPO}/archive/{ref}.zip"
    with urllib.request.urlopen(url, timeout=600) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(ROOT)
    (ROOT / top).rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ace-path", help="Use an existing ACE-Step 1.5 folder instead of downloading one.")
    ap.add_argument("--skip-models", action="store_true", help="Don't pre-download models.")
    ap.add_argument("--latest", action="store_true", help="Use ACE-Step's newest code instead of the tested version.")
    ap.add_argument("--no-ace", action="store_true", help="Only install the Workbench. Point it at your ACE-Step in Settings.")
    args = ap.parse_args()

    if args.no_ace:
        config.save({})
        say("Workbench installed. Start it, then open Settings > ACE-Step engine > ACE-Step folder")
        say("and paste the folder of your existing ACE-Step 1.5 install (press 'check' to verify it).")
        return

    if args.ace_path:
        ace_dir = Path(args.ace_path).expanduser().resolve()
        if not (ace_dir / "acestep").is_dir():
            raise SystemExit(f"  {ace_dir} does not look like ACE-Step 1.5 (no 'acestep' folder).")
        config.save({"ace": {"path": str(ace_dir)}})
        say(f"Using existing ACE-Step at {ace_dir}")
    else:
        ace_dir = config.ace_dir(config.load())
        if not (ace_dir / "acestep").is_dir():
            fetch_ace(ace_dir, args.latest)
        else:
            say(f"ACE-Step already present at {ace_dir}")

    if ace_patches.patch_xl_variant_map(ace_dir) == "patched":
        say("Applied a small fix to ACE-Step so it can train LoRAs on the XL models.")

    if not ace.venv_python(ace_dir).exists():
        uv = shutil.which("uv")
        if not uv:
            raise SystemExit("  uv is missing. Run install.bat / install.sh instead of install.py directly.")
        say("Installing ACE-Step's environment (PyTorch + CUDA, several GB, takes a while)...")
        run([uv, "sync"], ace_dir)
    else:
        say("ACE-Step environment already installed.")

    if not args.skip_models:
        ck = ace_dir / "checkpoints"
        if (ck / "acestep-v15-turbo").is_dir():
            say("Main models already downloaded.")
        else:
            say("Downloading the main models (turbo + 1.7B LM + VAE, about 10 GB)...")
            run([str(ace.venv_python(ace_dir)), "-m", "acestep.model_downloader"], ace_dir)
        say("More models (XL turbo, base, 0.6B/4B LM) download automatically when you pick them in Settings.")

    config.save({})  # writes data/settings.json with defaults if missing
    say("Install complete.")


if __name__ == "__main__":
    main()
