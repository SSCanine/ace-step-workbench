# ACE-Step Workbench

An all-in-one, local music studio around [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5), the open-source song generator.

**Write** songs with any AI (GPT, Claude, Codex, or a free local model), **create** them with ACE-Step, **fix** single words and phrases without losing the take, and **train** your own style LoRAs, all from one browser tab with a built-in guide and tooltips on every control.

| Tab | What it does |
|---|---|
| 📖 Guide | Step-by-step manual inside the app: captions, lyric tags, delivery tricks, settings, training, troubleshooting. |
| ✍️ Write | AI songwriter. Idea in, title + caption + lyrics + BPM/key/length out. Two targets: **ACE-Step (unfiltered)** and **Suno (safe)**. |
| 🎛️ Create | The generator: caption, lyrics, metadata, seed, batch, LoRA. Every result is saved with its exact seed and settings. |
| 🎚️ Studio | **Regenerate a song with the same seed**, edit lyrics, fix how words are sung (CAPS, backing vocals, stretched vowels, syllables, vocal tags, AI pronunciation fixes), and **re-sing just a few seconds** while keeping the rest. |
| 🎧 Library | Every song you've made, searchable, with its settings. |
| 🧪 Train | Build a dataset from your tracks, auto-label it with ACE-Step, train a LoRA or LoKr with a live loss chart. |
| ⚙️ Settings | ACE-Step models, AI writer provider and keys, folders, theme, tooltips. |

## What you need

- An NVIDIA GPU with 8 GB+ VRAM (12 GB+ recommended; 16 GB+ for comfortable LoRA training)
- Windows 10/11 or Linux, about 20 GB free disk
- An internet connection for the first install (ACE-Step + models, about 15 GB total)

## Install

**Windows:** double-click `install.bat`.
**Linux:** `./install.sh`

The installer:
1. installs [uv](https://docs.astral.sh/uv/) if you don't have it,
2. downloads ACE-Step 1.5 next to this folder (pinned to the version this release was tested with),
3. installs ACE-Step's environment (PyTorch + CUDA),
4. downloads the main models (turbo + 1.7B planner LM, about 10 GB).

### Already have ACE-Step 1.5?

Reuse it and its models instead of downloading a second copy:

1. Run `install.bat --no-ace` (Linux: `./install.sh --no-ace`). This installs only the Workbench.
2. Start the Workbench, open **Settings → ACE-Step engine**.
3. Paste your ACE-Step folder into **ACE-Step folder**, press **check** (it confirms the install and lists your downloaded models), then **Save** and **Start**.

You can change this folder any time in Settings. Leaving it empty uses the copy the installer puts next to the app.

Other installer options: `--skip-models` (download on first use), `--latest` (newest ACE-Step instead of the tested version).

## Start

**Windows:** double-click `start.bat`. **Linux:** `./start.sh`

Your browser opens http://127.0.0.1:3470. ACE-Step starts in the background; the dot at the bottom left turns green when it's ready (first start takes a few minutes).

Options: `--port 3500`, `--no-ace` (don't auto-start ACE-Step), `--host 0.0.0.0` (reach it from other devices on your network; anyone on the network can then use it).

## Your first song

1. **Write** tab → type an idea → **Write song**.
2. **Send to Create** → **Generate**.
3. Like it but one word is off? **Studio** → fix the word → select that moment on the waveform → **Re-sing selection**.

The full walkthrough is the **Guide** tab.

## Plug in an AI writer

Settings → **AI writer** → pick a provider → **Test**.

| Provider | Needs | Notes |
|---|---|---|
| Ollama, LM Studio, KoboldCpp, llama.cpp, text-generation-webui | The app running locally | Free, private, uncensored with the right model |
| OpenAI (GPT) | API key | |
| Claude API | Anthropic API key | Default `claude-opus-5-5`; `claude-haiku-5-5` is cheapest |
| Claude Code CLI | `claude` installed and logged in | Uses your Claude subscription, no key |
| Codex CLI | `codex` installed and logged in | Uses your ChatGPT plan, no key |
| OpenRouter, Groq, DeepSeek, any OpenAI-compatible server | API key / URL | |
| ACE-Step built-in LM | Nothing | Simple songs without any setup (ACE-Step mode only) |
| None | Nothing | Rule-based drafts |

**About "uncensored":** the Workbench adds no filter of its own. In ACE-Step mode the writer is told to write exactly what you ask for. Hosted AI services still apply their own policies; for unrestricted writing use a local model.

## Suno mode

Write tab → **Suno (safe)**. The writer follows Suno's rules (no artist names, Suno tag format, length limits) and a local safety pass removes "in the style of X" phrases, names from your blocklist, and (optionally) softens words Suno's filter flags. Every change is listed. Then use the copy buttons and **Open Suno**. Edit `data/suno_blocklist.txt` and `data/suno_soften.json` to tune it.

## Same seed, fix a phrase

Every song stores its seed and full settings.

- **Regenerate (same seed):** same seed + your edited lyrics → the same musical idea with your new words. Best with Thinking off.
- **Re-sing selection (repaint):** regenerate only the selected seconds; everything else stays identical.
- **Variations:** same settings, new seeds.

## Training a LoRA

Train tab → create a dataset with a trigger word → add 8-30 tracks you have the rights to → **Auto-label** (ACE-Step writes captions, BPM and key) → check them → pick a preset → **Start training**. ACE-Step pauses while training uses the GPU and restarts afterwards. The finished LoRA appears in Create's LoRA list; its trigger word is added to your caption automatically.

Under the hood this uses ACE-Step 1.5's own trainer (`train.py fixed`), and datasets use ACE-Step's own `dataset.json` format.

## Where things live

```
ace-step-workbench/
  workbench/        Python backend (FastAPI)
  web/              The app (plain HTML/JS, no build step) + guide.html
  ACE-Step-1.5/     ACE-Step itself (created by the installer, not in git)
  data/             Your stuff (not in git):
    settings.json   settings and API keys (stored locally only)
    songs/          every song + its settings
    datasets/       training datasets
    loras/          trained LoRAs
    trash/          anything you delete (nothing is hard-deleted)
```

All locations are set in **Settings**: the ACE-Step folder (ACE-Step engine card) and where datasets, trained LoRAs and extra LoRA folders live (Folders card). Nothing is hard-coded to a particular drive or user.

## FAQ

**Can I run ACE-Step myself and just use the Workbench as a front end?** Yes. Start ACE-Step's API (`uv run acestep-api`), untick "Start with Workbench" in Settings and set the API address.

**Generation is slow.** Turn Thinking off and fill BPM/key/length yourself. Close other GPU apps. A smaller planner LM (0.6B) or no LM helps on small GPUs.

**Can I use LoRAs I downloaded?** Put them in a folder and add it under Settings → Folders → More LoRA folders.

**Switching LoRAs restarts ACE-Step?** Yes, on purpose: ACE-Step 1.5 doesn't fully remove a LoRA when unloading, so a clean restart guarantees you hear only the LoRA you picked.

## Tests

```
.venv\Scripts\python -m pip install pytest      (once)
.venv\Scripts\python -m pytest -q tests
```

The tests use a mock ACE-Step server and a fake trainer, so they run without a GPU.

## Credits and license

Built on [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) (MIT). The Workbench itself is MIT licensed, see `LICENSE`. Only train on and generate from material you have the rights to use.
