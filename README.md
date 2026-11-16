# AI Trilogy Bot

Unattended pipeline that turns `story_idea.txt` into **three separate original 2D cartoon movies**. Each finished file is a 4K, 60 fps `.mp4` with voice, music, and sound effects.

| Movie | Length | File |
| --- | --- | --- |
| 1 | 96 minutes (inside the 90–100 minute window) | `output/final_movies/Movie_1_Full_4K_60fps.mp4` |
| 2 | 96 minutes | `output/final_movies/Movie_2_Full_4K_60fps.mp4` |
| 3 | 96 minutes | `output/final_movies/Movie_3_Full_4K_60fps.mp4` |

96 minutes is exact on purpose: 6 chapters × 120 shots × 8.0 seconds. Every shot stays inside the 6–10 second range. A test run is 8 shots × 7.5 seconds = 60 seconds.

Pictures are generated as flat cel cartoons (bold ink outlines, simple shapes). Prompts reject photorealism, real people, and celebrities. Character and location masters are reused for all three films so faces and places stay consistent.

Code and `progress_log.md` go to GitHub. Video and audio go to the private Hugging Face dataset `Fruityahhh/my-2d-trilogy`. `output/` is gitignored so a 4K file never hits GitHub.

## What you need

- A Kaggle notebook with a **T4** GPU and **Internet on**, or a local NVIDIA GPU with 4–8 GB VRAM (CPU offload turns on under 10 GB).
- Python 3.10+.
- `ffmpeg` (the Kaggle cell below installs it).
- `GEMINI_API_KEY`, `HF_TOKEN` (write), and `GITHUB_TOKEN` (repo contents write).

The full trilogy does **not** finish in one free Kaggle session. Run the same command again after a timeout. Finished shots, chapters, and movies are skipped.

Rough T4 time, as an estimate rather than a promise:

| Stage | 1-minute test | Full trilogy |
| --- | --- | --- |
| Story bible | a few minutes | 30–60 minutes |
| Master art | a few minutes | under 15 minutes |
| 240p / 4 fps render | 10–30 minutes | about 12–20 hours |
| 60 fps interpolation + 4K upscale | 20–60 minutes | about 15–40 hours |

Flat cel video compresses well, but a 96-minute 4K chapter pair can still fill a small disk. The pipeline upscales one chapter at a time, uploads it, and deletes the intermediate 240p/60 fps file. On Kaggle it also deletes the local final after Hugging Face confirms the upload. Set `KEEP_LOCAL_FINALS=1` to keep those files.

## Kaggle setup

1. Create a notebook. Set the accelerator to **GPU T4**. Turn **Internet** on.
2. Open **Add-ons → Secrets** and add:
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
   - `GITHUB_TOKEN`
   - Optional: `HF_REPO_ID` (default `Fruityahhh/my-2d-trilogy`), `GITHUB_REPO` (default `mittalvihaan14-maker/ai-trilogy-bot`).
3. The Hugging Face token needs permission to create or write a **private dataset**. The GitHub token needs contents write on `mittalvihaan14-maker/ai-trilogy-bot`.
4. Paste this into a cell and run it. Do **not** put keys in the notebook.

```python
import os, subprocess, sys
from pathlib import Path

os.chdir("/kaggle/working")
repo = Path("/kaggle/working/ai-trilogy-bot")
if not repo.exists():
    subprocess.check_call(["git", "clone", "https://github.com/mittalvihaan14-maker/ai-trilogy-bot.git"])
os.chdir(repo)
subprocess.check_call(["apt-get", "update"])
subprocess.check_call(["apt-get", "install", "-y", "ffmpeg"])
# Keep Kaggle's CUDA build of PyTorch. requirements.txt also lists torch for local GPUs.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
subprocess.check_call([sys.executable, "run_pipeline.py", "--test-mode"])
```

When the one-minute test looks right, stop the cell if it is still going and run the real job:

```python
subprocess.check_call([sys.executable, "run_pipeline.py", "--full-trilogy"])
```

Run that same command in later sessions. Add `--fresh` only when you want to throw away the story bible and start the trilogy over.

Optional: install `rife-ncnn-vulkan` if you want RIFE frame interpolation. Without it, the pipeline uses `ffmpeg minterpolate` (motion compensation, then a blend fallback). Set `USE_RIFE=0` to force minterpolate even when the RIFE binary is present. Set `FORCE_CPU_OFFLOAD=1` if a 16 GB card runs out of memory after IP-Adapter loads.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in GEMINI_API_KEY, HF_TOKEN, GITHUB_TOKEN
python run_pipeline.py --test-mode
```

Edit `story_idea.txt` before the first full run if you want a different original story. After shots have been rendered, the existing bible is kept so a later edit does not throw away finished video. Use `--fresh` to rebuild from the new idea.

## What each file does

| File | Role |
| --- | --- |
| `config.py` | Kaggle Secrets or `.env`, Hugging Face login, visual prompt rules, `git_auto_sync()` |
| `00_generate_story_bible.py` | `gemini-2.5-flash` writes characters, locations, and the shot grid |
| `01_generate_master_assets.py` | One PNG per character and location (`flat-2d-animerge`, fallback `Counterfeit-V2.5`, LCM-LoRA, 6 steps) |
| `02_audio_and_sound_design.py` | `edge-tts` dialogue, mood music, effect cues, mixed 4 fps-locked wavs |
| `03_render_240p_4fps.py` | img2img + IP-Adapter frames, mouth sprites, 240p chapter drafts |
| `04_sync_storage.py` | Dataset upload plus a GitHub push of scripts, the bible, and `progress_log.md` |
| `05_upscale_and_assemble_movies.py` | 60 fps, RealESRGAN to 4K, concat each movie, upload, free disk |
| `run_pipeline.py` | Runs the stages in order and prints the Final Delivery Report |

Stage scripts are safe to run on their own. `run_pipeline.py` is the one that keeps disk use chapter-sized.

## Models and mix

- Checkpoint: `stablediffusionapi/flat-2d-animerge`, then `gsdf/Counterfeit-V2.5` if that repo is unavailable.
- Speed: `latent-consistency/lcm-lora-sdv1-5`, `LCMScheduler`, 6 steps, `guidance_scale=1.5`, float16 on CUDA.
- Memory: `enable_attention_slicing()`, `enable_vae_slicing()`, and `enable_model_cpu_offload()` under 10 GB.
- Consistency: `h94/IP-Adapter` (`models/ip-adapter_sd15.bin`) for faces, location img2img for the background.
- Draft frames are generated at 424×240 (divisible by 8) and stored at 426×240.
- Upscale: `RealESRGAN_x4plus_anime_6B`, then resize to 3840×2160.
- Encode: `libx264 -crf 18`, `aac -b:a 192k`. Movie join: concat demuxer, `-c copy`.
- Mix: dialogue 100%, effects 70%, music 20%.

If the RealESRGAN weights cannot be loaded, that chapter is scaled with Lanczos and the delivery report says `lanczos-fallback` instead of hiding it.

## Delivery report

At the end, `run_pipeline.py` prints a block like this:

```text
============================================================
Final Delivery Report
============================================================
Movie 1 — The Map That Forgot the Sea
  Duration: 1:36:00 (5760.00 seconds)
  Video: 3840x2160 @ 60.00 fps | audio: aac
  Upscaler: RealESRGAN_x4plus_anime_6B
  Interpolation: ffmpeg-minterpolate-mci
  Hugging Face: https://huggingface.co/datasets/Fruityahhh/my-2d-trilogy/resolve/main/final_movies/Movie_1_Full_4K_60fps.mp4
...
============================================================
```

The dataset is private. Open the link while logged into the Hugging Face account that owns `Fruityahhh/my-2d-trilogy`.

Test uploads live under `test/` in that dataset, so a one-minute test cannot overwrite Movie 1.

## GitHub pushes

`git_auto_sync()` sets a repo-local git identity, commits `progress_log.md`, the scripts, and `output/story_bible.json` when that file is under 50 MB, then pushes to `https://github.com/mittalvihaan14-maker/ai-trilogy-bot`. The token is used for that one push and is not written into `.git/config`. `.env` and everything in `output/` except the small story bible stay untracked.
