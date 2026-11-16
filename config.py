#!/usr/bin/env python3
"""Secrets, paths, visual rules, and the GitHub checkpoint helper.

Kaggle notebooks read GEMINI_API_KEY, HF_TOKEN, and GITHUB_TOKEN from
UserSecretsClient. Local runs fall back to the process environment and .env.
Importing this module does not log into Hugging Face. Call bootstrap() at
process start.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
BASE_OUTPUT = ROOT / "output"
STORY_PATH = ROOT / "story_idea.txt"
PROGRESS_LOG = ROOT / "progress_log.md"

SD_MODEL_ID = "stablediffusionapi/flat-2d-animerge"
SD_FALLBACK_MODEL_ID = "gsdf/Counterfeit-V2.5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"

INFERENCE_STEPS = 6
GUIDANCE_SCALE = 1.5

# 426 is not divisible by 8, which Stable Diffusion requires. Frames are
# generated at 424x240 and resized to the 426x240 draft size.
GEN_WIDTH = 424
GEN_HEIGHT = 240
FRAME_WIDTH = 426
FRAME_HEIGHT = 240
FINAL_WIDTH = 3840
FINAL_HEIGHT = 2160
DRAFT_FPS = 4
FINAL_FPS = 60
MASTER_SIZE = 512
VRAM_OFFLOAD_BELOW_GB = 10.0

FULL_MOVIES = 3
FULL_CHAPTERS = 6
FULL_SHOTS_PER_CHAPTER = 120
FULL_SHOT_SECONDS = 8.0
TEST_MOVIES = 1
TEST_CHAPTERS = 1
TEST_SHOTS = 8
TEST_SHOT_SECONDS = 7.5
MOVIE_MIN_SECONDS = 90 * 60
MOVIE_MAX_SECONDS = 100 * 60

SAMPLE_RATE = 48000
DIALOGUE_GAIN = 1.0
SFX_GAIN = 0.70
MUSIC_GAIN = 0.20
VIDEO_CRF = "18"
AUDIO_BITRATE = "192k"

STYLE_SUFFIX = (
    ", 2d flipaclip hand-drawn cartoon animation, bold black ink outlines, "
    "flat cel shading, simple geometric cartoon shapes, vibrant clean colors"
)
NEGATIVE_PROMPT = (
    "photorealistic, real person, photo, celebrity, famous actor, 3d render, "
    "realistic skin, hyperdetailed, live action, blurry, deformed"
)

MOODS = (
    "hopeful",
    "tense",
    "sad",
    "action",
    "wonder",
    "peaceful",
    "comic",
    "dark",
)
NARRATOR_VOICE = "en-US-ChristopherNeural"
VOICE_POOL = (
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-IN-NeerjaNeural",
    "en-IE-EmilyNeural",
)
GIT_NAME = "ai-trilogy-bot"
GIT_EMAIL = "ai-trilogy-bot@users.noreply.github.com"
ESRGAN_WEIGHT_NAME = "RealESRGAN_x4plus_anime_6B.pth"
ESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)

LOGGER = logging.getLogger("trilogy")

_KAGGLE_CLIENT = None
_KAGGLE_TRIED = False
_TEST_MODE = False
_OUTPUT_DIR = None  # type: Optional[Path]
_BOOTSTRAPPED = False


def _kaggle_client():
    global _KAGGLE_CLIENT, _KAGGLE_TRIED
    if _KAGGLE_TRIED:
        return _KAGGLE_CLIENT
    _KAGGLE_TRIED = True
    try:
        from kaggle_secrets import UserSecretsClient

        _KAGGLE_CLIENT = UserSecretsClient()
    except Exception:
        _KAGGLE_CLIENT = None
    return _KAGGLE_CLIENT


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _secret(name: str, default: str = "") -> str:
    value = ""
    client = _kaggle_client()
    if client is not None:
        try:
            value = client.get_secret(name) or ""
        except Exception:
            value = ""
    if not str(value).strip():
        value = os.getenv(name, "") or ""
    value = str(value).strip()
    return value or default


_load_dotenv()
GEMINI_API_KEY = _secret("GEMINI_API_KEY")
HF_TOKEN = _secret("HF_TOKEN")
GITHUB_TOKEN = _secret("GITHUB_TOKEN")
HF_REPO_ID = _secret("HF_REPO_ID", "Fruityahhh/my-2d-trilogy")
GITHUB_REPO = _secret("GITHUB_REPO", "mittalvihaan14-maker/ai-trilogy-bot")


def scrub(text: str) -> str:
    """Remove loaded secrets from a log or exception string."""
    redacted = str(text)
    for secret in (GITHUB_TOKEN, HF_TOKEN, GEMINI_API_KEY):
        if secret and len(secret) > 4:
            redacted = redacted.replace(secret, "***")
    return redacted


def on_kaggle() -> bool:
    return Path("/kaggle/working").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def test_mode() -> bool:
    return _TEST_MODE


def output_dir() -> Path:
    global _OUTPUT_DIR
    if _OUTPUT_DIR is None:
        set_mode(False)
    return _OUTPUT_DIR  # type: ignore[return-value]


def final_movies_dir() -> Path:
    return output_dir() / "final_movies"


def set_mode(is_test: bool) -> None:
    global _TEST_MODE, _OUTPUT_DIR
    _TEST_MODE = bool(is_test)
    _OUTPUT_DIR = BASE_OUTPUT / "test" if _TEST_MODE else BASE_OUTPUT
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_movies_dir().mkdir(parents=True, exist_ok=True)
    _setup_logging()


def hf_repo_path(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    if _TEST_MODE:
        cleaned.insert(0, "test")
    return "/".join(cleaned)


def hf_file_url(repo_path: str) -> str:
    return "https://huggingface.co/datasets/{}/resolve/main/{}".format(HF_REPO_ID, repo_path)


def movie_filename(movie_number: int) -> str:
    if _TEST_MODE:
        return "Test_Movie_4K_60fps.mp4"
    return "Movie_{}_Full_4K_60fps.mp4".format(movie_number)


def _setup_logging() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    log_path = output_dir() / "pipeline.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def log_progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = "- {} {}".format(stamp, scrub(message))
    LOGGER.info("%s", scrub(message))
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def login_huggingface() -> bool:
    if not HF_TOKEN:
        LOGGER.warning("HF_TOKEN is not set. Hugging Face uploads are disabled.")
        return False
    from huggingface_hub import login

    login(token=HF_TOKEN, add_to_git_credential=False)
    LOGGER.info("Logged in to Hugging Face dataset %s", HF_REPO_ID)
    return True


def require_secrets(gemini: bool = True, hf: bool = True, github: bool = False) -> None:
    missing = []  # type: List[str]
    if gemini and not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if hf and not HF_TOKEN:
        missing.append("HF_TOKEN")
    if github and not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if missing:
        raise RuntimeError(
            "Missing secrets: {}. On Kaggle add them under Add-ons > Secrets. "
            "Locally put them in .env (see .env.example).".format(", ".join(missing))
        )
    if not github and not GITHUB_TOKEN:
        LOGGER.warning("GITHUB_TOKEN is not set. Progress logs will stay on this machine.")


def bootstrap(is_test: bool = False) -> None:
    """Prepare folders, caches, and the Hugging Face session."""
    global _BOOTSTRAPPED
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    if on_kaggle():
        cache = Path("/kaggle/working/hf_home")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    set_mode(is_test)
    if _BOOTSTRAPPED:
        return
    if HF_TOKEN:
        login_huggingface()
    else:
        LOGGER.warning("HF_TOKEN is not set. Continuing without Hugging Face login.")
    _BOOTSTRAPPED = True


def _run_git(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = scrub((completed.stderr or "") + (completed.stdout or ""))
        raise RuntimeError("git command failed ({}): {}".format(args[0], detail[-1500:]))
    return completed


def _ensure_git_repo() -> None:
    if not (ROOT / ".git").exists():
        _run_git(["git", "init"])
        _run_git(["git", "checkout", "-B", "main"])
    _run_git(["git", "config", "--local", "user.name", GIT_NAME])
    _run_git(["git", "config", "--local", "user.email", GIT_EMAIL])
    public = "https://github.com/{}.git".format(GITHUB_REPO.strip().strip("/"))
    remotes = _run_git(["git", "remote"], check=False)
    names = (remotes.stdout or "").split()
    if "origin" not in names:
        _run_git(["git", "remote", "add", "origin", public])
        return
    current = _run_git(["git", "remote", "get-url", "origin"], check=False)
    url = (current.stdout or "").strip()
    if (not url) or ("x-access-token" in url) or ("@" in url.split("github.com")[0]):
        _run_git(["git", "remote", "set-url", "origin", public])


def _stage_lightweight_files() -> None:
    tracked = [
        ".env.example",
        ".gitignore",
        "README.md",
        "requirements.txt",
        "requirements-kaggle.txt",
        "story_idea.txt",
        "progress_log.md",
        "config.py",
        "run_pipeline.py",
        "00_generate_story_bible.py",
        "01_generate_master_assets.py",
        "02_audio_and_sound_design.py",
        "03_render_240p_4fps.py",
        "04_sync_storage.py",
        "05_upscale_and_assemble_movies.py",
        "pipeline_lib",
    ]
    existing = [name for name in tracked if (ROOT / name).exists()]
    if existing:
        _run_git(["git", "add", "--"] + existing)
    bible = output_dir() / "story_bible.json"
    # Full-mode bible lives at output/story_bible.json. Test-mode bible stays local.
    full_bible = BASE_OUTPUT / "story_bible.json"
    if full_bible.exists() and full_bible.stat().st_size < 50 * 1024 * 1024:
        _run_git(["git", "add", "-f", "--", str(full_bible)])
    elif bible.exists() and (not test_mode()) and bible.stat().st_size < 50 * 1024 * 1024:
        _run_git(["git", "add", "-f", "--", str(bible)])


def _unstage_unsafe_files() -> None:
    staged = _run_git(["git", "diff", "--cached", "--name-only"], check=False)
    for name in (staged.stdout or "").splitlines():
        path = ROOT / name
        blocked = name == ".env" or name.endswith(".env") or name.startswith(".env")
        too_big = path.exists() and path.stat().st_size >= 50 * 1024 * 1024
        media = name.startswith("output/") and name != "output/story_bible.json"
        if blocked or too_big or media:
            _run_git(["git", "reset", "-q", "HEAD", "--", name], check=False)
            _run_git(["git", "reset", "-q", "--", name], check=False)
            LOGGER.warning("Left %s out of the git commit", name)


def git_auto_sync(commit_msg: str) -> bool:
    """Commit scripts, the story bible, and progress_log.md, then push.

    Git identity is stored only in this repo's local config. The token is
    passed to a one-shot push URL and is never written into .git/config.
    """
    if not GITHUB_TOKEN:
        LOGGER.warning("GITHUB_TOKEN missing; skipped GitHub push.")
        return False
    message = scrub(commit_msg).replace("\n", " ").strip() or "Update trilogy progress"
    try:
        _ensure_git_repo()
        _stage_lightweight_files()
        _unstage_unsafe_files()
        staged = _run_git(["git", "diff", "--cached", "--name-only"], check=False)
        if (staged.stdout or "").strip():
            _run_git(["git", "commit", "-m", message])
            LOGGER.info("Committed: %s", message)
        else:
            LOGGER.info("Git commit skipped; nothing new is staged.")
        repo = GITHUB_REPO.strip().strip("/")
        if repo.endswith(".git"):
            repo = repo[:-4]
        remote = "https://x-access-token:{}@github.com/{}.git".format(quote(GITHUB_TOKEN, safe=""), repo)
        push = _run_git(["git", "push", remote, "HEAD:main"], check=False)
        if push.returncode != 0:
            LOGGER.error("GitHub push failed: %s", scrub((push.stderr or "")[-1500:]))
            return False
        LOGGER.info("Pushed progress to github.com/%s", repo)
        return True
    except Exception as exc:
        LOGGER.error("git_auto_sync failed: %s", scrub(str(exc)))
        return False
