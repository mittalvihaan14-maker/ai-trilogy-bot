#!/usr/bin/env python3
"""Unattended orchestrator for the three-movie cartoon pipeline.

--test-mode renders one 60-second 4K/60fps movie through every stage.
--full-trilogy builds three separate 96-minute movies and resumes from the
last finished shot, chapter, or movie when a Kaggle session ends.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys

import config
from pipeline_lib import ffmpeg_tools
from pipeline_lib.hf_store import store
from pipeline_lib import layout
from pipeline_lib.report import emit
from pipeline_lib.state import State
from pipeline_lib import storyboard

LOGGER = logging.getLogger("trilogy")


def _spawn(script: str, args) -> None:
    command = [sys.executable, str(config.ROOT / script)] + list(args)
    LOGGER.info("Running %s", " ".join(command))
    subprocess.run(command, check=True)


def _mode_args(is_test: bool):
    return ["--test-mode"] if is_test else ["--full-trilogy"]


def _clean(is_test: bool) -> None:
    if is_test:
        shutil.rmtree(config.BASE_OUTPUT / "test", ignore_errors=True)
    else:
        for name in (
            "masters",
            "audio",
            "shots",
            "drafts",
            "frames",
            "work",
            "chapters_4k",
            "final_movies",
            "weights",
            "_hf",
        ):
            shutil.rmtree(config.BASE_OUTPUT / name, ignore_errors=True)
        for name in ("story_bible.json", "state.json", "pipeline.log"):
            path = config.BASE_OUTPUT / name
            if path.exists():
                path.unlink()
    config.set_mode(is_test)
    config.log_progress("Cleared the {} checkpoint.".format("test" if is_test else "full trilogy"))


def _chapter_ready(state: State, movie_id: int, chapter_id: int) -> bool:
    key = layout.chapter_key(movie_id, chapter_id)
    path = layout.chapter_4k(movie_id, chapter_id)
    if not state.contains("chapters_4k", key):
        return False
    if path.exists() and path.stat().st_size > 0:
        return True
    return store().remote_size(layout.relative(path)) is not None


def _movie_ready(state: State, movie_id: int) -> bool:
    record = state.data.get("movies", {}).get(str(movie_id)) or {}
    if not record.get("url"):
        return False
    return store().remote_size(layout.relative(layout.final_movie(movie_id))) is not None


def _log_device() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / float(1024 ** 3)
            LOGGER.info("GPU: %s (%.1f GB)", name, memory)
        else:
            LOGGER.info("No CUDA GPU visible. Stages that render will use the CPU or MPS fallback.")
    except Exception as exc:
        LOGGER.info("Torch is not importable yet (%s)", exc)


def run(is_test: bool, fresh: bool) -> int:
    config.bootstrap(is_test)
    config.require_secrets(gemini=True, hf=True, github=False)
    ffmpeg_tools.require_ffmpeg()
    _log_device()
    if fresh:
        _clean(is_test)
    label = "1-minute test" if is_test else "full trilogy"
    config.log_progress("Starting the {} run. Output: {}.".format(label, config.output_dir()))
    mode = _mode_args(is_test)
    _spawn("00_generate_story_bible.py", mode)
    _spawn("01_generate_master_assets.py", mode)
    bible = storyboard.load_bible()
    for line in storyboard.planned_report(bible):
        LOGGER.info("%s", line)
    for movie, chapter in storyboard.iter_chapters(bible):
        if _chapter_ready(State.load(), movie["id"], chapter["id"]):
            LOGGER.info("Skipping finished movie %s chapter %s", movie["id"], chapter["id"])
            continue
        scope = mode + ["--movie", str(movie["id"]), "--chapter", str(chapter["id"])]
        config.log_progress("Starting movie {} chapter {} — {}.".format(movie["id"], chapter["id"], chapter["title"]))
        _spawn("02_audio_and_sound_design.py", scope)
        _spawn("03_render_240p_4fps.py", scope)
        _spawn("04_sync_storage.py", scope)
        _spawn("05_upscale_and_assemble_movies.py", scope + ["--upscale-only"])
    for movie in bible["movies"]:
        if _movie_ready(State.load(), movie["id"]):
            LOGGER.info("Movie %s already published", movie["id"])
            continue
        _spawn("05_upscale_and_assemble_movies.py", mode + ["--assemble", str(movie["id"])])
    ok = emit(storyboard.load_bible(), State.load())
    return 0 if ok else 1


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Build the original 2D cartoon trilogy")
    mode = arg_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-mode", action="store_true", help="Render a 1-minute 4K 60fps test movie")
    mode.add_argument("--full-trilogy", action="store_true", help="Build all 3 movies at 90–100 minutes each")
    arg_parser.add_argument("--fresh", action="store_true", help="Delete this mode's checkpoint before starting")
    arguments = arg_parser.parse_args()
    try:
        code = run(arguments.test_mode, arguments.fresh)
    except Exception as exc:
        config.log_progress("ERROR: {}".format(config.scrub(str(exc))))
        try:
            config.git_auto_sync("Log pipeline error")
        except Exception:
            pass
        raise
    sys.exit(code)


if __name__ == "__main__":
    main()
