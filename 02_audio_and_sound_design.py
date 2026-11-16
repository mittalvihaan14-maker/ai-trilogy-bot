#!/usr/bin/env python3
"""Voice tracks, music, effects, and a mixed wav per shot.

Frame count is duration x 4. Full movies are locked at 8.0 seconds per shot
so six chapters land at 96 minutes. Speech that runs long is sped up to fit
the frame count. The mix is dialogue at 100%, effects at 70%, music at 20%.
"""

from __future__ import annotations

import logging

import config
from pipeline_lib.audio_fx import build_shot_audio
from pipeline_lib.cli import parser
from pipeline_lib.hf_store import publish_state, store
from pipeline_lib import layout
from pipeline_lib.state import State
from pipeline_lib import storyboard

LOGGER = logging.getLogger("trilogy")


def _ensure_shot(shot: dict, bible: dict, state: State) -> None:
    wav_path = layout.audio_wav(shot["id"])
    mouth_path = layout.mouth_json(shot["id"])
    if state.contains("audio_shots", shot["id"]) and wav_path.exists() and mouth_path.exists():
        return
    if not wav_path.exists():
        store().ensure_local(wav_path, layout.relative(wav_path))
    if not mouth_path.exists():
        store().ensure_local(mouth_path, layout.relative(mouth_path))
    if wav_path.exists() and mouth_path.exists():
        state.add("audio_shots", shot["id"])
        return
    LOGGER.info("Mixing %s (%s frames, %.1fs)", shot["id"], shot["frames"], shot["duration_sec"])
    build_shot_audio(shot, bible, wav_path)
    store().upload(wav_path, layout.relative(wav_path))
    store().upload(mouth_path, layout.relative(mouth_path))
    state.add("audio_shots", shot["id"])
    publish_state(state)


def generate(is_test: bool, movie_id=None, chapter_id=None) -> None:
    config.bootstrap(is_test)
    from pipeline_lib import ffmpeg_tools

    ffmpeg_tools.require_ffmpeg()
    bible = storyboard.load_bible()
    state = State.load()
    for line in storyboard.planned_report(bible):
        LOGGER.info("%s", line)
    for movie, chapter in storyboard.iter_chapters(bible):
        if movie_id and int(movie["id"]) != int(movie_id):
            continue
        if chapter_id and int(chapter["id"]) != int(chapter_id):
            continue
        frames = sum(int(shot["frames"]) for shot in chapter["shots"])
        seconds = frames / float(config.DRAFT_FPS)
        LOGGER.info(
            "Movie %s chapter %s audio: %s shots, %s frames, %.2fs",
            movie["id"],
            chapter["id"],
            len(chapter["shots"]),
            frames,
            seconds,
        )
        for shot in chapter["shots"]:
            _ensure_shot(shot, bible, state)
        if not is_test:
            total = float(movie["duration_sec"])
            if not (config.MOVIE_MIN_SECONDS <= total <= config.MOVIE_MAX_SECONDS):
                raise RuntimeError("Movie {} planned length {:.1f}s is outside 90–100 minutes".format(movie["id"], total))
    config.log_progress("Audio mix finished for the requested chapters. Draft rate is {} fps.".format(config.DRAFT_FPS))


def main() -> None:
    arguments = parser("Generate dialogue, music, effects, and mixed shot audio").parse_args()
    generate(arguments.test_mode, arguments.movie, arguments.chapter)


if __name__ == "__main__":
    main()
