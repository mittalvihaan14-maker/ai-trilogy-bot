"""Predictable artifact paths. Relative paths are also the Hugging Face keys."""

from __future__ import annotations

import re
from pathlib import Path

import config

_SHOT = re.compile(r"^m(\d+)c(\d+)s(\d+)$")


def shot_parts(shot_id: str):
    match = _SHOT.match(shot_id)
    if not match:
        raise ValueError("Bad shot id {}".format(shot_id))
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def relative(path: Path) -> str:
    return path.resolve().relative_to(config.output_dir().resolve()).as_posix()


def audio_wav(shot_id: str) -> Path:
    movie, chapter, _index = shot_parts(shot_id)
    return config.output_dir() / "audio" / "m{}".format(movie) / "c{:02d}".format(chapter) / "{}.wav".format(shot_id)


def mouth_json(shot_id: str) -> Path:
    wav = audio_wav(shot_id)
    return wav.with_suffix(".mouth.json")


def shot_video(shot_id: str) -> Path:
    movie, chapter, _index = shot_parts(shot_id)
    return config.output_dir() / "shots" / "m{}".format(movie) / "c{:02d}".format(chapter) / "{}.mp4".format(shot_id)


def frame_dir(shot_id: str) -> Path:
    movie, chapter, _index = shot_parts(shot_id)
    return config.output_dir() / "frames" / "m{}".format(movie) / "c{:02d}".format(chapter) / shot_id


def master_character(character_id: str) -> Path:
    return config.output_dir() / "masters" / "characters" / "{}.png".format(character_id)


def master_location(location_id: str) -> Path:
    return config.output_dir() / "masters" / "locations" / "{}.png".format(location_id)


def draft_video(movie_id: int, chapter_id: int) -> Path:
    return config.output_dir() / "drafts" / "m{}_c{:02d}_240p.mp4".format(movie_id, chapter_id)


def chapter_wav(movie_id: int, chapter_id: int) -> Path:
    return config.output_dir() / "audio" / "m{}_c{:02d}.wav".format(movie_id, chapter_id)


def mezzanine_video(movie_id: int, chapter_id: int) -> Path:
    return config.output_dir() / "work" / "m{}_c{:02d}_240p60.mp4".format(movie_id, chapter_id)


def chapter_4k(movie_id: int, chapter_id: int) -> Path:
    return config.output_dir() / "chapters_4k" / "movie_{}".format(movie_id) / "chapter_{:02d}.mp4".format(chapter_id)


def final_movie(movie_id: int) -> Path:
    return config.final_movies_dir() / config.movie_filename(movie_id)


def chapter_key(movie_id: int, chapter_id: int) -> str:
    return "m{}c{}".format(movie_id, chapter_id)
