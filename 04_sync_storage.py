#!/usr/bin/env python3
"""Upload finished lightweight artifacts and push the progress log.

Video files under output/ stay out of git. story_bible.json is committed only
when it is under 50MB. The GitHub remote stored on disk never contains the token.
"""

from __future__ import annotations

import logging

import config
from pipeline_lib.cli import parser
from pipeline_lib.hf_store import store

LOGGER = logging.getLogger("trilogy")
SKIP_DIRS = {"_hf", "weights", "frames", "work"}
KEEP_SUFFIXES = {".png", ".wav", ".json", ".mp4"}


def _iter_artifacts():
    root = config.output_dir()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in SKIP_DIRS:
            continue
        if path.suffix.lower() not in KEEP_SUFFIXES:
            continue
        if path.name.endswith(".ffmpeg.log") or path.name.endswith(".concat.txt"):
            continue
        yield path, relative


def sync(is_test: bool, movie_id=None, chapter_id=None) -> None:
    config.bootstrap(is_test)
    uploaded = 0
    for path, relative in _iter_artifacts():
        try:
            if store().upload(path, relative):
                uploaded += 1
        except Exception as exc:
            LOGGER.error("Sync failed for %s: %s", relative, config.scrub(str(exc)))
            raise
    if movie_id and chapter_id:
        message = "Sync movie {} chapter {} drafts, audio, and progress".format(movie_id, chapter_id)
    elif is_test:
        message = "Sync 1-minute test checkpoint"
    else:
        message = "Sync trilogy checkpoint"
    config.log_progress("{} ({} files checked).".format(message, uploaded))
    config.git_auto_sync(message)


def main() -> None:
    arguments = parser("Sync artifacts to Hugging Face and progress to GitHub").parse_args()
    sync(arguments.test_mode, arguments.movie, arguments.chapter)


if __name__ == "__main__":
    main()
