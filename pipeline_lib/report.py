"""Final delivery report."""

from __future__ import annotations

from typing import Dict, List

import config
from pipeline_lib.state import State


def format_duration(seconds: float) -> str:
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, secs)
    return "{}:{:02d}".format(minutes, secs)


def duration_ok(seconds: float, is_test: bool) -> bool:
    if is_test:
        return 58.0 <= seconds <= 62.0
    return config.MOVIE_MIN_SECONDS <= seconds <= config.MOVIE_MAX_SECONDS


def specs_ok(record: Dict) -> bool:
    width = int(record.get("width") or 0)
    height = int(record.get("height") or 0)
    fps = float(record.get("fps") or 0)
    return width == config.FINAL_WIDTH and height == config.FINAL_HEIGHT and abs(fps - config.FINAL_FPS) < 0.2


def build_lines(bible: Dict, state: State) -> List[str]:
    lines = [
        "============================================================",
        "Final Delivery Report",
        "============================================================",
    ]
    movies = bible.get("movies") or []
    complete = True
    for movie in movies:
        number = str(movie["id"])
        record = state.data.get("movies", {}).get(number)
        title = movie.get("title") or "Movie {}".format(number)
        if not record:
            complete = False
            lines.append("Movie {} — {}".format(number, title))
            lines.append("  Status: not finished")
            continue
        seconds = float(record.get("duration_sec") or 0)
        ok_time = duration_ok(seconds, config.test_mode())
        ok_specs = specs_ok(record)
        if not (ok_time and ok_specs):
            complete = False
        lines.append("Movie {} — {}".format(number, title))
        lines.append("  Duration: {} ({:.2f} seconds)".format(format_duration(seconds), seconds))
        lines.append(
            "  Video: {}x{} @ {:.2f} fps | audio: {}".format(
                record.get("width"), record.get("height"), float(record.get("fps") or 0), record.get("audio") or "aac"
            )
        )
        lines.append("  Upscaler: {}".format(record.get("upscaler") or "unknown"))
        lines.append("  Interpolation: {}".format(record.get("interpolator") or "unknown"))
        lines.append("  Hugging Face: {}".format(record.get("url")))
        if record.get("local_path"):
            lines.append("  Local file: {}".format(record.get("local_path")))
    lines.append("Dataset: https://huggingface.co/datasets/{}".format(config.HF_REPO_ID))
    lines.append("GitHub: https://github.com/{}".format(config.GITHUB_REPO))
    lines.append("Result: {}".format("complete" if complete else "incomplete"))
    lines.append("============================================================")
    return lines


def emit(bible: Dict, state: State) -> bool:
    lines = build_lines(bible, state)
    print("\n".join(lines))
    return "Result: complete" in lines
