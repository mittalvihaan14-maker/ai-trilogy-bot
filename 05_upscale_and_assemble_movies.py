#!/usr/bin/env python3
"""Turn each 240p/4fps chapter into a 4K/60fps chapter, then join the movie.

Interpolation runs first at 240p (RIFE when rife-ncnn-vulkan is installed,
otherwise ffmpeg minterpolate). RealESRGAN_x4plus_anime_6B then scales every
frame, and a Lanczos step finishes the resize to 3840x2160. Chapters are
joined with the concat demuxer and stream copy, so the join does not re-encode.
"""

from __future__ import annotations

import logging
import os
import shutil

import config
from pipeline_lib.audio_fx import fit_length, read_wav, write_wav
from pipeline_lib.cli import parser
from pipeline_lib import ffmpeg_tools
from pipeline_lib.hf_store import publish_state, store
from pipeline_lib import layout
from pipeline_lib.state import State
from pipeline_lib import storyboard

LOGGER = logging.getLogger("trilogy")


def _keep_local_finals() -> bool:
    if os.getenv("KEEP_LOCAL_FINALS", "").lower() in {"1", "true", "yes"}:
        return True
    return not config.on_kaggle()


def _chapter_audio(movie, chapter):
    dest = layout.chapter_wav(movie["id"], chapter["id"])
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import numpy as np

    pieces = []
    complete = True
    for shot in chapter["shots"]:
        wav_path = layout.audio_wav(shot["id"])
        if not wav_path.exists():
            store().ensure_local(wav_path, layout.relative(wav_path))
        if not wav_path.exists():
            complete = False
            break
        audio, _rate = read_wav(wav_path)
        pieces.append(audio)
    if complete and pieces:
        mixed = np.concatenate(pieces, axis=0)
    else:
        draft = layout.draft_video(movie["id"], chapter["id"])
        if not draft.exists():
            store().ensure_local(draft, layout.relative(draft))
        if not draft.exists():
            raise RuntimeError("No audio or draft for movie {} chapter {}".format(movie["id"], chapter["id"]))
        ffmpeg_tools.extract_wav(draft, dest)
        audio, _rate = read_wav(dest)
        mixed = audio
    target = int(round(float(chapter["duration_sec"]) * config.SAMPLE_RATE))
    write_wav(dest, fit_length(mixed, target))
    return dest


def _remote_ready(path) -> bool:
    return store().remote_size(layout.relative(path)) is not None


def _valid_4k(path) -> bool:
    if not path.exists() or path.stat().st_size < 100000:
        return False
    try:
        probed = ffmpeg_tools.probe(path)
    except Exception:
        return False
    return (
        probed.width == config.FINAL_WIDTH
        and probed.height == config.FINAL_HEIGHT
        and abs((probed.fps or 0) - config.FINAL_FPS) < 0.2
        and probed.duration > 1
    )


def _publish_chapter(dest, movie, chapter, state: State, probed, upscaler_name: str, interpolator: str) -> None:
    key = layout.chapter_key(movie["id"], chapter["id"])
    store().upload(dest, layout.relative(dest))
    state.add("chapters_4k", key)
    state.data.setdefault("chapter_meta", {})[key] = {
        "upscaler": upscaler_name,
        "interpolator": interpolator,
        "duration_sec": probed.duration,
    }
    publish_state(state)
    draft = layout.draft_video(movie["id"], chapter["id"])
    audio = layout.chapter_wav(movie["id"], chapter["id"])
    if draft.exists():
        draft.unlink()
    if audio.exists():
        audio.unlink()
    config.log_progress(
        "4K chapter ready: movie {} chapter {} ({}) via {}.".format(
            movie["id"], chapter["id"], chapter["title"], upscaler_name
        )
    )
    config.git_auto_sync("Complete movie {} chapter {} at 4K 60fps".format(movie["id"], chapter["id"]))


def upscale_chapter(movie, chapter, state: State) -> None:
    key = layout.chapter_key(movie["id"], chapter["id"])
    dest = layout.chapter_4k(movie["id"], chapter["id"])
    if state.contains("chapters_4k", key) and (dest.exists() or _remote_ready(dest)):
        LOGGER.info("4K chapter already stored: %s", dest.name)
        return
    if _valid_4k(dest):
        probed = ffmpeg_tools.probe(dest)
        meta = (state.data.get("chapter_meta") or {}).get(key, {})
        LOGGER.info("Reusing existing 4K chapter %s", dest.name)
        _publish_chapter(
            dest,
            movie,
            chapter,
            state,
            probed,
            meta.get("upscaler") or "existing-encode",
            meta.get("interpolator") or "existing-interpolation",
        )
        return
    if dest.exists():
        dest.unlink()
    draft = layout.draft_video(movie["id"], chapter["id"])
    if not draft.exists():
        store().ensure_local(draft, layout.relative(draft))
    if not draft.exists():
        raise RuntimeError("Missing 240p draft {}. Render it before upscaling.".format(draft))
    audio = _chapter_audio(movie, chapter)
    mezzanine = layout.mezzanine_video(movie["id"], chapter["id"])
    note = mezzanine.with_suffix(".interpolator.txt")
    if mezzanine.exists() and mezzanine.stat().st_size > 0 and note.exists():
        interpolator = note.read_text(encoding="utf-8").strip() or "reused-interpolation"
        LOGGER.info("Reusing 60 fps mezzanine %s", mezzanine.name)
    else:
        LOGGER.info("Interpolating %s to 60 fps", draft.name)
        interpolator = ffmpeg_tools.minterpolate_to_60(draft, mezzanine)
        note.write_text(interpolator, encoding="utf-8")
    info = ffmpeg_tools.probe(mezzanine)
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError("Could not read frames from {}".format(mezzanine))
    import numpy as np
    from tqdm import tqdm
    from pipeline_lib.upscaler import Upscaler

    upscaler = Upscaler()
    target = int(round(float(chapter["duration_sec"]) * config.FINAL_FPS))
    frame_bytes = info.width * info.height * 3
    reader = ffmpeg_tools.open_raw_reader(mezzanine)
    writer = ffmpeg_tools.open_raw_writer(dest, config.FINAL_WIDTH, config.FINAL_HEIGHT, config.FINAL_FPS, audio)
    last = None
    try:
        if writer.stdin is None or reader.stdout is None:
            raise RuntimeError("ffmpeg pipes did not open for {}".format(key))
        for _index in tqdm(range(target), desc="4K {}".format(key)):
            raw = ffmpeg_tools.read_exact(reader.stdout, frame_bytes)
            if raw is None:
                if last is None:
                    raise RuntimeError("Interpolation produced no frames for {}".format(key))
                frame = last
            else:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((info.height, info.width, 3)).copy()
                last = frame
            scaled = np.ascontiguousarray(upscaler.upscale(frame))
            writer.stdin.write(scaled.tobytes())
        ffmpeg_tools.close_writer(writer)
        writer = None
    except Exception:
        if writer is not None and writer.poll() is None:
            writer.kill()
            writer.wait()
        if dest.exists():
            dest.unlink()
        raise
    finally:
        if reader.poll() is None:
            reader.kill()
        reader.wait()
    if mezzanine.exists():
        mezzanine.unlink()
    if note.exists():
        note.unlink()
    probed = ffmpeg_tools.probe(dest)
    LOGGER.info(
        "Chapter %s encoded: %.2fs %sx%s %.2ffps via %s / %s",
        key,
        probed.duration,
        probed.width,
        probed.height,
        probed.fps,
        upscaler.kind,
        interpolator,
    )
    _publish_chapter(dest, movie, chapter, state, probed, upscaler.kind, interpolator)


def _chapter_styles(state: State, movie) -> tuple:
    upscalers = []
    interpolators = []
    for chapter in movie["chapters"]:
        meta = (state.data.get("chapter_meta") or {}).get(layout.chapter_key(movie["id"], chapter["id"]), {})
        if meta.get("upscaler"):
            upscalers.append(meta["upscaler"])
        if meta.get("interpolator"):
            interpolators.append(meta["interpolator"])
    return (
        ",".join(sorted(set(upscalers))) or "unknown",
        ",".join(sorted(set(interpolators))) or "unknown",
    )


def assemble_movie(movie, state: State) -> None:
    number = str(movie["id"])
    final_path = layout.final_movie(movie["id"])
    recorded = state.data.get("movies", {}).get(number) or {}
    if recorded.get("url") and _remote_ready(final_path):
        LOGGER.info("Movie %s already on Hugging Face", number)
        return
    work = config.output_dir() / "work"
    frames = config.output_dir() / "frames"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(frames, ignore_errors=True)
    chapters = []
    for chapter in movie["chapters"]:
        path = layout.chapter_4k(movie["id"], chapter["id"])
        if not path.exists():
            store().ensure_local(path, layout.relative(path))
        if not path.exists():
            raise RuntimeError("Movie {} is missing 4K chapter {}".format(number, chapter["id"]))
        chapters.append(path)
    LOGGER.info("Joining %s chapters into %s", len(chapters), final_path.name)
    ffmpeg_tools.concat_copy(chapters, final_path)
    probed = ffmpeg_tools.probe(final_path)
    repo_path = store().upload(final_path, layout.relative(final_path))
    url = config.hf_file_url(repo_path)
    upscaler_name, interpolator_name = _chapter_styles(state, movie)
    local_path = str(final_path)
    if not _keep_local_finals():
        for path in chapters:
            if path.exists():
                path.unlink()
        if final_path.exists():
            final_path.unlink()
        local_path = ""
    state.data.setdefault("movies", {})[number] = {
        "title": movie.get("title") or "Movie {}".format(number),
        "duration_sec": probed.duration,
        "width": probed.width,
        "height": probed.height,
        "fps": probed.fps,
        "audio": probed.audio_codec,
        "url": url,
        "local_path": local_path,
        "upscaler": upscaler_name,
        "interpolator": interpolator_name,
    }
    state.add("movies_done", number)
    publish_state(state)
    config.log_progress(
        "Movie {} finished: {} ({:.2f}s) {}".format(number, movie.get("title"), probed.duration, url)
    )
    config.git_auto_sync("Finish movie {} and publish the 4K file".format(number))


def upscale_all(is_test: bool, movie_id=None, chapter_id=None, upscale_only: bool = False, assemble=None) -> None:
    config.bootstrap(is_test)
    ffmpeg_tools.require_ffmpeg()
    bible = storyboard.load_bible()
    state = State.load()
    if assemble:
        for movie in bible["movies"]:
            if int(movie["id"]) == int(assemble):
                assemble_movie(movie, state)
                return
        raise RuntimeError("No movie {}".format(assemble))
    for movie, chapter in storyboard.iter_chapters(bible):
        if movie_id and int(movie["id"]) != int(movie_id):
            continue
        if chapter_id and int(chapter["id"]) != int(chapter_id):
            continue
        upscale_chapter(movie, chapter, state)
    if upscale_only:
        return
    for movie in bible["movies"]:
        if movie_id and int(movie["id"]) != int(movie_id):
            continue
        assemble_movie(movie, state)


def main() -> None:
    arg_parser = parser("Upscale chapters to 4K 60fps and assemble each movie")
    arg_parser.add_argument("--upscale-only", action="store_true", help="Encode one chapter and do not join the movie")
    arg_parser.add_argument("--assemble", type=int, help="Join the finished chapters of this movie number")
    arguments = arg_parser.parse_args()
    upscale_all(
        arguments.test_mode,
        arguments.movie,
        arguments.chapter,
        upscale_only=arguments.upscale_only,
        assemble=arguments.assemble,
    )


if __name__ == "__main__":
    main()
