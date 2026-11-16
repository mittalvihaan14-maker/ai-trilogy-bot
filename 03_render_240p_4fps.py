#!/usr/bin/env python3
"""Render each shot at 426x240 and 4 fps.

Location masters drive img2img so the place stays put. Character masters
drive IP-Adapter so the face stays put. Mouth sprites open and close from
the dialogue envelope. Finished shots are concatenated into a 240p chapter draft.
"""

from __future__ import annotations

import json
import logging
import shutil

from PIL import Image

import config
from pipeline_lib.cli import parser
from pipeline_lib import ffmpeg_tools
from pipeline_lib.hf_store import publish_state, store
from pipeline_lib import layout
from pipeline_lib.mouth import overlay
from pipeline_lib.state import State
from pipeline_lib import storyboard

LOGGER = logging.getLogger("trilogy")


def _reference(shot, bible):
    images = []
    for character_id in shot.get("character_ids") or []:
        path = layout.master_character(character_id)
        if not path.exists():
            store().ensure_local(path, layout.relative(path))
        if path.exists():
            images.append(Image.open(path).convert("RGB"))
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    height = 256
    resized = []
    for image in images:
        width = max(1, int(image.width * height / float(image.height)))
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (sum(image.width for image in resized), height), (255, 255, 255))
    x = 0
    for image in resized:
        sheet.paste(image, (x, 0))
        x += image.width
    return sheet


def _location_frame(shot, bible) -> Image.Image:
    path = layout.master_location(shot["location_id"])
    if not path.exists():
        store().ensure_local(path, layout.relative(path))
    if not path.exists():
        raise RuntimeError("Missing location master {}. Run 01_generate_master_assets.py first.".format(path))
    return Image.open(path).convert("RGB").resize((config.FRAME_WIDTH, config.FRAME_HEIGHT), Image.Resampling.LANCZOS)


def _location_seed(shot, bible) -> int:
    for location in bible["locations"]:
        if location["id"] == shot["location_id"]:
            return int(location.get("seed") or 1)
    return 1


def _render_shot(pipe, shot, bible) -> None:
    video = layout.shot_video(shot["id"])
    if video.exists() and video.stat().st_size > 0:
        return
    if store().ensure_local(video, layout.relative(video)):
        return
    wav_path = layout.audio_wav(shot["id"])
    if not wav_path.exists():
        store().ensure_local(wav_path, layout.relative(wav_path))
    if not wav_path.exists():
        raise RuntimeError("Missing audio for {}. Run 02_audio_and_sound_design.py first.".format(shot["id"]))
    mouth_path = layout.mouth_json(shot["id"])
    if mouth_path.exists():
        flags = json.loads(mouth_path.read_text(encoding="utf-8"))
    else:
        flags = [False] * int(shot["frames"])
    frames_dir = layout.frame_dir(shot["id"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    reference = _reference(shot, bible)
    location = _location_frame(shot, bible)
    previous = location
    base_seed = _location_seed(shot, bible)
    total = int(shot["frames"])
    LOGGER.info("Rendering %s (%s frames)", shot["id"], total)
    for index in range(total):
        frame_path = frames_dir / "f_{:04d}.png".format(index)
        if frame_path.exists():
            previous = Image.open(frame_path).convert("RGB")
            continue
        if index == 0:
            init = location
            strength = 0.62
        elif index % 8 == 0:
            init = Image.blend(previous.resize(location.size), location, 0.35)
            strength = 0.45
        else:
            init = previous
            strength = 0.32
        if reference is None:
            reference = location
        seed = (base_seed + int(shot["index"]) * 1009 + index * 17) % (2**31 - 1)
        image = pipe.frame(shot, bible, seed, init, reference, strength)
        image = overlay(image, list(shot.get("character_ids") or []), shot.get("speaker") or "", bool(flags[index] if index < len(flags) else False))
        image.save(frame_path, format="PNG")
        previous = image
    ffmpeg_tools.encode_shot(frames_dir, wav_path, video)
    shutil.rmtree(frames_dir, ignore_errors=True)


def _concat_chapter(movie, chapter, state: State) -> None:
    draft = layout.draft_video(movie["id"], chapter["id"])
    key = layout.chapter_key(movie["id"], chapter["id"])
    if draft.exists() and state.contains("draft_chapters", key):
        return
    paths = []
    for shot in chapter["shots"]:
        video = layout.shot_video(shot["id"])
        if not video.exists():
            store().ensure_local(video, layout.relative(video))
        if not video.exists():
            raise RuntimeError("Chapter draft is missing shot {}".format(shot["id"]))
        paths.append(video)
    LOGGER.info("Concatenating %s shots into %s", len(paths), draft.name)
    ffmpeg_tools.concat_copy(paths, draft)
    store().upload(draft, layout.relative(draft))
    state.add("draft_chapters", key)
    publish_state(state)
    for path in paths:
        if path.exists():
            path.unlink()


def render(is_test: bool, movie_id=None, chapter_id=None) -> None:
    config.bootstrap(is_test)
    ffmpeg_tools.require_ffmpeg()
    bible = storyboard.load_bible()
    state = State.load()
    from pipeline_lib.sd_engine import CartoonPipeline

    pipe = None
    try:
        for movie, chapter in storyboard.iter_chapters(bible):
            if movie_id and int(movie["id"]) != int(movie_id):
                continue
            if chapter_id and int(chapter["id"]) != int(chapter_id):
                continue
            key = layout.chapter_key(movie["id"], chapter["id"])
            draft = layout.draft_video(movie["id"], chapter["id"])
            if state.contains("draft_chapters", key) and (draft.exists() or store().ensure_local(draft, layout.relative(draft))):
                LOGGER.info("Draft already complete: %s", draft.name)
                for shot in chapter["shots"]:
                    video = layout.shot_video(shot["id"])
                    if video.exists():
                        video.unlink()
                continue
            for shot in chapter["shots"]:
                if state.contains("rendered_shots", shot["id"]) and layout.shot_video(shot["id"]).exists():
                    continue
                if pipe is None:
                    pipe = CartoonPipeline("img2img")
                _render_shot(pipe, shot, bible)
                video = layout.shot_video(shot["id"])
                store().upload(video, layout.relative(video))
                state.add("rendered_shots", shot["id"])
                publish_state(state)
            _concat_chapter(movie, chapter, state)
            config.log_progress(
                "240p draft ready for movie {} chapter {} ({}).".format(movie["id"], chapter["id"], chapter["title"])
            )
    finally:
        if pipe is not None:
            pipe.close()


def main() -> None:
    arguments = parser("Render 240p 4fps chapter drafts").parse_args()
    render(arguments.test_mode, arguments.movie, arguments.chapter)


if __name__ == "__main__":
    main()
