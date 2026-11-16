#!/usr/bin/env python3
"""Expand story_idea.txt into output/story_bible.json with Gemini 2.5 Flash.

Full mode: 3 movies, 6 chapters each, 120 shots of 8 seconds (96 minutes).
Test mode: 1 movie, 1 chapter, 8 shots of 7.5 seconds (60 seconds).
Shots that the model skips are filled with original hold shots so the clock
still lands on time. Restarting the script resumes at the first unfinished batch.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List

import config
from pipeline_lib.cli import parser
from pipeline_lib.state import State
from pipeline_lib import storyboard

LOGGER = logging.getLogger("trilogy")
SYSTEM = (
    "You are the story editor for an original general-audience 2D cartoon trilogy. "
    "Invent original characters only. Never use real people, celebrities, public figures, "
    "or characters from existing films, series, books, or games. "
    "Describe faces as exaggerated geometric cartoons. Reply with JSON only."
)


def parse_json(text: str):
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        obj = body.find("{")
        arr = body.find("[")
        if obj == -1 and arr == -1:
            raise
        if arr != -1 and (obj == -1 or arr < obj):
            end = body.rfind("]")
            return json.loads(body[arr : end + 1])
        end = body.rfind("}")
        return json.loads(body[obj : end + 1])


def gemini_json(prompt: str, max_tokens: int = 8192):
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    last_error = None
    for attempt in range(5):
        try:
            cfg = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                system_instruction=SYSTEM,
            )
            try:
                response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt, config=cfg)
            except TypeError:
                response = client.models.generate_content(
                    model="gemini-3.8-flash", contents=prompt, generation_config=cfg
                )
            text = getattr(response, "text", None) or ""
            if not text:
                raise RuntimeError("Gemini returned no text")
            return parse_json(text)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Gemini attempt %s failed: %s", attempt + 1, config.scrub(str(exc)))
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini request failed: {}".format(config.scrub(str(last_error))))


def unwrap_overview(payload: Dict) -> Dict:
    if not isinstance(payload, dict):
        raise RuntimeError("Story overview was not a JSON object")
    for key in ("bible", "series", "story", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict) and ("movies" in nested or "characters" in nested):
            return nested
    return payload


def unwrap_shots(payload) -> List[Dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("shots", "storyboard", "scenes"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    raise RuntimeError("Shot batch did not contain shots")


def overview_prompt(spec: Dict, idea: str) -> str:
    return (
        "Expand this original story idea into a cartoon trilogy outline.\n\n"
        "STORY IDEA:\n{idea}\n\n"
        "Return JSON with keys title, logline, characters, locations, movies.\n"
        "characters: {chars} items with name, visual (under 15 words, geometric cartoon, no realism), personality.\n"
        "locations: {places} recurring places with name and visual (under 15 words, empty of people).\n"
        "movies: exactly {movies} items. Each movie needs title, synopsis, and chapters.\n"
        "Each movie has exactly {chapters} chapters. Each chapter needs title, synopsis (60-90 words), "
        "and mood, one of {moods}.\n"
        "Each movie must have its own beginning, middle, and ending. Movie 3 pays off the trilogy.\n"
        "Do not include shots yet.\n"
    ).format(
        idea=idea.strip(),
        chars=4 if not spec["test_mode"] else 3,
        places=5 if not spec["test_mode"] else 2,
        movies=spec["movies"],
        chapters=spec["chapters"],
        moods=", ".join(config.MOODS),
    )


def shot_prompt(bible: Dict, movie: Dict, chapter: Dict, start: int, count: int, spec: Dict, previous: str) -> str:
    characters = ", ".join("{} ({})".format(item["id"], item["name"]) for item in bible["characters"])
    locations = ", ".join("{} ({})".format(item["id"], item["name"]) for item in bible["locations"])
    return (
        "Write {count} consecutive original cartoon shots, numbers {start} through {end}, "
        "for movie \"{movie}\" chapter \"{chapter}\".\n"
        "Movie synopsis: {movie_syn}\n"
        "Chapter synopsis: {chapter_syn}\n"
        "Previous shot: {previous}\n"
        "Allowed character ids: {characters}\n"
        "Allowed location ids: {locations}\n"
        "Each shot is exactly {seconds:.1f} seconds. action_prompt under 20 words. "
        "dialogue under 16 words, or an empty string when nobody speaks. "
        "speaker is a character id, narrator, or an empty string.\n"
        "mood is one of {moods}. sfx_cue is a short list of sounds such as wind, birds, footsteps, door, magic, rain.\n"
        "Return JSON {{\"shots\": [{{location_id, character_ids, action_prompt, speaker, dialogue, mood, sfx_cue}}]}}.\n"
        "Advance the chapter plot across the batch. Keep faces cartoon-simple.\n"
    ).format(
        count=count,
        start=start,
        end=start + count - 1,
        movie=movie["title"],
        chapter=chapter["title"],
        movie_syn=movie.get("synopsis") or "",
        chapter_syn=chapter.get("synopsis") or "",
        previous=previous or "This is the first shot of the chapter.",
        characters=characters,
        locations=locations,
        seconds=spec["shot_seconds"],
        moods=", ".join(config.MOODS),
    )


def fill_chapter(bible: Dict, movie: Dict, chapter: Dict, spec: Dict) -> None:
    target = spec["shots"]
    shots = [shot for shot in chapter.get("shots") or [] if isinstance(shot, dict)]
    batch_size = target if spec["test_mode"] else 12
    while len(shots) < target:
        count = min(batch_size, target - len(shots))
        start = len(shots) + 1
        previous = ""
        if shots:
            last = shots[-1]
            previous = "{} {}".format(last.get("action_prompt", ""), last.get("dialogue", "")).strip()
        try:
            payload = gemini_json(shot_prompt(bible, movie, chapter, start, count, spec, previous), max_tokens=6144)
            fresh = unwrap_shots(payload)
        except Exception as exc:
            LOGGER.warning(
                "Shot batch %s-%s failed (%s). Filling the gap with original hold shots.",
                start,
                start + count - 1,
                config.scrub(str(exc)),
            )
            fresh = []
        cleaned = []
        for offset, raw in enumerate(fresh[:count]):
            cleaned.append(storyboard._sanitize_shot(raw, bible, chapter, len(shots) + offset))
        while len(cleaned) < count:
            ordinal = len(shots) + len(cleaned)
            cleaned.append(storyboard._sanitize_shot(storyboard._filler_shot(bible, chapter, ordinal), bible, chapter, ordinal))
        shots.extend(cleaned)
        chapter["shots"] = shots
        storyboard.save_bible(bible)
        LOGGER.info(
            "Movie %s chapter %s: %s/%s shots written",
            movie["id"],
            chapter["id"],
            len(shots),
            target,
        )


def generate(is_test: bool, fresh: bool) -> Dict:
    config.bootstrap(is_test)
    config.require_secrets(gemini=True, hf=False, github=False)
    state = State.load()
    spec = storyboard.profile(is_test)
    path = storyboard.bible_path()
    current_hash = storyboard.idea_hash()
    if fresh and path.exists():
        path.unlink()
        state.data["sets"] = {}
        state.data.pop("bible_ready", None)
        state.data.pop("overview_ready", None)
        state.save()
    rendered = (state.data.get("sets") or {}).get("rendered_shots") or []
    if path.exists() and state.get("bible_ready") and state.get("idea_hash") == current_hash and not fresh:
        LOGGER.info("Story bible already complete: %s", path)
        return storyboard.load_bible()
    if path.exists() and rendered and state.get("idea_hash") not in (None, current_hash):
        LOGGER.warning("story_idea.txt changed after shots were rendered. Keeping the bible those shots belong to.")
        return storyboard.load_bible()
    if path.exists() and state.get("overview_ready") and state.get("idea_hash") == current_hash and not fresh:
        bible = storyboard.load_bible()
    else:
        idea = config.STORY_PATH.read_text(encoding="utf-8")
        LOGGER.info("Asking gemini-3.8-flash for the trilogy outline")
        bible = unwrap_overview(gemini_json(overview_prompt(spec, idea), max_tokens=8192))
        storyboard.ensure_outline(bible, is_test)
        storyboard.save_bible(bible)
        state.set("overview_ready", True)
        state.set("idea_hash", current_hash)
        bible = storyboard.load_bible()
    for movie, chapter in storyboard.iter_chapters(bible):
        key = "m{}c{}".format(movie["id"], chapter["id"])
        if state.contains("bible_chapters", key) and len(chapter.get("shots") or []) >= spec["shots"]:
            continue
        LOGGER.info("Writing shots for movie %s chapter %s (%s)", movie["id"], chapter["id"], chapter["title"])
        fill_chapter(bible, movie, chapter, spec)
        state.add("bible_chapters", key)
    bible = storyboard.normalize_bible(storyboard.load_bible(), is_test)
    storyboard.save_bible(bible)
    state.set("bible_ready", True)
    state.set("idea_hash", current_hash)
    for line in storyboard.planned_report(bible):
        config.log_progress(line)
    config.log_progress("Story bible saved ({} bytes).".format(path.stat().st_size))
    return bible


def main() -> None:
    arguments = parser("Generate the trilogy story bible").parse_args()
    generate(arguments.test_mode, arguments.fresh)


if __name__ == "__main__":
    main()
