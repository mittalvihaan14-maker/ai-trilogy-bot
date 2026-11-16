"""Story-bible timing lock.

Full movies are 6 chapters x 120 shots x 8.0 seconds = 96 minutes.
That sits inside the required 90–100 minute window, and every shot stays
inside the 6–10 second range. Test mode is 8 shots x 7.5 seconds = 60 seconds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

import config
from pipeline_lib.prompts import clip_words

CAMERAS = (
    "wide establishing view",
    "slow push-in",
    "gentle pan across the background",
    "close view of a simple cartoon prop",
    "silhouette against a flat sky",
    "small idle bounce",
    "over-the-shoulder cartoon angle",
    "low angle looking up at a simple shape",
)


def profile(is_test: bool) -> Dict:
    if is_test:
        shots = config.TEST_SHOTS
        shot_seconds = config.TEST_SHOT_SECONDS
        chapters = config.TEST_CHAPTERS
        movies = config.TEST_MOVIES
    else:
        shots = config.FULL_SHOTS_PER_CHAPTER
        shot_seconds = config.FULL_SHOT_SECONDS
        chapters = config.FULL_CHAPTERS
        movies = config.FULL_MOVIES
    return {
        "test_mode": bool(is_test),
        "movies": movies,
        "chapters": chapters,
        "shots": shots,
        "shot_seconds": float(shot_seconds),
        "chapter_seconds": float(shots * shot_seconds),
        "movie_seconds": float(chapters * shots * shot_seconds),
        "frames_per_shot": int(round(shot_seconds * config.DRAFT_FPS)),
    }


def bible_path():
    return config.output_dir() / "story_bible.json"


def load_bible() -> Dict:
    return json.loads(bible_path().read_text(encoding="utf-8"))


def save_bible(bible: Dict) -> None:
    path = bible_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(bible, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def idea_hash() -> str:
    text = config.STORY_PATH.read_text(encoding="utf-8") if config.STORY_PATH.exists() else ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str, fallback: str) -> str:
    cleaned = []
    for char in str(text or "").lower():
        if char.isascii() and char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug[:24] or fallback).strip("_") or fallback


def _unique_slug(text: str, fallback: str, used: set) -> str:
    base = _slug(text, fallback)
    candidate = base
    number = 2
    while candidate in used:
        suffix = "_{}".format(number)
        candidate = base[: 24 - len(suffix)] + suffix
        number += 1
    used.add(candidate)
    return candidate


def _take_exact(items: List[Dict], count: int, make_filler) -> List[Dict]:
    if count <= 0:
        return []
    if not items:
        return [make_filler(index) for index in range(count)]
    if len(items) == count:
        return [dict(item) for item in items]
    if len(items) > count:
        if count == 1:
            return [dict(items[0])]
        picked = []
        last = len(items) - 1
        for index in range(count):
            source = int(round(index * last / float(count - 1)))
            picked.append(dict(items[source]))
        return picked
    padded = [dict(item) for item in items]
    while len(padded) < count:
        padded.append(make_filler(len(padded)))
    return padded


def _default_character(index: int) -> Dict:
    palette = (
        ("Pip Lark", "square lantern face, teal coat, round ink eyes, short black hair"),
        ("Nia Quill", "tall triangle silhouette, amber scarf, oval eyes, long straight hair"),
        ("Bram Holt", "wide trapezoid torso, rust mail coat, tiny round glasses"),
        ("Oddment", "small folded-ticket fox, cream triangles, dot eyes, stub tail"),
    )
    name, visual = palette[index % len(palette)]
    return {"name": name, "visual": visual, "personality": "loyal and curious"}


def _default_location(index: int) -> Dict:
    places = (
        ("Lumenfold Square", "flat pastel plaza, round paper sun, simple arched doors"),
        ("Round Harbor", "circular teal bay, blocky boats, bold white wave lines"),
        ("Clockwork Orchard", "orange ticking trees, round fruit, striped terraces"),
        ("Atlas Binding", "giant open book, ink rivers, floating paper stairs"),
    )
    name, visual = places[index % len(places)]
    return {"name": name, "visual": visual}


def _sanitize_character(raw: Dict, index: int, used: set) -> Dict:
    fallback = _default_character(index)
    name = clip_words(raw.get("name") or fallback["name"], 4) or fallback["name"]
    return {
        "id": _unique_slug(raw.get("id") or name, "hero_{}".format(index + 1), used),
        "name": name,
        "visual": clip_words(raw.get("visual") or raw.get("description") or fallback["visual"], 18),
        "personality": clip_words(raw.get("personality") or fallback["personality"], 16),
    }


def _sanitize_location(raw: Dict, index: int, used: set) -> Dict:
    fallback = _default_location(index)
    name = clip_words(raw.get("name") or fallback["name"], 5) or fallback["name"]
    return {
        "id": _unique_slug(raw.get("id") or name, "place_{}".format(index + 1), used),
        "name": name,
        "visual": clip_words(raw.get("visual") or raw.get("description") or fallback["visual"], 18),
    }


def _filler_shot(bible: Dict, chapter: Dict, ordinal: int) -> Dict:
    locations = bible["locations"]
    characters = bible["characters"]
    location = locations[ordinal % len(locations)]
    cast = [characters[ordinal % len(characters)]["id"]]
    if len(characters) > 1 and ordinal % 2 == 0:
        cast.append(characters[(ordinal + 1) % len(characters)]["id"])
    synopsis = clip_words(chapter.get("synopsis") or chapter.get("title") or "the story continues", 16)
    return {
        "location_id": location["id"],
        "character_ids": cast,
        "action_prompt": "{}, {}".format(CAMERAS[ordinal % len(CAMERAS)], synopsis),
        "speaker": "",
        "dialogue": "",
        "mood": chapter.get("mood") or "peaceful",
        "sfx_cue": "soft wind",
    }


def _sanitize_shot(raw: Dict, bible: Dict, chapter: Dict, ordinal: int) -> Dict:
    if not isinstance(raw, dict):
        raw = {}
    location_ids = [item["id"] for item in bible["locations"]]
    location_names = {str(item["name"]).lower(): item["id"] for item in bible["locations"]}
    character_ids = [item["id"] for item in bible["characters"]]
    character_names = {str(item["name"]).lower(): item["id"] for item in bible["characters"]}
    location_id = raw.get("location_id")
    if location_id not in location_ids:
        location_id = location_names.get(str(location_id or "").lower(), location_ids[ordinal % len(location_ids)])
    cast = []
    for character_id in raw.get("character_ids") or []:
        resolved = character_id if character_id in character_ids else character_names.get(str(character_id).lower())
        if resolved and resolved not in cast:
            cast.append(resolved)
    if not cast:
        cast = [character_ids[ordinal % len(character_ids)]]
    cast = cast[:3]
    dialogue = str(raw.get("dialogue") or "").replace("<", " ").replace(">", " ").strip()
    if len(dialogue) > 240:
        dialogue = dialogue[:240].rsplit(" ", 1)[0]
    speaker = str(raw.get("speaker") or "").strip()
    if speaker.lower() in character_names:
        speaker = character_names[speaker.lower()]
    if speaker not in cast and speaker != "narrator":
        speaker = cast[0] if dialogue else ""
    if not dialogue:
        speaker = ""
    mood = str(raw.get("mood") or chapter.get("mood") or "peaceful").strip().lower()
    if mood not in config.MOODS:
        for candidate in config.MOODS:
            if candidate in mood:
                mood = candidate
                break
        else:
            mood = "peaceful"
    sfx = str(raw.get("sfx_cue") or raw.get("sfx") or "soft wind").strip()
    return {
        "location_id": location_id,
        "character_ids": cast,
        "action_prompt": clip_words(raw.get("action_prompt") or raw.get("action") or "Characters hold a cartoon pose", 22),
        "speaker": speaker,
        "dialogue": dialogue,
        "mood": mood,
        "sfx_cue": clip_words(sfx, 12) or "soft wind",
    }


def _lock_shots(shots: List[Dict], movie_number: int, chapter_number: int, spec: Dict) -> List[Dict]:
    locked = []
    for index, shot in enumerate(shots, start=1):
        item = dict(shot)
        item["id"] = "m{}c{}s{:03d}".format(movie_number, chapter_number, index)
        item["index"] = index
        item["duration_sec"] = spec["shot_seconds"]
        item["frames"] = spec["frames_per_shot"]
        locked.append(item)
    return locked


def ensure_outline(bible: Dict, is_test: bool) -> Dict:
    """Pad movies and chapters to the required grid without inventing shots."""
    spec = profile(is_test)
    used_characters = set()  # type: set
    raw_characters = bible.get("characters") if isinstance(bible.get("characters"), list) else []
    characters = [
        _sanitize_character(item if isinstance(item, dict) else {}, index, used_characters)
        for index, item in enumerate(raw_characters[:8])
    ]
    if len(characters) < 3:
        while len(characters) < 3:
            characters.append(_sanitize_character(_default_character(len(characters)), len(characters), used_characters))
    used_locations = set()  # type: set
    raw_locations = bible.get("locations") if isinstance(bible.get("locations"), list) else []
    locations = [
        _sanitize_location(item if isinstance(item, dict) else {}, index, used_locations)
        for index, item in enumerate(raw_locations[:8])
    ]
    if not locations:
        locations.append(_sanitize_location(_default_location(0), 0, used_locations))

    def make_movie(index: int) -> Dict:
        return {
            "title": "Original Chapter Road {}".format(index + 1),
            "synopsis": clip_words(bible.get("logline") or "The companions choose a new road.", 40),
            "chapters": [],
        }

    raw_movies = bible.get("movies") if isinstance(bible.get("movies"), list) else []
    movies = _take_exact([item for item in raw_movies if isinstance(item, dict)], spec["movies"], make_movie)
    normalized_movies = []
    for movie_index, movie in enumerate(movies, start=1):
        def make_chapter(index: int, movie=movie) -> Dict:
            return {
                "title": "{} — part {}".format(clip_words(movie.get("title") or "Movie", 6), index + 1),
                "synopsis": clip_words(movie.get("synopsis") or bible.get("logline") or "The map keeps moving.", 40),
                "mood": "hopeful",
                "shots": [],
            }

        raw_chapters = movie.get("chapters") if isinstance(movie.get("chapters"), list) else []
        chapters = _take_exact([item for item in raw_chapters if isinstance(item, dict)], spec["chapters"], make_chapter)
        clean_chapters = []
        for chapter_index, chapter in enumerate(chapters, start=1):
            mood = str(chapter.get("mood") or "hopeful").lower()
            if mood not in config.MOODS:
                mood = "hopeful"
            clean_chapters.append(
                {
                    "id": chapter_index,
                    "title": clip_words(chapter.get("title") or "Chapter {}".format(chapter_index), 8),
                    "synopsis": clip_words(chapter.get("synopsis") or movie.get("synopsis") or "", 80),
                    "mood": mood,
                    "shots": chapter.get("shots") if isinstance(chapter.get("shots"), list) else [],
                }
            )
        normalized_movies.append(
            {
                "id": movie_index,
                "title": clip_words(movie.get("title") or "Movie {}".format(movie_index), 8),
                "synopsis": clip_words(movie.get("synopsis") or bible.get("logline") or "", 80),
                "chapters": clean_chapters,
            }
        )
    bible["title"] = clip_words(bible.get("title") or "The Lantern Cartographers", 8)
    bible["logline"] = clip_words(bible.get("logline") or "Original cartoon companions redraw a living map.", 40)
    bible["characters"] = characters
    bible["locations"] = locations
    bible["movies"] = normalized_movies
    return bible


def normalize_bible(bible: Dict, is_test: bool) -> Dict:
    spec = profile(is_test)
    ensure_outline(bible, is_test)
    for character in bible["characters"]:
        character["seed"] = stable_seed("character", character["id"])
    for index, character in enumerate(bible["characters"]):
        character["voice"] = config.VOICE_POOL[index % len(config.VOICE_POOL)]
    for location in bible["locations"]:
        location["seed"] = stable_seed("location", location["id"])
    for movie in bible["movies"]:
        for chapter in movie["chapters"]:
            raw_shots = [shot for shot in chapter.get("shots") or [] if isinstance(shot, dict)]

            def make_filler(ordinal: int, chapter=chapter) -> Dict:
                return _filler_shot(bible, chapter, ordinal)

            shots = _take_exact(raw_shots, spec["shots"], make_filler)
            clean = [
                _sanitize_shot(shot, bible, chapter, ordinal) for ordinal, shot in enumerate(shots)
            ]
            chapter["shots"] = _lock_shots(clean, movie["id"], chapter["id"], spec)
            chapter["duration_sec"] = spec["chapter_seconds"]
            chapter["frames"] = spec["shots"] * spec["frames_per_shot"]
        movie["duration_sec"] = spec["movie_seconds"]
        movie["frames_at_4fps"] = spec["chapters"] * spec["shots"] * spec["frames_per_shot"]
    if not is_test:
        seconds = spec["movie_seconds"]
        if not (config.MOVIE_MIN_SECONDS <= seconds <= config.MOVIE_MAX_SECONDS):
            raise RuntimeError("Locked movie length {}s is outside 90–100 minutes".format(seconds))
    bible["runtime"] = {
        "mode": "test" if is_test else "full",
        "idea_hash": idea_hash(),
        "movie_count": spec["movies"],
        "chapters_per_movie": spec["chapters"],
        "shots_per_chapter": spec["shots"],
        "shot_seconds": spec["shot_seconds"],
        "chapter_seconds": spec["chapter_seconds"],
        "movie_seconds": spec["movie_seconds"],
        "draft_fps": config.DRAFT_FPS,
        "final_fps": config.FINAL_FPS,
        "final_resolution": [config.FINAL_WIDTH, config.FINAL_HEIGHT],
    }
    return bible


def iter_chapters(bible: Dict):
    for movie in bible.get("movies") or []:
        for chapter in movie.get("chapters") or []:
            yield movie, chapter


def find_chapter(bible: Dict, movie_id: int, chapter_id: int):
    for movie, chapter in iter_chapters(bible):
        if int(movie["id"]) == int(movie_id) and int(chapter["id"]) == int(chapter_id):
            return movie, chapter
    raise KeyError("Missing movie {} chapter {}".format(movie_id, chapter_id))


def planned_report(bible: Dict) -> List[str]:
    lines = []
    runtime = bible.get("runtime") or {}
    lines.append(
        "Mode {mode}: {movie_count} movie(s), {chapters_per_movie} chapter(s), "
        "{shots_per_chapter} shots x {shot_seconds:.1f}s = {movie_seconds:.0f}s per movie.".format(
            **{
                "mode": runtime.get("mode", "full"),
                "movie_count": runtime.get("movie_count", 0),
                "chapters_per_movie": runtime.get("chapters_per_movie", 0),
                "shots_per_chapter": runtime.get("shots_per_chapter", 0),
                "shot_seconds": float(runtime.get("shot_seconds") or 0),
                "movie_seconds": float(runtime.get("movie_seconds") or 0),
            }
        )
    )
    for movie in bible.get("movies") or []:
        frames = int(movie.get("frames_at_4fps") or 0)
        lines.append(
            "Movie {}: {} — {:.0f}s, {} draft frames at {} fps.".format(
                movie["id"], movie["title"], float(movie["duration_sec"]), frames, config.DRAFT_FPS
            )
        )
    return lines


def self_check() -> None:
    from pipeline_lib.prompts import character_master_prompt, shot_prompt

    short = {
        "title": "Check",
        "logline": "A test of the timing lock.",
        "characters": [{"name": "Pip", "visual": "square teal face and round eyes"}],
        "locations": [{"name": "Harbor", "visual": "round teal bay and blocky boats"}],
        "movies": [
            {
                "title": "Only One",
                "synopsis": "The harbor forgets the sea.",
                "chapters": [
                    {
                        "title": "Opening",
                        "synopsis": "Boats wake up in a meadow.",
                        "shots": [
                            {
                                "action_prompt": "Pip lifts the square lantern",
                                "dialogue": "The sea moved inland.",
                                "speaker": "pip",
                                "location_id": "missing",
                                "character_ids": [],
                                "mood": "hopeful",
                                "sfx_cue": "soft wind",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    full = normalize_bible(json.loads(json.dumps(short)), False)
    assert len(full["movies"]) == 3
    assert len(full["characters"]) >= 3
    for movie in full["movies"]:
        assert abs(movie["duration_sec"] - 5760) < 0.01
        assert config.MOVIE_MIN_SECONDS <= movie["duration_sec"] <= config.MOVIE_MAX_SECONDS
        assert len(movie["chapters"]) == 6
        assert movie["frames_at_4fps"] == 6 * 120 * 32
        for chapter in movie["chapters"]:
            assert len(chapter["shots"]) == 120
            assert abs(sum(shot["duration_sec"] for shot in chapter["shots"]) - 960) < 0.01
            for shot in chapter["shots"]:
                assert shot["frames"] == 32
                assert 6 <= shot["duration_sec"] <= 10
    assert config.STYLE_SUFFIX in shot_prompt(full["movies"][0]["chapters"][0]["shots"][0], full)
    assert "photorealistic" in config.NEGATIVE_PROMPT
    assert "flipaclip" in character_master_prompt(full["characters"][0]).lower()
    tiny = normalize_bible(json.loads(json.dumps(short)), True)
    assert len(tiny["movies"]) == 1
    assert len(tiny["movies"][0]["chapters"]) == 1
    assert len(tiny["movies"][0]["chapters"][0]["shots"]) == 8
    assert abs(tiny["movies"][0]["duration_sec"] - 60) < 0.01
    assert tiny["movies"][0]["chapters"][0]["shots"][0]["frames"] == 30
    ids = [shot["id"] for shot in tiny["movies"][0]["chapters"][0]["shots"]]
    assert ids[0] == "m1c1s001" and ids[-1] == "m1c1s008"
    print("storyboard self-check ok")


if __name__ == "__main__":
    self_check()
