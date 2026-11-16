"""Prompt assembly for the locked 2D cartoon look."""

from __future__ import annotations

from typing import Dict, Iterable, List

import config


def clip_words(text: str, limit: int) -> str:
    words = str(text or "").replace("\n", " ").split()
    return " ".join(words[:limit]).strip()


def with_style(prompt: str) -> str:
    body = str(prompt or "").strip().rstrip(" ,.")
    if "flipaclip" in body.lower():
        return body
    return body + config.STYLE_SUFFIX


def character_names(shot: Dict, bible: Dict) -> List[str]:
    wanted = set(shot.get("character_ids") or [])
    names = []
    for character in bible.get("characters") or []:
        if character.get("id") in wanted:
            names.append(str(character.get("name") or character.get("id")))
    return names[:3]


def location_name(shot: Dict, bible: Dict) -> str:
    wanted = shot.get("location_id")
    for location in bible.get("locations") or []:
        if location.get("id") == wanted:
            return str(location.get("name") or location.get("id"))
    return str(wanted or "cartoon location")


def shot_prompt(shot: Dict, bible: Dict) -> str:
    action = clip_words(shot.get("action_prompt") or "Characters hold a simple cartoon pose", 22)
    names = ", ".join(character_names(shot, bible)) or "the cartoon cast"
    place = location_name(shot, bible)
    return with_style("{}, {} in {}".format(action, names, place))


def character_master_prompt(character: Dict) -> str:
    name = character.get("name") or character.get("id")
    visual = clip_words(character.get("visual") or "exaggerated geometric cartoon hero", 18)
    return with_style(
        "{}, {}, single cartoon character, front view, full body, centered, "
        "plain flat background".format(name, visual)
    )


def location_master_prompt(location: Dict) -> str:
    name = location.get("name") or location.get("id")
    visual = clip_words(location.get("visual") or "simple colorful cartoon background", 18)
    return with_style("{}, {}, empty wide background, no characters, no people".format(name, visual))


def contact_ids(values: Iterable[str]) -> str:
    return ", ".join(str(value) for value in values)
