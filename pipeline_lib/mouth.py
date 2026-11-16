"""Flat cartoon mouth sprites driven by the dialogue envelope."""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image, ImageDraw


def sprite(open_mouth: bool, size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if open_mouth:
        draw.ellipse((2, 4, size - 3, size - 3), fill=(25, 8, 12, 255), outline=(0, 0, 0, 255), width=max(2, size // 12))
        draw.ellipse((size // 5, size // 4, size - size // 5, size - size // 4), fill=(186, 70, 78, 255))
    else:
        draw.arc((3, size // 4, size - 4, size - 4), 200, 340, fill=(0, 0, 0, 255), width=max(2, size // 10))
    return image


def anchors(count: int, width: int, height: int) -> List[Tuple[int, int]]:
    count = max(1, count)
    y = int(height * 0.64)
    if count == 1:
        return [(width // 2, y)]
    return [(int(width * (index + 1) / (count + 1)), y) for index in range(count)]


def overlay(frame: Image.Image, character_ids: List[str], speaker: str, mouth_open: bool) -> Image.Image:
    image = frame.convert("RGBA")
    points = anchors(len(character_ids) or 1, image.width, image.height)
    size = max(16, image.width // 14)
    open_sprite = sprite(True, size)
    closed_sprite = sprite(False, size)
    if speaker not in character_ids:
        return image.convert("RGB")
    speaker_index = character_ids.index(speaker)
    for index, (x, y) in enumerate(points):
        stamp = open_sprite if (mouth_open and index == speaker_index) else closed_sprite
        image.paste(stamp, (int(x - size / 2), int(y - size / 2)), stamp)
    return image.convert("RGB")
