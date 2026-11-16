"""Shared command-line flags for the stage scripts."""

from __future__ import annotations

import argparse


def parser(description: str) -> argparse.ArgumentParser:
    args = argparse.ArgumentParser(description=description)
    mode = args.add_mutually_exclusive_group()
    mode.add_argument("--test-mode", action="store_true", help="1-minute 4K test movie")
    mode.add_argument("--full-trilogy", action="store_true", help="Three 96-minute movies")
    args.add_argument("--fresh", action="store_true", help="Discard this mode's checkpoint and start over")
    args.add_argument("--movie", type=int, help="Limit work to one movie number")
    args.add_argument("--chapter", type=int, help="Limit work to one chapter number")
    return args


def selected(arguments, movie_id: int, chapter_id: int) -> bool:
    if arguments.movie and int(arguments.movie) != int(movie_id):
        return False
    if arguments.chapter and int(arguments.chapter) != int(chapter_id):
        return False
    return True
