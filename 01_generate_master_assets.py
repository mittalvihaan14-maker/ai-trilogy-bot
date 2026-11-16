#!/usr/bin/env python3
"""One master PNG per character and per recurring location.

These images are the IP-Adapter and img2img references for every shot in all
three movies, so a face designed here is the face that returns later.
"""

from __future__ import annotations

import logging

import config
from pipeline_lib.cli import parser
from pipeline_lib import layout
from pipeline_lib.hf_store import publish_state, store
from pipeline_lib.state import State
from pipeline_lib import storyboard

LOGGER = logging.getLogger("trilogy")


def generate(is_test: bool) -> None:
    config.bootstrap(is_test)
    bible = storyboard.load_bible()
    state = State.load()
    pending = []
    for character in bible["characters"]:
        path = layout.master_character(character["id"])
        if not (path.exists() and state.contains("masters_characters", character["id"])):
            if not store().ensure_local(path, layout.relative(path)):
                pending.append(("character", character, path))
            else:
                state.add("masters_characters", character["id"])
    for location in bible["locations"]:
        path = layout.master_location(location["id"])
        if not (path.exists() and state.contains("masters_locations", location["id"])):
            if not store().ensure_local(path, layout.relative(path)):
                pending.append(("location", location, path))
            else:
                state.add("masters_locations", location["id"])
    if not pending:
        config.log_progress("Master art already complete.")
        return
    from pipeline_lib.sd_engine import CartoonPipeline

    pipe = CartoonPipeline("txt2img")
    try:
        for kind, item, path in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "character":
                image = pipe.master_character(item)
                bucket = "masters_characters"
            else:
                image = pipe.master_location(item)
                bucket = "masters_locations"
            image.save(path, format="PNG")
            LOGGER.info("Master %s -> %s", item["id"], path)
            store().upload(path, layout.relative(path))
            state.add(bucket, item["id"])
            publish_state(state)
    finally:
        pipe.close()
    config.log_progress(
        "Master art ready: {} characters, {} locations. Model {}.".format(
            len(bible["characters"]), len(bible["locations"]), pipe.model_id
        )
    )


def main() -> None:
    arguments = parser("Generate master character and location PNGs").parse_args()
    generate(arguments.test_mode)


if __name__ == "__main__":
    main()
