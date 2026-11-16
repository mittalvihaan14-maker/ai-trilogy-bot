"""Small JSON checkpoint. Artifacts on disk or Hugging Face are the source of truth."""

from __future__ import annotations

import json
from typing import Any, Dict

import config


class State(object):
    def __init__(self, data: Dict, path) -> None:
        self.data = data
        self.path = path

    @classmethod
    def load(cls) -> "State":
        path = config.output_dir() / "state.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {
                "version": 1,
                "mode": "test" if config.test_mode() else "full",
                "sets": {},
                "movies": {},
            }
        mode = "test" if config.test_mode() else "full"
        if data.get("mode") not in (None, mode):
            raise RuntimeError(
                "Checkpoint mode is '{}' but this run is '{}'. "
                "Pass --fresh to start this mode over.".format(data.get("mode"), mode)
            )
        data["mode"] = mode
        data.setdefault("sets", {})
        data.setdefault("movies", {})
        return cls(data, path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def contains(self, bucket: str, key: str) -> bool:
        return key in self.data.setdefault("sets", {}).setdefault(bucket, [])

    def add(self, bucket: str, key: str) -> None:
        items = self.data.setdefault("sets", {}).setdefault(bucket, [])
        if key not in items:
            items.append(key)
            self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()
