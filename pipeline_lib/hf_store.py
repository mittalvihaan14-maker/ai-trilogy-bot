"""Private Hugging Face dataset storage for heavy media and checkpoints."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import config

LOGGER = logging.getLogger("trilogy")
_STORE = None  # type: Optional["Store"]


class Store(object):
    def __init__(self) -> None:
        self.api = None
        self.repo_ready = False

    def enabled(self) -> bool:
        return bool(config.HF_TOKEN)

    def _api(self):
        if not self.enabled():
            return None
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi(token=config.HF_TOKEN)
        return self.api

    def ensure_repo(self) -> bool:
        api = self._api()
        if api is None:
            return False
        if self.repo_ready:
            return True
        api.create_repo(
            repo_id=config.HF_REPO_ID,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        self._ensure_card(api)
        self.repo_ready = True
        LOGGER.info("Hugging Face dataset ready: %s", config.HF_REPO_ID)
        return True

    def _ensure_card(self, api) -> None:
        try:
            infos = api.get_paths_info(config.HF_REPO_ID, ["README.md"], repo_type="dataset")
            if infos:
                return
        except Exception:
            return
        card = (
            "---\n"
            "license: other\n"
            "tags:\n"
            "- cartoon\n"
            "- animation\n"
            "---\n\n"
            "# Original 2D trilogy\n\n"
            "Generated cartoon movies for the `ai-trilogy-bot` pipeline. "
            "Characters are original geometric designs. The set does not contain "
            "photographs of real people.\n\n"
            "Finished films are in `final_movies/`.\n"
        )
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.HF_REPO_ID,
            repo_type="dataset",
        )

    def remote_size(self, repo_path: str) -> Optional[int]:
        api = self._api()
        if api is None:
            return None
        try:
            infos = api.get_paths_info(config.HF_REPO_ID, [repo_path], repo_type="dataset")
        except Exception:
            return None
        if not infos:
            return None
        return getattr(infos[0], "size", None)

    def exists(self, repo_path: str) -> bool:
        return self.remote_size(repo_path) is not None

    def upload(self, local_path: Path, *parts: str) -> str:
        """Upload one file. Returns the repo path, or an empty string if skipped."""
        local_path = Path(local_path)
        if not local_path.exists() or local_path.stat().st_size <= 0:
            return ""
        if not self.ensure_repo():
            LOGGER.warning("Skipped upload of %s; Hugging Face is not configured.", local_path.name)
            return ""
        repo_path = config.hf_repo_path(*parts)
        size = local_path.stat().st_size
        remote = self.remote_size(repo_path)
        if remote is not None and remote == size:
            LOGGER.info("Already on Hugging Face (%s bytes): %s", size, repo_path)
            return repo_path
        api = self._api()
        last_error = None  # type: Optional[Exception]
        for attempt in range(3):
            try:
                LOGGER.info("Uploading %s -> %s", local_path.name, repo_path)
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=config.HF_REPO_ID,
                    repo_type="dataset",
                )
                return repo_path
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Upload attempt %s failed: %s", attempt + 1, config.scrub(str(exc)))
                time.sleep(2 ** attempt)
        raise RuntimeError("Could not upload {}: {}".format(repo_path, config.scrub(str(last_error))))

    def ensure_local(self, dest: Path, *parts: str) -> bool:
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            return True
        if not self.enabled():
            return False
        repo_path = config.hf_repo_path(*parts)
        if self.remote_size(repo_path) is None:
            return False
        from huggingface_hub import hf_hub_download

        LOGGER.info("Downloading %s", repo_path)
        downloaded = hf_hub_download(
            repo_id=config.HF_REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            token=config.HF_TOKEN,
            local_dir=str(config.output_dir() / "_hf"),
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)
        return dest.exists() and dest.stat().st_size > 0


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def publish_state(state_obj) -> None:
    state_obj.save()
    try:
        store().upload(state_obj.path, "state.json")
    except Exception as exc:
        LOGGER.warning("Could not upload state.json: %s", config.scrub(str(exc)))
