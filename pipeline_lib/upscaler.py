"""RealESRGAN_x4plus_anime_6B, then a resize to 3840x2160.

The official checkpoint is about 17MB. If it cannot load, the frame is scaled
with Lanczos and the delivery report says so instead of pretending otherwise.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config

LOGGER = logging.getLogger("trilogy")


class Upscaler(object):
    def __init__(self) -> None:
        self.kind = "lanczos-fallback"
        self.model = None
        self.device = "cpu"
        self.dtype = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            from pipeline_lib.rrdb import RRDBNet
        except Exception as exc:
            LOGGER.warning("RealESRGAN dependencies missing (%s). Using Lanczos.", exc)
            return
        weight = self._weights()
        if weight is None:
            return
        try:
            try:
                state = torch.load(str(weight), map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(str(weight), map_location="cpu")
            if isinstance(state, dict):
                if "params_ema" in state:
                    state = state["params_ema"]
                elif "params" in state:
                    state = state["params"]
            model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32)
            model.load_state_dict(state, strict=True)
            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.float16
                model = model.to(device="cuda", dtype=torch.float16)
            else:
                self.device = "cpu"
                self.dtype = torch.float32
            model.eval()
            self.model = model
            self.kind = config.ESRGAN_WEIGHT_NAME.replace(".pth", "")
            LOGGER.info("Loaded %s on %s", self.kind, self.device)
        except Exception as exc:
            LOGGER.warning("Could not load RealESRGAN weights (%s). Using Lanczos.", exc)
            self.model = None
            self.kind = "lanczos-fallback"

    def _weights(self) -> Optional[Path]:
        dest = config.output_dir() / "weights" / config.ESRGAN_WEIGHT_NAME
        if dest.exists() and dest.stat().st_size > 1_000_000:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            LOGGER.info("Downloading %s", config.ESRGAN_WEIGHT_NAME)
            urllib.request.urlretrieve(config.ESRGAN_URL, dest)
        except Exception as exc:
            LOGGER.warning("RealESRGAN download failed: %s", exc)
            return None
        if dest.exists() and dest.stat().st_size > 1_000_000:
            return dest
        return None

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(frame, mode="RGB")
        if self.model is None:
            return np.asarray(image.resize((config.FINAL_WIDTH, config.FINAL_HEIGHT), Image.Resampling.LANCZOS))
        import torch

        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        tensor = tensor / 255.0
        with torch.inference_mode():
            scaled = self.model(tensor)
        scaled = scaled.float().clamp(0, 1).mul(255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        resized = Image.fromarray(scaled, mode="RGB").resize((config.FINAL_WIDTH, config.FINAL_HEIGHT), Image.Resampling.LANCZOS)
        return np.asarray(resized)
