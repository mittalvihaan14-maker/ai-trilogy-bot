"""Stable Diffusion 1.5 + LCM-LoRA at 6 steps, with IP-Adapter on shots.

Attention slicing and VAE slicing are always on. Model CPU offload is enabled
when the GPU has under 10GB of VRAM, which covers 4GB–8GB cards. A T4 (16GB)
keeps the model on GPU. Set FORCE_CPU_OFFLOAD=1 if a 16GB card still runs out
of memory after IP-Adapter loads.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

import config
from pipeline_lib.prompts import character_master_prompt, location_master_prompt, shot_prompt

LOGGER = logging.getLogger("trilogy")


def vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / float(1024 ** 3)


class CartoonPipeline(object):
    def __init__(self, kind: str) -> None:
        if kind not in ("txt2img", "img2img"):
            raise ValueError(kind)
        self.kind = kind
        self.pipe = None
        self.model_id = ""
        self.pipe, self.model_id = self._load()

    def _load(self):
        import torch
        from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        cls = StableDiffusionImg2ImgPipeline if self.kind == "img2img" else StableDiffusionPipeline
        errors = []
        pipe = None
        model_id = ""
        for candidate in (config.SD_MODEL_ID, config.SD_FALLBACK_MODEL_ID):
            try:
                pipe = self._from_pretrained(cls, candidate, dtype)
                model_id = candidate
                break
            except Exception as exc:
                errors.append("{}: {}".format(candidate, exc))
                LOGGER.warning("Could not load %s (%s)", candidate, exc)
        if pipe is None:
            raise RuntimeError("No Stable Diffusion checkpoint loaded. {}".format(" | ".join(errors)))
        pipe.load_lora_weights(config.LCM_LORA_ID)
        try:
            pipe.fuse_lora()
        except Exception as exc:
            LOGGER.info("LoRA left unfused (%s)", exc)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        if self.kind == "img2img":
            pipe.load_ip_adapter(
                config.IP_ADAPTER_REPO,
                subfolder=config.IP_ADAPTER_SUBFOLDER,
                weight_name=config.IP_ADAPTER_WEIGHT,
            )
            pipe.set_ip_adapter_scale(0.7)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        self._place(pipe)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        LOGGER.info("SD pipeline %s on %s (%s)", self.kind, model_id, "float16" if dtype == torch.float16 else "float32")
        return pipe, model_id

    def _from_pretrained(self, cls, model_id: str, dtype):
        try:
            return cls.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
        except Exception as exc:
            if "safety" not in str(exc).lower():
                raise
            return cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                safety_checker=None,
                requires_safety_checker=False,
            )

    def _place(self, pipe) -> None:
        import torch

        force = os.getenv("FORCE_CPU_OFFLOAD", "").lower() in {"1", "true", "yes"}
        memory = vram_gb()
        if torch.cuda.is_available() and (force or memory < config.VRAM_OFFLOAD_BELOW_GB):
            LOGGER.info("VRAM %.1fGB; enabling model CPU offload", memory)
            pipe.enable_model_cpu_offload()
            return
        if torch.cuda.is_available():
            pipe.to("cuda")
            return
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            pipe.to("mps")
            return
        LOGGER.warning("No CUDA GPU detected. Rendering on CPU will be very slow.")
        pipe.to("cpu")

    def _generate(self, prompt: str, seed: int, init: Optional[Image.Image] = None, reference: Optional[Image.Image] = None, strength: float = 0.55) -> Image.Image:
        import torch

        generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**31 - 1))
        kwargs = {
            "prompt": prompt,
            "negative_prompt": config.NEGATIVE_PROMPT,
            "num_inference_steps": config.INFERENCE_STEPS,
            "guidance_scale": config.GUIDANCE_SCALE,
            "generator": generator,
        }
        if self.kind == "txt2img":
            kwargs["width"] = config.MASTER_SIZE
            kwargs["height"] = config.MASTER_SIZE
        else:
            kwargs["image"] = init.convert("RGB").resize((config.GEN_WIDTH, config.GEN_HEIGHT), Image.Resampling.LANCZOS)
            kwargs["strength"] = strength
            if reference is not None:
                kwargs["ip_adapter_image"] = reference.convert("RGB")
        try:
            result = self.pipe(**kwargs).images[0]
        except TypeError:
            if "ip_adapter_image" in kwargs and not isinstance(kwargs["ip_adapter_image"], list):
                kwargs["ip_adapter_image"] = [kwargs["ip_adapter_image"]]
                result = self.pipe(**kwargs).images[0]
            else:
                raise
        if self.kind == "img2img" and result.size != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
            result = result.resize((config.FRAME_WIDTH, config.FRAME_HEIGHT), Image.Resampling.LANCZOS)
        return result.convert("RGB")

    def master_character(self, character: dict) -> Image.Image:
        return self._generate(character_master_prompt(character), int(character["seed"]))

    def master_location(self, location: dict) -> Image.Image:
        return self._generate(location_master_prompt(location), int(location["seed"]))

    def frame(self, shot: dict, bible: dict, seed: int, init: Image.Image, reference: Optional[Image.Image], strength: float) -> Image.Image:
        return self._generate(shot_prompt(shot, bible), seed, init=init, reference=reference, strength=strength)

    def close(self) -> None:
        pipe = self.pipe
        self.pipe = None
        del pipe
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
