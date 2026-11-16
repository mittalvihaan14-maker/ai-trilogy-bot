"""Dialogue, procedural score, procedural effects, and the shot mix.

Levels are dialogue 100%, sound effects 70%, and music at the ducked 20% bed.
Shot length is already locked, so speech is fit inside the frame count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from pipeline_lib import ffmpeg_tools

LOGGER = logging.getLogger("trilogy")

MOOD_PRESETS = {
    "hopeful": {"bpm": 104, "root": 220.0, "third": 4},
    "tense": {"bpm": 132, "root": 196.0, "third": 3},
    "sad": {"bpm": 72, "root": 174.0, "third": 3},
    "action": {"bpm": 150, "root": 246.0, "third": 4},
    "wonder": {"bpm": 92, "root": 261.0, "third": 4},
    "peaceful": {"bpm": 80, "root": 196.0, "third": 4},
    "comic": {"bpm": 126, "root": 294.0, "third": 4},
    "dark": {"bpm": 66, "root": 146.0, "third": 3},
}


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if width != 2:
        raise RuntimeError("Expected 16-bit wav at {}".format(path))
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data, rate


def write_wav(path: Path, audio: np.ndarray, rate: int = config.SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    if clipped.ndim == 1:
        clipped = np.stack([clipped, clipped], axis=1)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def silence(n_samples: int) -> np.ndarray:
    return np.zeros((n_samples, 2), dtype=np.float32)


def fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if len(audio) == n_samples:
        return audio.astype(np.float32)
    if len(audio) > n_samples:
        return audio[:n_samples].astype(np.float32)
    pad = np.zeros((n_samples - len(audio), audio.shape[1]), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad], axis=0)


def _lowpass(noise: np.ndarray) -> np.ndarray:
    kernel = (0.08 * (0.92 ** np.arange(512))).astype(np.float32)
    kernel /= kernel.sum()
    return np.convolve(noise, kernel, mode="same").astype(np.float32)


def synth_music(duration: float, mood: str, seed: int) -> np.ndarray:
    preset = MOOD_PRESETS.get(mood, MOOD_PRESETS["peaceful"])
    rate = config.SAMPLE_RATE
    count = int(round(duration * rate))
    timeline = np.arange(count, dtype=np.float32) / rate
    bar = 4.0 * (60.0 / preset["bpm"])
    chords = (0, 7, preset["third"] + 5, 5)
    chord_index = np.floor(timeline / bar).astype(np.int64) % len(chords)
    signal = np.zeros(count, dtype=np.float32)
    for index, semitone in enumerate(chords):
        mask = (chord_index == index).astype(np.float32)
        fundamental = preset["root"] * (2.0 ** (semitone / 12.0))
        tones = (
            fundamental / 2.0,
            fundamental,
            fundamental * (2.0 ** (preset["third"] / 12.0)),
            fundamental * (2.0 ** (7 / 12.0)),
        )
        chord = np.zeros(count, dtype=np.float32)
        for tone in tones:
            chord += np.sin(2.0 * np.pi * tone * timeline).astype(np.float32)
        signal += chord * mask
    pulse = (0.75 + 0.25 * np.sin(2.0 * np.pi * (preset["bpm"] / 60.0) * timeline)).astype(np.float32)
    signal *= pulse
    rng = np.random.default_rng(seed)
    signal += rng.normal(0, 0.02, count).astype(np.float32)
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = signal / peak * 0.8
    return np.stack([signal, signal * 0.92], axis=1)


def synth_sfx(cue: str, duration: float, seed: int) -> np.ndarray:
    rate = config.SAMPLE_RATE
    count = int(round(duration * rate))
    timeline = np.arange(count, dtype=np.float32) / rate
    rng = np.random.default_rng(seed + 91)
    text = (cue or "").lower()
    layers = []  # type: List[np.ndarray]
    if any(word in text for word in ("wind", "soft", "air", "room")):
        layers.append(_lowpass(rng.normal(0, 1, count).astype(np.float32)) * 0.35)
    if "bird" in text:
        chirp = np.sin(2 * np.pi * (880 + 380 * np.sin(2 * np.pi * 3 * timeline)) * timeline)
        gate = (np.sin(2 * np.pi * 2.2 * timeline) > 0.55).astype(np.float32)
        layers.append(chirp.astype(np.float32) * gate * 0.25)
    if any(word in text for word in ("foot", "step", "walk")):
        hits = np.zeros(count, dtype=np.float32)
        step = int(rate * 0.45)
        for start in range(0, count, step):
            length = min(int(rate * 0.05), count - start)
            env = np.linspace(1, 0, length, dtype=np.float32)
            hits[start : start + length] += rng.normal(0, 1, length).astype(np.float32) * env
        layers.append(_lowpass(hits) * 1.4)
    if any(word in text for word in ("door", "knock", "impact", "hit", "thud")):
        env = np.exp(-timeline * 8).astype(np.float32)
        layers.append(np.sin(2 * np.pi * 90 * timeline).astype(np.float32) * env * 0.8)
    if any(word in text for word in ("magic", "spark", "chime")):
        rise = np.sin(2 * np.pi * np.linspace(400, 1400, count) * timeline).astype(np.float32)
        layers.append(rise * np.linspace(0.1, 0.5, count).astype(np.float32))
    if "rain" in text or "water" in text:
        layers.append(_lowpass(rng.normal(0, 1, count).astype(np.float32)) * 0.45)
    if "thunder" in text:
        env = np.exp(-timeline * 1.5).astype(np.float32)
        layers.append(_lowpass(rng.normal(0, 1, count).astype(np.float32)) * env * 1.5)
    if "whoosh" in text or "swish" in text:
        sweep = _lowpass(rng.normal(0, 1, count).astype(np.float32))
        envelope = np.sin(np.linspace(0, np.pi, count)).astype(np.float32)
        layers.append(sweep * envelope)
    if "crowd" in text:
        layers.append(_lowpass(rng.normal(0, 1, count).astype(np.float32)) * 0.3)
    if not layers:
        layers.append(_lowpass(rng.normal(0, 1, count).astype(np.float32)) * 0.12)
    mix = np.sum(np.stack(layers, axis=0), axis=0)
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix / peak * 0.75
    return np.stack([mix, mix], axis=1)


def mouth_flags(dialogue: np.ndarray, n_frames: int) -> List[bool]:
    if n_frames <= 0:
        return []
    mono = dialogue.mean(axis=1) if dialogue.ndim == 2 else dialogue
    total = len(mono)
    values = []
    for index in range(n_frames):
        start = int(index * total / n_frames)
        end = max(start + 1, int((index + 1) * total / n_frames))
        chunk = mono[start:end]
        values.append(float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0)
    array = np.array(values, dtype=np.float32)
    audible = array[array > 0.008]
    if len(audible) == 0:
        return [False] * n_frames
    threshold = max(0.02, float(np.quantile(audible, 0.4)))
    return [bool(value >= threshold) for value in values]


def _speak(text: str, voice: str, mp3_path: Path, rate: str) -> None:
    import edge_tts

    async def _save() -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(mp3_path))

    asyncio.run(_save())


def render_speech(text: str, voice: str, work_dir: Path, max_seconds: float, seed_name: str) -> Optional[np.ndarray]:
    spoken = (text or "").strip()
    if not spoken:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = work_dir / "{}.mp3".format(seed_name)
    wav_path = work_dir / "{}.speech.wav".format(seed_name)
    voices = [voice, config.NARRATOR_VOICE, "en-US-JennyNeural"]
    last_error = None  # type: Optional[Exception]
    audio = None
    for candidate in voices:
        try:
            _speak(spoken, candidate, mp3_path, "+0%")
            ffmpeg_tools.mp3_to_wav(mp3_path, wav_path)
            audio, rate = read_wav(wav_path)
            duration = len(audio) / float(rate)
            if duration > max_seconds and max_seconds > 0:
                percent = int(min(45, max(0, round((duration / max_seconds - 1.0) * 100))))
                if percent > 0:
                    _speak(spoken, candidate, mp3_path, "+{}%".format(percent))
                    ffmpeg_tools.mp3_to_wav(mp3_path, wav_path)
                    audio, rate = read_wav(wav_path)
                    duration = len(audio) / float(rate)
                if duration > max_seconds:
                    factor = min(2.0, duration / max_seconds)
                    sped = work_dir / "{}.sped.wav".format(seed_name)
                    ffmpeg_tools.atempo_wav(wav_path, sped, factor)
                    audio, _rate = read_wav(sped)
            break
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Voice %s failed: %s", candidate, config.scrub(str(exc)))
            audio = None
    if audio is None:
        raise RuntimeError("Speech synthesis failed for {}: {}".format(seed_name, last_error))
    return audio


def mix_shot(dialogue: Optional[np.ndarray], sfx: np.ndarray, music: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    bed_sfx = fit_length(sfx, n_samples) * config.SFX_GAIN
    bed_music = fit_length(music, n_samples) * config.MUSIC_GAIN
    speech_bus = silence(n_samples)
    if dialogue is not None and len(dialogue):
        lead = int(0.3 * config.SAMPLE_RATE)
        placed = silence(n_samples)
        speech = dialogue.astype(np.float32) * config.DIALOGUE_GAIN
        if speech.ndim == 1:
            speech = np.stack([speech, speech], axis=1)
        stop = min(n_samples, lead + len(speech))
        placed[lead:stop] = speech[: stop - lead]
        speech_bus = placed
    mixed = speech_bus + bed_sfx + bed_music
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.astype(np.float32), speech_bus.astype(np.float32)


def build_shot_audio(shot: Dict, bible: Dict, dest_wav: Path) -> None:
    """Write the mixed wav and a sibling .mouth.json aligned to 4 fps frames."""
    frames = int(shot["frames"])
    duration = frames / float(config.DRAFT_FPS)
    n_samples = int(round(duration * config.SAMPLE_RATE))
    voice = config.NARRATOR_VOICE
    speaker = shot.get("speaker") or ""
    for character in bible.get("characters") or []:
        if character.get("id") == speaker:
            voice = character.get("voice") or voice
            break
    work = dest_wav.parent / "_speech"
    dialogue = render_speech(shot.get("dialogue") or "", voice, work, max(1.0, duration - 0.6), shot["id"])
    seed = int(shot.get("index") or 1) + sum(ord(char) for char in shot["id"])
    music = synth_music(duration, shot.get("mood") or "peaceful", seed)
    sfx = synth_sfx(shot.get("sfx_cue") or "", duration, seed)
    mixed, speech_bus = mix_shot(dialogue, sfx, music, n_samples)
    write_wav(dest_wav, mixed)
    flags = mouth_flags(speech_bus, frames)
    dest_wav.with_suffix(".mouth.json").write_text(json.dumps(flags), encoding="utf-8")
    if work.exists():
        for leftover in work.glob(shot["id"] + "*"):
            leftover.unlink()
