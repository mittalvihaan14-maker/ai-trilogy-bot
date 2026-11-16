"""ffmpeg helpers. Uses a system ffmpeg when it exists, otherwise the
binary shipped with imageio-ffmpeg.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("trilogy")


@dataclass
class MediaInfo(object):
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is not installed. On Kaggle run `apt-get install -y ffmpeg`, "
            "or `pip install imageio-ffmpeg`."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def require_ffmpeg() -> str:
    exe = ffmpeg_exe()
    subprocess.run([exe, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    LOGGER.info("ffmpeg: %s", exe)
    return exe


def run_ffmpeg(args: List[str]) -> None:
    cmd = [ffmpeg_exe()] + args
    LOGGER.info("ffmpeg %s", " ".join(args[:8]))
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError("ffmpeg failed: {}".format(detail))


def probe(path: Path) -> MediaInfo:
    completed = subprocess.run(
        [ffmpeg_exe(), "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    text = completed.stderr or ""
    duration = 0.0
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        duration = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    width = 0
    height = 0
    fps = 0.0
    video_codec = ""
    audio_codec = ""
    for line in text.splitlines():
        if "Video:" in line:
            video_codec = line.split("Video:", 1)[1].split(",", 1)[0].strip()
            size = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if size:
                width = int(size.group(1))
                height = int(size.group(2))
            rate = re.search(r"(\d+(?:\.\d+)?)\s*fps", line) or re.search(r"(\d+(?:\.\d+)?)\s*tbr", line)
            if rate:
                fps = float(rate.group(1))
        elif "Audio:" in line and not audio_codec:
            audio_codec = line.split("Audio:", 1)[1].split(",", 1)[0].strip()
    return MediaInfo(duration, width, height, fps, video_codec, audio_codec)


def write_concat_list(paths: List[Path], list_path: Path) -> None:
    lines = ["file '{}'".format(path.resolve().as_posix()) for path in paths]
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_copy(paths: List[Path], dest: Path) -> None:
    """Lossless concat. Every input must already share codec parameters."""
    if not paths:
        raise RuntimeError("concat_copy received no inputs")
    list_path = dest.with_suffix(".concat.txt")
    write_concat_list(paths, list_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(
            [
                "-y",
                "-fflags",
                "+genpts",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(dest),
            ]
        )
    except RuntimeError:
        LOGGER.warning("faststart concat failed; writing a stream-copy file without faststart")
        run_ffmpeg(
            ["-y", "-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(dest)]
        )
    finally:
        if list_path.exists():
            list_path.unlink()


def encode_shot(frame_dir: Path, wav_path: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-framerate",
            "4",
            "-start_number",
            "0",
            "-i",
            str(frame_dir / "f_%04d.png"),
            "-i",
            str(wav_path),
            "-vf",
            "scale=426:240:flags=lanczos,setsar=1,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-r",
            "4",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(dest),
        ]
    )


def extract_wav(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", str(dest)])


def mp3_to_wav(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(src), "-ac", "2", "-ar", "48000", str(dest)])


def atempo_wav(src: Path, dest: Path, factor: float) -> None:
    """Speed speech up. factor > 1 is faster. ffmpeg atempo accepts 0.5–2.0."""
    factor = max(0.5, min(2.0, factor))
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-y", "-i", str(src), "-filter:a", "atempo={:.4f}".format(factor), "-ac", "2", "-ar", "48000", str(dest)])


def minterpolate_to_60(src: Path, dest: Path) -> str:
    """Return the interpolator name that produced dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _rife_available():
        try:
            _rife_interpolate(src, dest)
            return "rife-ncnn-vulkan"
        except Exception as exc:
            LOGGER.warning("RIFE failed (%s); falling back to ffmpeg minterpolate", exc)
    motion = "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    try:
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(src),
                "-an",
                "-vf",
                motion,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ]
        )
        return "ffmpeg-minterpolate-mci"
    except RuntimeError as exc:
        LOGGER.warning("Motion-compensated minterpolate failed (%s); using blend", exc)
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(src),
                "-an",
                "-vf",
                "minterpolate=fps=60:mi_mode=blend",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ]
        )
        return "ffmpeg-minterpolate-blend"


def _rife_available() -> bool:
    import os

    if os.getenv("USE_RIFE", "1").lower() in {"0", "false", "no"}:
        return False
    return shutil.which("rife-ncnn-vulkan") is not None


def _rife_interpolate(src: Path, dest: Path) -> None:
    """4 fps -> 60 fps (15x) with rife-ncnn-vulkan, then delete the frame folders."""
    binary = shutil.which("rife-ncnn-vulkan")
    work = dest.parent / (dest.stem + "_rife")
    incoming = work / "in"
    outgoing = work / "out"
    incoming.mkdir(parents=True, exist_ok=True)
    outgoing.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(["-y", "-i", str(src), "-vsync", "0", str(incoming / "f_%06d.png")])
        completed = subprocess.run(
            [binary, "-i", str(incoming), "-o", str(outgoing), "-n", "15"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", "replace")[-1000:])
        run_ffmpeg(
            [
                "-y",
                "-framerate",
                "60",
                "-i",
                str(outgoing / "f_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ]
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def open_raw_reader(path: Path) -> subprocess.Popen:
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def open_raw_writer(dest: Path, width: int, height: int, fps: int, audio: Optional[Path]) -> subprocess.Popen:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "{}x{}".format(width, height),
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio is not None:
        cmd.extend(["-i", str(audio)])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
        ]
    )
    if audio is not None:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest"])
    cmd.append(str(dest))
    log_path = dest.with_suffix(".ffmpeg.log")
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_handle)
    process._trilogy_log = log_handle  # type: ignore[attr-defined]
    return process


def close_writer(process: subprocess.Popen) -> None:
    if process.stdin:
        process.stdin.close()
    code = process.wait()
    handle = getattr(process, "_trilogy_log", None)
    if handle:
        handle.close()
    if code != 0:
        raise RuntimeError("ffmpeg encoder exited with status {}".format(code))


def read_exact(stream, size: int):
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
