from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F


InterpolationMode = Literal["nearest", "bilinear", "bicubic", "trilinear"]
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


@dataclass(frozen=True)
class ImageInfo:
    path: str
    width: int
    height: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    duration_seconds: float
    has_audio: bool
    tags: dict[str, str]
    prompt: str | None = None
    prompt_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
            "duration_seconds": self.duration_seconds,
            "has_audio": self.has_audio,
            "tags": self.tags,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
        }


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    samples: int
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "duration_seconds": self.duration_seconds,
        }


def resolve_video_path(workspace: Path, raw_path: str) -> Path:
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    root = workspace.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Input video must be inside the workspace")
    if not resolved.is_file():
        raise ValueError("Input video does not exist")
    if resolved.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported input video extension: {resolved.suffix or '(none)'}")
    return resolved


def resolve_image_path(workspace: Path, raw_path: str) -> Path:
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    root = workspace.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Input image must be inside the workspace")
    if not resolved.is_file():
        raise ValueError("Input image does not exist")
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported input image extension: {resolved.suffix or '(none)'}")
    return resolved


def resolve_audio_path(workspace: Path, raw_path: str) -> Path:
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    root = workspace.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Input audio must be inside the workspace")
    if not resolved.is_file():
        raise ValueError("Input audio does not exist")
    if resolved.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported input audio extension: {resolved.suffix or '(none)'}")
    return resolved


def _ratio(value: str | None, default: float = 24.0) -> float:
    if not value:
        return default
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(value)
    except (ValueError, ZeroDivisionError):
        return default
    return result if math.isfinite(result) and result > 0 else default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found on PATH") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "ffprobe failed")
    try:
        return json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ffprobe returned invalid metadata") from exc


def _prompt_from_object(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("prompt", "positive_prompt", "text"):
            prompt = value.get(key)
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        for nested in value.values():
            prompt = _prompt_from_object(nested)
            if prompt:
                return prompt
    elif isinstance(value, list):
        for nested in value:
            prompt = _prompt_from_object(nested)
            if prompt:
                return prompt
    return None


def _prompt_from_text(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    text = value.strip()
    try:
        prompt = _prompt_from_object(json.loads(text))
    except json.JSONDecodeError:
        prompt = None
    return prompt or text


def read_video_prompt(path: Path, tags: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    sidecar = path.with_suffix(".metadata.json")
    if sidecar.is_file():
        try:
            prompt = _prompt_from_object(json.loads(sidecar.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError):
            prompt = None
        if prompt:
            return prompt, "sidecar"

    tags = tags or {}
    for key in ("comment", "description", "title"):
        prompt = _prompt_from_text(tags.get(key))
        if prompt:
            return prompt, f"mp4.{key}"
    return None, None


def probe_video(path: Path) -> VideoInfo:
    data = _ffprobe(path)
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise ValueError("Input file has no video stream")
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Input video has invalid dimensions")
    fps = _ratio(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or ""))
    format_data = data.get("format") or {}
    duration = _float(video.get("duration"), _float(format_data.get("duration")))
    frame_value = video.get("nb_frames")
    try:
        frames = int(frame_value) if frame_value not in (None, "N/A") else 0
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0 and duration > 0:
        frames = max(1, round(duration * fps))
    if frames <= 0:
        raise ValueError("Input video has no readable frame count")
    tags: dict[str, str] = {}
    for source in (format_data.get("tags"), video.get("tags")):
        if isinstance(source, dict):
            tags.update({str(key).lower(): str(value) for key, value in source.items()})
    prompt, prompt_source = read_video_prompt(path, tags)
    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration_seconds=duration or frames / fps,
        has_audio=any(item.get("codec_type") == "audio" for item in streams if isinstance(item, dict)),
        tags=tags,
        prompt=prompt,
        prompt_source=prompt_source,
    )


def probe_image(path: Path) -> ImageInfo:
    data = _ffprobe(path)
    streams = data.get("streams") or []
    image = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(image, dict):
        raise ValueError("Input file has no image stream")
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Input image has invalid dimensions")
    return ImageInfo(path=str(path), width=width, height=height)


def probe_audio(path: Path) -> AudioInfo:
    data = _ffprobe(path)
    streams = data.get("streams") or []
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(audio, dict):
        raise ValueError("Input file has no audio stream")
    sample_rate = int(audio.get("sample_rate") or 0)
    channels = int(audio.get("channels") or 0)
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("Input audio has invalid sample rate or channel count")
    format_data = data.get("format") or {}
    duration = _float(audio.get("duration"), _float(format_data.get("duration")))
    if duration <= 0:
        duration_ts = _float(audio.get("duration_ts"))
        time_base = _ratio(str(audio.get("time_base") or ""), default=0.0)
        duration = duration_ts * time_base
    if duration <= 0:
        raise ValueError("Input audio has no readable duration")
    return AudioInfo(
        path=str(path),
        sample_rate=sample_rate,
        channels=channels,
        samples=max(1, round(duration * sample_rate)),
        duration_seconds=duration,
    )


def read_image(path: Path, info: ImageInfo | None = None, max_bytes: int = 128 * 1024**2) -> np.ndarray:
    info = info or probe_image(path)
    expected_bytes = info.width * info.height * 3
    if expected_bytes > max_bytes:
        raise ValueError(
            f"Input image is too large for the decode budget: "
            f"{expected_bytes / 1024**2:.1f} MiB > {max_bytes / 1024**2:.1f} MiB"
        )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg image decode failed")
    if len(result.stdout) < expected_bytes:
        raise ValueError("ffmpeg decoded an incomplete image")
    raw = np.frombuffer(result.stdout[:expected_bytes], dtype=np.uint8)
    image = raw.reshape(info.height, info.width, 3).copy()
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None, :, None], dtype=np.float32) / 255.0


def read_reference_image(
    path: Path,
    target_height: int,
    target_width: int,
    max_bytes: int = 512 * 1024**2,
) -> np.ndarray:
    """Decode an image at Ref2VA's LANCZOS-normalized geometry as uint8 HWC."""
    if target_height < 1 or target_width < 1:
        raise ValueError("Reference image dimensions must be positive")
    expected_bytes = target_width * target_height * 3
    if expected_bytes > max_bytes:
        raise ValueError(
            f"Reference image exceeds the decode budget: "
            f"{expected_bytes / 1024**2:.1f} MiB > {max_bytes / 1024**2:.1f} MiB"
        )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={target_width}:{target_height}:flags=lanczos",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg image decode failed")
    if len(result.stdout) < expected_bytes:
        raise ValueError("ffmpeg decoded an incomplete reference image")
    raw = np.frombuffer(result.stdout[:expected_bytes], dtype=np.uint8)
    return np.ascontiguousarray(raw.reshape(target_height, target_width, 3).copy())


def read_video_frames(path: Path, info: VideoInfo | None = None, max_bytes: int = 2 * 1024**3) -> np.ndarray:
    info = info or probe_video(path)
    expected_bytes = info.frames * info.width * info.height * 3
    if expected_bytes > max_bytes:
        raise ValueError(
            f"Input video is too large for the in-memory preprocessing budget: "
            f"{expected_bytes / 1024**3:.2f} GiB > {max_bytes / 1024**3:.2f} GiB"
        )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg video decode failed")
    frame_size = info.width * info.height * 3
    actual_frames = len(result.stdout) // frame_size
    if actual_frames < 1:
        raise ValueError("ffmpeg decoded no video frames")
    actual_frames = min(actual_frames, info.frames)
    raw = np.frombuffer(result.stdout[: actual_frames * frame_size], dtype=np.uint8)
    frames = raw.reshape(actual_frames, info.height, info.width, 3).copy()
    return np.ascontiguousarray(frames.transpose(3, 0, 1, 2)[None], dtype=np.float32) / 255.0


def read_reference_video_frames(
    path: Path,
    info: VideoInfo | None = None,
    target_fps: float = 24.0,
    target_height: int | None = None,
    target_width: int | None = None,
    max_frames: int | None = None,
    max_bytes: int = 2 * 1024**3,
) -> np.ndarray:
    info = info or probe_video(path)
    if target_fps <= 0:
        raise ValueError("Target reference frame rate must be positive")
    if (target_height is None) != (target_width is None):
        raise ValueError("Reference target height and width must be provided together")
    height = target_height or info.height
    width = target_width or info.width
    frame_count = max(1, round(info.duration_seconds * target_fps))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    expected_bytes = frame_count * width * height * 3
    if expected_bytes > max_bytes:
        raise ValueError(
            f"Reference video exceeds the decode budget: "
            f"{expected_bytes / 1024**3:.2f} GiB > {max_bytes / 1024**3:.2f} GiB"
        )
    filters = [f"fps={target_fps:g}"]
    if (height, width) != (info.height, info.width):
        filters.append(f"scale={width}:{height}:flags=lanczos")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg video decode failed")
    frame_size = width * height * 3
    actual_frames = len(result.stdout) // frame_size
    if actual_frames < 1:
        raise ValueError("ffmpeg decoded no reference video frames")
    raw = np.frombuffer(result.stdout[: actual_frames * frame_size], dtype=np.uint8)
    return raw.reshape(actual_frames, height, width, 3).copy()


def read_audio_waveform(
    path: Path,
    target_sample_rate: int = 32_000,
    target_channels: int = 2,
    max_duration_seconds: float | None = None,
    max_bytes: int = 512 * 1024**2,
) -> np.ndarray:
    if target_sample_rate <= 0 or target_channels <= 0:
        raise ValueError("Target audio sample rate and channel count must be positive")
    info = probe_audio(path)
    duration = info.duration_seconds
    if max_duration_seconds is not None:
        if max_duration_seconds <= 0:
            raise ValueError("Audio decode duration must be positive")
        duration = min(duration, max_duration_seconds)
    expected_samples = max(1, math.ceil(duration * target_sample_rate))
    expected_bytes = expected_samples * target_channels * np.dtype(np.float32).itemsize
    if expected_bytes > max_bytes:
        raise ValueError(
            f"Reference audio exceeds the decode budget: "
            f"{expected_bytes / 1024**2:.1f} MiB > {max_bytes / 1024**2:.1f} MiB"
        )
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0"]
    if max_duration_seconds is not None:
        command.extend(("-t", f"{max_duration_seconds:.9g}"))
    command.extend(
        (
            "-ac",
            str(target_channels),
            "-ar",
            str(target_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        )
    )
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg audio decode failed")
    values = np.frombuffer(result.stdout, dtype="<f4")
    samples = values.size // target_channels
    if samples < 1:
        raise ValueError("ffmpeg decoded no audio samples")
    values = values[: samples * target_channels].reshape(samples, target_channels).T.copy()
    return np.ascontiguousarray(values, dtype=np.float32)


def resize_video_spatiotemporal(
    frames: np.ndarray,
    target_frames: int,
    target_height: int,
    target_width: int,
    mode: InterpolationMode,
) -> np.ndarray:
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[1] != 3:
        raise ValueError("Video frames must have shape [1, 3, T, H, W]")
    if target_frames < 1 or target_height < 1 or target_width < 1:
        raise ValueError("Interpolation target dimensions must be positive")
    tensor = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
    if mode == "nearest":
        resized = F.interpolate(tensor, size=(target_frames, target_height, target_width), mode="nearest")
    elif mode == "trilinear":
        resized = F.interpolate(
            tensor,
            size=(target_frames, target_height, target_width),
            mode="trilinear",
            align_corners=False,
        )
    elif mode in {"bilinear", "bicubic"}:
        temporal = tensor.permute(0, 3, 4, 1, 2).reshape(-1, 3, tensor.shape[2])
        temporal = F.interpolate(temporal, size=target_frames, mode="linear", align_corners=False)
        temporal = temporal.reshape(1, tensor.shape[3], tensor.shape[4], 3, target_frames).permute(0, 3, 4, 1, 2)
        spatial = temporal.permute(0, 2, 1, 3, 4).reshape(-1, 3, tensor.shape[3], tensor.shape[4])
        spatial = F.interpolate(spatial, size=(target_height, target_width), mode=mode, align_corners=False)
        resized = spatial.reshape(1, target_frames, 3, target_height, target_width).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    return resized.clamp(0.0, 1.0).numpy()


def center_pad_video(frames: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if frames.shape[-2] > target_height or frames.shape[-1] > target_width:
        raise ValueError("Target padding canvas is smaller than the video")
    pad_height = target_height - frames.shape[-2]
    pad_width = target_width - frames.shape[-1]
    if pad_height == 0 and pad_width == 0:
        return frames
    tensor = torch.from_numpy(frames)
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    return F.pad(tensor, (left, right, top, bottom, 0, 0), mode="replicate").numpy()


def prepare_frame_condition(
    image: np.ndarray,
    output_height: int,
    output_width: int,
    padded_height: int,
    padded_width: int,
) -> np.ndarray:
    if image.ndim != 5 or image.shape[:3] != (1, 3, 1):
        raise ValueError("Condition image must have shape [1, 3, 1, H, W]")
    resized = resize_video_spatiotemporal(
        image,
        1,
        output_height,
        output_width,
        "bicubic",
    )
    return center_pad_video(resized, padded_height, padded_width) * 2.0 - 1.0


def prepare_super_resolution_segment(
    frames: np.ndarray,
    start: int,
    stop: int,
    segment_frames: int,
    vae_frames: int,
    output_height: int,
    output_width: int,
    padded_height: int,
    padded_width: int,
    interpolation: InterpolationMode,
) -> tuple[np.ndarray, int]:
    if not 0 <= start < frames.shape[2]:
        raise ValueError("Super-resolution segment starts outside the input video")
    actual_frames = min(stop, frames.shape[2]) - start
    if actual_frames < 1:
        raise ValueError("Super-resolution segment has no input frames")
    segment = frames[:, :, start : start + actual_frames]
    if actual_frames < segment_frames:
        segment = np.concatenate(
            (segment, np.repeat(segment[:, :, -1:], segment_frames - actual_frames, axis=2)),
            axis=2,
        )
    resized = resize_video_spatiotemporal(
        segment,
        vae_frames,
        output_height,
        output_width,
        interpolation,
    )
    padded = center_pad_video(resized, padded_height, padded_width)
    return padded * 2.0 - 1.0, actual_frames
