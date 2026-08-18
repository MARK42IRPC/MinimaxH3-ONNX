from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np


ReferenceKind = Literal["image", "video", "audio"]

H3_FPS = 24.0
H3_AUDIO_SAMPLE_RATE = 32_000
H3_AUDIO_CHANNELS = 2
H3_AUDIO_LATENTS_PER_SECOND = 40
H3_CANVAS_MULTIPLE = 32
H3_CANVAS_SHORT_EDGE = 768
H3_CANVAS_MAX_PIXELS = 768 * 1344
H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
H3_MIN_DURATION_SECONDS = 5.0
H3_MAX_DURATION_SECONDS = 15.0
H3_MIN_REFERENCE_DURATION_SECONDS = 2.0
H3_MAX_REFERENCE_DURATION_SECONDS = 15.0

H3_VIDEO_TAG = 0
H3_TEXT_TAG = 1
H3_AUDIO_TAG = 2

VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_PAD_TOKEN_ID = 151655
VIDEO_PAD_TOKEN_ID = 151656

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


class ReferenceLike(Protocol):
    kind: ReferenceKind
    has_audio: bool


@dataclass(frozen=True)
class ReferenceRequest:
    kind: ReferenceKind
    path: str
    role: str = "reference"

    @classmethod
    def from_value(cls, value: ReferenceRequest | Mapping[str, Any]) -> ReferenceRequest:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("Each reference must be an object with kind and path")
        kind = str(value.get("kind") or value.get("type") or "").strip().lower()
        if kind not in {"image", "video", "audio"}:
            raise ValueError(f"Unknown reference kind: {kind or '(empty)'}")
        path = str(value.get("path") or value.get("uri") or "").strip()
        if not path:
            raise ValueError("Reference path must not be empty")
        role = str(value.get("role") or "reference").strip().lower()
        return cls(kind=kind, path=path, role=role)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ReferenceSpec:
    kind: ReferenceKind
    path: str
    has_audio: bool = False
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "has_audio": self.has_audio,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class ReferencePresentation:
    token_ids: np.ndarray
    token_tags: np.ndarray
    # Qwen's modality tags are separate from H3's packed-row tags.  The
    # processor marks only image/video pad tokens; vision delimiters remain
    # text tokens for M-RoPE grouping.
    mm_token_type_ids: np.ndarray


@dataclass(frozen=True)
class PackedReferenceLayout:
    position_ids: np.ndarray
    token_tags: np.ndarray
    video_indices: np.ndarray
    audio_indices: np.ndarray
    text_indices: np.ndarray
    num_condition_video_rows: int
    num_condition_audio_rows: int

    @property
    def target_video_indices(self) -> np.ndarray:
        return self.video_indices[self.num_condition_video_rows :]

    @property
    def target_audio_indices(self) -> np.ndarray:
        return self.audio_indices[self.num_condition_audio_rows :]


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int = H3_CANVAS_MULTIPLE,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
    max_pixels: int = H3_CANVAS_MAX_PIXELS,
    min_aspect_ratio: float = 1 / 4,
    max_aspect_ratio: float = 4,
) -> tuple[int, int]:
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} "
            f"to {max_aspect_ratio:g}:1, got {aspect_width}:{aspect_height}"
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        width, height = width * scale, height * scale
    return (
        max(canvas_multiple, round(height / canvas_multiple) * canvas_multiple),
        max(canvas_multiple, round(width / canvas_multiple) * canvas_multiple),
    )


def align_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"num_frames must be of the form {frames_per_chunk} * n + {latents_per_chunk}, got {num_frames}"
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int,
    fps: float = H3_FPS,
    latents_per_second: int = H3_AUDIO_LATENTS_PER_SECOND,
) -> int:
    return int(round(num_frames / fps * latents_per_second))


def reference_labels(references: Sequence[ReferenceLike]) -> tuple[tuple[str, ...], ...]:
    counts = {"image": 0, "video": 0, "audio": 0}
    labels: list[tuple[str, ...]] = []
    for reference in references:
        current: list[str] = []
        if reference.has_audio:
            counts["audio"] += 1
            current.append(f"Audio {counts['audio']}")
        if reference.kind == "image":
            counts["image"] += 1
            current.append(f"Picture {counts['image']}")
        elif reference.kind == "video":
            counts["video"] += 1
            current.append(f"Video {counts['video']}")
        labels.append(tuple(current))
    return tuple(labels)


def resolve_reference_specs(
    workspace: Path,
    values: Sequence[ReferenceRequest | Mapping[str, Any]],
) -> tuple[ReferenceSpec, ...]:
    from h3_workbench.media_input import (
        probe_audio,
        probe_image,
        probe_video,
        resolve_audio_path,
        resolve_image_path,
        resolve_video_path,
    )

    requests = tuple(ReferenceRequest.from_value(value) for value in values)
    if not requests:
        raise ValueError("Ref2VA needs at least one reference")
    if len(requests) > 12:
        raise ValueError(f"MiniMax-H3 accepts at most 12 references in total, got {len(requests)}")
    if any(request.role != "reference" for request in requests):
        raise ValueError("MiniMax-H3 Ref2VA conditions must use role='reference'")

    kinds = [request.kind for request in requests]
    for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
        count = kinds.count(kind)
        if count > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {count}")
    if set(kinds) == {"audio"}:
        raise ValueError("An audio reference must be paired with at least one image or video reference")

    resolved: list[ReferenceSpec] = []
    video_duration = 0.0
    audio_duration = 0.0
    for request in requests:
        if request.kind == "image":
            path = resolve_image_path(workspace, request.path)
            info = probe_image(path)
            ratio = info.width / info.height
            if not 0.25 <= ratio <= 4.0:
                raise ValueError(f"A reference image must be within 1:4 and 4:1, got {info.width}x{info.height}")
            resolved.append(
                ReferenceSpec(kind="image", path=str(path), width=info.width, height=info.height)
            )
            continue

        if request.kind == "video":
            path = resolve_video_path(workspace, request.path)
            info = probe_video(path)
            _validate_reference_duration("video", info.duration_seconds)
            video_duration += info.duration_seconds
            resolved.append(
                ReferenceSpec(
                    kind="video",
                    path=str(path),
                    has_audio=info.has_audio,
                    duration_seconds=info.duration_seconds,
                    width=info.width,
                    height=info.height,
                    fps=info.fps,
                )
            )
            continue

        path = resolve_audio_path(workspace, request.path)
        info = probe_audio(path)
        _validate_reference_duration("audio", info.duration_seconds)
        audio_duration += info.duration_seconds
        resolved.append(
            ReferenceSpec(
                kind="audio",
                path=str(path),
                has_audio=True,
                duration_seconds=info.duration_seconds,
                sample_rate=info.sample_rate,
                channels=info.channels,
            )
        )

    if video_duration > H3_MAX_REFERENCE_DURATION_SECONDS + 1e-6:
        raise ValueError(
            f"Reference videos may total at most {H3_MAX_REFERENCE_DURATION_SECONDS:g} seconds, "
            f"got {video_duration:.3g}"
        )
    if audio_duration > H3_MAX_REFERENCE_DURATION_SECONDS + 1e-6:
        raise ValueError(
            f"Independent audio references may total at most {H3_MAX_REFERENCE_DURATION_SECONDS:g} seconds, "
            f"got {audio_duration:.3g}"
        )

    labels = reference_labels(resolved)
    return tuple(replace(reference, labels=label) for reference, label in zip(resolved, labels))


def _validate_reference_duration(kind: str, duration: float) -> None:
    if not math.isfinite(duration) or not H3_MIN_REFERENCE_DURATION_SECONDS <= duration <= H3_MAX_REFERENCE_DURATION_SECONDS:
        raise ValueError(
            f"Each {kind} reference must run from {H3_MIN_REFERENCE_DURATION_SECONDS:g} to "
            f"{H3_MAX_REFERENCE_DURATION_SECONDS:g} seconds, got {duration:.3g}"
        )


def sample_video_condition_frames(
    frame_count: int,
    fps: float = H3_FPS,
    sample_fps: float = 2.0,
    temporal_patch: int = 2,
) -> tuple[np.ndarray, tuple[float, ...]]:
    if frame_count < 1:
        raise ValueError("A reference video must contain at least one frame")
    if fps <= 0 or sample_fps <= 0:
        raise ValueError("Reference and conditioner frame rates must be positive")
    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frame_count:
        index = round(cursor)
        if not indices or index > indices[-1]:
            indices.append(index)
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"A reference video sampled at {sample_fps:g} fps needs at least {minimum} frames at {fps:g} fps"
        )
    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = tuple(
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    )
    return np.asarray(indices, dtype=np.int64), block_timestamps


def _tokenize_text(tokenizer: Any, value: str) -> list[int]:
    encoded = tokenizer(value, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if encoded is None:
        raise ValueError("Tokenizer did not return input_ids")
    if isinstance(encoded, np.ndarray):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _special_token_id(tokenizer: Any, token: str, fallback: int) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        value = convert(token)
        if value is not None and int(value) >= 0:
            return int(value)
    return fallback


def build_reference_presentation(
    tokenizer: Any,
    prompt: str,
    references: Sequence[ReferenceLike],
    image_token_counts: Sequence[int],
    video_block_token_counts: Sequence[int],
    video_block_timestamps: Sequence[Sequence[float]],
    text_tag: int = H3_TEXT_TAG,
    video_tag: int = H3_VIDEO_TAG,
) -> ReferencePresentation:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Ref2VA prompt must be a non-empty string")
    if len(image_token_counts) != sum(reference.kind == "image" for reference in references):
        raise ValueError("Image token counts do not match the reference list")
    video_count = sum(reference.kind == "video" for reference in references)
    if len(video_block_token_counts) != video_count or len(video_block_timestamps) != video_count:
        raise ValueError("Video token geometry does not match the reference list")

    vision_start = _special_token_id(tokenizer, "<|vision_start|>", VISION_START_TOKEN_ID)
    vision_end = _special_token_id(tokenizer, "<|vision_end|>", VISION_END_TOKEN_ID)
    image_pad = _special_token_id(tokenizer, "<|image_pad|>", IMAGE_PAD_TOKEN_ID)
    video_pad = _special_token_id(tokenizer, "<|video_pad|>", VIDEO_PAD_TOKEN_ID)
    token_ids: list[int] = []
    token_tags: list[int] = []
    mm_token_type_ids: list[int] = []

    def emit_text(value: str) -> None:
        ids = _tokenize_text(tokenizer, value)
        token_ids.extend(ids)
        token_tags.extend([text_tag] * len(ids))
        mm_token_type_ids.extend([0] * len(ids))

    def emit_vision(pad_token: int, count: int, modality: int) -> None:
        if count < 1:
            raise ValueError("A vision block must contain at least one merged token")
        ids = [vision_start, *([pad_token] * count), vision_end]
        token_ids.extend(ids)
        token_tags.extend([video_tag] * len(ids))
        mm_token_type_ids.extend([0, *([modality] * count), 0])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit_text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            emit_text(f"<Picture {counts['image']}>: ")
            emit_vision(image_pad, int(image_token_counts[counts["image"] - 1]), 1)
        elif reference.kind == "video":
            counts["video"] += 1
            video_index = counts["video"] - 1
            emit_text(f"<Video {counts['video']}>: ")
            for timestamp in video_block_timestamps[video_index]:
                emit_text(f"<{timestamp:.1f} seconds>")
                emit_vision(video_pad, int(video_block_token_counts[video_index]), 2)
    emit_text(prompt)
    return ReferencePresentation(
        token_ids=np.asarray(token_ids, dtype=np.int64),
        token_tags=np.asarray(token_tags, dtype=np.int64),
        mm_token_type_ids=np.asarray(mm_token_type_ids, dtype=np.int64),
    )


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False, dtype=np.float64) * _ROPE_SPATIAL_SCALE


def _temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.asarray(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(spans[:-1], dtype=np.float64)))


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    height_values, width_values = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack((height_values.reshape(-1), width_values.reshape(-1)), axis=-1), width_grid


def _fill_audio_positions(
    position_ids: np.ndarray,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: np.ndarray,
    audio_channels: int,
) -> None:
    if rows.stop == rows.start:
        return
    time = rotary_time + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[rows, 0] = np.tile(time, audio_channels)
    # The official layout anchors each stereo channel to one side of the
    # spatial grid.  Interpolating between the endpoints changes the second
    # channel when more than two channels are ever configured and, for the
    # released stereo model, does not reproduce the reference coordinates.
    channel_positions = np.concatenate(
        (
            np.full(1, float(width_grid[0]), dtype=np.float64),
            np.full(audio_channels - 1, float(width_grid[-1]), dtype=np.float64),
        )
    )
    position_ids[rows, 2] = np.repeat(channel_positions, num_audio_latents)


def build_ref2va_packed_layout(
    text_token_tags: np.ndarray,
    references: Sequence[ReferenceLike],
    condition_video_shapes: Sequence[tuple[int, int, int]],
    condition_audio_row_counts: Sequence[int],
    target_video_shape: tuple[int, int, int],
    num_target_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = H3_AUDIO_CHANNELS,
    audio_tag: int = H3_AUDIO_TAG,
    video_tag: int = H3_VIDEO_TAG,
) -> PackedReferenceLayout:
    text_token_tags = np.asarray(text_token_tags, dtype=np.int64).reshape(-1)
    _, patch_h, patch_w = patch_size
    target_frames, target_height, target_width = target_video_shape
    if target_height % patch_h or target_width % patch_w:
        raise ValueError("Target latent geometry is not divisible by the transformer patch")

    visual_count = sum(reference.kind in {"image", "video"} for reference in references)
    audio_count = sum(reference.has_audio for reference in references)
    if len(condition_video_shapes) != visual_count:
        raise ValueError("Visual latent geometry does not match the reference list")
    if len(condition_audio_row_counts) != audio_count:
        raise ValueError("Audio latent geometry does not match the reference list")
    if any(rows < 0 or rows % audio_channels for rows in condition_audio_row_counts):
        raise ValueError("Reference audio rows must be non-negative and divisible by the channel count")

    num_target_video_rows = target_frames * (target_height // patch_h) * (target_width // patch_w)
    num_target_audio_rows = num_target_audio_latents * audio_channels
    num_reference_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in condition_video_shapes
    )
    num_reference_audio_rows = sum(condition_audio_row_counts)
    sequence_length = (
        text_token_tags.size
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )
    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[: text_token_tags.size, 0] = np.arange(text_token_tags.size, dtype=np.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(target_height, target_width, patch_h, patch_w)

    visual_geometry = iter(condition_video_shapes)
    audio_row_counts = iter(condition_audio_row_counts)
    video_indices: list[np.ndarray] = []
    audio_indices: list[np.ndarray] = []
    cursor = text_token_tags.size
    rotary_time = float(text_token_tags.size)
    for reference in references:
        if reference.kind == "image":
            frames, height, width = next(visual_geometry)
            rows_count = frames * (height // patch_h) * (width // patch_w)
            rows = slice(cursor, cursor + rows_count)
            cursor = rows.stop
            video_indices.append(np.arange(rows.start, rows.stop, dtype=np.int64))
            frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = np.tile(frame_grid, (frames, 1))
            rotary_time += 1.0
            continue

        if reference.kind == "audio":
            audio_rows_count = next(audio_row_counts)
            audio_latents = audio_rows_count // audio_channels
            rows = slice(cursor, cursor + audio_rows_count)
            cursor = rows.stop
            audio_indices.append(np.arange(rows.start, rows.stop, dtype=np.int64))
            _fill_audio_positions(position_ids, rows, audio_latents, rotary_time, target_width_grid, audio_channels)
            rotary_time += float(audio_latents)
            continue

        if reference.kind != "video":
            raise ValueError(f"Unknown reference kind: {reference.kind!r}")
        audio_rows_count = next(audio_row_counts) if reference.has_audio else 0
        audio_latents = audio_rows_count // audio_channels
        frames, height, width = next(visual_geometry)
        video_rows_count = frames * (height // patch_h) * (width // patch_w)
        audio_rows = slice(cursor, cursor + audio_rows_count)
        video_rows = slice(audio_rows.stop, audio_rows.stop + video_rows_count)
        cursor = video_rows.stop
        audio_indices.append(np.arange(audio_rows.start, audio_rows.stop, dtype=np.int64))
        video_indices.append(np.arange(video_rows.start, video_rows.stop, dtype=np.int64))
        frame_grid, width_grid = _frame_position_grid(height, width, patch_h, patch_w)
        _fill_audio_positions(position_ids, audio_rows, audio_latents, rotary_time, width_grid, audio_channels)
        frame_time = _temporal_position_grid(frames, rotary_time)
        position_ids[video_rows, 0] = np.repeat(frame_time, frame_grid.shape[0])
        position_ids[video_rows, 1:] = np.tile(frame_grid, (frames, 1))
        video_span = sum(
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(frames)
        )
        rotary_time += max(float(audio_latents), video_span)

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_target_audio_latents,
        rotary_time,
        target_width_grid,
        audio_channels,
    )
    target_frame_time = _temporal_position_grid(target_frames, rotary_time)
    position_ids[video_start:, 0] = np.repeat(target_frame_time, target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = np.tile(target_frame_grid, (target_frames, 1))

    video_indices.append(np.arange(video_start, sequence_length, dtype=np.int64))
    audio_indices.append(np.arange(audio_start, video_start, dtype=np.int64))
    packed_video_indices = np.concatenate(video_indices) if video_indices else np.empty(0, dtype=np.int64)
    packed_audio_indices = np.concatenate(audio_indices) if audio_indices else np.empty(0, dtype=np.int64)
    text_indices = np.arange(text_token_tags.size, dtype=np.int64)
    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_indices] = text_token_tags
    token_tags[packed_audio_indices] = audio_tag
    token_tags[packed_video_indices] = video_tag

    return PackedReferenceLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=packed_video_indices,
        audio_indices=packed_audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


def build_row_timesteps(
    layout: PackedReferenceLayout,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float = 0.999,
    condition_audio_timestep: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    row_timesteps = np.full(layout.token_tags.shape[0], video_timestep, dtype=np.float32)
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
    return np.unique(row_timesteps, return_inverse=True)
