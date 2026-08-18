from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from h3_workbench.media_input import (
    read_audio_waveform,
    read_reference_image,
    read_reference_video_frames,
)
from h3_workbench.media_output import (
    encode_audio_waveform_onnx,
    encode_image_frame,
    encode_video_frames,
)
from h3_workbench.qwen_transformer import qwen_mrope_position_ids
from h3_workbench.qwen_vision import Qwen3VLVisionEncoder, VisionEncoding
from h3_workbench.reference import (
    H3_AUDIO_SAMPLE_RATE,
    H3_CANVAS_MAX_PIXELS,
    H3_CANVAS_MULTIPLE,
    H3_CANVAS_SHORT_EDGE,
    H3_FPS,
    H3_REFERENCE_IMAGE_SHORT_EDGE,
    IMAGE_PAD_TOKEN_ID,
    VIDEO_PAD_TOKEN_ID,
    ReferencePresentation,
    ReferenceSpec,
    build_ref2va_packed_layout,
    build_reference_presentation,
    resolve_canvas_size,
    sample_video_condition_frames,
)


ReferenceActivity = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class NormalizedReferenceMedia:
    spec: ReferenceSpec
    pixels: np.ndarray | None
    waveform: np.ndarray | None

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def has_audio(self) -> bool:
        return self.spec.has_audio


@dataclass(frozen=True)
class ReferenceVisionCondition:
    presentation: ReferencePresentation
    position_ids: np.ndarray
    visual_condition: dict[str, object]
    encodings: tuple[VisionEncoding, ...]
    image_grid_thw: np.ndarray
    video_grid_thw: np.ndarray
    video_timestamps: tuple[tuple[float, ...], ...]


def _emit(callback: ReferenceActivity | None, operation: str, **details: object) -> None:
    if callback is not None:
        callback({"module": "Ref2VA", "operation": operation, **details})


def reference_image_size(width: int, height: int, short_edge: int = H3_REFERENCE_IMAGE_SHORT_EDGE) -> tuple[int, int]:
    if width < 1 or height < 1 or short_edge < 1:
        raise ValueError("Reference image dimensions must be positive")
    scale = short_edge / min(width, height)
    target_height = max(H3_CANVAS_MULTIPLE, round(height * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
    target_width = max(H3_CANVAS_MULTIPLE, round(width * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
    return target_height, target_width


def snap_reference_video_frames(frame_count: int) -> int:
    """Snap a decoded reference down to the Video VAE's native temporal form."""
    if frame_count < 1:
        raise ValueError("A reference video must contain at least one frame")
    if frame_count < 5:
        raise ValueError("A reference video must contain at least 5 frames for Ref2VA")
    return max(1, (frame_count - 5) // 17) * 17 + 5


def normalize_reference_media(
    references: Sequence[ReferenceSpec],
    target_frames: int,
    callback: ReferenceActivity | None = None,
) -> tuple[NormalizedReferenceMedia, ...]:
    if target_frames < 1:
        raise ValueError("Ref2VA target frame count must be positive")
    duration = target_frames / H3_FPS
    normalized: list[NormalizedReferenceMedia] = []
    for index, spec in enumerate(references, 1):
        _emit(callback, "Normalizing reference", index=index, kind=spec.kind, path=spec.path)
        pixels: np.ndarray | None = None
        waveform: np.ndarray | None = None
        if spec.kind == "image":
            assert spec.width is not None and spec.height is not None
            height, width = reference_image_size(spec.width, spec.height)
            pixels = read_reference_image(Path(spec.path), height, width)
        elif spec.kind == "video":
            assert spec.width is not None and spec.height is not None
            height, width = resolve_canvas_size(
                spec.width,
                spec.height,
                H3_CANVAS_MULTIPLE,
                H3_CANVAS_SHORT_EDGE,
                H3_CANVAS_MAX_PIXELS,
            )
            pixels = read_reference_video_frames(
                Path(spec.path),
                target_fps=H3_FPS,
                target_height=height,
                target_width=width,
                max_frames=target_frames,
            )
        if spec.has_audio:
            waveform = read_audio_waveform(
                Path(spec.path),
                target_sample_rate=H3_AUDIO_SAMPLE_RATE,
                target_channels=2,
                max_duration_seconds=duration,
            )
        normalized.append(NormalizedReferenceMedia(spec, pixels, waveform))
    return tuple(normalized)


def _token_id(tokenizer: Any, token: str, fallback: int) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        value = convert(token)
        if value is not None and int(value) >= 0:
            return int(value)
    return fallback


def _concat_features(encodings: Sequence[VisionEncoding], deepstack_index: int | None = None) -> np.ndarray:
    if not encodings:
        return np.empty((0, 5120), dtype=np.float32)
    if deepstack_index is None:
        return np.ascontiguousarray(np.concatenate([encoding.features for encoding in encodings], axis=0), dtype=np.float32)
    return np.ascontiguousarray(
        np.concatenate(
            [encoding.deepstack_features[deepstack_index] for encoding in encodings],
            axis=0,
        ),
        dtype=np.float32,
    )


def encode_reference_vision(
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    references: Sequence[NormalizedReferenceMedia],
    *,
    prefer_cuda: bool,
    callback: ReferenceActivity | None = None,
) -> ReferenceVisionCondition:
    image_encodings: list[VisionEncoding] = []
    video_encodings: list[VisionEncoding] = []
    all_encodings: list[VisionEncoding] = []
    image_token_counts: list[int] = []
    video_token_counts: list[int] = []
    video_timestamps: list[tuple[float, ...]] = []
    with Qwen3VLVisionEncoder(checkpoint, prefer_cuda=prefer_cuda) as encoder:
        for index, reference in enumerate(references, 1):
            if reference.kind == "audio":
                continue
            if reference.pixels is None:
                raise ValueError(f"Reference {index} has no decoded visual pixels")
            if reference.kind == "image":
                encoding = encoder.encode_image(reference.pixels)
                image_encodings.append(encoding)
                image_token_counts.append(encoding.geometry.merged_tokens)
            else:
                indices, timestamps = sample_video_condition_frames(
                    reference.pixels.shape[0], H3_FPS, sample_fps=2.0, temporal_patch=2
                )
                encoding = encoder.encode_sampled_video(
                    reference.pixels[indices],
                    source_frames=reference.pixels.shape[0],
                    sampled_indices=indices,
                )
                video_encodings.append(encoding)
                video_token_counts.append(encoding.geometry.tokens_per_temporal_block)
                video_timestamps.append(timestamps)
            all_encodings.append(encoding)
            _emit(
                callback,
                "Qwen visual reference encoded",
                index=index,
                kind=reference.kind,
                merged_tokens=encoding.geometry.merged_tokens,
                grid_thw=list(encoding.geometry.grid_thw),
            )

    presentation = build_reference_presentation(
        tokenizer,
        prompt,
        [reference.spec for reference in references],
        image_token_counts,
        video_token_counts,
        video_timestamps,
    )
    image_pad = _token_id(tokenizer, "<|image_pad|>", IMAGE_PAD_TOKEN_ID)
    video_pad = _token_id(tokenizer, "<|video_pad|>", VIDEO_PAD_TOKEN_ID)
    image_mask = presentation.token_ids == image_pad
    video_mask = presentation.token_ids == video_pad
    image_grid = np.asarray(
        [encoding.geometry.grid_thw for encoding in image_encodings], dtype=np.int64
    ).reshape(-1, 3)
    video_grid = np.asarray(
        [encoding.geometry.grid_thw for encoding in video_encodings], dtype=np.int64
    ).reshape(-1, 3)
    visual_condition: dict[str, object] = {
        "image_mask": image_mask,
        "video_mask": video_mask,
        "image_features": _concat_features(image_encodings),
        "video_features": _concat_features(video_encodings),
        "image_deepstack": tuple(_concat_features(image_encodings, index) for index in range(3)),
        "video_deepstack": tuple(_concat_features(video_encodings, index) for index in range(3)),
    }
    position_ids = qwen_mrope_position_ids(
        presentation.token_ids,
        image_grid,
        video_grid,
        mm_token_type_ids=presentation.mm_token_type_ids,
    )
    return ReferenceVisionCondition(
        presentation=presentation,
        position_ids=position_ids,
        visual_condition=visual_condition,
        encodings=tuple(all_encodings),
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        video_timestamps=tuple(video_timestamps),
    )


def _video_pixels_for_vae(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float32) / 127.5 - 1.0
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"Reference video pixels must be THWC uint8, got {values.shape}")
    return np.ascontiguousarray(values.transpose(3, 0, 1, 2)[None], dtype=np.float16)


def _image_pixels_for_vae(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float32) / 127.5 - 1.0
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Reference image pixels must be HWC uint8, got {values.shape}")
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None, :, None], dtype=np.float16)


def encode_reference_latents(
    video_encoder: Any,
    audio_directory: Path,
    references: Sequence[NormalizedReferenceMedia],
    *,
    prefer_cuda: bool,
    posterior_seed: int = 42,
    callback: ReferenceActivity | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    condition_video: list[np.ndarray] = []
    condition_audio: list[np.ndarray] = []
    for index, reference in enumerate(references, 1):
        if reference.kind == "image":
            assert reference.pixels is not None
            def image_callback(details: dict[str, object], index: int = index) -> None:
                payload = dict(details)
                operation = str(payload.pop("operation", "Video VAE"))
                _emit(callback, operation, index=index, **payload)

            latent = encode_image_frame(
                video_encoder,
                _image_pixels_for_vae(reference.pixels),
                callback=image_callback,
                offload_after=False,
                sample_posterior=True,
                posterior_seed=posterior_seed,
            )
            condition_video.append(latent)
        elif reference.kind == "video":
            assert reference.pixels is not None
            frame_count = snap_reference_video_frames(reference.pixels.shape[0])
            def video_callback(details: dict[str, object], index: int = index) -> None:
                payload = dict(details)
                operation = str(payload.pop("operation", "Video VAE"))
                _emit(callback, operation, index=index, **payload)

            latent = encode_video_frames(
                video_encoder,
                _video_pixels_for_vae(reference.pixels[:frame_count]),
                callback=video_callback,
                offload_after=False,
                sample_posterior=True,
                posterior_seed=posterior_seed,
            )
            condition_video.append(latent)
        if reference.has_audio:
            if reference.waveform is None:
                raise ValueError(f"Reference {index} declares audio but no waveform was decoded")
            def audio_callback(details: dict[str, object], index: int = index) -> None:
                payload = dict(details)
                operation = str(payload.pop("operation", "Audio VAE"))
                _emit(callback, operation, index=index, **payload)

            latent = encode_audio_waveform_onnx(
                audio_directory,
                reference.waveform,
                prefer_cuda=prefer_cuda,
                callback=audio_callback,
            )
            condition_audio.append(np.ascontiguousarray(latent, dtype=np.float32))
    return tuple(condition_video), tuple(condition_audio)


def build_reference_layout(
    references: Sequence[NormalizedReferenceMedia],
    presentation: ReferencePresentation,
    condition_video_latents: Sequence[np.ndarray],
    condition_audio_latents: Sequence[np.ndarray],
    target_video_shape: tuple[int, int, int],
    target_audio_latents: int,
) -> Any:
    visual_shapes = [tuple(int(value) for value in latent.shape[2:5]) for latent in condition_video_latents]
    audio_rows = [int(latent.shape[2] * latent.shape[3]) for latent in condition_audio_latents]
    return build_ref2va_packed_layout(
        presentation.token_tags,
        [reference.spec for reference in references],
        visual_shapes,
        audio_rows,
        target_video_shape,
        target_audio_latents,
    )
