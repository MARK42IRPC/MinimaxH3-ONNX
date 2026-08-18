from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F


VisionKind = Literal["image", "video"]

QWEN_PATCH_SIZE = 16
QWEN_TEMPORAL_PATCH_SIZE = 2
QWEN_MERGE_SIZE = 2
QWEN_IMAGE_MIN_PIXELS = 65_536
QWEN_IMAGE_MAX_PIXELS = 16_777_216
QWEN_VIDEO_MIN_PIXELS = 4_096
QWEN_VIDEO_MAX_PIXELS = 25_165_824
QWEN_VISION_HIDDEN_SIZE = 1152
QWEN_VISION_OUT_SIZE = 5120
QWEN_VISION_DEPTH = 27
QWEN_VISION_HEADS = 16
QWEN_VISION_INTERMEDIATE_SIZE = 4304
QWEN_VISION_POSITION_EMBEDDINGS = 2304
QWEN_DEEPSTACK_INDEXES = (8, 16, 24)


@dataclass(frozen=True)
class VisionGeometry:
    kind: VisionKind
    grid_thw: tuple[int, int, int]
    resized_height: int
    resized_width: int
    source_frames: int
    sampled_indices: tuple[int, ...] = ()

    @property
    def merged_tokens(self) -> int:
        temporal, height, width = self.grid_thw
        return temporal * (height // QWEN_MERGE_SIZE) * (width // QWEN_MERGE_SIZE)

    @property
    def tokens_per_temporal_block(self) -> int:
        _, height, width = self.grid_thw
        return (height // QWEN_MERGE_SIZE) * (width // QWEN_MERGE_SIZE)


@dataclass(frozen=True)
class VisionEncoding:
    features: np.ndarray
    deepstack_features: tuple[np.ndarray, ...]
    geometry: VisionGeometry

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != QWEN_VISION_OUT_SIZE:
            raise ValueError(f"Unexpected Qwen vision feature shape: {self.features.shape}")
        if self.features.shape[0] != self.geometry.merged_tokens:
            raise ValueError(
                "Qwen vision feature count does not match the processor grid: "
                f"{self.features.shape[0]} != {self.geometry.merged_tokens}"
            )
        for feature in self.deepstack_features:
            if feature.shape != self.features.shape:
                raise ValueError("Qwen DeepStack feature shape does not match merged features")


def smart_resize_image(
    height: int,
    width: int,
    *,
    factor: int = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE,
    min_pixels: int = QWEN_IMAGE_MIN_PIXELS,
    max_pixels: int = QWEN_IMAGE_MAX_PIXELS,
) -> tuple[int, int]:
    """Match Qwen2-VL's smart_resize used by the official H3 image processor."""
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    if factor < 1 or min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError("Invalid Qwen image resize limits")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int = QWEN_TEMPORAL_PATCH_SIZE,
    factor: int = QWEN_PATCH_SIZE * QWEN_MERGE_SIZE,
    min_pixels: int = QWEN_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN_VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    """Match Qwen3-VL's video smart_resize implementation."""
    if num_frames < temporal_factor:
        raise ValueError(f"num_frames must be at least {temporal_factor}")
    if height < 1 or width < 1:
        raise ValueError("Video dimensions must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    if min(height, width) < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def sample_video_indices(
    frame_count: int,
    fps: float,
    target_fps: float = 2.0,
    *,
    min_frames: int = 4,
    max_frames: int = 768,
) -> np.ndarray:
    """Use Qwen3-VL's uniform ``int(duration * fps)`` frame sampling."""
    if frame_count < 1 or fps <= 0 or target_fps <= 0:
        raise ValueError("Video frame count and rates must be positive")
    requested = int(frame_count / fps * target_fps)
    requested = min(max(requested, min_frames), max_frames, frame_count)
    requested = max(1, requested)
    return np.linspace(0, frame_count - 1, requested).round().astype(np.int64)


def paired_video_timestamps(
    indices: np.ndarray,
    fps: float,
    temporal_patch_size: int = QWEN_TEMPORAL_PATCH_SIZE,
) -> tuple[float, ...]:
    values = [float(index) / fps for index in np.asarray(indices).reshape(-1)]
    if not values:
        raise ValueError("At least one sampled video frame is required")
    remainder = len(values) % temporal_patch_size
    if remainder:
        values.extend([values[-1]] * (temporal_patch_size - remainder))
    return tuple(
        (values[index] + values[index + temporal_patch_size - 1]) / 2.0
        for index in range(0, len(values), temporal_patch_size)
    )


def _as_rgb_image(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 5:
        if array.shape[0] != 1 or array.shape[1] != 3 or array.shape[2] != 1:
            raise ValueError("Image tensor must have shape [1, 3, 1, H, W]")
        array = array[0, :, 0].transpose(1, 2, 0)
    elif array.ndim == 4:
        if array.shape[0] == 1 and array.shape[1] == 3:
            array = array[0].transpose(1, 2, 0)
        elif array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        else:
            raise ValueError("Image tensor must be RGB CHW or HWC")
    elif array.ndim != 3:
        raise ValueError("Image input must be a 3D HWC/CHW or 5D VAE tensor")
    if array.shape[-1] != 3:
        if array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        else:
            raise ValueError("Image input must contain three RGB channels")
    array = np.asarray(array, dtype=np.float32)
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _as_video_thwc(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 5:
        if array.shape[0] != 1 or array.shape[1] != 3:
            raise ValueError("Video tensor must have shape [1, 3, T, H, W]")
        array = array[0].transpose(1, 2, 3, 0)
    elif array.ndim == 4:
        if array.shape[-1] == 3:
            pass
        elif array.shape[0] == 3:
            array = array.transpose(1, 2, 3, 0)
        else:
            raise ValueError("Video input must be THWC or CT HW")
    else:
        raise ValueError("Video input must be a 4D THWC/CTHW or 5D VAE tensor")
    if array.shape[-1] != 3:
        raise ValueError("Video input must contain three RGB channels")
    array = np.asarray(array, dtype=np.float32)
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()


def _normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    values = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    return (values - 0.5) / 0.5


def patchify_image(
    image: np.ndarray,
    *,
    patch_size: int = QWEN_PATCH_SIZE,
    temporal_patch_size: int = QWEN_TEMPORAL_PATCH_SIZE,
    merge_size: int = QWEN_MERGE_SIZE,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Apply the official Qwen2-VL image patch order."""
    rgb = _as_rgb_image(image)
    height, width = smart_resize_image(
        rgb.shape[0],
        rgb.shape[1],
        factor=patch_size * merge_size,
    )
    values = _normalize_rgb(_resize_rgb(rgb, height, width)).unsqueeze(0)
    grid_h, grid_w = height // patch_size, width // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Resized image grid is not divisible by merge_size")
    patches = values.reshape(
        1,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flattened = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(1, grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    )
    return np.ascontiguousarray(flattened[0].numpy(), dtype=np.float32), (1, grid_h, grid_w)


def patchify_video(
    video: np.ndarray,
    *,
    patch_size: int = QWEN_PATCH_SIZE,
    temporal_patch_size: int = QWEN_TEMPORAL_PATCH_SIZE,
    merge_size: int = QWEN_MERGE_SIZE,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Apply the official Qwen3-VL video resize, normalization and patch order."""
    frames = _as_video_thwc(video)
    source_frames, source_height, source_width, _ = frames.shape
    height, width = smart_resize_video(
        source_frames,
        source_height,
        source_width,
        temporal_factor=temporal_patch_size,
        factor=patch_size * merge_size,
    )
    resized = np.stack([_resize_rgb(frame, height, width) for frame in frames], axis=0)
    values = torch.from_numpy(np.ascontiguousarray(resized)).permute(0, 3, 1, 2)
    if (padding := -values.shape[0] % temporal_patch_size):
        values = torch.cat((values, values[-1:].expand(padding, -1, -1, -1)), dim=0)
    grid_t = values.shape[0] // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Resized video grid is not divisible by merge_size")
    patches = values.reshape(
        1,
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flattened = patches.reshape(
        1,
        grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )
    flattened = (flattened - 0.5) / 0.5
    return np.ascontiguousarray(flattened[0].numpy(), dtype=np.float32), (grid_t, grid_h, grid_w)


def vision_checkpoint_keys(checkpoint: Path) -> tuple[str, ...]:
    """Return visual keys after removing the checkpoint's ``visual.`` prefix."""
    from h3_workbench.main_transformer import _StreamingSafeTensorFile

    reader = _StreamingSafeTensorFile(checkpoint)
    return tuple(sorted(key.removeprefix("visual.") for key in reader._entries if key.startswith("visual.")))


def qwen_vision_config() -> Any:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    return Qwen3VLVisionConfig(
        depth=QWEN_VISION_DEPTH,
        hidden_size=QWEN_VISION_HIDDEN_SIZE,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=QWEN_VISION_INTERMEDIATE_SIZE,
        num_heads=QWEN_VISION_HEADS,
        in_channels=3,
        patch_size=QWEN_PATCH_SIZE,
        spatial_merge_size=QWEN_MERGE_SIZE,
        temporal_patch_size=QWEN_TEMPORAL_PATCH_SIZE,
        out_hidden_size=QWEN_VISION_OUT_SIZE,
        num_position_embeddings=QWEN_VISION_POSITION_EMBEDDINGS,
        deepstack_visual_indexes=list(QWEN_DEEPSTACK_INDEXES),
    )


def _device_for(prefer_cuda: bool, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Qwen3VLVisionEncoder:
    """Load and run the H3 Qwen3-VL visual tower independently of the text tower."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        prefer_cuda: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        config: Any | None = None,
        loader: Callable[[Any, Path, torch.device, torch.dtype], None] | None = None,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise ValueError(f"Qwen visual checkpoint does not exist: {self.checkpoint}")
        self.device = _device_for(prefer_cuda, device)
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.config = config or qwen_vision_config()
        self._model: Any | None = None
        self._loader = loader

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        config = self.config
        # Reference images are encoded at a 2048-pixel short edge. Eager
        # attention materializes the full score matrix at that resolution
        # (more than 14 GiB for a typical portrait), while PyTorch SDPA keeps
        # the same full-attention semantics without that allocation.
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3VLVisionModel(config)
        model.to_empty(device=self.device)
        for parameter in model.parameters():
            parameter.data = torch.empty(parameter.shape, device=self.device, dtype=self.dtype)
        for buffer in model.buffers():
            if buffer.device.type == "meta":
                buffer.data = torch.empty(buffer.shape, device=self.device, dtype=self.dtype)
        if self._loader is not None:
            self._loader(model, self.checkpoint, self.device, self.dtype)
        else:
            self._load_checkpoint_tensors(model)
        # ``inv_freq`` is a non-persistent buffer and therefore is not present in
        # the H3 checkpoint. Recreate it after to_empty() allocated the module.
        rotary = getattr(model, "rotary_pos_emb", None)
        if rotary is not None and hasattr(rotary, "inv_freq"):
            dim = int(rotary.dim)
            theta = float(rotary.theta)
            values = 1.0 / (theta ** (torch.arange(0, dim, 2, device=self.device, dtype=torch.float32) / dim))
            rotary.inv_freq = values
        model.eval().requires_grad_(False)
        self._model = model
        return model

    def _load_checkpoint_tensors(self, model: Any) -> None:
        from h3_workbench.main_transformer import _StreamingSafeTensorFile

        reader = _StreamingSafeTensorFile(self.checkpoint)
        targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        expected = set(targets)
        found: set[str] = set()
        for key in reader._entries:
            if not key.startswith("visual."):
                continue
            name = key.removeprefix("visual.")
            target = targets.get(name)
            if target is None:
                raise ValueError(f"Qwen visual checkpoint contains unknown tensor: {key}")
            source = reader.get_tensor(key)
            converted = np.asarray(source, dtype=np.float32).copy()
            tensor = torch.from_numpy(converted).to(device=self.device, dtype=self.dtype)
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(f"Qwen visual tensor shape mismatch for {name}: {tensor.shape} != {target.shape}")
            target.data.copy_(tensor)
            found.add(name)
            del source, converted, tensor
        missing = sorted(expected - found - {"rotary_pos_emb.inv_freq"})
        if missing:
            raise ValueError(f"Qwen visual checkpoint is missing tensors: {missing[:8]}")

    def _ensure_model(self) -> Any:
        return self._model if self._model is not None else self._load_model()

    def _run(self, patches: np.ndarray, grid: tuple[int, int, int], geometry: VisionGeometry) -> VisionEncoding:
        model = self._ensure_model()
        pixel_values = torch.from_numpy(np.ascontiguousarray(patches)).to(device=self.device, dtype=torch.float32)
        grid_thw = torch.tensor([grid], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = model(pixel_values, grid_thw)
            merged = output.pooler_output if hasattr(output, "pooler_output") else output[1]
            deepstack = output.deepstack_features if hasattr(output, "deepstack_features") else output[2]
            features = np.asarray(merged.float().cpu().numpy(), dtype=np.float32)
            deep = tuple(np.asarray(value.float().cpu().numpy(), dtype=np.float32) for value in (deepstack or ()))
        return VisionEncoding(features=features, deepstack_features=deep, geometry=geometry)

    def encode_image(self, image: np.ndarray) -> VisionEncoding:
        patches, grid = patchify_image(image)
        source = _as_rgb_image(image)
        geometry = VisionGeometry(
            kind="image",
            grid_thw=grid,
            resized_height=smart_resize_image(source.shape[0], source.shape[1])[0],
            resized_width=smart_resize_image(source.shape[0], source.shape[1])[1],
            source_frames=1,
        )
        return self._run(patches, grid, geometry)

    def encode_video(
        self,
        video: np.ndarray,
        *,
        fps: float = 24.0,
        target_fps: float = 2.0,
    ) -> VisionEncoding:
        frames = _as_video_thwc(video)
        indices = sample_video_indices(frames.shape[0], fps, target_fps)
        sampled = frames[indices]
        return self.encode_sampled_video(
            sampled,
            source_frames=int(frames.shape[0]),
            sampled_indices=indices,
        )

    def encode_sampled_video(
        self,
        video: np.ndarray,
        *,
        source_frames: int | None = None,
        sampled_indices: np.ndarray | Sequence[int] = (),
    ) -> VisionEncoding:
        """Encode frames sampled by H3's reference-video policy.

        The official pipeline samples at 2 FPS before invoking the Qwen video
        processor.  Keeping that sampling outside this method lets callers
        preserve the exact frame indices and timestamp labels from the H3
        setup block.
        """
        frames = _as_video_thwc(video)
        patches, grid = patchify_video(frames)
        height, width = smart_resize_video(frames.shape[0], frames.shape[1], frames.shape[2])
        geometry = VisionGeometry(
            kind="video",
            grid_thw=grid,
            resized_height=height,
            resized_width=width,
            source_frames=int(source_frames if source_frames is not None else frames.shape[0]),
            sampled_indices=tuple(int(value) for value in np.asarray(sampled_indices).reshape(-1)),
        )
        return self._run(patches, grid, geometry)

    def close(self) -> None:
        model = self._model
        self._model = None
        if model is not None:
            del model
            gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "Qwen3VLVisionEncoder":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def encode_reference_vision(
    checkpoint: Path,
    kind: VisionKind,
    media: np.ndarray,
    *,
    fps: float = 24.0,
    prefer_cuda: bool = True,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> VisionEncoding:
    """Encode one reference and release the visual tower before returning."""
    encoder = Qwen3VLVisionEncoder(checkpoint, prefer_cuda=prefer_cuda)
    try:
        if callback is not None:
            callback({"module": "Qwen Vision", "operation": "Encoding reference", "kind": kind})
        return encoder.encode_image(media) if kind == "image" else encoder.encode_video(media, fps=fps)
    finally:
        encoder.close()
