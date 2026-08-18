from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from h3_workbench.vendor.audio_vae import MiniMaxH3AudioVAE
from h3_workbench.vendor.video_vae import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LATENTS_MEAN,
    LATENTS_STD,
    MiniMaxH3VideoVAE,
)
from h3_workbench.profiles import video_vae_output_frames


def _video_vae_posterior_latents(
    moments: np.ndarray,
    *,
    sample_posterior: bool,
    posterior_seed: int,
) -> np.ndarray:
    """Convert Video VAE moments to normalized conditioning latents."""
    values = np.asarray(moments, dtype=np.float32)
    if values.ndim != 5 or values.shape[1] != len(LATENTS_MEAN) * 2:
        raise ValueError(f"Unexpected Video VAE moments shape: {values.shape}")
    mean, logvar = np.split(values, 2, axis=1)
    if sample_posterior:
        # Match diffusers' DiagonalGaussianDistribution and H3's fresh CPU
        # generator per reference.  The released pipeline rounds the sampled
        # latent to FP16 before applying latent mean/std normalization.
        generator = torch.Generator(device="cpu").manual_seed(int(posterior_seed))
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32).numpy()
        std = np.exp(0.5 * np.clip(logvar, -30.0, 20.0)).astype(np.float32, copy=False)
        latent = mean + std * noise
        latent = latent.astype(np.float16).astype(np.float32)
    else:
        latent = mean
    latent_mean = np.asarray(LATENTS_MEAN, dtype=np.float32).reshape(1, -1, 1, 1, 1)
    latent_std = np.asarray(LATENTS_STD, dtype=np.float32).reshape(1, -1, 1, 1, 1)
    return np.ascontiguousarray((latent - latent_mean) / latent_std, dtype=np.float32)


def _video_vae_temporal_windows(latent_frames: int) -> tuple[list[tuple[int, int]], int]:
    """Return the reference VAE's seven-token windows at a five-token stride."""
    if latent_frames < 2:
        raise ValueError("The sharded ONNX Video VAE requires at least two temporal latent tokens.")
    pseudo_total = latent_frames + 3
    pad_tokens = (-pseudo_total) % 5
    pseudo_total += pad_tokens
    num_chunks = pseudo_total // 5 - 1
    if num_chunks < 1:
        pad_tokens += 5
        num_chunks += 1
    padded_frames = latent_frames + pad_tokens
    windows = [(index * 5, index * 5 + 7) for index in range(num_chunks)]
    assert not windows or windows[-1][1] == padded_frames
    return windows, video_vae_output_frames(latent_frames)


def _split_tiles(input_len: int, tile_size: int = 256, overlap_min: int = 64, ratio: int = 16) -> tuple[list[int], list[int], list[int]]:
    if tile_size >= input_len:
        return [0], [input_len], []
    count = max(2, int(np.ceil(input_len / tile_size)))
    while tile_size * count - overlap_min * (count - 1) < input_len:
        count += 1
    remaining = tile_size * count - overlap_min * (count - 1) - input_len
    overlaps = [overlap_min] * (count - 1)
    for index in range(remaining // ratio):
        overlaps[index % (count - 1)] += ratio
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return starts, [tile_size] * count, overlaps


def _blend_numpy(left: np.ndarray, right: np.ndarray, extent: int, axis: int) -> np.ndarray:
    extent = min(left.shape[axis], right.shape[axis], extent)
    positions = np.arange(extent, dtype=np.float32)
    weight_left = 1.0 - positions / extent
    weight_right = positions / extent
    shape = [1] * left.ndim
    shape[axis] = extent
    weight_left = weight_left.reshape(shape)
    weight_right = weight_right.reshape(shape)
    left_slice = [slice(None)] * left.ndim
    left_slice[axis] = slice(-extent, None)
    right_slice = [slice(None)] * right.ndim
    right_slice[axis] = slice(0, extent)
    blended = left[tuple(left_slice)] * weight_left + right[tuple(right_slice)] * weight_right
    if extent < right.shape[axis]:
        right_rest = [slice(None)] * right.ndim
        right_rest[axis] = slice(extent, None)
        return np.concatenate((blended, right[tuple(right_rest)]), axis=axis)
    return blended


def _release_io_binding(binding: object | None) -> None:
    if binding is None:
        return
    for method_name in ("clear_binding_inputs", "clear_binding_outputs"):
        method = getattr(binding, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:  # noqa: BLE001 - cleanup must not replace the inference error
                pass


def _decode_video_vae_blocks_persistent(
    directory: Path,
    runner: object,
    hidden_tiles: list[np.ndarray],
    rotary_tiles: list[np.ndarray],
    block_count: int,
    callback: Callable[[dict[str, object]], None] | None,
) -> list[np.ndarray]:
    import onnxruntime as ort

    from h3_workbench.video_vae_persistent import (
        PERSISTENT_VIDEO_VAE_TOPOLOGY,
        load_video_vae_block_weights,
        preload_video_vae_block_weights,
    )
    from h3_workbench.device_profile import selected_device_index

    session = None
    ram_cache = preload_video_vae_block_weights(directory, range(block_count))
    hidden_devices: list[ort.OrtValue] = []
    rotary_devices: list[ort.OrtValue] = []
    try:
        session = runner.session(directory / PERSISTENT_VIDEO_VAE_TOPOLOGY)  # type: ignore[attr-defined]
        device_index = selected_device_index()
        hidden_devices = [ort.OrtValue.ortvalue_from_numpy(value, "cuda", device_index) for value in hidden_tiles]
        rotary_devices = [ort.OrtValue.ortvalue_from_numpy(value, "cuda", device_index) for value in rotary_tiles]
        shapes = [value.shape for value in hidden_tiles]
        dtypes = [value.dtype for value in hidden_tiles]
        hidden_tiles.clear()
        rotary_tiles.clear()

        if callback is not None and ram_cache is not None:
            callback(
                {
                    "module": "Video VAE",
                    "operation": "Weights preloaded to host RAM",
                    "ram_cache_bytes": ram_cache.bytes,
                    "ram_cache_load_seconds": round(ram_cache.load_seconds, 3),
                    "blocks": block_count,
                }
            )
        if ram_cache is not None:
            block_iterator = ((ram_cache.get(index), None) for index in range(block_count))
        else:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-vae-weight-prefetch")
            pending = executor.submit(load_video_vae_block_weights, directory, 0)
            block_iterator = (
                (
                    pending.result(),
                    executor.submit(load_video_vae_block_weights, directory, index + 1)
                    if index + 1 < block_count
                    else None,
                )
                for index in range(block_count)
            )
        try:
            for index, (loaded, next_pending) in enumerate(block_iterator):
                weight_devices: dict[str, ort.OrtValue] = {}
                if next_pending is not None:
                    pending = next_pending
                try:
                    weight_devices = {
                        name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", device_index)
                        for name, value in loaded.feeds().items()
                    }
                    for tile_index in range(len(hidden_devices)):
                        if callback is not None:
                            callback(
                                {
                                    "module": "Video VAE",
                                    "operation": "Persistent transformer block",
                                    "current": index + 1,
                                    "total": block_count,
                                    "tile": tile_index + 1,
                                    "tiles": len(hidden_devices),
                                    "weight_load_seconds": round(
                                        0.0 if ram_cache is not None else loaded.load_seconds,
                                        3,
                                    ),
                                    "ram_cache": ram_cache is not None,
                                }
                            )
                        output = ort.OrtValue.ortvalue_from_shape_and_type(
                            shapes[tile_index], dtypes[tile_index], "cuda", device_index
                        )
                        binding = None
                        try:
                            binding = session.io_binding()
                            for weight_name, weight_value in weight_devices.items():
                                binding.bind_ortvalue_input(weight_name, weight_value)
                            del weight_name, weight_value
                            binding.bind_ortvalue_input("hidden_states", hidden_devices[tile_index])
                            binding.bind_ortvalue_input("rotary_table", rotary_devices[tile_index])
                            binding.bind_ortvalue_output("hidden_states_out", output)
                            binding.synchronize_inputs()
                            session.run_with_iobinding(binding)
                            binding.synchronize_outputs()
                            hidden_devices[tile_index] = output
                        finally:
                            _release_io_binding(binding)
                            binding = None
                finally:
                    if ram_cache is None:
                        loaded.close()
                    weight_devices.clear()
        finally:
            if ram_cache is None:
                executor.shutdown(wait=True)
        return [value.numpy() for value in hidden_devices]
    finally:
        hidden_devices.clear()
        rotary_devices.clear()
        session = None
        gc.collect()
        if ram_cache is not None:
            ram_cache.close()


def decode_video_latents_onnx(
    directory: Path,
    latents: np.ndarray,
    output_height: int,
    output_width: int | None = None,
    prefer_cuda: bool = True,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    """Decode short or long clips with spatial tiles and reference temporal overlap."""
    from h3_workbench.inference_runtime import ORTGraphRunner

    runner = ORTGraphRunner(prefer_cuda=prefer_cuda)
    try:
        return _decode_video_latents_onnx_with_runner(
            directory,
            latents,
            output_height,
            output_width,
            runner,
            callback,
        )
    finally:
        runner.close()


def _pad_video_latents_to_tile(latents: np.ndarray, minimum: int = 16) -> np.ndarray:
    """Center-pad sub-tile latent canvases for the fixed 256 px VAE boundary graphs."""
    pad_height = max(0, minimum - latents.shape[-2])
    pad_width = max(0, minimum - latents.shape[-1])
    if pad_height == 0 and pad_width == 0:
        return latents
    top = pad_height // 2
    left = pad_width // 2
    return np.pad(
        latents,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (top, pad_height - top),
            (left, pad_width - left),
        ),
        mode="constant",
    )


def _decode_video_latents_onnx_with_runner(
    directory: Path,
    latents: np.ndarray,
    output_height: int,
    output_width: int | None,
    runner: object,
    callback: Callable[[dict[str, object]], None] | None,
) -> np.ndarray:
    if latents.ndim != 5 or latents.shape[2] < 2:
        raise ValueError("The ONNX Video VAE expects at least two temporal latent tokens.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = tuple(manifest["profiles"]["latents"][2:])
    if expected != (5, 16, 16):
        raise ValueError(f"Unsupported Video VAE tile profile: {expected}")

    latents = _pad_video_latents_to_tile(latents, expected[-1])
    padded_height, padded_width = latents.shape[-2] * 16, latents.shape[-1] * 16
    output_width = padded_width if output_width is None else output_width
    if output_height > padded_height or output_width > padded_width:
        raise ValueError("Requested crop is larger than the latent canvas")
    y_starts, y_lengths, y_overlaps = _split_tiles(padded_height)
    x_starts, x_lengths, x_overlaps = _split_tiles(padded_width)

    temporal_windows, output_frames = _video_vae_temporal_windows(latents.shape[2])
    if any(stop - start == 7 for start, stop in temporal_windows):
        required = (directory / "video_decoder_prelude_t7.onnx", directory / "video_decoder_head_t7.onnx")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                "Long-video ONNX VAE support is incomplete: video_decoder_prelude_t7.onnx and "
                "video_decoder_head_t7.onnx are required."
            )

    # Restore the causal temporal padding and the final alignment tokens used
    # by the reference decoder's 5-token stride.
    pad_tokens = temporal_windows[-1][1] - latents.shape[2]
    padded_latents = np.concatenate(
        (latents, np.repeat(latents[:, :, -1:], pad_tokens, axis=2)),
        axis=2,
    ).astype(np.float16, copy=False)
    tile_inputs: list[tuple[int, int, int, np.ndarray]] = []
    for temporal_index, (temporal_start, temporal_stop) in enumerate(temporal_windows):
        for row, y_start in enumerate(y_starts):
            for column, x_start in enumerate(x_starts):
                tile_inputs.append(
                    (
                        temporal_index,
                        row,
                        column,
                        padded_latents[
                            :, :, temporal_start:temporal_stop,
                            y_start // 16 : y_start // 16 + y_lengths[row] // 16,
                            x_start // 16 : x_start // 16 + x_lengths[column] // 16,
                        ],
                    )
                )

    preludes = {
        length: runner.session(
            directory / ("video_decoder_prelude_t7.onnx" if length == 7 else "video_decoder_prelude.onnx")
        )
        for length in {tile.shape[2] for _, _, _, tile in tile_inputs}
    }
    hidden_tiles: list[np.ndarray] = []
    rotary_tiles: list[np.ndarray] = []
    try:
        for tile_index, (temporal_index, _, _, tile) in enumerate(tile_inputs):
            if callback is not None:
                callback(
                    {
                        "module": "Video VAE",
                        "operation": "GPU decoder prelude",
                        "tile": tile_index + 1,
                        "tiles": len(tile_inputs),
                        "temporal_chunk": temporal_index + 1,
                        "temporal_chunks": len(temporal_windows),
                        "provider": runner.provider,
                    }
                )
            hidden, rotary = preludes[tile.shape[2]].run(None, {"latents": tile})
            hidden_tiles.append(hidden)
            rotary_tiles.append(rotary)
    finally:
        del preludes
        gc.collect()

    block_count = len(manifest["blocks"])
    from h3_workbench.video_vae_persistent import persistent_video_vae_ready

    persistent_enabled = os.environ.get("H3_VIDEO_VAE_PERSISTENT", "1") != "0"
    if (
        persistent_enabled
        and runner.provider == "CUDAExecutionProvider"
        and persistent_video_vae_ready(directory, range(block_count), dynamic_batch=False)
    ):
        hidden_tiles = _decode_video_vae_blocks_persistent(
            directory,
            runner,
            hidden_tiles,
            rotary_tiles,
            block_count,
            callback,
        )
        rotary_tiles.clear()
        gc.collect()
    else:
        for index in range(block_count):
            session = runner.session(directory / f"video_decoder_block_{index:02d}.onnx")
            try:
                for tile_index in range(len(hidden_tiles)):
                    if callback is not None:
                        callback(
                            {
                                "module": "Video VAE",
                                "operation": "Transformer block",
                                "current": index + 1,
                                "total": block_count,
                                "tile": tile_index + 1,
                                "tiles": len(hidden_tiles),
                            }
                        )
                    hidden_tiles[tile_index] = session.run(
                        None,
                        {"hidden_states": hidden_tiles[tile_index], "rotary_table": rotary_tiles[tile_index]},
                    )[0]
            finally:
                del session
                gc.collect()

    heads = {
        length: runner.session(directory / ("video_decoder_head_t7.onnx" if length == 7 else "video_decoder_head.onnx"))
        for length in {tile.shape[2] for _, _, _, tile in tile_inputs}
    }
    decoded_windows: list[np.ndarray] = []
    previous_overlap: np.ndarray | None = None
    try:
        tile_cursor = 0
        for temporal_index, (temporal_start, temporal_stop) in enumerate(temporal_windows):
            temporal_length = temporal_stop - temporal_start
            decoded_tiles: dict[tuple[int, int], np.ndarray] = {}
            for row in range(len(y_starts)):
                for column in range(len(x_starts)):
                    if callback is not None:
                        callback(
                            {
                                "module": "Video VAE",
                                "operation": "GPU decoder head",
                                "tile": tile_cursor + 1,
                                "tiles": len(tile_inputs),
                                "temporal_chunk": temporal_index + 1,
                                "temporal_chunks": len(temporal_windows),
                                "provider": runner.provider,
                            }
                        )
                    decoded_tiles[(row, column)] = heads[temporal_length].run(
                        None, {"hidden_states": hidden_tiles[tile_cursor]}
                    )[0]
                    tile_cursor += 1

            decoded = _assemble_video_vae_tiles(
                decoded_tiles,
                y_starts,
                y_overlaps,
                x_starts,
                x_overlaps,
                padded_height,
                padded_width,
            )
            main = decoded[:, :, 3:20]
            if previous_overlap is not None:
                main = _blend_numpy(previous_overlap, main, 5, axis=2)
            decoded_windows.append(main)
            previous_overlap = decoded[:, :, 23:28].copy() if temporal_length == 7 else None
        if previous_overlap is not None:
            decoded_windows.append(previous_overlap)
    finally:
        del heads
        gc.collect()

    canvas = np.concatenate(decoded_windows, axis=2)[:, :, :output_frames]
    top = max(0, (padded_height - output_height) // 2)
    left = max(0, (padded_width - output_width) // 2)
    result = canvas[:, :, :, top : top + output_height, left : left + output_width]
    return result


def _assemble_video_vae_tiles(
    decoded_tiles: dict[tuple[int, int], np.ndarray],
    y_starts: list[int],
    y_overlaps: list[int],
    x_starts: list[int],
    x_overlaps: list[int],
    padded_height: int,
    padded_width: int,
    canvas_dtype: np.dtype | type[np.floating] | None = np.float32,
) -> np.ndarray:
    canvas: np.ndarray | None = None
    row_tails: list[np.ndarray | None] = []
    out_y = 0
    for row in range(len(y_starts)):
        new_tails: list[np.ndarray | None] = []
        left_tail: np.ndarray | None = None
        out_x = 0
        for column in range(len(x_starts)):
            tile = decoded_tiles[(row, column)]
            if row < len(y_starts) - 1:
                new_tails.append(tile[..., -y_overlaps[row] :, :].copy())
            next_left_tail = tile[..., :, -x_overlaps[column] :].copy() if column < len(x_starts) - 1 else None
            if row > 0:
                tile = _blend_numpy(row_tails[column], tile, y_overlaps[row - 1], axis=-2)
            if column > 0:
                tile = _blend_numpy(left_tail, tile, x_overlaps[column - 1], axis=-1)
            left_tail = next_left_tail
            if row < len(y_starts) - 1:
                tile = tile[..., :-y_overlaps[row], :]
            if column < len(x_starts) - 1:
                tile = tile[..., :, :-x_overlaps[column]]
            if canvas is None:
                dtype = tile.dtype if canvas_dtype is None else canvas_dtype
                canvas = np.empty((*tile.shape[:-2], padded_height, padded_width), dtype=dtype)
            canvas[..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]] = tile
            out_x += tile.shape[-1]
        row_tails = new_tails
        out_y += tile.shape[-2]

    assert canvas is not None
    return canvas


def _load_video_vae(checkpoint: Path, prefer_cuda: bool = True) -> MiniMaxH3VideoVAE:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("meta"):
            model = MiniMaxH3VideoVAE(tiling=True)
    finally:
        torch.set_default_dtype(previous_dtype)

    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    del state

    # These buffers are deliberately absent from the checkpoint. A meta-device
    # construction must materialize them before the first decode.
    model.pixel_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float16).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(IMAGENET_STD, dtype=torch.float16).view(1, 3, 1, 1, 1)
    inv_freq_count = model.decoder.pos_embed.inv_freq.numel()
    model.decoder.pos_embed.inv_freq = 1.0 / 100.0 ** torch.arange(
        0.0,
        1.0,
        1.0 / inv_freq_count,
        dtype=torch.float32,
    )
    if prefer_cuda and torch.cuda.is_available():
        model.decoder.offload_device = "cuda"
    return model


def decode_video_latents(
    checkpoint: Path,
    latents: np.ndarray,
    output_height: int = 360,
    output_width: int | None = None,
    prefer_cuda: bool = True,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    if callback is not None:
        callback({"module": "Video VAE", "operation": "Loading PyTorch checkpoint"})
    model = _load_video_vae(checkpoint, prefer_cuda)
    model.eval().requires_grad_(False)
    if callback is not None:
        callback({"module": "Video VAE", "operation": "PyTorch temporal/spatial decode"})
    with torch.inference_mode():
        pixels = model.decode(torch.from_numpy(latents).to(torch.float16)).cpu().numpy()
    del model
    gc.collect()
    height = pixels.shape[-2]
    width = pixels.shape[-1]
    output_width = width if output_width is None else output_width
    top = max(0, (height - output_height) // 2)
    left = max(0, (width - output_width) // 2)
    return pixels[:, :, :, top : top + output_height, left : left + output_width]


def _video_vae_encoder_overlap(use_cuda: bool, total_vram_bytes: int) -> int:
    setting = os.environ.get("H3_VAE_ENCODER_OVERLAP", "auto").strip().lower()
    if setting == "auto":
        return 32 if use_cuda and total_vram_bytes <= 6 * 1024**3 else 64
    try:
        overlap = int(setting)
    except ValueError as exc:
        raise ValueError(
            "H3_VAE_ENCODER_OVERLAP must be auto or a positive integer"
        ) from exc
    if overlap < 16 or overlap % 16 != 0:
        raise ValueError(
            "Video VAE encoder tile_overlap_min must be a multiple of 16 and at least 16"
        )
    return overlap


def video_vae_temporal_encoder_ready(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        encoder = manifest["temporal_encoder"]
        graphs = encoder["graphs"]
        return (
            encoder.get("layout") == "progressive_staged_cuda_v1"
            and encoder.get("validation_passed") is True
            and int(encoder.get("clip_frames", 0)) == 17
            and int(encoder.get("token_drop", -1)) == 3
            and int(encoder.get("spatial_ratio", 0)) == 16
            and int(encoder.get("temporal_tokens", 0)) == 5
            and encoder.get("channels")
            == {"prelude": 128, "late": 256, "tail": 1024, "moments": 48}
            and set(graphs) == {"prelude", "late", "tail", "head"}
            and all(
                (directory / str(graphs[name])).is_file()
                for name in ("prelude", "late", "tail", "head")
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def audio_vae_encoder_ready(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        validation = manifest["validation"]
        graphs = set(manifest["graphs"])
        return (
            manifest.get("component") == "audio_vae"
            and manifest.get("layout") == "dual_graph_autoencoder"
            and manifest.get("validation_passed") is True
            and isinstance(validation.get("encoder"), dict)
            and int(manifest.get("sample_rate", 0)) == 32_000
            and int(manifest.get("samples_per_latent", 0)) == 800
            and int(manifest.get("audio_channels", 0)) == 2
            and int(manifest.get("latent_channels", 0)) == 32
            and {"audio_encoder.onnx", "audio_decoder.onnx"}.issubset(graphs)
            and (directory / "audio_encoder.onnx").is_file()
            and (directory / "audio_decoder.onnx").is_file()
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def select_video_vae_encoder_backend(
    *,
    checkpoint_available: bool,
    onnx_available: bool,
    low_vram_cuda: bool,
    setting: str | None = None,
) -> tuple[str, str]:
    """Require the staged ONNX encoder so auto mode cannot select a known slow path."""
    del checkpoint_available, low_vram_cuda
    requested = (
        setting if setting is not None else os.environ.get("H3_VAE_ENCODER_BACKEND", "auto")
    ).strip().lower()
    requested = {"onnx": "onnxruntime", "torch": "pytorch"}.get(requested, requested)
    if requested == "pytorch":
        raise ValueError(
            "The PyTorch Video VAE encoder backend is disabled; export the staged ONNX encoder"
        )
    if requested not in {"auto", "onnxruntime"}:
        raise ValueError("H3_VAE_ENCODER_BACKEND must be auto or onnxruntime")
    if onnx_available:
        reason = (
            "forced staged ONNX encoder by H3_VAE_ENCODER_BACKEND"
            if requested == "onnxruntime"
            else "validated staged ONNX encoder is available"
        )
        return "onnxruntime", reason
    raise RuntimeError(
        "Super-resolution requires a validated staged ONNX Video VAE encoder; "
        "run export-video-encoder for the Video VAE product"
    )


class _MonolithicONNXVideoVAEEncoder:
    """Persistent ORT session for the production temporal Video VAE encoder."""

    _h3_onnx_runtime = True

    def __init__(self, directory: Path, prefer_cuda: bool = True):
        if not video_vae_temporal_encoder_ready(directory):
            raise ValueError(f"Production temporal Video VAE encoder is not ready: {directory}")
        from h3_workbench.inference_runtime import ORTGraphRunner

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        encoder = manifest["temporal_encoder"]
        self.directory = directory.resolve()
        self.clip_length = int(encoder["clip_frames"])
        self.token_drop = int(encoder["token_drop"])
        self.vae_ratio = int(encoder["spatial_ratio"])
        self._runner = ORTGraphRunner(prefer_cuda=prefer_cuda, prefetch_depth=0)
        self._session = self._runner.session(self.directory / str(encoder["graph"]))
        use_cuda = self._runner.provider == "CUDAExecutionProvider"
        free_bytes = 0
        total_bytes = 0
        if use_cuda and torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except RuntimeError:
                pass
        if total_bytes >= 10 * 1024**3:
            self.tile_size = 512
        elif total_bytes >= 3 * 1024**3:
            self.tile_size = 384
        elif free_bytes >= 1536 * 1024**2:
            self.tile_size = 320
        else:
            self.tile_size = 256
        self.tile_overlap_min = _video_vae_encoder_overlap(use_cuda, total_bytes)
        self.provider = self._runner.provider

    def close(self) -> None:
        self._session = None
        self._runner.close()

    def _encode_with_tile_size(
        self,
        pixels: np.ndarray,
        tile_size: int,
        callback: Callable[[dict[str, object]], None] | None,
    ) -> np.ndarray:
        y_starts, y_lengths, y_overlaps = _split_tiles(
            int(pixels.shape[3]),
            tile_size,
            self.tile_overlap_min,
            self.vae_ratio,
        )
        x_starts, x_lengths, x_overlaps = _split_tiles(
            int(pixels.shape[4]),
            tile_size,
            self.tile_overlap_min,
            self.vae_ratio,
        )
        tiles_per_clip = len(y_starts) * len(x_starts)
        temporal_clips = int(np.ceil(pixels.shape[2] / self.clip_length))
        clip_moments: list[np.ndarray] = []
        for temporal_index in range(temporal_clips):
            if callback is not None:
                callback(
                    {
                        "module": "Video VAE",
                        "operation": "ORT encoder temporal clip",
                        "event": "temporal_clip_start",
                        "temporal_clip": temporal_index + 1,
                        "temporal_clips": temporal_clips,
                        "provider": self.provider,
                    }
                )
            clip = pixels[
                :,
                :,
                temporal_index * self.clip_length : (temporal_index + 1) * self.clip_length,
                :,
                :,
            ]
            if clip.shape[2] < self.clip_length:
                clip = np.concatenate(
                    (clip, np.repeat(clip[:, :, -1:], self.clip_length - clip.shape[2], axis=2)),
                    axis=2,
                )
            encoded_tiles: dict[tuple[int, int], np.ndarray] = {}
            tile_number = 0
            for row, (y_start, y_length) in enumerate(zip(y_starts, y_lengths, strict=True)):
                for column, (x_start, x_length) in enumerate(zip(x_starts, x_lengths, strict=True)):
                    tile_number += 1
                    details = {
                        "temporal_clip": temporal_index + 1,
                        "temporal_clips": temporal_clips,
                        "tile": tile_number,
                        "tiles": tiles_per_clip,
                        "row": row + 1,
                        "rows": len(y_starts),
                        "column": column + 1,
                        "columns": len(x_starts),
                        "provider": self.provider,
                    }
                    if callback is not None:
                        callback(
                            {
                                "module": "Video VAE",
                                "operation": "ORT encoder tile",
                                "event": "tile_start",
                                **details,
                            }
                        )
                    started = time.perf_counter()
                    tile = np.ascontiguousarray(
                        clip[
                            :,
                            :,
                            :,
                            y_start : y_start + y_length,
                            x_start : x_start + x_length,
                        ],
                        dtype=np.float16,
                    )
                    encoded_tiles[(row, column)] = self._session.run(None, {"pixels": tile})[0]
                    if callback is not None:
                        callback(
                            {
                                "module": "Video VAE",
                                "operation": "ORT encoder tile complete",
                                "event": "tile_complete",
                                "elapsed_seconds": time.perf_counter() - started,
                                **details,
                            }
                        )
            clip_moments.append(
                _assemble_video_vae_tiles(
                    encoded_tiles,
                    y_starts,
                    [value // self.vae_ratio for value in y_overlaps],
                    x_starts,
                    [value // self.vae_ratio for value in x_overlaps],
                    int(pixels.shape[3]) // self.vae_ratio,
                    int(pixels.shape[4]) // self.vae_ratio,
                )
            )
        moments = np.concatenate(clip_moments, axis=2)
        if self.token_drop > 0:
            moments = moments[:, :, :-self.token_drop]
        mean = np.split(moments.astype(np.float32, copy=False), 2, axis=1)[0]
        latent_mean = np.asarray(LATENTS_MEAN, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        latent_std = np.asarray(LATENTS_STD, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        return (mean - latent_mean) / latent_std

    def encode(
        self,
        pixels: np.ndarray,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> np.ndarray:
        candidates = [value for value in (512, 448, 384, 320, 256) if value <= self.tile_size]
        last_error: Exception | None = None
        for tile_size in candidates:
            try:
                result = self._encode_with_tile_size(pixels, tile_size, callback)
                self.tile_size = tile_size
                return result
            except RuntimeError as exc:
                if not any(marker in str(exc).lower() for marker in ("out of memory", "cuda error 2")):
                    raise
                last_error = exc
                if callback is not None and tile_size != candidates[-1]:
                    callback(
                        {
                            "module": "Video VAE",
                            "operation": "Reducing ORT encoder tile after CUDA OOM",
                            "from_tile_size": tile_size,
                            "to_tile_size": candidates[candidates.index(tile_size) + 1],
                        }
                    )
        assert last_error is not None
        raise last_error


class _BatchedStagedONNXVideoVAEEncoder:
    """Three persistent ORT stages with CUDA-resident intermediate activations."""

    _h3_onnx_runtime = True

    def __init__(self, directory: Path, prefer_cuda: bool = True):
        if not video_vae_temporal_encoder_ready(directory):
            raise ValueError(f"Staged temporal Video VAE encoder is not ready: {directory}")
        from h3_workbench.inference_runtime import ORTGraphRunner

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        encoder = manifest["temporal_encoder"]
        self.directory = directory.resolve()
        self.clip_length = int(encoder["clip_frames"])
        self.token_drop = int(encoder["token_drop"])
        self.vae_ratio = int(encoder["spatial_ratio"])
        self.temporal_tokens = int(encoder["temporal_tokens"])
        self.channels = {name: int(value) for name, value in encoder["channels"].items()}
        self._graphs = {name: self.directory / str(value) for name, value in encoder["graphs"].items()}
        self._runner = ORTGraphRunner(prefer_cuda=prefer_cuda, prefetch_depth=0)
        self.provider = self._runner.provider
        self._sessions: dict[str, object] = {}
        self._load_sessions()

        use_cuda = self.provider == "CUDAExecutionProvider"
        free_bytes = 0
        total_bytes = 0
        if use_cuda and torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except RuntimeError:
                pass
        if total_bytes >= 10 * 1024**3:
            self.tile_size = 512
        elif total_bytes >= 3 * 1024**3:
            self.tile_size = 384
        elif free_bytes >= 1536 * 1024**2:
            self.tile_size = 320
        else:
            self.tile_size = 256
        self.tile_overlap_min = _video_vae_encoder_overlap(use_cuda, total_bytes)

    def _load_sessions(self) -> None:
        try:
            self._sessions = {
                name: self._runner.session(self._graphs[name])
                for name in ("prelude", "late", "head")
            }
        except Exception:
            self._sessions.clear()
            gc.collect()
            raise

    def _reload_sessions(self) -> None:
        self._sessions.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._load_sessions()

    def close(self) -> None:
        self._sessions.clear()
        gc.collect()
        self._runner.close()

    @staticmethod
    def _emit(
        callback: Callable[[dict[str, object]], None] | None,
        operation: str,
        **details: object,
    ) -> None:
        if callback is not None:
            callback({"module": "Video VAE", "operation": operation, **details})

    def _run_cuda_stage(
        self,
        session: object,
        input_name: str,
        input_value: object,
        output_name: str,
        output_shape: tuple[int, ...],
    ) -> object:
        import onnxruntime as ort

        output = ort.OrtValue.ortvalue_from_shape_and_type(
            output_shape,
            np.float16,
            "cuda",
            self._runner.device_index,
        )
        binding = None
        try:
            binding = session.io_binding()  # type: ignore[attr-defined]
            binding.bind_ortvalue_input(input_name, input_value)
            binding.bind_ortvalue_output(output_name, output)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)  # type: ignore[attr-defined]
            binding.synchronize_outputs()
            return output
        finally:
            _release_io_binding(binding)

    def _encode_clip_cuda(
        self,
        clip: np.ndarray,
        tile_size: int,
        callback: Callable[[dict[str, object]], None] | None,
        temporal_index: int,
        temporal_clips: int,
    ) -> np.ndarray:
        import onnxruntime as ort

        y_starts, y_lengths, y_overlaps = _split_tiles(
            int(clip.shape[3]), tile_size, self.tile_overlap_min, self.vae_ratio
        )
        x_starts, x_lengths, x_overlaps = _split_tiles(
            int(clip.shape[4]), tile_size, self.tile_overlap_min, self.vae_ratio
        )
        if len(set(y_lengths)) != 1 or len(set(x_lengths)) != 1:
            raise ValueError("Staged Video VAE encoder requires uniform spatial tile shapes")
        tile_entries = [
            (row, column, y_start, y_length, x_start, x_length)
            for row, (y_start, y_length) in enumerate(zip(y_starts, y_lengths, strict=True))
            for column, (x_start, x_length) in enumerate(zip(x_starts, x_lengths, strict=True))
        ]
        latent_height = y_lengths[0] // self.vae_ratio
        latent_width = x_lengths[0] // self.vae_ratio
        single_shape = (
            1,
            self.channels["prelude"],
            self.temporal_tokens,
            latent_height,
            latent_width,
        )
        prelude_shape = (len(tile_entries), *single_shape[1:])
        prelude_batch = ort.OrtValue.ortvalue_from_shape_and_type(
            prelude_shape,
            np.float16,
            "cuda",
            self._runner.device_index,
        )
        tile_bytes = int(np.prod(single_shape, dtype=np.int64)) * np.dtype(np.float16).itemsize
        prelude_session = self._sessions["prelude"]
        for tile_index, (row, column, y_start, y_length, x_start, x_length) in enumerate(tile_entries):
            details = {
                "event": "tile_start",
                "temporal_clip": temporal_index + 1,
                "temporal_clips": temporal_clips,
                "tile": tile_index + 1,
                "tiles": len(tile_entries),
                "row": row + 1,
                "rows": len(y_starts),
                "column": column + 1,
                "columns": len(x_starts),
                "provider": self.provider,
                "stage": "prelude",
            }
            self._emit(callback, "Staged encoder prelude tile", **details)
            started = time.perf_counter()
            tile = np.ascontiguousarray(
                clip[:, :, :, y_start : y_start + y_length, x_start : x_start + x_length],
                dtype=np.float16,
            )
            input_value = ort.OrtValue.ortvalue_from_numpy(
                tile,
                "cuda",
                self._runner.device_index,
            )
            binding = None
            try:
                binding = prelude_session.io_binding()  # type: ignore[attr-defined]
                binding.bind_ortvalue_input("pixels", input_value)
                binding.bind_output(
                    "prelude_activation",
                    "cuda",
                    self._runner.device_index,
                    np.float16,
                    single_shape,
                    prelude_batch.data_ptr() + tile_index * tile_bytes,
                )
                binding.synchronize_inputs()
                prelude_session.run_with_iobinding(binding)  # type: ignore[attr-defined]
                binding.synchronize_outputs()
            finally:
                _release_io_binding(binding)
            self._emit(
                callback,
                "Staged encoder prelude tile complete",
                **{**details, "event": "tile_complete", "elapsed_seconds": time.perf_counter() - started},
            )

        self._emit(
            callback,
            "Staged encoder late tile batch",
            temporal_clip=temporal_index + 1,
            temporal_clips=temporal_clips,
            tiles=len(tile_entries),
            stage="late",
            provider=self.provider,
        )
        late_shape = (
            len(tile_entries),
            self.channels["late"],
            self.temporal_tokens,
            latent_height,
            latent_width,
        )
        late_batch = self._run_cuda_stage(
            self._sessions["late"],
            "prelude_activation",
            prelude_batch,
            "late_activation",
            late_shape,
        )
        del prelude_batch

        self._emit(
            callback,
            "Staged encoder head tile batch",
            temporal_clip=temporal_index + 1,
            temporal_clips=temporal_clips,
            tiles=len(tile_entries),
            stage="head",
            provider=self.provider,
        )
        moments_shape = (
            len(tile_entries),
            self.channels["moments"],
            self.temporal_tokens,
            latent_height,
            latent_width,
        )
        moments_device = self._run_cuda_stage(
            self._sessions["head"],
            "late_activation",
            late_batch,
            "moments",
            moments_shape,
        )
        del late_batch
        moments_batch = moments_device.numpy()
        del moments_device
        encoded_tiles = {
            (row, column): moments_batch[index : index + 1]
            for index, (row, column, *_rest) in enumerate(tile_entries)
        }
        return _assemble_video_vae_tiles(
            encoded_tiles,
            y_starts,
            [value // self.vae_ratio for value in y_overlaps],
            x_starts,
            [value // self.vae_ratio for value in x_overlaps],
            int(clip.shape[3]) // self.vae_ratio,
            int(clip.shape[4]) // self.vae_ratio,
        )

    def _encode_clip_cpu(
        self,
        clip: np.ndarray,
        tile_size: int,
        callback: Callable[[dict[str, object]], None] | None,
        temporal_index: int,
        temporal_clips: int,
    ) -> np.ndarray:
        y_starts, y_lengths, y_overlaps = _split_tiles(
            int(clip.shape[3]), tile_size, self.tile_overlap_min, self.vae_ratio
        )
        x_starts, x_lengths, x_overlaps = _split_tiles(
            int(clip.shape[4]), tile_size, self.tile_overlap_min, self.vae_ratio
        )
        entries = [
            (row, column, y_start, y_length, x_start, x_length)
            for row, (y_start, y_length) in enumerate(zip(y_starts, y_lengths, strict=True))
            for column, (x_start, x_length) in enumerate(zip(x_starts, x_lengths, strict=True))
        ]
        preludes: list[np.ndarray] = []
        for tile_index, (row, column, y_start, y_length, x_start, x_length) in enumerate(entries):
            self._emit(
                callback,
                "Staged encoder CPU prelude tile",
                temporal_clip=temporal_index + 1,
                temporal_clips=temporal_clips,
                tile=tile_index + 1,
                tiles=len(entries),
                row=row + 1,
                column=column + 1,
                provider=self.provider,
                stage="prelude",
            )
            tile = np.ascontiguousarray(
                clip[:, :, :, y_start : y_start + y_length, x_start : x_start + x_length],
                dtype=np.float16,
            )
            preludes.append(
                self._sessions["prelude"].run(None, {"pixels": tile})[0]  # type: ignore[attr-defined]
            )
        prelude_batch = np.concatenate(preludes, axis=0)
        late_batch = self._sessions["late"].run(  # type: ignore[attr-defined]
            None, {"prelude_activation": prelude_batch}
        )[0]
        moments_batch = self._sessions["head"].run(  # type: ignore[attr-defined]
            None, {"late_activation": late_batch}
        )[0]
        encoded_tiles = {
            (row, column): moments_batch[index : index + 1]
            for index, (row, column, *_rest) in enumerate(entries)
        }
        return _assemble_video_vae_tiles(
            encoded_tiles,
            y_starts,
            [value // self.vae_ratio for value in y_overlaps],
            x_starts,
            [value // self.vae_ratio for value in x_overlaps],
            int(clip.shape[3]) // self.vae_ratio,
            int(clip.shape[4]) // self.vae_ratio,
        )

    def _encode_with_tile_size(
        self,
        pixels: np.ndarray,
        tile_size: int,
        callback: Callable[[dict[str, object]], None] | None,
    ) -> np.ndarray:
        temporal_clips = int(np.ceil(pixels.shape[2] / self.clip_length))
        clip_moments: list[np.ndarray] = []
        for temporal_index in range(temporal_clips):
            self._emit(
                callback,
                "Staged encoder temporal clip",
                event="temporal_clip_start",
                temporal_clip=temporal_index + 1,
                temporal_clips=temporal_clips,
                provider=self.provider,
                tile_size=tile_size,
            )
            clip = pixels[
                :,
                :,
                temporal_index * self.clip_length : (temporal_index + 1) * self.clip_length,
                :,
                :,
            ]
            if clip.shape[2] < self.clip_length:
                clip = np.concatenate(
                    (clip, np.repeat(clip[:, :, -1:], self.clip_length - clip.shape[2], axis=2)),
                    axis=2,
                )
            if self.provider == "CUDAExecutionProvider":
                clip_moments.append(
                    self._encode_clip_cuda(clip, tile_size, callback, temporal_index, temporal_clips)
                )
            else:
                clip_moments.append(
                    self._encode_clip_cpu(clip, tile_size, callback, temporal_index, temporal_clips)
                )
        moments = np.concatenate(clip_moments, axis=2)
        if self.token_drop > 0:
            moments = moments[:, :, :-self.token_drop]
        mean = np.split(moments.astype(np.float32, copy=False), 2, axis=1)[0]
        latent_mean = np.asarray(LATENTS_MEAN, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        latent_std = np.asarray(LATENTS_STD, dtype=np.float32).reshape(1, -1, 1, 1, 1)
        return (mean - latent_mean) / latent_std

    def encode(
        self,
        pixels: np.ndarray,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> np.ndarray:
        candidates = [value for value in (512, 448, 384, 320, 256) if value <= self.tile_size]
        last_error: Exception | None = None
        for tile_size in candidates:
            try:
                result = self._encode_with_tile_size(pixels, tile_size, callback)
                self.tile_size = tile_size
                return result
            except RuntimeError as exc:
                if not any(marker in str(exc).lower() for marker in ("out of memory", "cuda error 2")):
                    raise
                last_error = exc
                if tile_size != candidates[-1]:
                    next_tile = candidates[candidates.index(tile_size) + 1]
                    self._emit(
                        callback,
                        "Rebuilding staged encoder after CUDA OOM",
                        from_tile_size=tile_size,
                        to_tile_size=next_tile,
                    )
                    self._reload_sessions()
        assert last_error is not None
        raise last_error


class ONNXVideoVAEEncoder:
    """Progressively tile early levels and stream compact encoder stages."""

    _h3_onnx_runtime = True

    def __init__(self, directory: Path, prefer_cuda: bool = True):
        if not video_vae_temporal_encoder_ready(directory):
            raise ValueError(f"Progressive temporal Video VAE encoder is not ready: {directory}")
        from h3_workbench.inference_runtime import ORTGraphRunner, _preload_tensorrt_dlls

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        encoder = manifest["temporal_encoder"]
        self.directory = directory.resolve()
        self.clip_length = int(encoder["clip_frames"])
        self.token_drop = int(encoder["token_drop"])
        self.vae_ratio = int(encoder["spatial_ratio"])
        self.temporal_tokens = int(encoder["temporal_tokens"])
        self.channels = {name: int(value) for name, value in encoder["channels"].items()}
        self._graphs = {name: self.directory / str(value) for name, value in encoder["graphs"].items()}
        self._runner = ORTGraphRunner(prefer_cuda=prefer_cuda, prefetch_depth=0)
        self.provider = self._runner.provider
        if self.provider == "CUDAExecutionProvider":
            import onnxruntime as ort

            self._runner.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._runner.enable_mem_pattern = True
            self._runner.cudnn_conv_use_max_workspace = True
        self.tensorrt_enabled = False
        self.tensorrt_fallback_reason: str | None = None
        self._tensorrt_cache_root = Path(
            os.environ.get(
                "H3_VAE_TENSORRT_CACHE",
                str(Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "h3-workbench" / "tensorrt-cache" / "video-encoder"),
            )
        )
        if (
            self.provider == "CUDAExecutionProvider"
            and os.environ.get("H3_VAE_TENSORRT", "1") != "0"
        ):
            try:
                _preload_tensorrt_dlls()
                import onnxruntime as ort

                self.tensorrt_enabled = "TensorrtExecutionProvider" in ort.get_available_providers()
                if self.tensorrt_enabled:
                    self._tensorrt_cache_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001 - CUDA remains the validated fallback
                self.tensorrt_fallback_reason = str(exc)
        self.tile_overlap_min = 32 if self.provider == "CUDAExecutionProvider" else 64
        self.stage1_overlap = 16 if self.provider == "CUDAExecutionProvider" else 32
        total_bytes = 0
        if self.provider == "CUDAExecutionProvider" and torch.cuda.is_available():
            try:
                _, total_bytes = torch.cuda.mem_get_info()
            except RuntimeError:
                pass
        if total_bytes >= 10 * 1024**3:
            self.tile_height, self.tile_width = 768, 1024
            self.stage1_tile_height, self.stage1_tile_width = 384, 512
        elif total_bytes >= 3 * 1024**3:
            self.tile_height, self.tile_width = 208, 264
            self.stage1_tile_height, self.stage1_tile_width = 136, 144
        else:
            self.tile_height, self.tile_width = 256, 320
            self.stage1_tile_height, self.stage1_tile_width = 128, 128
        self.tile_size = self.tile_height

    def _session(self, stage: str) -> object:
        if not self.tensorrt_enabled or stage not in {"prelude", "late"}:
            return self._runner.session(self._graphs[stage])
        import onnxruntime as ort

        cache = self._tensorrt_cache_root / stage
        cache.mkdir(parents=True, exist_ok=True)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        provider_options = [
            {
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": str(cache),
                "trt_fp16_enable": "True",
                "trt_builder_optimization_level": "3",
                "trt_max_workspace_size": str(1024**3),
            },
            {
                "device_id": str(self._runner.device_index),
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_use_max_workspace": "1",
            },
        ]
        try:
            session = ort.InferenceSession(
                str(self._graphs[stage]),
                sess_options=options,
                providers=providers,
                provider_options=provider_options,
            )
            if "TensorrtExecutionProvider" not in session.get_providers():
                raise RuntimeError("ONNX Runtime did not activate TensorRTExecutionProvider")
            return session
        except Exception as exc:  # noqa: BLE001 - keep the CUDA path operational
            self.tensorrt_enabled = False
            self.tensorrt_fallback_reason = str(exc)
            return self._runner.session(self._graphs[stage])

    def close(self) -> None:
        self._runner.close()
        gc.collect()

    @staticmethod
    def _emit(
        callback: Callable[[dict[str, object]], None] | None,
        operation: str,
        **details: object,
    ) -> None:
        if callback is not None:
            callback({"module": "Video VAE", "operation": operation, **details})

    def _run_cuda_to_device(
        self,
        session: object,
        input_name: str,
        values: np.ndarray | object,
        output_name: str,
        output_shape: tuple[int, ...],
    ) -> object:
        import onnxruntime as ort

        input_value = (
            values
            if isinstance(values, ort.OrtValue)
            else ort.OrtValue.ortvalue_from_numpy(
                np.ascontiguousarray(values, dtype=np.float16),
                "cuda",
                self._runner.device_index,
            )
        )
        output = ort.OrtValue.ortvalue_from_shape_and_type(
            output_shape,
            np.float16,
            "cuda",
            self._runner.device_index,
        )
        binding = None
        try:
            binding = session.io_binding()  # type: ignore[attr-defined]
            binding.bind_ortvalue_input(input_name, input_value)
            binding.bind_ortvalue_output(output_name, output)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)  # type: ignore[attr-defined]
            binding.synchronize_outputs()
            return output
        finally:
            _release_io_binding(binding)

    def _run_to_host(
        self,
        session: object,
        input_name: str,
        values: np.ndarray,
        output_name: str,
        output_shape: tuple[int, ...],
    ) -> np.ndarray:
        if self.provider != "CUDAExecutionProvider":
            return session.run(  # type: ignore[attr-defined]
                [output_name],
                {input_name: np.ascontiguousarray(values, dtype=np.float16)},
            )[0]
        output = self._run_cuda_to_device(
            session,
            input_name,
            values,
            output_name,
            output_shape,
        )
        return output.numpy()  # type: ignore[no-any-return,union-attr]

    def _run_tiled_stage(
        self,
        session: object,
        values: np.ndarray,
        *,
        input_name: str,
        output_name: str,
        output_channels: int,
        output_frames: int,
        tile_height: int,
        tile_width: int,
        overlap: int,
        stage: str,
        temporal_index: int,
        temporal_clips: int,
        callback: Callable[[dict[str, object]], None] | None,
        spatial_downsample: int = 2,
    ) -> np.ndarray:
        if spatial_downsample < 1:
            raise ValueError("Video VAE stage spatial_downsample must be positive")
        if values.shape[3] % spatial_downsample or values.shape[4] % spatial_downsample:
            raise ValueError(
                "Video VAE stage input dimensions must be divisible by spatial_downsample: "
                f"{values.shape[3:5]} / {spatial_downsample}"
            )
        y_starts, y_lengths, y_overlaps = _split_tiles(
            int(values.shape[3]), tile_height, overlap, spatial_downsample
        )
        x_starts, x_lengths, x_overlaps = _split_tiles(
            int(values.shape[4]), tile_width, overlap, spatial_downsample
        )
        entries = [
            (row, column, y_start, y_length, x_start, x_length)
            for row, (y_start, y_length) in enumerate(zip(y_starts, y_lengths, strict=True))
            for column, (x_start, x_length) in enumerate(zip(x_starts, x_lengths, strict=True))
        ]
        outputs: dict[tuple[int, int], np.ndarray] = {}
        for tile_index, (row, column, y_start, y_length, x_start, x_length) in enumerate(entries):
            details = {
                "event": "tile_start",
                "temporal_clip": temporal_index + 1,
                "temporal_clips": temporal_clips,
                "tile": tile_index + 1,
                "tiles": len(entries),
                "row": row + 1,
                "rows": len(y_starts),
                "column": column + 1,
                "columns": len(x_starts),
                "stage": stage,
                "provider": self.provider,
            }
            self._emit(callback, f"Progressive encoder {stage} tile", **details)
            started = time.perf_counter()
            tile = values[
                :,
                :,
                :,
                y_start : y_start + y_length,
                x_start : x_start + x_length,
            ]
            output_shape = (
                1,
                output_channels,
                output_frames,
                y_length // spatial_downsample,
                x_length // spatial_downsample,
            )
            outputs[(row, column)] = self._run_to_host(
                session,
                input_name,
                tile,
                output_name,
                output_shape,
            )
            self._emit(
                callback,
                f"Progressive encoder {stage} tile complete",
                **{**details, "event": "tile_complete", "elapsed_seconds": time.perf_counter() - started},
            )
        return _assemble_video_vae_tiles(
            outputs,
            y_starts,
            [value // spatial_downsample for value in y_overlaps],
            x_starts,
            [value // spatial_downsample for value in x_overlaps],
            int(values.shape[3]) // spatial_downsample,
            int(values.shape[4]) // spatial_downsample,
            canvas_dtype=None,
        )

    def _encode_clip(
        self,
        clip: np.ndarray,
        tile_height: int,
        tile_width: int,
        temporal_index: int,
        temporal_clips: int,
        callback: Callable[[dict[str, object]], None] | None,
    ) -> np.ndarray:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-encoder-stage")
        session = self._session("prelude")
        next_session = executor.submit(self._session, "late")
        try:
            try:
                stage0 = self._run_tiled_stage(
                    session,
                    clip,
                    input_name="pixels",
                    output_name="stage0_activation",
                    output_channels=self.channels["prelude"],
                    output_frames=self.clip_length,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    overlap=self.tile_overlap_min,
                    stage="stage 0",
                    temporal_index=temporal_index,
                    temporal_clips=temporal_clips,
                    callback=callback,
                )
            finally:
                del session
                gc.collect()

            session = next_session.result()
            next_session = executor.submit(self._session, "tail")
            try:
                stage1 = self._run_tiled_stage(
                    session,
                    stage0,
                    input_name="stage0_activation",
                    output_name="stage1_activation",
                    output_channels=self.channels["late"],
                    output_frames=9,
                    tile_height=(
                        self.stage1_tile_height
                        if tile_height == self.tile_height
                        else max(96, tile_height // 2)
                    ),
                    tile_width=(
                        self.stage1_tile_width
                        if tile_width == self.tile_width
                        else max(96, tile_width // 2)
                    ),
                    overlap=self.stage1_overlap,
                    stage="stage 1",
                    temporal_index=temporal_index,
                    temporal_clips=temporal_clips,
                    callback=callback,
                )
            finally:
                del stage0
                del session
                gc.collect()

            tail_shape = (
                1,
                self.channels["tail"],
                self.temporal_tokens,
                int(stage1.shape[3]) // 4,
                int(stage1.shape[4]) // 4,
            )
            self._emit(
                callback,
                "Progressive encoder compact tail",
                temporal_clip=temporal_index + 1,
                temporal_clips=temporal_clips,
                stage="tail",
                provider=self.provider,
            )
            session = next_session.result()
            next_session = executor.submit(self._session, "head")
            try:
                # The compact tail still receives the full /4 activation map
                # for a large 2048px reference image. Keep that map on the
                # host and run the /4 -> /16 tail one spatial tile at a time;
                # transferring the whole activation would exceed a 4 GiB GPU.
                tail = self._run_tiled_stage(
                    session,
                    stage1,
                    input_name="stage1_activation",
                    output_name="tail_activation",
                    output_channels=self.channels["tail"],
                    output_frames=self.temporal_tokens,
                    tile_height=(
                        self.stage1_tile_height
                        if tile_height == self.tile_height
                        else max(96, tile_height // 2)
                    ),
                    tile_width=(
                        self.stage1_tile_width
                        if tile_width == self.tile_width
                        else max(96, tile_width // 2)
                    ),
                    overlap=self.stage1_overlap,
                    stage="tail",
                    temporal_index=temporal_index,
                    temporal_clips=temporal_clips,
                    callback=callback,
                    spatial_downsample=4,
                )
            finally:
                del stage1
                del session
                gc.collect()

            moments_shape = (
                1,
                self.channels["moments"],
                self.temporal_tokens,
                tail_shape[3],
                tail_shape[4],
            )
            self._emit(
                callback,
                "Progressive encoder head",
                temporal_clip=temporal_index + 1,
                temporal_clips=temporal_clips,
                stage="head",
                provider=self.provider,
            )
            session = next_session.result()
            try:
                if self.provider == "CUDAExecutionProvider":
                    moments_device = self._run_cuda_to_device(
                        session,
                        "tail_activation",
                        tail,
                        "moments",
                        moments_shape,
                    )
                    moments = moments_device.numpy()  # type: ignore[union-attr]
                    del moments_device
                else:
                    moments = session.run(  # type: ignore[attr-defined]
                        ["moments"], {"tail_activation": tail}
                    )[0]
            finally:
                del tail
                del session
                gc.collect()
            return moments
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _encode_with_tiles(
        self,
        pixels: np.ndarray,
        tile_height: int,
        tile_width: int,
        callback: Callable[[dict[str, object]], None] | None,
        *,
        sample_posterior: bool,
        posterior_seed: int,
        first_token_only: bool,
    ) -> np.ndarray:
        temporal_clips = int(np.ceil(pixels.shape[2] / self.clip_length))
        clip_moments: list[np.ndarray] = []
        for temporal_index in range(temporal_clips):
            self._emit(
                callback,
                "Progressive encoder temporal clip",
                event="temporal_clip_start",
                temporal_clip=temporal_index + 1,
                temporal_clips=temporal_clips,
                provider=self.provider,
                tile_height=tile_height,
                tile_width=tile_width,
            )
            clip = pixels[
                :,
                :,
                temporal_index * self.clip_length : (temporal_index + 1) * self.clip_length,
                :,
                :,
            ]
            if clip.shape[2] < self.clip_length:
                clip = np.concatenate(
                    (clip, np.repeat(clip[:, :, -1:], self.clip_length - clip.shape[2], axis=2)),
                    axis=2,
                )
            clip_moments.append(
                self._encode_clip(
                    clip,
                    tile_height,
                    tile_width,
                    temporal_index,
                    temporal_clips,
                    callback,
                )
            )
        moments = np.concatenate(clip_moments, axis=2)
        if self.token_drop > 0:
            moments = moments[:, :, :-self.token_drop]
        if first_token_only:
            moments = moments[:, :, :1]
        return _video_vae_posterior_latents(
            moments,
            sample_posterior=sample_posterior,
            posterior_seed=posterior_seed,
        )

    def encode(
        self,
        pixels: np.ndarray,
        callback: Callable[[dict[str, object]], None] | None = None,
        *,
        sample_posterior: bool = False,
        posterior_seed: int = 42,
        first_token_only: bool = False,
    ) -> np.ndarray:
        candidates = [
            (self.tile_height, self.tile_width),
            (192, 256),
            (160, 224),
            (128, 192),
        ]
        candidates = list(dict.fromkeys(candidates))
        last_error: Exception | None = None
        for index, (tile_height, tile_width) in enumerate(candidates):
            try:
                result = self._encode_with_tiles(
                    pixels,
                    tile_height,
                    tile_width,
                    callback,
                    sample_posterior=sample_posterior,
                    posterior_seed=posterior_seed,
                    first_token_only=first_token_only,
                )
                self.tile_height, self.tile_width = tile_height, tile_width
                self.tile_size = tile_height
                return result
            except RuntimeError as exc:
                error_text = str(exc).lower()
                if not any(
                    marker in error_text
                    for marker in (
                        "out of memory",
                        "cuda error 2",
                        "failed to allocate memory",
                        "bfc arena",
                        "memcpyfromhost",
                    )
                ):
                    raise
                last_error = exc
                if index + 1 < len(candidates):
                    self._emit(
                        callback,
                        "Reducing progressive encoder tiles after CUDA OOM",
                        from_tile=[tile_height, tile_width],
                        to_tile=list(candidates[index + 1]),
                    )
        assert last_error is not None
        raise last_error

    def encode_image(
        self,
        pixels: np.ndarray,
        callback: Callable[[dict[str, object]], None] | None = None,
        *,
        sample_posterior: bool = False,
        posterior_seed: int = 42,
    ) -> np.ndarray:
        """Encode one image using the causal token equivalent to encode_images()."""
        if pixels.ndim != 5 or pixels.shape[:3] != (1, 3, 1):
            raise ValueError("Image encoder expects pixels with shape [1, 3, 1, H, W]")
        # The validated staged graphs have a fixed 17-frame temporal axis. For
        # a repeated image, causal token zero is numerically equivalent to the
        # official process_image=True path; later tokens contain video history.
        repeated = np.repeat(pixels, self.clip_length, axis=2)
        latent = self.encode(
            np.ascontiguousarray(repeated, dtype=np.float16),
            callback,
            sample_posterior=sample_posterior,
            posterior_seed=posterior_seed,
            first_token_only=True,
        )
        if latent.shape[2] < 1:
            raise RuntimeError("Video VAE image encode returned no latent token")
        return np.ascontiguousarray(latent[:, :, :1], dtype=np.float32)


def load_video_vae_onnx_encoder(directory: Path, prefer_cuda: bool = True) -> ONNXVideoVAEEncoder:
    return ONNXVideoVAEEncoder(directory, prefer_cuda=prefer_cuda)


def load_video_vae_for_encoding(
    checkpoint: Path,
    prefer_cuda: bool = True,
    tile_size: int | None = None,
    tile_overlap_min: int | None = None,
) -> MiniMaxH3VideoVAE:
    """Load only the Video VAE encoder subgraph, leaving the decoder unmapped."""
    from h3_workbench.device_profile import selected_device_index, torch_cuda_architecture_supported

    device_index = selected_device_index() if torch.cuda.is_available() else 0
    cuda_device = f"cuda:{device_index}"
    use_cuda = (
        prefer_cuda
        and torch.cuda.is_available()
        and torch_cuda_architecture_supported(device_index)
    )
    cuda_fallback_reason = None
    total_vram_bytes = 0
    if prefer_cuda and torch.cuda.is_available() and not use_cuda:
        cuda_fallback_reason = (
            "PyTorch build does not contain kernels for the selected CUDA architecture; "
            "using CPU Video VAE encoding"
        )
    free_bytes = 0
    if use_cuda:
        free_bytes, total_vram_bytes = torch.cuda.mem_get_info(device_index)
    if tile_size is None:
        tile_size = 256
        if use_cuda:
            if free_bytes >= 10 * 1024**3:
                tile_size = 512
            elif free_bytes >= 3 * 1024**3:
                tile_size = 384
            elif free_bytes >= 1536 * 1024**2:
                tile_size = 320
    if tile_overlap_min is None:
        # On low-VRAM devices this removes one complete spatial row for 736p
        # input (3x3 encoder tiles become 2x3). Larger devices retain 64px.
        tile_overlap_min = _video_vae_encoder_overlap(use_cuda, total_vram_bytes)
    if tile_overlap_min < 16 or tile_overlap_min % 16 != 0:
        raise ValueError(
            "Video VAE encoder tile_overlap_min must be a multiple of 16 and at least 16"
        )
    if tile_size < 256 or tile_size % 16 != 0:
        raise ValueError("Video VAE encoder tile_size must be a multiple of 16 and at least 256")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("meta"):
            model = MiniMaxH3VideoVAE(
                tiling=True,
                tile_size=tile_size,
                tile_overlap_min=tile_overlap_min,
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    encoder_keys = {
        name
        for name in model.state_dict()
        if name.startswith("encoder.") or name.startswith("quant_conv.") or name in {"latents_mean", "latents_std"}
    }
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for name in sorted(encoder_keys & available):
            state[name] = handle.get_tensor(name)
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected Video VAE encoder weights: {sorted(unexpected)}")
    del state

    model.pixel_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float16).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor(IMAGENET_STD, dtype=torch.float16).view(1, 3, 1, 1, 1)
    del model.decoder
    del model.post_quant_conv
    model._h3_prefer_cuda = use_cuda  # type: ignore[attr-defined]
    model._h3_cuda_device = cuda_device  # type: ignore[attr-defined]
    model._h3_cuda_fallback_reason = cuda_fallback_reason  # type: ignore[attr-defined]
    model._h3_tile_size = tile_size  # type: ignore[attr-defined]
    model._h3_tile_overlap = tile_overlap_min  # type: ignore[attr-defined]
    model._h3_channels_last_3d = bool(use_cuda)  # type: ignore[attr-defined]
    model._h3_cuda_oom = False  # type: ignore[attr-defined]
    if use_cuda:
        try:
            # CUDA Conv3d has a fast channels-last-3d path. CPU fallback stays
            # contiguous because this format is not consistently faster there.
            model.to(device=cuda_device, memory_format=torch.channels_last_3d)
            torch.backends.cudnn.benchmark = True
        except RuntimeError as exc:
            if not any(marker in str(exc).lower() for marker in ("out of memory", "cuda error 2")):
                raise
            model._h3_cuda_oom = True  # type: ignore[attr-defined]
            model.to(device="cpu", memory_format=torch.contiguous_format)
            torch.cuda.empty_cache()
    else:
        model.to("cpu")
    model.eval().requires_grad_(False)
    return model


def encode_image_frame(
    model: MiniMaxH3VideoVAE | ONNXVideoVAEEncoder,
    pixels: np.ndarray,
    callback: Callable[[dict[str, object]], None] | None = None,
    offload_after: bool = True,
    *,
    sample_posterior: bool = False,
    posterior_seed: int = 42,
) -> np.ndarray:
    """Encode one reference image into one normalized Video VAE token."""
    if pixels.ndim != 5 or pixels.shape[:3] != (1, 3, 1):
        raise ValueError("Image encoder expects pixels with shape [1, 3, 1, H, W]")
    if isinstance(model, ONNXVideoVAEEncoder):
        if callback is not None:
            callback(
                {
                    "module": "Video VAE",
                    "operation": "Encoding single-image conditioning",
                    "frames": 1,
                    "height": int(pixels.shape[3]),
                    "width": int(pixels.shape[4]),
                    "provider": model.provider,
                    "backend": "onnxruntime",
                    "protocol": "causal_token_zero",
                }
            )
        result = model.encode_image(
            np.ascontiguousarray(pixels, dtype=np.float16),
            callback,
            sample_posterior=sample_posterior,
            posterior_seed=posterior_seed,
        )
    else:
        result = encode_video_frames(
            model,
            pixels,
            callback=callback,
            offload_after=offload_after,
            sample_posterior=sample_posterior,
            posterior_seed=posterior_seed,
        )
    if result.shape[2] != 1:
        raise RuntimeError(f"Video VAE image encode returned {result.shape[2]} latent tokens")
    if not np.isfinite(result).all():
        invalid = int((~np.isfinite(result)).sum())
        raise FloatingPointError(f"Non-finite Video VAE image encoder output: {invalid} invalid values")
    return np.ascontiguousarray(result, dtype=np.float32)


def encode_video_frames(
    model: MiniMaxH3VideoVAE | ONNXVideoVAEEncoder,
    pixels: np.ndarray,
    callback: Callable[[dict[str, object]], None] | None = None,
    offload_after: bool = True,
    *,
    sample_posterior: bool = False,
    posterior_seed: int = 42,
) -> np.ndarray:
    """Encode ``[1, 3, T, H, W]`` pixels into normalized VAE latents.

    ``offload_after`` is intentionally configurable for segmented super
    resolution.  Encoding all segments as one GPU phase avoids moving the
    (large) PyTorch encoder weights to CPU and back for every segment.  The
    default remains ``True`` so callers that interleave encoding with another
    GPU workload retain the previous low-VRAM behaviour.
    """
    if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError("Video encoder expects pixels with shape [1, 3, T, H, W]")
    if isinstance(model, ONNXVideoVAEEncoder):
        if callback is not None:
            callback(
                {
                    "module": "Video VAE",
                    "operation": "Encoding temporal/spatial conditioning",
                    "frames": int(pixels.shape[2]),
                    "height": int(pixels.shape[3]),
                    "width": int(pixels.shape[4]),
                    "provider": model.provider,
                    "tile_size": model.tile_size,
                    "tile_overlap": model.tile_overlap_min,
                    "backend": "onnxruntime",
                }
            )
        result = model.encode(
            np.ascontiguousarray(pixels, dtype=np.float16),
            callback,
            sample_posterior=sample_posterior,
            posterior_seed=posterior_seed,
        )
        if not np.isfinite(result).all():
            invalid = int((~np.isfinite(result)).sum())
            raise FloatingPointError(f"Non-finite Video VAE encoder output: {invalid} invalid values")
        return result

    def is_cuda_failure(error: RuntimeError) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "out of memory",
                "cuda error 2",
                "no kernel image",
                "not implemented for",
                "not supported on this gpu",
            )
        )

    current_device = next(model.parameters()).device
    if (
        current_device.type == "cpu"
        and getattr(model, "_h3_prefer_cuda", False)
        and not getattr(model, "_h3_cuda_oom", False)
    ):
        try:
            model.to(
                device=getattr(model, "_h3_cuda_device", "cuda"),
                memory_format=torch.channels_last_3d,
            )
        except RuntimeError as exc:
            if not is_cuda_failure(exc):
                raise
            model._h3_cuda_oom = True  # type: ignore[attr-defined]
            model.to("cpu")
            torch.cuda.empty_cache()
    if callback is not None:
        tile_size = int(getattr(model, "_h3_tile_size", getattr(model, "tile_size", 256)))
        tile_overlap = int(
            getattr(model, "_h3_tile_overlap", getattr(model, "tile_overlap_min", 64))
        )
        y_tiles = len(model.split_tiles(int(pixels.shape[3]))[0])
        x_tiles = len(model.split_tiles(int(pixels.shape[4]))[0])
        callback(
            {
                "module": "Video VAE",
                "operation": "Encoding temporal/spatial conditioning",
                "frames": int(pixels.shape[2]),
                "height": int(pixels.shape[3]),
                "width": int(pixels.shape[4]),
                "provider": str(next(model.parameters()).device),
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "tiles_per_temporal_clip": y_tiles * x_tiles,
                "fallback_reason": getattr(model, "_h3_cuda_fallback_reason", None),
            }
        )
    device = next(model.parameters()).device
    input_tensor = torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float16))
    if device.type == "cuda":
        try:
            # Temporal chunks are copied one at a time. A pinned source makes
            # those copies asynchronous and avoids a pageable-memory stall at
            # every 17-frame boundary.
            input_tensor = input_tensor.pin_memory()
        except RuntimeError:
            pass

    def encode_progress(details: dict[str, object]) -> None:
        if callback is None:
            return
        event = str(details.get("event", ""))
        if event == "tile_complete":
            operation = "Video VAE encoder tile complete"
        elif event == "tile_start":
            operation = "Video VAE encoder tile"
        else:
            operation = "Video VAE encoder temporal clip"
        callback({"module": "Video VAE", "operation": operation, **details})

    def next_tile_size(size: int) -> int | None:
        candidates = (512, 448, 384, 320, 256)
        smaller = [candidate for candidate in candidates if candidate < size]
        return max(smaller) if smaller else None

    latent = None
    while latent is None:
        try:
            with torch.inference_mode():
                latent = model.encode(
                    input_tensor,
                    device=device,
                    callback=encode_progress,
                    sample_posterior=sample_posterior,
                    posterior_seed=posterior_seed,
                )
        except RuntimeError as exc:
            if device.type != "cuda" or not is_cuda_failure(exc):
                raise
            current_tile = int(getattr(model, "tile_size", getattr(model, "_h3_tile_size", 256)))
            smaller_tile = next_tile_size(current_tile)
            if smaller_tile is not None:
                model.tile_size = smaller_tile
                model._h3_tile_size = smaller_tile  # type: ignore[attr-defined]
                model.tile_overlap_min = min(
                    int(getattr(model, "tile_overlap_min", 64)),
                    32,
                )
                model._h3_tile_overlap = model.tile_overlap_min  # type: ignore[attr-defined]
                model._h3_cuda_fallback_reason = (
                    f"encoder tile {current_tile} exceeded available VRAM; retrying at {smaller_tile}"
                )  # type: ignore[attr-defined]
                if callback is not None:
                    callback(
                        {
                            "module": "Video VAE",
                            "operation": "Reducing encoder tile after CUDA OOM",
                            "from_tile_size": current_tile,
                            "to_tile_size": smaller_tile,
                        }
                    )
                torch.cuda.empty_cache()
                continue
            # Only give up on CUDA after every safe tile size has failed. The
            # previous implementation fell back after the first OOM and could
            # silently turn a GPU job into a very slow CPU encode.
            model._h3_cuda_oom = True  # type: ignore[attr-defined]
            model._h3_cuda_fallback_reason = str(exc)  # type: ignore[attr-defined]
            model.to(device="cpu", memory_format=torch.contiguous_format)
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            with torch.inference_mode():
                latent = model.encode(
                    input_tensor,
                    device=device,
                    callback=encode_progress,
                    sample_posterior=sample_posterior,
                    posterior_seed=posterior_seed,
                )
    result = latent.detach().cpu().numpy().astype(np.float32, copy=False)
    if offload_after and next(model.parameters()).device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
    if not np.isfinite(result).all():
        invalid = int((~np.isfinite(result)).sum())
        raise FloatingPointError(f"Non-finite Video VAE encoder output: {invalid} invalid values")
    return result


def decode_audio_latents(
    checkpoint: Path,
    latents: np.ndarray,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    if callback is not None:
        callback({"module": "Audio VAE", "operation": "Loading checkpoint"})
    model = MiniMaxH3AudioVAE()
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    if callback is not None:
        callback({"module": "Audio VAE", "operation": "Decoding waveform"})
    with torch.inference_mode():
        waveform = model.decode(torch.from_numpy(latents).float()).clamp(-1.0, 1.0).cpu().numpy()
    del model
    gc.collect()
    return waveform


def encode_audio_waveform_onnx(
    directory: Path,
    waveform: np.ndarray,
    prefer_cuda: bool = True,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    """Encode a 32 kHz stereo waveform with the complete Audio VAE encoder graph."""
    from h3_workbench.inference_runtime import ORTGraphRunner

    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("Audio VAE input must have shape [2, samples] or [batch, 2, samples]")
    right_pad = (-values.shape[-1]) % 800
    if right_pad:
        values = np.pad(values, ((0, 0), (0, 0), (0, right_pad)))
    if callback is not None:
        callback({"module": "Audio VAE", "operation": "Loading complete ONNX encoder"})
    if not audio_vae_encoder_ready(directory):
        raise RuntimeError(
            "Reference audio requires audio_encoder.onnx; re-export the Audio VAE product with encoder support"
        )
    runner = ORTGraphRunner(prefer_cuda=prefer_cuda, prefetch_depth=1)
    session = runner.session(directory / "audio_encoder.onnx")
    try:
        if callback is not None:
            callback(
                {
                    "module": "Audio VAE",
                    "operation": "Encoding reference waveform",
                    "provider": runner.provider,
                    "audio_samples": int(values.shape[-1]),
                }
            )
        latents = session.run(None, {"waveform": np.ascontiguousarray(values)})[0]
        if not np.isfinite(latents).all():
            invalid = int((~np.isfinite(latents)).sum())
            raise FloatingPointError(f"Non-finite Audio VAE encoder output: {invalid} invalid values")
        return np.asarray(latents, dtype=np.float32)
    finally:
        del session
        runner.close()
        gc.collect()


def decode_audio_latents_onnx(
    directory: Path,
    latents: np.ndarray,
    prefer_cuda: bool = True,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> np.ndarray:
    """Decode the complete stereo latent sequence with one ONNX graph."""
    from h3_workbench.inference_runtime import ORTGraphRunner

    if callback is not None:
        callback({"module": "Audio VAE", "operation": "Loading complete ONNX decoder"})
    runner = ORTGraphRunner(prefer_cuda=prefer_cuda, prefetch_depth=1)
    session = runner.session(directory / "audio_decoder.onnx")
    try:
        if callback is not None:
            callback(
                {
                    "module": "Audio VAE",
                    "operation": "Decoding complete latent sequence",
                    "provider": runner.provider,
                    "latent_frames": int(latents.shape[-1]),
                }
            )
        waveform = session.run(None, {"latents": latents.astype(np.float32, copy=False)})[0]
        if not np.isfinite(waveform).all():
            invalid = int((~np.isfinite(waveform)).sum())
            raise FloatingPointError(f"Non-finite Audio VAE output: {invalid} invalid values")
        return np.clip(waveform, -1.0, 1.0)
    finally:
        del session
        runner.close()
        gc.collect()


def _write_wave(path: Path, waveform: np.ndarray, sample_rate: int = 32000) -> None:
    samples = np.clip(waveform[0].T, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def output_metadata_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def write_mp4(
    path: Path,
    pixels: np.ndarray,
    waveform: np.ndarray,
    fps: float = 24,
    metadata: dict[str, object] | None = None,
) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.clip(pixels[0].transpose(1, 2, 3, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
    height, width = frames.shape[1:3]
    with tempfile.TemporaryDirectory(prefix="h3-media-") as temp:
        audio_path = Path(temp) / "audio.wav"
        _write_wave(audio_path, waveform)
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
        ]
        if metadata is not None:
            embedded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            command.extend(
                (
                    "-metadata",
                    "title=MiniMax H3 Edge Workbench output",
                    "-metadata",
                    f"comment={embedded}",
                )
            )
        command.append(str(path))
        result = subprocess.run(command, input=frames.tobytes(), capture_output=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    if metadata is None:
        return None
    metadata_path = output_metadata_path(path)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata_path


def write_mp4_with_audio_source(
    path: Path,
    pixels: np.ndarray,
    audio_source: Path,
    fps: float = 24,
    metadata: dict[str, object] | None = None,
) -> Path | None:
    """Encode generated frames and reuse the input video's audio stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.clip(pixels[0].transpose(1, 2, 3, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
    height, width = frames.shape[1:3]
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
    ]
    if metadata is not None:
        embedded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        command.extend(
            (
                "-metadata",
                "title=MiniMax H3 Edge Workbench output",
                "-metadata",
                f"comment={embedded}",
            )
        )
    command.append(str(path))
    result = subprocess.run(command, input=frames.tobytes(), capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    if metadata is None:
        return None
    metadata_path = output_metadata_path(path)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata_path
