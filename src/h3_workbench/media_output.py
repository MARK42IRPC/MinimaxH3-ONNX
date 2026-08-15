from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from h3_workbench.vendor.audio_vae import MiniMaxH3AudioVAE
from h3_workbench.vendor.video_vae import IMAGENET_MEAN, IMAGENET_STD, MiniMaxH3VideoVAE
from h3_workbench.profiles import video_vae_output_frames


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
    )

    session = None
    hidden_devices: list[ort.OrtValue] = []
    rotary_devices: list[ort.OrtValue] = []
    try:
        session = runner.session(directory / PERSISTENT_VIDEO_VAE_TOPOLOGY)  # type: ignore[attr-defined]
        hidden_devices = [ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0) for value in hidden_tiles]
        rotary_devices = [ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0) for value in rotary_tiles]
        shapes = [value.shape for value in hidden_tiles]
        dtypes = [value.dtype for value in hidden_tiles]
        hidden_tiles.clear()
        rotary_tiles.clear()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-vae-weight-prefetch") as executor:
            pending = executor.submit(load_video_vae_block_weights, directory, 0)
            for index in range(block_count):
                loaded = pending.result()
                weight_devices: dict[str, ort.OrtValue] = {}
                if index + 1 < block_count:
                    pending = executor.submit(load_video_vae_block_weights, directory, index + 1)
                try:
                    weight_devices = {
                        name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
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
                                    "weight_load_seconds": round(loaded.load_seconds, 3),
                                }
                            )
                        output = ort.OrtValue.ortvalue_from_shape_and_type(
                            shapes[tile_index], dtypes[tile_index], "cuda", 0
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
                    loaded.close()
                    weight_devices.clear()
        return [value.numpy() for value in hidden_devices]
    finally:
        hidden_devices.clear()
        rotary_devices.clear()
        session = None
        gc.collect()


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
                canvas = np.empty((*tile.shape[:-2], padded_height, padded_width), dtype=np.float32)
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
    fps: int = 24,
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
