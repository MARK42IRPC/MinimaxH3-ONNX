from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from safetensors.torch import load_file

from h3_workbench.acceleration import inspect_acceleration_lora, load_silu_temb_grid
from h3_workbench.model_registry import inspect_checkpoint, scan_models
from h3_workbench.main_transformer import (
    ATTENTION_INNER,
    HEAD_DIM,
    HEADS,
    CheckpointReader,
    MainConditioning,
    MainEmbeddings,
    MainHead,
    RefinerNorm,
)
from h3_workbench.qwen_transformer import QwenCheckpointReader
from h3_workbench.vendor.audio_vae import MiniMaxH3AudioVAE
from h3_workbench.vendor.video_vae import IMAGENET_MEAN, IMAGENET_STD, MiniMaxH3VideoVAE

ProgressCallback = Callable[[float, str], None]
OPSET_VERSION = 20


def _progress(callback: ProgressCallback | None, value: float, message: str) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, value)), message)


def _load_model(path: Path, component: str) -> nn.Module:
    dtype = torch.float32 if component == "audio_vae" else torch.float16
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model: nn.Module
        if component == "audio_vae":
            model = MiniMaxH3AudioVAE()
        elif component == "video_vae":
            with torch.device("meta"):
                model = MiniMaxH3VideoVAE(tiling=False)
        else:
            raise ValueError(f"Unsupported component: {component}")
    finally:
        torch.set_default_dtype(previous_dtype)

    state = load_file(str(path), device="cpu")
    incompatible = model.load_state_dict(state, strict=True, assign=component == "video_vae")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state
    if component == "video_vae":
        assert isinstance(model, MiniMaxH3VideoVAE)
        model.pixel_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float16).view(1, 3, 1, 1, 1)
        model.pixel_std = torch.tensor(IMAGENET_STD, dtype=torch.float16).view(1, 3, 1, 1, 1)
        count = model.decoder.pos_embed.inv_freq.numel()
        model.decoder.pos_embed.inv_freq = 1.0 / 100.0 ** torch.arange(
            0.0, 1.0, 1.0 / count, dtype=torch.float32
        )
    model.eval().requires_grad_(False)
    gc.collect()
    return model


def _export_graph(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    dynamo: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.with_name(f"{path.name}.data").unlink(missing_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamo=dynamo,
            external_data=True,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
    onnx.checker.check_model(str(path), full_check=False)
    if dynamo:
        torch._dynamo.reset()
    gc.collect()


def _ort_run(path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    if path.with_name(f"{path.name}.data").exists():
        with tempfile.TemporaryDirectory(prefix="h3-ort-") as temp:
            temp_dir = Path(temp)
            input_path = temp_dir / "inputs.npz"
            output_path = temp_dir / "outputs.npz"
            np.savez(input_path, **inputs)
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            result = subprocess.run(
                [sys.executable, "-m", "h3_workbench.ort_worker", str(path), str(input_path), str(output_path)],
                capture_output=True,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
                raise RuntimeError(f"Isolated ONNX Runtime validation failed: {details}")
            with np.load(output_path, allow_pickle=False) as archive:
                return [archive[f"output_{index}"].copy() for index in range(len(archive.files))]

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    # ORT 1.22's basic optimizer can incorrectly prune a Cast feeding the
    # expanded FP16 RMSNorm graph. Keep validation faithful to the exported graph.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    try:
        return session.run(None, inputs)
    finally:
        del session
        gc.collect()


def _metrics(expected: torch.Tensor | np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected_np = expected.detach().cpu().float().numpy() if isinstance(expected, torch.Tensor) else expected.astype(np.float32)
    actual_np = actual.astype(np.float32)
    delta = np.abs(expected_np - actual_np)
    denominator = np.linalg.norm(expected_np.reshape(-1)) * np.linalg.norm(actual_np.reshape(-1))
    expected_norm = np.linalg.norm(expected_np.reshape(-1))
    cosine = float(np.dot(expected_np.reshape(-1), actual_np.reshape(-1)) / denominator) if denominator else 1.0
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "cosine": cosine,
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / expected_norm) if expected_norm else 0.0,
    }


def _require_close(
    name: str,
    metrics: dict[str, float],
    max_abs: float,
    min_cosine: float = 0.999,
    max_relative_l2: float | None = None,
) -> None:
    relative_failed = max_relative_l2 is not None and metrics["relative_l2"] > max_relative_l2
    finite = all(np.isfinite(value) for value in metrics.values())
    if not finite or metrics["max_abs"] > max_abs or metrics["cosine"] < min_cosine or relative_failed:
        raise RuntimeError(
            f"{name} validation failed: max_abs={metrics['max_abs']:.6g}, "
            f"cosine={metrics['cosine']:.6g}, relative_l2={metrics['relative_l2']:.6g}"
        )


class AudioEncoder(nn.Module):
    def __init__(self, model: MiniMaxH3AudioVAE):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model.encode(waveform)


class AudioDecoder(nn.Module):
    def __init__(self, model: MiniMaxH3AudioVAE):
        super().__init__()
        self.model = model

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents)


class VideoEncoder(nn.Module):
    def __init__(self, model: MiniMaxH3VideoVAE):
        super().__init__()
        self.encoder = model.encoder
        self.quant_conv = model.quant_conv
        for module in self.encoder.modules():
            if hasattr(module, "causal_padding"):
                module.export_causal_frames = 1
        self.register_buffer("pixel_mean", model.pixel_mean)
        self.register_buffer("pixel_std", model.pixel_std)
        self.register_buffer("latents_mean", model.latents_mean)
        self.register_buffer("latents_std", model.latents_std)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        normalized = (pixels + 1.0) * 0.5
        normalized = (normalized - self.pixel_mean.to(normalized)) / self.pixel_std.to(normalized)
        moments = self.quant_conv(self.encoder(normalized))[:, :, -1:, :, :]
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - latent_mean) / latent_std


class VideoDecoderPrelude(nn.Module):
    def __init__(self, model: MiniMaxH3VideoVAE, latent_shape: tuple[int, int, int] = (5, 16, 16)):
        super().__init__()
        self.post_quant_conv = model.post_quant_conv
        self.x_embedder = model.decoder.x_embedder
        self.register_tokens = model.decoder.register_tokens
        self.register_buffer("latents_mean", model.latents_mean)
        self.register_buffer("latents_std", model.latents_std)
        from h3_workbench.vendor.video_vae import create_token_ids

        ids = create_token_ids(latent_shape, torch.device("cpu"), model.latents_mean.dtype)
        suffix_ids = torch.zeros((1, 1 + self.register_tokens.shape[1], 3), dtype=ids.dtype)
        with torch.inference_mode():
            rotary_table = model.decoder.pos_embed(torch.cat((ids, suffix_ids), dim=1))
        self.register_buffer("rotary_table", rotary_table)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(latents)
        z = self.post_quant_conv(latents * latent_std + latent_mean)
        batch = z.shape[0]
        hidden = self.x_embedder(z.flatten(2).transpose(1, 2))
        suffix = torch.zeros_like(hidden[:, 0:1, :])
        hidden = torch.cat((hidden, self.register_tokens.to(hidden).expand(batch, -1, -1), suffix), dim=1)
        return hidden, self.rotary_table.to(hidden).expand(batch, -1, -1, -1, -1, -1)


class VideoDecoderBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor, rotary_table: torch.Tensor) -> torch.Tensor:
        # Upstream blocks use in-place residual updates. A shard boundary must
        # not mutate the caller's activation because export and validation reuse it.
        return self.block(hidden_states.clone(), rotary_table)


class VideoDecoderHead(nn.Module):
    def __init__(self, model: MiniMaxH3VideoVAE, latent_shape: tuple[int, int, int]):
        super().__init__()
        self.norm_out = model.decoder.norm_out
        self.proj_out = model.decoder.proj_out
        self.patch_size = model.decoder.patch_size
        self.patch_size_t = model.decoder.patch_size_t
        self.out_channels = model.decoder.out_channels
        self.latent_shape = latent_shape
        self.register_buffer("pixel_mean", model.pixel_mean)
        self.register_buffer("pixel_std", model.pixel_std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latent_time, latent_height, latent_width = self.latent_shape
        patch_count = latent_time * latent_height * latent_width
        output = self.proj_out(self.norm_out(hidden_states))[:, :patch_count, :]
        output = output.view(
            hidden_states.shape[0], latent_time, latent_height, latent_width,
            self.out_channels, self.patch_size_t, self.patch_size, self.patch_size,
        )
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        output = output.reshape(
            hidden_states.shape[0], self.out_channels,
            latent_time * self.patch_size_t, latent_height * self.patch_size, latent_width * self.patch_size,
        )
        output = output.float() * self.pixel_std.float()
        return (output + self.pixel_mean.float()).clamp(0.0, 1.0)


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_audio(path: Path, output_dir: Path, callback: ProgressCallback | None = None) -> dict[str, Any]:
    _progress(callback, 0.03, "Loading Audio VAE")
    model = _load_model(path, "audio_vae")
    assert isinstance(model, MiniMaxH3AudioVAE)
    torch.manual_seed(7)
    latent_frames = 29
    latents = torch.randn(1, 32, 2, latent_frames, dtype=torch.float32)
    decoder = AudioDecoder(model)

    with torch.inference_mode():
        expected_decoded = decoder(latents)

    decoder_path = output_dir / "audio_decoder.onnx"
    _progress(callback, 0.25, "Exporting complete audio decoder")
    _export_graph(
        decoder,
        (latents,),
        decoder_path,
        ["latents"],
        ["waveform"],
        {"latents": {3: "latent_frames"}, "waveform": {2: "audio_samples"}},
    )

    _progress(callback, 0.82, "Validating complete Audio VAE decoder with ONNX Runtime")
    actual_decoded = _ort_run(decoder_path, {"latents": latents.numpy()})[0]
    validation = {"decoder": _metrics(expected_decoded, actual_decoded)}
    _require_close("Audio decoder", validation["decoder"], max_abs=1e-3, min_cosine=0.9999)
    manifest = {
        "format": "h3-workbench-onnx-v1",
        "source": str(path.resolve()),
        "component": "audio_vae",
        "layout": "single_graph_decoder",
        "activation_dtype": "float32",
        "opset": OPSET_VERSION,
        "profiles": {
            "validation_latents": [1, 32, 2, latent_frames],
            "validation_waveform": list(expected_decoded.shape),
        },
        "dynamic_dimensions": ["latent_frames", "audio_samples"],
        "graphs": [decoder_path.name],
        "validation": validation,
        "validation_passed": True,
    }
    _write_manifest(output_dir, manifest)
    _progress(callback, 1.0, "Single-graph Audio VAE decoder export and validation completed")
    return manifest


def _parse_blocks(spec: str, count: int) -> list[int]:
    if spec.lower() == "all":
        return list(range(count))
    blocks = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if not blocks or blocks[0] < 0 or blocks[-1] >= count:
        raise ValueError(f"Video block selection must be 'all' or indices from 0 to {count - 1}")
    return blocks


def export_video(
    path: Path,
    output_dir: Path,
    block_spec: str = "all",
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _progress(callback, 0.01, "Loading Video VAE")
    model = _load_model(path, "video_vae")
    assert isinstance(model, MiniMaxH3VideoVAE)
    blocks = _parse_blocks(block_spec, len(model.decoder.transformer_blocks))
    torch.manual_seed(11)
    tile_latent_shape = (5, 16, 16)
    tile_pixels_shape = (1, 3, 20, 256, 256)
    pixels = torch.randn(1, 3, 1, 32, 32, dtype=torch.float16).clamp(-1, 1)
    latents = torch.randn(1, 24, *tile_latent_shape, dtype=torch.float16)
    encoder = VideoEncoder(model)
    prelude = VideoDecoderPrelude(model, tile_latent_shape)
    head = VideoDecoderHead(model, tile_latent_shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / "video_encoder.onnx"
    prelude_path = output_dir / "video_decoder_prelude.onnx"
    head_path = output_dir / "video_decoder_head.onnx"

    with torch.inference_mode():
        expected_encoded = encoder(pixels)
        hidden, rotary = prelude(latents)

    _progress(callback, 0.06, "Exporting video encoder")
    _export_graph(encoder, (pixels,), encoder_path, ["pixels"], ["latents"])
    _progress(callback, 0.14, "Exporting video decoder prelude")
    _export_graph(prelude, (latents,), prelude_path, ["latents"], ["hidden_states", "rotary_table"])

    block_graphs: list[str] = []
    block_metrics: dict[str, dict[str, float]] = {}
    all_blocks = len(model.decoder.transformer_blocks)
    smoke_hidden = hidden[:, :8]
    smoke_rotary = rotary[:, :8]
    for position, index in enumerate(blocks):
        wrapper = VideoDecoderBlock(model.decoder.transformer_blocks[index])
        with torch.inference_mode():
            expected_hidden = wrapper(smoke_hidden, smoke_rotary)
        block_path = output_dir / f"video_decoder_block_{index:02d}.onnx"
        start = 0.18 + 0.58 * position / max(1, len(blocks))
        _progress(callback, start, f"Exporting video decoder block {index + 1}/{all_blocks}")
        _export_graph(
            wrapper,
            (smoke_hidden, smoke_rotary),
            block_path,
            ["hidden_states", "rotary_table"],
            ["hidden_states_out"],
            dynamic_axes={
                "hidden_states": {1: "sequence"},
                "rotary_table": {1: "sequence"},
                "hidden_states_out": {1: "sequence"},
            },
        )
        actual_hidden = _ort_run(
            block_path,
            {"hidden_states": smoke_hidden.cpu().numpy(), "rotary_table": smoke_rotary.cpu().numpy()},
        )[0]
        block_metrics[str(index)] = _metrics(expected_hidden, actual_hidden)
        block_graphs.append(block_path.name)

    _progress(callback, 0.80, "Exporting video decoder head")
    _export_graph(head, (hidden,), head_path, ["hidden_states"], ["pixels"])

    _progress(callback, 0.90, "Validating video shards with ONNX Runtime")
    actual_encoded = _ort_run(encoder_path, {"pixels": pixels.cpu().numpy()})[0]
    prelude_outputs = _ort_run(prelude_path, {"latents": latents.cpu().numpy()})
    validation: dict[str, Any] = {
        "encoder": _metrics(expected_encoded, actual_encoded),
        "prelude_hidden": _metrics(prelude(latents)[0], prelude_outputs[0]),
        "prelude_rotary": _metrics(prelude(latents)[1], prelude_outputs[1]),
        "blocks": block_metrics,
    }
    _require_close("Video encoder", validation["encoder"], max_abs=1e-2, min_cosine=0.999)
    _require_close("Video decoder prelude", validation["prelude_hidden"], max_abs=1e-2, min_cosine=0.999)
    _require_close("Video rotary table", validation["prelude_rotary"], max_abs=1e-4, min_cosine=0.999)
    for index, metrics in block_metrics.items():
        _require_close(f"Video decoder block {index}", metrics, max_abs=1e-2, min_cosine=0.999)

    with torch.inference_mode():
        expected_head = head(hidden)
    actual_head = _ort_run(head_path, {"hidden_states": hidden.cpu().numpy()})[0]
    validation["head"] = _metrics(expected_head, actual_head)
    _require_close("Video decoder head", validation["head"], max_abs=1e-2, min_cosine=0.999)

    manifest = {
        "format": "h3-workbench-onnx-v1",
        "source": str(path.resolve()),
        "component": "video_vae",
        "opset": OPSET_VERSION,
        "profiles": {
            "pixels": [1, 3, 1, 32, 32],
            "latents": [1, 24, *tile_latent_shape],
            "tile_pixels": list(tile_pixels_shape),
            "output_frames": 17,
            "temporal_warmup_frames": 3,
        },
        "blocks": blocks,
        "graphs": [encoder_path.name, prelude_path.name, *block_graphs, head_path.name],
        "validation": validation,
        "validation_passed": True,
    }
    _write_manifest(output_dir, manifest)
    if blocks == list(range(all_blocks)):
        from h3_workbench.video_vae_persistent import build_persistent_video_vae_topology

        export_video_long_temporal_graphs(
            path,
            output_dir,
            lambda value, message: _progress(callback, 0.92 + 0.04 * value, message),
        )
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        _progress(callback, 0.97, "Validating all Video VAE blocks for persistent execution")
        build_persistent_video_vae_topology(output_dir, validate_blocks=blocks)
        manifest["persistent_decoder"] = {
            "topology": "runtime_persistent_video_decoder_block.onnx",
            "manifest": "runtime_persistent_video_decoder_manifest.json",
            "validated_blocks": blocks,
        }
        _write_manifest(output_dir, manifest)
    _progress(callback, 1.0, "Video VAE sharded export and validation completed")
    return manifest


def export_video_long_temporal_graphs(
    path: Path,
    output_dir: Path,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Add the small 7-token boundary graphs used by long-video decoding."""
    _progress(callback, 0.05, "Loading Video VAE for long-video boundary graphs")
    model = _load_model(path, "video_vae")
    assert isinstance(model, MiniMaxH3VideoVAE)
    latent_shape = (7, 16, 16)
    latents = torch.randn(1, 24, *latent_shape, dtype=torch.float16)
    prelude = VideoDecoderPrelude(model, latent_shape)
    head = VideoDecoderHead(model, latent_shape)
    output_dir.mkdir(parents=True, exist_ok=True)
    prelude_path = output_dir / "video_decoder_prelude_t7.onnx"
    head_path = output_dir / "video_decoder_head_t7.onnx"

    _progress(callback, 0.45, "Exporting 7-token Video VAE prelude")
    _export_graph(prelude, (latents,), prelude_path, ["latents"], ["hidden_states", "rotary_table"])
    with torch.inference_mode():
        hidden, rotary = prelude(latents)
        expected_pixels = head(hidden)
    _progress(callback, 0.75, "Exporting 7-token Video VAE head")
    _export_graph(head, (hidden,), head_path, ["hidden_states"], ["pixels"])

    actual_hidden, actual_rotary = _ort_run(prelude_path, {"latents": latents.cpu().numpy()})
    actual_pixels = _ort_run(head_path, {"hidden_states": hidden.cpu().numpy()})[0]
    long_validation = {
        "prelude_hidden": _metrics(hidden, actual_hidden),
        "prelude_rotary": _metrics(rotary, actual_rotary),
        "head": _metrics(expected_pixels, actual_pixels),
    }
    _require_close("Long-video prelude", long_validation["prelude_hidden"], max_abs=1e-2, min_cosine=0.999)
    _require_close("Long-video rotary", long_validation["prelude_rotary"], max_abs=1e-4, min_cosine=0.999)
    _require_close("Long-video head", long_validation["head"], max_abs=1e-2, min_cosine=0.999)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graphs = list(manifest.get("graphs", []))
    for graph in (prelude_path.name, head_path.name):
        if graph not in graphs:
            graphs.append(graph)
    manifest["graphs"] = graphs
    manifest["long_video"] = {
        "temporal_stride_tokens": 5,
        "temporal_window_tokens": 7,
        "overlap_frames": 5,
        "graphs": [prelude_path.name, head_path.name],
        "validation": long_validation,
    }
    _write_manifest(output_dir, manifest)
    _progress(callback, 1.0, "Long-video Video VAE boundary graphs completed")
    return manifest


def _main_dynamic_axes(*names: str) -> dict[str, dict[int, str]]:
    return {name: {0: "sequence"} for name in names}


def _export_main_shard(
    kind: str,
    index: int,
    checkpoint: Path,
    model_path: Path,
    inputs: dict[str, np.ndarray],
    lora_path: Path | None = None,
    lora_strength: float = 1.0,
    gpu_native_fp16: bool = True,
) -> torch.Tensor:
    with tempfile.TemporaryDirectory(prefix="h3-export-") as temp:
        temp_dir = Path(temp)
        input_path = temp_dir / "inputs.npz"
        expected_path = temp_dir / "expected.npz"
        np.savez(input_path, **inputs)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        command = [
                sys.executable,
                "-m",
                "h3_workbench.main_export_worker",
                kind,
                str(index),
                str(checkpoint),
                str(model_path),
                str(input_path),
                str(expected_path),
            ]
        if lora_path is not None:
            command.extend(("--lora", str(lora_path), "--lora-strength", str(lora_strength)))
        if gpu_native_fp16:
            command.append("--gpu-native-fp16")
        result = subprocess.run(
            command,
            capture_output=True,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"Isolated main shard export failed: {details}")
        with np.load(expected_path, allow_pickle=False) as archive:
            return torch.from_numpy(archive["output_0"].copy())


def _export_qwen_shard(
    kind: str,
    index: int,
    checkpoint: Path,
    model_path: Path,
    inputs: dict[str, np.ndarray],
    gpu_native_fp16: bool = True,
) -> list[torch.Tensor]:
    with tempfile.TemporaryDirectory(prefix="qwen-export-") as temp:
        temp_dir = Path(temp)
        input_path = temp_dir / "inputs.npz"
        expected_path = temp_dir / "expected.npz"
        np.savez(input_path, **inputs)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        command = [
                sys.executable,
                "-m",
                "h3_workbench.qwen_export_worker",
                kind,
                str(index),
                str(checkpoint),
                str(model_path),
                str(input_path),
                str(expected_path),
            ]
        if gpu_native_fp16:
            command.append("--gpu-native-fp16")
        result = subprocess.run(
            command,
            capture_output=True,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"Isolated Qwen shard export failed: {details}")
        with np.load(expected_path, allow_pickle=False) as archive:
            return [torch.from_numpy(archive[f"output_{index}"].copy()) for index in range(len(archive.files))]


def export_main(
    path: Path,
    output_dir: Path,
    block_spec: str = "all",
    callback: ProgressCallback | None = None,
    lora_path: Path | None = None,
    lora_strength: float = 1.0,
) -> dict[str, Any]:
    blocks = _parse_blocks(block_spec, 50)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(19)

    video_patches = torch.randn(1, 96, dtype=torch.float32)
    audio_patches = torch.randn(1, 32, dtype=torch.float32)
    text_states = torch.randn(1, 5120, dtype=torch.float16)
    hidden_states = torch.randn(3, 5376, dtype=torch.float32) * 0.25
    timesteps = torch.tensor([0.5], dtype=torch.float32)
    position_ids = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [2.0, 1.0, 1.0]], dtype=torch.float32)
    modulation_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    turbo_grid = (
        torch.from_numpy(load_silu_temb_grid(lora_path)) if lora_path is not None else None
    )

    graphs: list[str] = []
    validation: dict[str, Any] = {"blocks": {}}

    _progress(callback, 0.01, "Loading main-model input projections")
    reader = CheckpointReader(path)
    embeddings = MainEmbeddings(reader).eval()
    embeddings_path = output_dir / "main_embeddings.onnx"
    with torch.inference_mode():
        expected_embeddings = embeddings(video_patches, audio_patches, text_states)
    _export_graph(
        embeddings,
        (video_patches, audio_patches, text_states),
        embeddings_path,
        ["video_patches", "audio_patches", "text_states"],
        ["video_embeddings", "audio_embeddings", "text_embeddings"],
        {
            "video_patches": {0: "video_sequence"},
            "audio_patches": {0: "audio_sequence"},
            "text_states": {0: "text_sequence"},
            "video_embeddings": {0: "video_sequence"},
            "audio_embeddings": {0: "audio_sequence"},
            "text_embeddings": {0: "text_sequence"},
        },
    )
    actual_embeddings = _ort_run(
        embeddings_path,
        {
            "video_patches": video_patches.numpy(),
            "audio_patches": audio_patches.numpy(),
            "text_states": text_states.numpy(),
        },
    )
    validation["embeddings"] = {
        name: _metrics(expected, actual)
        for name, expected, actual in zip(("video", "audio", "text"), expected_embeddings, actual_embeddings, strict=True)
    }
    for name, metrics in validation["embeddings"].items():
        _require_close(f"Main {name} embeddings", metrics, max_abs=2e-2, min_cosine=0.999)
    graphs.append(embeddings_path.name)
    text_hidden = expected_embeddings[2]
    del embeddings, expected_embeddings, actual_embeddings, reader
    gc.collect()

    for index in range(2):
        _progress(callback, 0.04 + index * 0.025, f"Exporting token refiner {index + 1}/2 attention")
        attention_path = output_dir / f"main_token_refiner_block_{index:02d}_attention.onnx"
        expected_attention = _export_main_shard(
            "refiner_attention",
            index,
            path,
            attention_path,
            {"hidden_states": text_hidden.numpy()},
            lora_path,
            lora_strength,
        )
        actual_attention = _ort_run(attention_path, {"hidden_states": text_hidden.numpy()})[0]
        attention_metrics = _metrics(expected_attention, actual_attention)
        _require_close(
            f"Token refiner {index} attention",
            attention_metrics,
            max_abs=64.0,
            min_cosine=0.999,
            max_relative_l2=2e-3,
        )
        graphs.append(attention_path.name)

        _progress(callback, 0.052 + index * 0.025, f"Exporting token refiner {index + 1}/2 MLP")
        mlp_path = output_dir / f"main_token_refiner_block_{index:02d}_mlp.onnx"
        expected_mlp = _export_main_shard(
            "refiner_mlp",
            index,
            path,
            mlp_path,
            {"hidden_states": expected_attention.numpy()},
            lora_path,
            lora_strength,
        )
        actual_mlp = _ort_run(mlp_path, {"hidden_states": expected_attention.numpy()})[0]
        mlp_metrics = _metrics(expected_mlp, actual_mlp)
        _require_close(
            f"Token refiner {index} MLP",
            mlp_metrics,
            max_abs=64.0,
            min_cosine=0.999,
            max_relative_l2=2e-3,
        )
        validation.setdefault("token_refiner", {})[str(index)] = {
            "attention": attention_metrics,
            "mlp": mlp_metrics,
        }
        graphs.append(mlp_path.name)
        text_hidden = expected_mlp
        del expected_attention, actual_attention, expected_mlp, actual_mlp
        gc.collect()

    reader = CheckpointReader(path)
    refiner_norm = RefinerNorm(reader.tensor("token_refiner.final_norm.weight")).eval()
    refiner_norm_path = output_dir / "main_token_refiner_norm.onnx"
    with torch.inference_mode():
        expected_norm = refiner_norm(text_hidden)
    _export_graph(
        refiner_norm,
        (text_hidden,),
        refiner_norm_path,
        ["hidden_states"],
        ["hidden_states_out"],
        _main_dynamic_axes("hidden_states", "hidden_states_out"),
    )
    actual_norm = _ort_run(refiner_norm_path, {"hidden_states": text_hidden.numpy()})[0]
    validation["token_refiner_norm"] = _metrics(expected_norm, actual_norm)
    _require_close("Token refiner norm", validation["token_refiner_norm"], max_abs=1e-2, min_cosine=0.9999)
    graphs.append(refiner_norm_path.name)
    del refiner_norm, expected_norm, actual_norm, text_hidden, reader
    gc.collect()

    _progress(callback, 0.10, "Exporting timestep curve and RoPE")
    reader = CheckpointReader(path)
    conditioning = MainConditioning(reader, turbo_grid).eval()
    conditioning_path = output_dir / "main_conditioning.onnx"
    with torch.inference_mode():
        conditioning_outputs = conditioning(timesteps, position_ids)
        timestep_embedding, rotary_table = conditioning_outputs[:2]
        silu_timestep_embedding = conditioning_outputs[2] if len(conditioning_outputs) == 3 else None
    _export_graph(
        conditioning,
        (timesteps, position_ids),
        conditioning_path,
        ["timesteps", "position_ids"],
        ["timestep_embedding", "rotary_table", "silu_timestep_embedding"]
        if turbo_grid is not None
        else ["timestep_embedding", "rotary_table"],
        {
            "timesteps": {0: "timestep_count"},
            "position_ids": {0: "sequence"},
            "timestep_embedding": {0: "timestep_count"},
            "rotary_table": {1: "sequence"},
            **({"silu_timestep_embedding": {0: "timestep_count"}} if turbo_grid is not None else {}),
        },
    )
    actual_conditioning = _ort_run(
        conditioning_path,
        {"timesteps": timesteps.numpy(), "position_ids": position_ids.numpy()},
    )
    validation["conditioning"] = {
        "timestep_embedding": _metrics(timestep_embedding, actual_conditioning[0]),
        "rotary_table": _metrics(rotary_table, actual_conditioning[1]),
    }
    if silu_timestep_embedding is not None:
        validation["conditioning"]["silu_timestep_embedding"] = _metrics(
            silu_timestep_embedding, actual_conditioning[2]
        )
    _require_close("Main timestep curve", validation["conditioning"]["timestep_embedding"], 1e-5, 0.99999)
    _require_close("Main RoPE", validation["conditioning"]["rotary_table"], 2e-3, 0.9999)
    graphs.append(conditioning_path.name)
    del conditioning, actual_conditioning, reader
    gc.collect()

    for position, index in enumerate(blocks):
        start = 0.13 + 0.79 * position / max(1, len(blocks))
        _progress(callback, start, f"Dequantizing and exporting main block {index + 1}/50 streaming QKV")
        attention_qkv_path = output_dir / f"main_block_{index:02d}_attention_qkv.onnx"
        attention_inputs = {
            "hidden_states": hidden_states.numpy(),
            "timestep_embedding": timestep_embedding.numpy(),
            "modulation_ids": modulation_ids.numpy(),
            "rotary_table": rotary_table.numpy(),
            **(
                {"silu_timestep_embedding": silu_timestep_embedding.numpy()}
                if silu_timestep_embedding is not None
                else {}
            ),
        }
        expected_qkv = _export_main_shard(
            "dit_attention_qkv",
            index,
            path,
            attention_qkv_path,
            attention_inputs,
            lora_path,
            lora_strength,
        )
        actual_qkv = _ort_run(attention_qkv_path, attention_inputs)[0]
        qkv_metrics = _metrics(expected_qkv, actual_qkv)
        # fp32 reference vs fp16 attention GEMM internals: absolute noise is
        # fp16 half-ulp at QKV magnitude; relative_l2 (~2^-11) is the real gate.
        _require_close(
            f"Main block {index} streaming QKV",
            qkv_metrics,
            max_abs=64.0,
            min_cosine=0.999,
            max_relative_l2=2e-3,
        )
        query, key, value = expected_qkv.chunk(3, dim=-1)
        sequence = query.shape[0]
        query_heads = query.reshape(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0)
        key_heads = key.reshape(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0)
        value_heads = value.reshape(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(query_heads.float(), key_heads.float().transpose(-2, -1)) * (HEAD_DIM**-0.5)
        attended = torch.matmul(torch.softmax(scores, dim=-1).to(value_heads.dtype), value_heads)
        attended = attended.transpose(1, 2).reshape(sequence, ATTENTION_INNER)

        attention_output_path = output_dir / f"main_block_{index:02d}_attention_output.onnx"
        expected_attention = _export_main_shard(
            "dit_attention_output",
            index,
            path,
            attention_output_path,
            {
                "hidden_states": hidden_states.numpy(),
                "attended": attended.numpy(),
                "timestep_embedding": timestep_embedding.numpy(),
                "modulation_ids": modulation_ids.numpy(),
                **(
                    {"silu_timestep_embedding": silu_timestep_embedding.numpy()}
                    if silu_timestep_embedding is not None
                    else {}
                ),
            },
            lora_path,
            lora_strength,
        )
        actual_attention = _ort_run(
            attention_output_path,
            {
                "hidden_states": hidden_states.numpy(),
                "attended": attended.numpy(),
                "timestep_embedding": timestep_embedding.numpy(),
                "modulation_ids": modulation_ids.numpy(),
                **(
                    {"silu_timestep_embedding": silu_timestep_embedding.numpy()}
                    if silu_timestep_embedding is not None
                    else {}
                ),
            },
        )[0]
        attention_output_metrics = _metrics(expected_attention, actual_attention)
        _require_close(
            f"Main block {index} streaming output",
            attention_output_metrics,
            max_abs=256.0,
            min_cosine=0.999,
            max_relative_l2=2e-3,
        )
        graphs.extend((attention_qkv_path.name, attention_output_path.name))

        _progress(callback, start + 0.39 / max(1, len(blocks)), f"Exporting main block {index + 1}/50 MLP")
        mlp_path = output_dir / f"main_block_{index:02d}_mlp.onnx"
        expected_mlp = _export_main_shard(
            "dit_mlp",
            index,
            path,
            mlp_path,
            {
                "hidden_states": expected_attention.numpy(),
                "timestep_embedding": timestep_embedding.numpy(),
                "modulation_ids": modulation_ids.numpy(),
                **(
                    {"silu_timestep_embedding": silu_timestep_embedding.numpy()}
                    if silu_timestep_embedding is not None
                    else {}
                ),
            },
            lora_path,
            lora_strength,
            gpu_native_fp16=False,
        )
        actual_mlp = _ort_run(
            mlp_path,
            {
                "hidden_states": expected_attention.numpy(),
                "timestep_embedding": timestep_embedding.numpy(),
                "modulation_ids": modulation_ids.numpy(),
                **(
                    {"silu_timestep_embedding": silu_timestep_embedding.numpy()}
                    if silu_timestep_embedding is not None
                    else {}
                ),
            },
        )[0]
        mlp_metrics = _metrics(expected_mlp, actual_mlp)
        _require_close(
            f"Main block {index} MLP",
            mlp_metrics,
            max_abs=256.0,
            min_cosine=0.999,
            max_relative_l2=2e-3,
        )
        validation["blocks"][str(index)] = {
            "attention_qkv": qkv_metrics,
            "attention_output": attention_output_metrics,
            "mlp": mlp_metrics,
        }
        graphs.append(mlp_path.name)
        del expected_qkv, actual_qkv, expected_attention, actual_attention, expected_mlp, actual_mlp
        gc.collect()

    _progress(callback, 0.94, "Exporting main video/audio output head")
    reader = CheckpointReader(path, lora_path, lora_strength)
    head = MainHead(reader).eval()
    head_path = output_dir / "main_head.onnx"
    video_hidden = hidden_states[:1]
    audio_hidden = hidden_states[1:2]
    head_silu = silu_timestep_embedding if silu_timestep_embedding is not None else None
    with torch.inference_mode():
        expected_head = head(
            video_hidden,
            audio_hidden,
            timestep_embedding,
            timestep_embedding,
            head_silu,
            head_silu,
        )
    head_inputs = (video_hidden, audio_hidden, timestep_embedding, timestep_embedding)
    head_input_names = [
        "video_hidden",
        "audio_hidden",
        "video_timestep_embedding",
        "audio_timestep_embedding",
    ]
    if head_silu is not None:
        head_inputs += (head_silu, head_silu)
        head_input_names += ["video_silu_timestep_embedding", "audio_silu_timestep_embedding"]
    _export_graph(
        head,
        head_inputs,
        head_path,
        head_input_names,
        ["video_patches", "audio_patches"],
        {
            "video_hidden": {0: "video_sequence"},
            "audio_hidden": {0: "audio_sequence"},
            "video_patches": {0: "video_sequence"},
            "audio_patches": {0: "audio_sequence"},
            **(
                {
                    "video_silu_timestep_embedding": {0: "timestep_count"},
                    "audio_silu_timestep_embedding": {0: "timestep_count"},
                }
                if head_silu is not None
                else {}
            ),
        },
    )
    actual_head = _ort_run(
        head_path,
        {
            "video_hidden": video_hidden.numpy(),
            "audio_hidden": audio_hidden.numpy(),
            "video_timestep_embedding": timestep_embedding.numpy(),
            "audio_timestep_embedding": timestep_embedding.numpy(),
            **(
                {
                    "video_silu_timestep_embedding": head_silu.numpy(),
                    "audio_silu_timestep_embedding": head_silu.numpy(),
                }
                if head_silu is not None
                else {}
            ),
        },
    )
    validation["head"] = {
        "video": _metrics(expected_head[0], actual_head[0]),
        "audio": _metrics(expected_head[1], actual_head[1]),
    }
    for name, metrics in validation["head"].items():
        _require_close(f"Main {name} head", metrics, max_abs=2e-2, min_cosine=0.999)
    graphs.append(head_path.name)

    manifest = {
        "format": "h3-workbench-onnx-v1",
        "source": str(path.resolve()),
        "component": "fl2va_transformer",
        "source_quantization": "float8_e4m3fn_scaled",
        "conversion": "fp8_dequant_fp16_storage_hybrid_fp16_attention_fp32_mlp",
        "activation_dtype": "mixed_fp16_attention_fp32_mlp",
        "attention": {
            "format": "streaming_qkv_output",
            "kernel": "runtime_sdpa",
            "query_chunk_tokens": 256,
        },
        "ignored_source_parameters": ["input_scale"],
        "opset": OPSET_VERSION,
        "architecture": {
            "hidden_size": 5376,
            "layers": 50,
            "heads": 56,
            "head_dim": 128,
            "ffn_size": 14336,
            "curve_dim": 8,
        },
        "profiles": {"smoke_sequence": 3, "smoke_timestep_count": 1},
        "dynamic_dimensions": ["sequence", "video_sequence", "audio_sequence", "text_sequence", "timestep_count"],
        "blocks": blocks,
        "graphs": graphs,
        "validation": validation,
        "validation_passed": True,
    }
    if lora_path is not None:
        acceleration = inspect_acceleration_lora(lora_path, lora_strength)
        manifest["acceleration"] = {
            **acceleration.to_dict(),
            "recommended_steps": [4, 5, 6, 7, 8],
            "default_steps": 4,
            "backbone_application": "static_fp16_merge",
            "pruned_adaln_application": "runtime_silu_temb_grid_injection",
            "coverage": f"{acceleration.tensor_pairs}/{acceleration.tensor_pairs}",
        }
    _write_manifest(output_dir, manifest)
    _progress(callback, 1.0, "Main model sharded export and validation completed")
    return manifest


def export_qwen(
    path: Path,
    output_dir: Path,
    block_spec: str = "all",
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    blocks = _parse_blocks(block_spec, 50)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_quantization = QwenCheckpointReader(path).source_quantization
    if source_quantization in {"int8_per_channel", "int8_tensorwise_convrot"}:
        if blocks != list(range(50)):
            raise ValueError("Qwen INT8 virtual slicing always validates all 50 layers")
        return _export_qwen_int8_virtual(path, output_dir, callback)

    token_ids = torch.tensor([1, 42, 1000, 151935], dtype=torch.int64)
    sequence = token_ids.shape[0]
    positions = torch.arange(sequence, dtype=torch.float32)
    frequencies = 1.0 / (500_000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128.0))
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    attention_mask = torch.full((1, 1, sequence, sequence), -10_000.0, dtype=torch.float32)
    attention_mask = torch.triu(attention_mask, diagonal=1)

    graphs: list[str] = []
    validation: dict[str, Any] = {"blocks": {}}
    _progress(callback, 0.01, "Exporting Qwen INT8 embedding")
    embedding_path = output_dir / "qwen_embedding.onnx"
    expected_embedding = _export_qwen_shard(
        "embedding",
        0,
        path,
        embedding_path,
        {"token_ids": token_ids.numpy()},
    )[0]
    actual_embedding = _ort_run(embedding_path, {"token_ids": token_ids.numpy()})[0]
    validation["embedding"] = _metrics(expected_embedding, actual_embedding)
    _require_close("Qwen embedding", validation["embedding"], 1e-5, 0.99999, 1e-6)
    hidden_states = expected_embedding
    graphs.append(embedding_path.name)

    for position, index in enumerate(blocks):
        progress = 0.05 + 0.88 * position / max(1, len(blocks))
        layer_validation: dict[str, Any] = {}

        _progress(callback, progress, f"Exporting Qwen layer {index + 1}/50 attention")
        attention_path = output_dir / f"qwen_layer_{index:02d}_attention.onnx"
        expected_attention = _export_qwen_shard(
            "attention",
            index,
            path,
            attention_path,
            {
                "hidden_states": hidden_states.numpy(),
                "cosine": cosine.numpy(),
                "sine": sine.numpy(),
                "attention_mask": attention_mask.numpy(),
            },
        )[0]
        actual_attention = _ort_run(
            attention_path,
            {
                "hidden_states": hidden_states.numpy(),
                "cosine": cosine.numpy(),
                "sine": sine.numpy(),
                "attention_mask": attention_mask.numpy(),
            },
        )[0]
        layer_validation["attention"] = _metrics(expected_attention, actual_attention)
        _require_close(f"Qwen layer {index} attention", layer_validation["attention"], 1.0, 0.999, 2e-3)
        graphs.append(attention_path.name)

        _progress(callback, progress + 0.22 / max(1, len(blocks)), f"Exporting Qwen layer {index + 1}/50 gate")
        gate_path = output_dir / f"qwen_layer_{index:02d}_gate.onnx"
        expected_normalized, expected_gate = _export_qwen_shard(
            "gate",
            index,
            path,
            gate_path,
            {"hidden_states": expected_attention.numpy()},
        )
        actual_normalized, actual_gate = _ort_run(gate_path, {"hidden_states": expected_attention.numpy()})
        layer_validation["norm"] = _metrics(expected_normalized, actual_normalized)
        layer_validation["gate"] = _metrics(expected_gate, actual_gate)
        _require_close(f"Qwen layer {index} norm", layer_validation["norm"], 1e-4, 0.9999, 1e-4)
        _require_close(f"Qwen layer {index} gate", layer_validation["gate"], 1.0, 0.999, 2e-3)
        graphs.append(gate_path.name)

        _progress(callback, progress + 0.44 / max(1, len(blocks)), f"Exporting Qwen layer {index + 1}/50 up")
        up_path = output_dir / f"qwen_layer_{index:02d}_up.onnx"
        expected_up = _export_qwen_shard(
            "up",
            index,
            path,
            up_path,
            {"normalized_states": expected_normalized.numpy()},
        )[0]
        actual_up = _ort_run(up_path, {"normalized_states": expected_normalized.numpy()})[0]
        layer_validation["up"] = _metrics(expected_up, actual_up)
        _require_close(f"Qwen layer {index} up", layer_validation["up"], 1.0, 0.999, 2e-3)
        graphs.append(up_path.name)

        _progress(callback, progress + 0.66 / max(1, len(blocks)), f"Exporting Qwen layer {index + 1}/50 down")
        down_path = output_dir / f"qwen_layer_{index:02d}_down.onnx"
        expected_down = _export_qwen_shard(
            "down",
            index,
            path,
            down_path,
            {
                "hidden_states": expected_attention.numpy(),
                "gate": expected_gate.numpy(),
                "up": expected_up.numpy(),
            },
        )[0]
        actual_down = _ort_run(
            down_path,
            {
                "hidden_states": expected_attention.numpy(),
                "gate": expected_gate.numpy(),
                "up": expected_up.numpy(),
            },
        )[0]
        layer_validation["down"] = _metrics(expected_down, actual_down)
        _require_close(f"Qwen layer {index} down", layer_validation["down"], 1.0, 0.999, 2e-3)
        graphs.append(down_path.name)
        validation["blocks"][str(index)] = layer_validation
        hidden_states = expected_down

    manifest = {
        "format": "h3-workbench-onnx-v1",
        "source": str(path.resolve()),
        "component": "text_encoder",
        "source_quantization": source_quantization,
        "conversion": f"{source_quantization}_dequant_fp16_storage_gpu_native_fp16_gemm",
        "activation_dtype": "float32",
        "architecture": {
            "hidden_size": 5120,
            "layers": 50,
            "heads": 64,
            "kv_heads": 8,
            "head_dim": 128,
            "intermediate_size": 25600,
        },
        "rope_inputs": "external_cosine_sine",
        "text_only": True,
        "blocks": blocks,
        "graphs": graphs,
        "validation": validation,
        "validation_passed": True,
    }
    _write_manifest(output_dir, manifest)
    _progress(callback, 1.0, "Qwen text-tower export and validation completed")
    return manifest


def _export_qwen_int8_virtual(
    path: Path,
    output_dir: Path,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build and validate the compact runtime topology for an INT8 Qwen source."""
    from h3_workbench.qwen_attention_benchmark import ATTENTION_WEIGHT_TO_SOURCE, _inputs
    from h3_workbench.qwen_int8_graph import build_int8_qdq_graph, build_weight_input_topology
    from h3_workbench.qwen_virtual_slicer import build_virtual_qwen_product
    from h3_workbench.qwen_virtual_validation import validate_virtual_qwen_product

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "validation.jsonl"
    with tempfile.TemporaryDirectory(prefix="h3-qwen-topology-", dir=output_dir.parent) as temporary_name:
        temporary = Path(temporary_name)
        hidden = np.random.default_rng(19).standard_normal((4, 5120), dtype=np.float32) * 0.25

        _progress(callback, 0.04, "Building Qwen attention reference graph")
        attention_fp16 = temporary / "qwen_layer_00_attention.onnx"
        _export_qwen_shard("attention", 0, path, attention_fp16, _inputs(4, 0))
        attention_qdq = temporary / "qwen_layer_00_attention_int8.onnx"
        build_int8_qdq_graph(
            attention_fp16,
            path,
            attention_qdq,
            block=0,
            weight_to_source=ATTENTION_WEIGHT_TO_SOURCE,
        )
        attention_names = {"input_norm", "q_norm", "k_norm"} | {
            name
            for weight in ATTENTION_WEIGHT_TO_SOURCE
            for name in (f"{weight}.int8", f"{weight}.scale")
        }
        attention_topology = temporary / "runtime_qwen_attention_int8.onnx"
        build_weight_input_topology(attention_qdq, attention_topology, attention_names)

        _progress(callback, 0.28, "Building fused Qwen MLP reference graph")
        mlp_fp16 = temporary / "qwen_layer_00_mlp.onnx"
        _export_qwen_shard("mlp", 0, path, mlp_fp16, {"hidden_states": hidden.astype(np.float32)})
        mlp_qdq = temporary / "qwen_layer_00_mlp_int8.onnx"
        build_int8_qdq_graph(mlp_fp16, path, mlp_qdq, block=0)
        mlp_names = {
            "norm",
            "gate.linear.weight.int8",
            "gate.linear.weight.scale",
            "up.linear.weight.int8",
            "up.linear.weight.scale",
            "down.linear.weight.int8",
            "down.linear.weight.scale",
        }
        mlp_topology = temporary / "runtime_qwen_mlp_int8.onnx"
        build_weight_input_topology(mlp_qdq, mlp_topology, mlp_names)

        _progress(callback, 0.52, "Publishing zero-copy Qwen virtual slices")
        build_virtual_qwen_product(path, output_dir, attention_topology, mlp_topology)

    _progress(callback, 0.58, "Validating Qwen layers 1, 25, and 50")
    validation = validate_virtual_qwen_product(
        output_dir,
        log_path,
        blocks=(0, 24, 49),
        tokens=4,
        relative_l2_max=1e-3,
        run_full_chain=True,
    )
    if not validation.get("validation_passed"):
        raise RuntimeError(f"Qwen virtual validation failed: {validation}")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    _progress(callback, 1.0, "Qwen virtual slicing and validation completed")
    return manifest


def export_checkpoint(
    path: Path,
    output_dir: Path,
    video_blocks: str = "all",
    callback: ProgressCallback | None = None,
    lora_path: Path | None = None,
    lora_strength: float = 1.0,
) -> dict[str, Any]:
    path = path.resolve()
    workspace = path.parent
    record = inspect_checkpoint(path, workspace)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if record.component == "audio_vae":
        return export_audio(path, output_dir, callback)
    if record.component == "video_vae":
        return export_video(path, output_dir, video_blocks, callback)
    if record.component in {"fl2va_transformer", "ref2va_transformer"}:
        return export_main(path, output_dir, video_blocks, callback, lora_path, lora_strength)
    if record.component == "text_encoder":
        return export_qwen(path, output_dir, video_blocks, callback)
    raise ValueError(f"No exporter is available for component '{record.component}'")


def _main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3 ONNX exporter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("workspace", nargs="?", type=Path, default=Path.cwd())
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("checkpoint", type=Path)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--video-blocks", "--main-blocks", dest="video_blocks", default="all")
    export_parser.add_argument("--lora", type=Path)
    export_parser.add_argument("--lora-strength", type=float, default=1.0)
    long_video_parser = subparsers.add_parser("export-video-long")
    long_video_parser.add_argument("checkpoint", type=Path)
    long_video_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "inspect":
        print(json.dumps([item.to_dict() for item in scan_models(args.workspace.resolve())], ensure_ascii=False, indent=2))
        return

    started = time.monotonic()
    if args.command == "export-video-long":
        result = export_video_long_temporal_graphs(
            args.checkpoint.resolve(),
            args.output.resolve(),
            lambda value, message: print(f"[{value:6.1%}] {message}", flush=True),
        )
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = export_checkpoint(
        args.checkpoint,
        args.output,
        args.video_blocks,
        lambda value, message: print(f"[{value:6.1%}] {message}", flush=True),
        args.lora,
        args.lora_strength,
    )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
