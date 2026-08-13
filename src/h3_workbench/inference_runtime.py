from __future__ import annotations

import gc
import ctypes
import json
import math
import os
import sys
import time
from collections.abc import Callable
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as torch_functional

from h3_workbench.acceleration import shifted_flow_sigmas
from h3_workbench.fl2va_runtime_graphs import (
    fp16_attention_output_ready,
    is_attention_output_graph,
    runtime_attention_output_graph,
    source_graph_name,
)
from h3_workbench.memory_planner import (
    main_model_shards,
    plan_shard_batches,
    plan_streaming_shard_batches,
    probe_gpu_memory,
    streaming_kv_bytes,
)
from h3_workbench.profiles import GenerationProfile, PROFILE_360P_17F
from h3_workbench.shard_cache import ShardPrefetchCache, default_prefetch_depth, graph_storage_bytes
from h3_workbench.qwen_persistent import QwenWeightInputs, persistent_qwen_ready

_DLL_DIRECTORY_HANDLES: list[object] = []
GIB = 1024**3


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    ]


def probe_host_commit_memory() -> tuple[int, int]:
    """Return current committed bytes and the system commit limit on Windows."""
    if os.name != "nt":
        return 0, 0
    information = _PerformanceInformation()
    information.cb = ctypes.sizeof(information)
    try:
        get_performance_info = ctypes.windll.psapi.GetPerformanceInfo
        if not get_performance_info(ctypes.byref(information), information.cb):
            return 0, 0
    except (AttributeError, OSError):
        return 0, 0
    page_size = int(information.PageSize)
    return int(information.CommitTotal) * page_size, int(information.CommitLimit) * page_size


def host_prefetch_budget_bytes(reserve_bytes: int = 4 * GIB) -> int | None:
    committed, limit = probe_host_commit_memory()
    if limit <= 0:
        return None
    return max(0, limit - committed - reserve_bytes)


def _preload_cuda_dlls() -> None:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        cuda_root = os.environ.get("CUDA_PATH")
        if cuda_root:
            cuda_bin = Path(cuda_root) / "bin"
            if cuda_bin.is_dir():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(cuda_bin)))
        site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        package_bins = [
            site_packages / "cublas" / "bin",
            site_packages / "cuda_nvrtc" / "bin",
            site_packages / "cudnn" / "bin",
        ]
        for directory in package_bins:
            if directory.is_dir():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        win_dll = getattr(ctypes, "WinDLL", None)
        tensor_ir = site_packages / "cudnn" / "bin" / "cudnn_engines_tensor_ir64_9.dll"
        if win_dll is not None and tensor_ir.is_file():
            try:
                win_dll(str(tensor_ir))
            except OSError:
                pass
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload(directory="")


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(latent: np.ndarray) -> np.ndarray:
    batch, channels, frames, height, width = latent.shape
    values = latent.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
    values = values.transpose(0, 2, 3, 5, 1, 4, 6)
    return values.reshape(batch * frames * (height // 2) * (width // 2), channels * 4)


def unpatchify_video(rows: np.ndarray, frames: int, height: int, width: int) -> np.ndarray:
    values = rows.reshape(1, frames, height // 2, width // 2, 24, 1, 2, 2)
    values = values.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return values.reshape(1, 24, frames, height, width)


def pack_audio(latent: np.ndarray) -> np.ndarray:
    return latent[0].transpose(1, 2, 0).reshape(-1, 32)


def unpack_audio(rows: np.ndarray) -> np.ndarray:
    frames = rows.shape[0] // 2
    return rows.reshape(2, frames, 32).transpose(2, 0, 1)[None]


def streamed_attention(packed: np.ndarray, use_cuda: bool, query_chunk_tokens: int = 256) -> np.ndarray:
    """Run exact full-sequence attention without materializing an N x N matrix."""
    if packed.ndim == 3 and packed.shape[1] == 56 and packed.shape[2] % 3 == 0:
        query, key, value = np.split(packed, 3, axis=2)
        head_dim = query.shape[2]
        query = query.reshape(query.shape[0], -1)
        key = key.reshape(key.shape[0], -1)
        value = value.reshape(value.shape[0], -1)
    elif packed.ndim == 2 and packed.shape[1] % 3 == 0:
        width = packed.shape[1] // 3
        query, key, value = np.split(packed, 3, axis=1)
        head_dim = width // 56
    else:
        raise ValueError(f"Unexpected packed QKV shape: {packed.shape}")
    width = query.shape[1]
    if width % 56:
        raise ValueError(f"Packed QKV width is not divisible by 56 heads: {width}")
    head_dim = width // 56
    device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    key_tensor = torch.from_numpy(key).view(-1, 56, head_dim).transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    value_tensor = torch.from_numpy(value).view(-1, 56, head_dim).transpose(0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    output = np.empty((query.shape[0], width), dtype=np.float32)
    try:
        for start in range(0, query.shape[0], query_chunk_tokens):
            stop = min(start + query_chunk_tokens, query.shape[0])
            query_tensor = torch.from_numpy(query[start:stop]).view(-1, 56, head_dim).transpose(0, 1).unsqueeze(0)
            query_tensor = query_tensor.to(device=device, dtype=dtype)
            attended = torch_functional.scaled_dot_product_attention(query_tensor, key_tensor, value_tensor)
            rows = attended.squeeze(0).transpose(0, 1).reshape(stop - start, width)
            output[start:stop] = rows.to(device="cpu", dtype=torch.float32).numpy()
            del query_tensor, attended, rows
    finally:
        del key_tensor, value_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def _ensure_streamed_sdpa_graph(path: Path) -> Path:
    if path.is_file():
        return path
    from onnx import TensorProto, helper

    query = helper.make_tensor_value_info("query", TensorProto.FLOAT16, [1, 56, "query_tokens", 128])
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT16, [1, 56, "sequence_tokens", 128])
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT16, [1, 56, "sequence_tokens", 128])
    score_scale = helper.make_tensor_value_info("score_scale", TensorProto.FLOAT, [])
    value_scale = helper.make_tensor_value_info("value_scale", TensorProto.FLOAT, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["query_tokens", 56, 128])
    shape = helper.make_tensor("output_shape", TensorProto.INT64, [3], [-1, 56, 128])
    nodes = [
        helper.make_node("Transpose", ["key"], ["key_transposed"], perm=[0, 1, 3, 2]),
        # Q and K arrive normalized to a bounded range. Their FP16 MatMul
        # therefore cannot overflow; restore the original score scale in FP32
        # before Softmax. Probabilities and normalized V are both bounded, so
        # their product is safe in FP16; restore the original V scale in FP32.
        helper.make_node("MatMul", ["query", "key_transposed"], ["normalized_scores_fp16"]),
        helper.make_node("Cast", ["normalized_scores_fp16"], ["normalized_scores"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["normalized_scores", "score_scale"], ["scaled_scores"]),
        helper.make_node("Softmax", ["scaled_scores"], ["probabilities"], axis=3),
        helper.make_node("Cast", ["probabilities"], ["probabilities_fp16"], to=TensorProto.FLOAT16),
        helper.make_node("MatMul", ["probabilities_fp16", "value"], ["normalized_attended_fp16"]),
        helper.make_node("Cast", ["normalized_attended_fp16"], ["normalized_attended"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["normalized_attended", "value_scale"], ["attended"]),
        helper.make_node("Transpose", ["attended"], ["rows"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["rows", "output_shape"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "h3_streamed_sdpa_normalized",
        [query, key, value, score_scale, value_scale],
        [output],
        [shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


class ORTStreamingAttention:
    """CUDA SDPA fallback that works when the installed PyTorch wheel is CPU-only."""

    def __init__(self, directory: Path, runner: "ORTGraphRunner"):
        self.runner = runner
        self.graph_path = _ensure_streamed_sdpa_graph(
            directory / "runtime_streamed_sdpa_fp16_normalized_fp32_softmax_fp16_value.onnx"
        )

    @staticmethod
    def _normalize(values: np.ndarray, limit: float = 8.0) -> tuple[np.ndarray, float]:
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        scale = max(1.0, maximum / limit)
        return np.ascontiguousarray(values / scale, dtype=np.float16), scale

    def __call__(
        self,
        packed: np.ndarray,
        query_chunk_tokens: int = 256,
        output_dtype: np.dtype | type[np.floating] = np.float32,
    ) -> np.ndarray:
        # The long-sequence score workspace can consume most of a 4 GB GPU. A
        # persistent session retains that CUDA arena and slows all later block
        # graphs through WDDM shared-memory migration.
        session = self.runner.session(self.graph_path)
        if packed.ndim == 3 and packed.shape[1] == 56 and packed.shape[2] % 3 == 0:
            rows = packed
        elif packed.ndim == 2 and packed.shape[1] % 3 == 0:
            rows = packed.reshape(packed.shape[0], 56, packed.shape[1] // 56)
        else:
            raise ValueError(f"Unexpected packed QKV shape: {packed.shape}")
        head_width = rows.shape[2] // 3
        if head_width != 128:
            raise ValueError(f"Expected 128-wide Q/K/V heads, got {head_width}")
        key, key_scale = self._normalize(rows[:, :, head_width : 2 * head_width].transpose(1, 0, 2)[None])
        key_value = ort.OrtValue.ortvalue_from_numpy(key, "cuda", 0)
        del key
        value, value_scale = self._normalize(rows[:, :, 2 * head_width :].transpose(1, 0, 2)[None])
        value_value = ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
        del value
        output = np.empty((rows.shape[0], 56 * head_width), dtype=np.float32)
        try:
            for start in range(0, rows.shape[0], query_chunk_tokens):
                stop = min(start + query_chunk_tokens, rows.shape[0])
                query, query_scale = self._normalize(rows[start:stop, :, :head_width].transpose(1, 0, 2)[None])
                io_binding = session.io_binding()
                io_binding.bind_ortvalue_input("query", ort.OrtValue.ortvalue_from_numpy(query, "cuda", 0))
                io_binding.bind_ortvalue_input("key", key_value)
                io_binding.bind_ortvalue_input("value", value_value)
                io_binding.bind_cpu_input(
                    "score_scale",
                    np.asarray(query_scale * key_scale / math.sqrt(head_width), dtype=np.float32),
                )
                io_binding.bind_cpu_input("value_scale", np.asarray(value_scale, dtype=np.float32))
                target = output[start:stop].reshape(stop - start, 56, head_width)
                io_binding.bind_output("output", "cpu", 0, np.float32, target.shape, target.ctypes.data)
                session.run_with_iobinding(io_binding)
                del query, io_binding
        finally:
            del key_value, value_value, session
            gc.collect()
        return output.astype(output_dtype, copy=False)


def _axis_from_sqrt_area(dimension: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dimension / sqrt_area
    count = dimension // patch
    return (np.arange(count, dtype=np.float64) * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0


def _finite_guard(label: str, values: np.ndarray) -> None:
    # Numerical checks are cheap relative to a main-model shard and must remain
    # active when the server is launched directly rather than through the BAT.
    if os.environ.get("H3_VALIDATE_FINITE", "1") != "0" and not np.isfinite(values).all():
        invalid = int((~np.isfinite(values)).sum())
        raise FloatingPointError(f"Non-finite {label}: {invalid} invalid values")


def select_fl2va_chunk_sizes(free_vram_bytes: int, dynamic: bool = True) -> dict[str, int]:
    if not dynamic or free_vram_bytes <= 0:
        return {"qkv": 256, "attention_output": 256, "mlp": 256}
    gib = 1024**3
    if free_vram_bytes >= int(2.75 * gib):
        return {"qkv": 1024, "attention_output": 2048, "mlp": 512}
    if free_vram_bytes >= int(2.25 * gib):
        return {"qkv": 1024, "attention_output": 1024, "mlp": 256}
    if free_vram_bytes >= int(1.75 * gib):
        return {"qkv": 512, "attention_output": 512, "mlp": 256}
    return {"qkv": 256, "attention_output": 256, "mlp": 256}


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ("out of memory", "cuda failure 2", "cuda error 2", "failed to allocate"))


def packed_position_ids(profile: GenerationProfile, text_tokens: int) -> np.ndarray:
    latent_h, latent_w = profile.video_latent_height, profile.video_latent_width
    area = math.sqrt(latent_h * latent_w)
    height_axis = _axis_from_sqrt_area(latent_h, 2, area)
    width_axis = _axis_from_sqrt_area(latent_w, 2, area)
    hh, ww = np.meshgrid(height_axis, width_axis, indexing="ij")
    frame = np.stack((hh.reshape(-1), ww.reshape(-1)), axis=-1)

    text = np.zeros((text_tokens, 3), dtype=np.float32)
    text[:, 0] = np.arange(text_tokens, dtype=np.float32)
    cursor = float(text_tokens)
    audio_frames = profile.audio_latent_frames
    audio = np.zeros((audio_frames * 2, 3), dtype=np.float32)
    audio[:, 0] = np.tile(cursor + np.arange(audio_frames, dtype=np.float32), 2)
    audio[:audio_frames, 2] = width_axis[0]
    audio[audio_frames:, 2] = width_axis[-1]

    frame_pattern = np.asarray([1.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float32)
    spans = np.resize(frame_pattern, profile.video_latent_frames) * (5.0 / 3.0)
    frame_times = cursor + np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(spans[:-1])))
    video = np.empty((profile.video_latent_frames, frame.shape[0], 3), dtype=np.float32)
    video[:, :, 0] = frame_times[:, None]
    video[:, :, 1:] = frame[None]
    return np.concatenate((text, audio, video.reshape(-1, 3)), axis=0)


def modulation_ids(profile: GenerationProfile, text_tokens: int, sigma_video: float) -> tuple[np.ndarray, np.ndarray]:
    sigma_audio = time_shift_sigma(sigma_video, 12.0, 3.0)
    time_video, time_audio = 1.0 - sigma_video, 1.0 - sigma_audio
    unique_times = np.asarray(sorted({time_video, time_audio}), dtype=np.float32)
    video_row = int(np.argmin(np.abs(unique_times - time_video))) * 3
    audio_row = int(np.argmin(np.abs(unique_times - time_audio))) * 3
    ids = np.concatenate(
        (
            np.full(text_tokens, video_row + 1, dtype=np.int64),
            np.full(profile.audio_tokens, audio_row + 2, dtype=np.int64),
            np.full(profile.video_tokens, video_row, dtype=np.int64),
        )
    )
    return unique_times, ids


class ORTGraphRunner:
    def __init__(
        self,
        prefer_cuda: bool = True,
        l2_cache_bytes: int | None = None,
        prefetch_depth: int | None = None,
    ):
        providers = ort.get_available_providers()
        self.provider = "CUDAExecutionProvider" if prefer_cuda and "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
        if prefetch_depth is None:
            prefetch_depth = default_prefetch_depth()
        self.shard_cache = ShardPrefetchCache(l2_cache_bytes, prefetch_depth)
        self._l1_prefetch_hits = 0
        self._l1_prefetch_waits = 0
        self._l1_prefetch_wait_seconds = 0.0
        if self.provider == "CUDAExecutionProvider":
            _preload_cuda_dlls()

    def session(self, path: Path | None = None, serialized_model: bytes | None = None) -> ort.InferenceSession:
        if path is None and serialized_model is None:
            raise ValueError("Either path or serialized_model is required")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        provider_options = [{}]
        if self.provider == "CUDAExecutionProvider":
            provider_options = [{"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "0"}]
        session = ort.InferenceSession(
            serialized_model if serialized_model is not None else str(path),
            sess_options=options,
            providers=[self.provider],
            provider_options=provider_options,
        )
        if self.provider == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "ONNX Runtime silently fell back to CPU. Check CUDA 12, cuDNN 9, and the MSVC runtime."
            )
        return session

    def run(self, path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        session = self.session(path)
        try:
            return session.run(None, inputs)
        finally:
            del session
            gc.collect()

    def adaptive_session_batches(
        self,
        groups: list[list[Path]],
        before_batch: Callable[[list[Path]], None] | None = None,
        loading_callback: Callable[[dict[str, object]], None] | None = None,
        session_prefetch_depth: int = 0,
        session_prefetch_budget_bytes: int = 0,
        prefetch_barrier: Callable[[list[Path]], bool] | None = None,
        dynamic_group_planner: Callable[[list[Path]], list[list[Path]]] | None = None,
    ) -> Iterator[list[tuple[Path, ort.InferenceSession]]]:
        if session_prefetch_depth > 0 and self.provider == "CUDAExecutionProvider":
            yield from self._prefetched_session_batches(
                groups,
                before_batch,
                loading_callback,
                session_prefetch_depth,
                session_prefetch_budget_bytes,
                prefetch_barrier,
                dynamic_group_planner,
            )
            return

        queue = list(groups)
        while queue:
            if dynamic_group_planner is not None:
                remaining = [path for group in queue for path in group]
                queue = dynamic_group_planner(remaining)
            paths = queue.pop(0)
            upcoming = [path for group in queue for path in group]
            if loading_callback is not None:
                loading_callback({"operation": "L2 prefetch", "path": paths[0], "batch_size": len(paths)})
            self.shard_cache.stage([*paths, *upcoming])
            if before_batch is not None:
                before_batch(paths)
            for index, path in enumerate(paths):
                wait_started = time.perf_counter()
                self.shard_cache.wait(path)
                if loading_callback is not None:
                    loading_callback(
                        {
                            "operation": "L2 shard ready",
                            "path": path,
                            "batch_index": index + 1,
                            "batch_size": len(paths),
                            "elapsed_seconds": round(time.perf_counter() - wait_started, 3),
                            **self.cache_stats(),
                        }
                    )
            sessions: list[tuple[Path, ort.InferenceSession]] = []
            try:
                for index, path in enumerate(paths):
                    if loading_callback is not None:
                        loading_callback(
                            {
                                "operation": "Building CUDA session",
                                "path": path,
                                "batch_index": index + 1,
                                "batch_size": len(paths),
                            }
                        )
                    session_started = time.perf_counter()
                    sessions.append((path, self.session(path)))
                    if loading_callback is not None:
                        loading_callback(
                            {
                                "operation": "CUDA session ready",
                                "path": path,
                                "batch_index": index + 1,
                                "batch_size": len(paths),
                                "elapsed_seconds": round(time.perf_counter() - session_started, 3),
                            }
                        )
            except Exception:
                del sessions
                gc.collect()
                if self.provider != "CUDAExecutionProvider" or len(paths) <= 1:
                    raise
                split = max(1, len(paths) // 2)
                queue.insert(0, paths[split:])
                queue.insert(0, paths[:split])
                continue
            try:
                yield sessions
            finally:
                del sessions
                gc.collect()

    def _load_session_batch(
        self,
        paths: list[Path],
        loading_callback: Callable[[dict[str, object]], None] | None,
        prefetch_ahead: int,
    ) -> list[tuple[Path, ort.InferenceSession]]:
        sessions: list[tuple[Path, ort.InferenceSession]] = []
        try:
            for index, path in enumerate(paths):
                wait_started = time.perf_counter()
                self.shard_cache.wait(path)
                if loading_callback is not None:
                    loading_callback(
                        {
                            "operation": "L2 shard ready",
                            "path": path,
                            "batch_index": index + 1,
                            "batch_size": len(paths),
                            "prefetch_ahead": prefetch_ahead,
                            "elapsed_seconds": round(time.perf_counter() - wait_started, 3),
                        }
                    )
                    loading_callback(
                        {
                            "operation": "L1 prefetch loading",
                            "path": path,
                            "batch_index": index + 1,
                            "batch_size": len(paths),
                            "prefetch_ahead": prefetch_ahead,
                        }
                    )
                session_started = time.perf_counter()
                # The L2 cache keeps the graph and external data mapped and hot.
                # Let ORT parse them once directly; parsing to protobuf bytes here
                # would make the CPU deserialize and serialize hundreds of MiB
                # before ORT immediately deserializes the same model again.
                sessions.append((path, self.session(path)))
                if loading_callback is not None:
                    loading_callback(
                        {
                            "operation": "L1 prefetch ready",
                            "path": path,
                            "batch_index": index + 1,
                            "batch_size": len(paths),
                            "prefetch_ahead": prefetch_ahead,
                            "elapsed_seconds": round(time.perf_counter() - session_started, 3),
                        }
                    )
            return sessions
        except Exception:
            del sessions
            gc.collect()
            raise

    def _prefetched_session_batches(
        self,
        groups: list[list[Path]],
        before_batch: Callable[[list[Path]], None] | None,
        loading_callback: Callable[[dict[str, object]], None] | None,
        depth: int,
        budget_bytes: int,
        prefetch_barrier: Callable[[list[Path]], bool] | None,
        dynamic_group_planner: Callable[[list[Path]], list[list[Path]]] | None,
    ) -> Iterator[list[tuple[Path, ort.InferenceSession]]]:
        queue = list(groups)
        pending: list[tuple[list[Path], int, Future[list[tuple[Path, ort.InferenceSession]]]]] = []
        resident_bytes = 0
        # Session construction is mostly protobuf parsing, CUDA kernel setup,
        # and weight upload. Two or three independent builders let those CPU
        # phases overlap the active CUDA graph instead of serializing all gaps.
        build_workers = max(1, min(depth + 1, 3))
        executor = ThreadPoolExecutor(max_workers=build_workers, thread_name_prefix="h3-l1-prefetch")

        def estimate(paths: list[Path]) -> int:
            return max(1, int(sum(graph_storage_bytes(path) for path in paths) * 1.2))

        def enqueue(paths: list[Path], estimated_bytes: int) -> None:
            nonlocal resident_bytes
            ahead = len(pending)
            if loading_callback is not None:
                loading_callback(
                    {
                        "operation": "L1 prefetch queued",
                        "path": paths[0],
                        "batch_size": len(paths),
                        "prefetch_ahead": ahead,
                        "estimated_resident_bytes": estimated_bytes,
                        "l1_prefetch_budget_bytes": budget_bytes,
                    }
                )
            future = executor.submit(self._load_session_batch, paths, loading_callback, ahead)
            pending.append((paths, estimated_bytes, future))
            resident_bytes += estimated_bytes

        def fill() -> None:
            nonlocal queue, resident_bytes
            # Advance the RAM window once per scheduler pass. Calling stage()
            # once for every queued L1 future makes adjacent builders evict and
            # restage one another's mmap before either can consume it.
            if dynamic_group_planner is not None and queue:
                remaining = [path for group in queue for path in group]
                queue = dynamic_group_planner(remaining)
            staged_paths = [path for paths, _, _ in pending for path in paths]
            staged_paths.extend(path for group in queue for path in group)
            self.shard_cache.stage(staged_paths)
            while queue and len(pending) < depth + 1:
                paths = queue[0]
                # A barrier must run with every earlier and prefetched session
                # released. FL2VA uses this between QKV and its full-sequence
                # SDPA workspace.
                if prefetch_barrier is not None and prefetch_barrier(paths):
                    break
                estimated_bytes = estimate(paths)
                if pending and budget_bytes > 0 and resident_bytes + estimated_bytes > budget_bytes:
                    break
                queue.pop(0)
                enqueue(paths, estimated_bytes)

        try:
            fill()
            while pending or queue:
                if not pending:
                    paths = queue.pop(0)
                    is_barrier = prefetch_barrier is not None and prefetch_barrier(paths)
                    if before_batch is not None:
                        before_batch(paths)
                    enqueue(paths, estimate(paths))
                    if not is_barrier:
                        fill()
                paths, estimated_bytes, future = pending.pop(0)
                ready = future.done()
                wait_started = time.perf_counter()
                sessions = future.result()
                wait_seconds = time.perf_counter() - wait_started
                if ready:
                    self._l1_prefetch_hits += 1
                else:
                    self._l1_prefetch_waits += 1
                    self._l1_prefetch_wait_seconds += wait_seconds
                if loading_callback is not None:
                    loading_callback(
                        {
                            "operation": "L1 prefetch hit" if ready else "L1 prefetch wait",
                            "path": paths[0],
                            "batch_size": len(paths),
                            "overlap_hit": ready,
                            "wait_seconds": round(wait_seconds, 3),
                            "prefetched_batches": len(pending),
                            "estimated_resident_bytes": resident_bytes,
                            "l1_prefetch_budget_bytes": budget_bytes,
                        }
                    )
                is_barrier = prefetch_barrier is not None and prefetch_barrier(paths)
                if before_batch is not None and not is_barrier:
                    before_batch(paths)
                if not is_barrier:
                    fill()
                try:
                    yield sessions
                finally:
                    del sessions
                    resident_bytes -= estimated_bytes
                    gc.collect()
                    fill()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for _, _, future in pending:
                if future.cancel() or not future.done():
                    continue
                try:
                    sessions = future.result()
                except Exception:
                    continue
                del sessions
            gc.collect()

    def cache_stats(self) -> dict[str, int | float]:
        return {
            **self.shard_cache.snapshot(),
            "l1_prefetch_hits": self._l1_prefetch_hits,
            "l1_prefetch_waits": self._l1_prefetch_waits,
            "l1_prefetch_wait_seconds": round(self._l1_prefetch_wait_seconds, 3),
        }

    def close(self) -> None:
        self.shard_cache.close()


class QwenTextRuntime:
    def __init__(self, directory: Path, runner: ORTGraphRunner, l1_prefetch_shards: int = 2):
        self.directory = directory
        self.runner = runner
        if not 0 <= l1_prefetch_shards <= 4:
            raise ValueError("l1_prefetch_shards must be from 0 to 4")
        self.l1_prefetch_shards = l1_prefetch_shards
        self.persistent = QwenWeightInputs(directory) if persistent_qwen_ready(directory) else None

    def _encode_persistent(
        self,
        token_ids: np.ndarray,
        callback: Callable[[str, int, int], None] | None,
    ) -> np.ndarray:
        assert self.persistent is not None
        kinds = ("embedding", "attention", "gate", "up", "down")
        sessions = {kind: self.runner.session(self.persistent.graph(kind)) for kind in kinds}
        try:
            hidden = sessions["embedding"].run(
                None, {"token_ids": token_ids.astype(np.int64), **self.persistent.inputs("embedding")}
            )[0]
            sequence = token_ids.shape[0]
            positions = np.arange(sequence, dtype=np.float32)
            inv_freq = 1.0 / (500_000.0 ** (np.arange(0, 128, 2, dtype=np.float32) / 128.0))
            angles = np.outer(positions, inv_freq)
            angles = np.concatenate((angles, angles), axis=-1)
            cosine, sine = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
            mask = np.triu(np.full((1, 1, sequence, sequence), -10_000.0, dtype=np.float32), k=1)
            normalized: np.ndarray | None = None
            gate: np.ndarray | None = None
            up: np.ndarray | None = None
            for layer in range(50):
                if callback is not None:
                    callback("Attention", layer + 1, 50)
                hidden = sessions["attention"].run(
                    None,
                    {
                        "hidden_states": hidden,
                        "cosine": cosine,
                        "sine": sine,
                        "attention_mask": mask,
                        **self.persistent.inputs("attention", layer),
                    },
                )[0]
                if callback is not None:
                    callback("Gate", layer + 1, 50)
                normalized, gate = sessions["gate"].run(
                    None, {"hidden_states": hidden, **self.persistent.inputs("gate", layer)}
                )
                if callback is not None:
                    callback("MLP Up", layer + 1, 50)
                up = sessions["up"].run(
                    None, {"normalized_states": normalized, **self.persistent.inputs("up", layer)}
                )[0]
                if callback is not None:
                    callback("MLP Down", layer + 1, 50)
                hidden = sessions["down"].run(
                    None,
                    {
                        "hidden_states": hidden,
                        "gate": gate,
                        "up": up,
                        **self.persistent.inputs("down", layer),
                    },
                )[0]
            return hidden
        finally:
            sessions.clear()
            gc.collect()

    def encode_token_ids(
        self,
        token_ids: np.ndarray,
        callback: Callable[[str, int, int], None] | None = None,
        activity_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> np.ndarray:
        if self.persistent is not None and self.runner.provider == "CUDAExecutionProvider":
            return self._encode_persistent(token_ids, callback)
        if callback is not None:
            callback("Embedding", 0, 50)
        hidden = self.runner.run(self.directory / "qwen_embedding.onnx", {"token_ids": token_ids.astype(np.int64)})[0]
        sequence = token_ids.shape[0]
        positions = np.arange(sequence, dtype=np.float32)
        inv_freq = 1.0 / (500_000.0 ** (np.arange(0, 128, 2, dtype=np.float32) / 128.0))
        angles = np.outer(positions, inv_freq)
        angles = np.concatenate((angles, angles), axis=-1)
        cosine, sine = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        mask = np.triu(np.full((1, 1, sequence, sequence), -10_000.0, dtype=np.float32), k=1)
        graph_paths = [
            self.directory / f"qwen_layer_{index:02d}_{kind}.onnx"
            for index in range(50)
            for kind in ("attention", "gate", "up", "down")
        ]
        # Keep one transformer layer resident as a dependency-safe unit. This
        # gives the background builders the duration of four CUDA graphs to
        # prepare the next layer, while the VRAM budget still bounds how many
        # layers may be queued at once.
        groups = [graph_paths[index : index + 4] for index in range(0, len(graph_paths), 4)]
        normalized: np.ndarray | None = None
        gate: np.ndarray | None = None
        up: np.ndarray | None = None

        def loading_activity(details: dict[str, object]) -> None:
            if activity_callback is None:
                return
            path = Path(str(details.pop("path")))
            parts = path.stem.split("_")
            layer = int(parts[2]) + 1
            kind = " ".join(parts[3:]).replace("attention", "Attention").replace("gate", "Gate")
            kind = kind.replace("up", "MLP Up").replace("down", "MLP Down") or "Layer"
            operation = str(details.pop("operation"))
            activity_callback(
                {
                    "module": "Qwen Loader",
                    "operation": f"{operation}: {kind}",
                    "prefetch_layer": layer,
                    "prefetch_total": 50,
                    **details,
                    **self.runner.cache_stats(),
                }
            )

        snapshot = probe_gpu_memory()
        prefetch_budget = max(0, snapshot.free_bytes - 768 * 1024**2)
        host_budget = host_prefetch_budget_bytes()
        if host_budget is not None:
            prefetch_budget = min(prefetch_budget, host_budget)
        # Qwen session construction is storage/CPU bound on this machine and
        # did not produce L1 hits. It remains opt-in for faster host systems.
        l1_enabled = os.environ.get("H3_QWEN_L1_PREFETCH", "0") == "1"
        prefetch_depth = self.l1_prefetch_shards if prefetch_budget > 0 and l1_enabled else 0
        for session_batch in self.runner.adaptive_session_batches(
            groups,
            loading_callback=loading_activity,
            session_prefetch_depth=prefetch_depth,
            session_prefetch_budget_bytes=prefetch_budget,
        ):
            for path, session in session_batch:
                layer = int(path.stem.split("_")[2]) + 1
                if path.name.endswith("_attention.onnx"):
                    if callback is not None:
                        callback("Attention", layer, 50)
                    hidden = session.run(
                        None,
                        {"hidden_states": hidden, "cosine": cosine, "sine": sine, "attention_mask": mask},
                    )[0]
                elif path.name.endswith("_gate.onnx"):
                    if callback is not None:
                        callback("Gate", layer, 50)
                    normalized, gate = session.run(None, {"hidden_states": hidden})
                elif path.name.endswith("_up.onnx"):
                    if normalized is None:
                        raise RuntimeError("Qwen MLP Up encountered without normalized states")
                    if callback is not None:
                        callback("MLP Up", layer, 50)
                    up = session.run(None, {"normalized_states": normalized})[0]
                else:
                    if gate is None or up is None:
                        raise RuntimeError("Qwen MLP Down encountered without gate/up states")
                    if callback is not None:
                        callback("MLP Down", layer, 50)
                    hidden = session.run(None, {"hidden_states": hidden, "gate": gate, "up": up})[0]
                    normalized = gate = up = None
        return hidden


class H3MainRuntime:
    def __init__(
        self,
        directory: Path,
        runner: ORTGraphRunner,
        profile: GenerationProfile = PROFILE_360P_17F,
        attention_query_chunk: int = 128,
        activity_callback: Callable[[dict[str, object]], None] | None = None,
        l1_prefetch_shards: int = 2,
    ):
        self.directory = directory
        self.runner = runner
        self.profile = profile
        if attention_query_chunk not in {32, 64, 128, 256, 512}:
            raise ValueError("attention_query_chunk must be one of 32, 64, 128, 256, or 512")
        self.attention_query_chunk = attention_query_chunk
        if not 0 <= l1_prefetch_shards <= 4:
            raise ValueError("l1_prefetch_shards must be from 0 to 4")
        self.l1_prefetch_shards = l1_prefetch_shards
        self.activity_callback = activity_callback
        self.sampling_step = 0
        self.sampling_steps = 0
        self.audio_fallback_reason: str | None = None
        self.dynamic_chunks = os.environ.get("H3_FL2VA_DYNAMIC_CHUNKS", "1") != "0"
        self.chunk_io_binding = os.environ.get("H3_FL2VA_IO_BINDING", "1") != "0"
        self.chunk_sizes = {"qkv": 256, "attention_output": 256, "mlp": 256}
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        self.streaming_attention = manifest.get("attention", {}).get("format") == "streaming_qkv_output"
        self.turbo_adaln = (
            manifest.get("acceleration", {}).get("pruned_adaln_application")
            == "runtime_silu_temb_grid_injection"
        )
        self.fp16_attention_output = (
            self.streaming_attention
            and runner.provider == "CUDAExecutionProvider"
            and fp16_attention_output_ready(directory)
        )
        self.ort_streamed_attention = (
            ORTStreamingAttention(directory, runner)
            if self.streaming_attention and runner.provider == "CUDAExecutionProvider"
            else None
        )

    def report_activity(
        self,
        module: str,
        operation: str,
        current: int | None = None,
        total: int | None = None,
        **details: object,
    ) -> None:
        if self.activity_callback is None:
            return
        payload: dict[str, object] = {"module": module, "operation": operation, **details}
        if current is not None:
            payload["current"] = current
        if total is not None:
            payload["total"] = total
        if self.sampling_step:
            payload["sampling_step"] = self.sampling_step
            payload["sampling_steps"] = self.sampling_steps
        self.activity_callback(payload)

    def prepare_text(self, text_states: np.ndarray) -> np.ndarray:
        self.report_activity("FL2VA Token Refiner", "Input projection")
        dummy_video = np.zeros((1, 96), dtype=np.float32)
        dummy_audio = np.zeros((1, 32), dtype=np.float32)
        _, _, hidden = self.runner.run(
            self.directory / "main_embeddings.onnx",
            {"video_patches": dummy_video, "audio_patches": dummy_audio, "text_states": text_states.astype(np.float16)},
        )
        for index in range(2):
            self.report_activity("FL2VA Token Refiner", "Attention", index + 1, 2)
            hidden = self.runner.run(
                self.directory / f"main_token_refiner_block_{index:02d}_attention.onnx",
                {"hidden_states": hidden},
            )[0]
            self.report_activity("FL2VA Token Refiner", "MLP", index + 1, 2)
            hidden = self.runner.run(
                self.directory / f"main_token_refiner_block_{index:02d}_mlp.onnx",
                {"hidden_states": hidden},
            )[0]
        self.report_activity("FL2VA Token Refiner", "Final norm")
        return self.runner.run(self.directory / "main_token_refiner_norm.onnx", {"hidden_states": hidden})[0]

    def denoise_step(
        self,
        video_latent: np.ndarray,
        audio_latent: np.ndarray,
        text_states: np.ndarray,
        sigma_video: float,
        text_is_refined: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        text_hidden = text_states if text_is_refined else self.prepare_text(text_states)
        self.report_activity("FL2VA", "Input embeddings")
        video_rows = patchify_video(video_latent.astype(np.float32))
        audio_rows = pack_audio(audio_latent.astype(np.float32))
        video_hidden, audio_hidden, _ = self.runner.run(
            self.directory / "main_embeddings.onnx",
            {
                "video_patches": video_rows,
                "audio_patches": audio_rows,
                "text_states": np.zeros((1, 5120), dtype=np.float16),
            },
        )
        hidden = np.concatenate((text_hidden, audio_hidden, video_hidden), axis=0).astype(np.float32)
        times, mod_ids = modulation_ids(self.profile, text_hidden.shape[0], sigma_video)
        positions = packed_position_ids(self.profile, text_hidden.shape[0])
        self.report_activity("FL2VA", "Timestep and RoPE")
        conditioning_outputs = self.runner.run(
            self.directory / "main_conditioning.onnx",
            {"timesteps": times, "position_ids": positions},
        )
        timestep_embedding, rotary = conditioning_outputs[:2]
        silu_timestep_embedding = conditioning_outputs[2] if self.turbo_adaln else None

        snapshot = probe_gpu_memory()
        latest_vram_free = snapshot.free_bytes
        self.chunk_sizes = select_fl2va_chunk_sizes(snapshot.free_bytes, self.dynamic_chunks)
        shards = main_model_shards(self.directory)
        batches = plan_shard_batches(shards, self.profile, snapshot.free_bytes) if snapshot.free_bytes else []
        groups = [[runtime_attention_output_graph(self.directory, name)] for name, _ in shards]
        if self.streaming_attention:
            # A QKV barrier prevents unsafe reordering. Up to three adjacent
            # dependency-safe sessions can otherwise share L1 on a 4 GB GPU;
            # the byte planner remains the final admission control.
            max_sessions = min(3, max(1, self.l1_prefetch_shards + 1))
            streaming_batches = plan_streaming_shard_batches(
                shards,
                self.profile,
                snapshot.free_bytes,
                max_sessions=max_sessions,
            )
            if streaming_batches and self.runner.provider == "CUDAExecutionProvider":
                groups = [
                    [runtime_attention_output_graph(self.directory, item.graph) for item in batch.shards]
                    for batch in streaming_batches
                ]
        elif batches and self.runner.provider == "CUDAExecutionProvider":
            groups = [[self.directory / item.graph for item in batch.shards] for batch in batches]
        attention_input: np.ndarray | None = None
        streaming_packed: np.ndarray | None = None
        attended: np.ndarray | None = None
        shard_total = len(shards)
        shard_current = 0

        def before_batch(paths: list[Path]) -> None:
            nonlocal attended
            if not self.streaming_attention or not is_attention_output_graph(paths[0]):
                return
            if attended is not None:
                return
            if attention_input is None or streaming_packed is None:
                raise RuntimeError("Streaming attention output encountered without QKV input")
            block = int(paths[0].name.split("_")[2]) + 1
            self.report_activity(
                "FL2VA",
                "Streaming SDPA",
                block,
                50,
                shard=shard_current + 1,
                shards=shard_total,
                vram_free_bytes=latest_vram_free,
                **self.runner.cache_stats(),
            )
            # No ORT session is resident here, leaving the maximum space for K/V.
            if self.ort_streamed_attention is not None:
                attended = self.ort_streamed_attention(
                    streaming_packed,
                    self.attention_query_chunk,
                    output_dtype=np.float32,
                )
            else:
                attended = streamed_attention(
                    streaming_packed,
                    self.runner.provider == "CUDAExecutionProvider",
                    self.attention_query_chunk,
                )
            _finite_guard(f"block {block} SDPA", attended)

        shard_positions = {name: index + 1 for index, (name, _) in enumerate(shards)}

        def loading_activity(details: dict[str, object]) -> None:
            path = Path(str(details.pop("path")))
            shard = shard_positions.get(source_graph_name(path), shard_current + 1)
            block = int(path.name.split("_")[2]) + 1
            operation = str(details.pop("operation"))
            runtime_details = {**details, **self.runner.cache_stats()}
            self.report_activity(
                "FL2VA Loader",
                operation,
                block,
                50,
                shard=shard,
                shards=shard_total,
                vram_free_bytes=latest_vram_free,
                **runtime_details,
            )

        attention_reserve = (
            streaming_kv_bytes(self.profile)
            if self.streaming_attention
            else self.profile.attention_workspace_bytes
        )
        prefetch_budget = max(0, snapshot.free_bytes - 768 * 1024**2 - attention_reserve)
        host_budget = host_prefetch_budget_bytes()
        if host_budget is not None:
            prefetch_budget = min(prefetch_budget, host_budget)
        # The native 15s trace recorded zero L1 hits and repeated waits. Keep
        # active attention buffers in VRAM and retain the effective L2 mmap
        # cache; session prefetch remains available as an explicit opt-in.
        l1_enabled = os.environ.get("H3_FL2VA_L1_PREFETCH", "0") == "1"
        prefetch_depth = self.l1_prefetch_shards if prefetch_budget > 0 and l1_enabled else 0

        def dynamic_group_planner(paths: list[Path]) -> list[list[Path]]:
            """Re-plan only unexecuted graphs against the current VRAM watermark."""
            nonlocal latest_vram_free
            current_snapshot = probe_gpu_memory()
            latest_vram_free = current_snapshot.free_bytes
            usable = max(1, latest_vram_free - 768 * 1024**2 - attention_reserve)
            max_sessions = min(3, max(1, self.l1_prefetch_shards + 1)) if self.streaming_attention else len(paths)
            planned: list[list[Path]] = []
            current: list[Path] = []
            current_bytes = 0

            def flush() -> None:
                nonlocal current, current_bytes
                if current:
                    planned.append(current)
                    current = []
                    current_bytes = 0

            for path in paths:
                estimated = max(1, int(graph_storage_bytes(path) * 1.2))
                if self.streaming_attention and is_attention_output_graph(path):
                    flush()
                    planned.append([path])
                    continue
                if current and (len(current) >= max_sessions or current_bytes + estimated > usable):
                    flush()
                current.append(path)
                current_bytes += estimated
                # QKV must finish before the full-sequence SDPA barrier.
                if self.streaming_attention and path.name.endswith("_attention_qkv.onnx"):
                    flush()
            flush()
            return planned

        for session_batch in self.runner.adaptive_session_batches(
            groups,
            before_batch,
            loading_activity,
            session_prefetch_depth=prefetch_depth,
            session_prefetch_budget_bytes=prefetch_budget,
            prefetch_barrier=lambda paths: is_attention_output_graph(paths[0]),
            dynamic_group_planner=dynamic_group_planner if self.runner.provider == "CUDAExecutionProvider" else None,
        ):
            try:
                l1_details: dict[str, object] = {
                    "l1_sessions": len(session_batch),
                    "l1_weight_bytes": sum(graph_storage_bytes(path) for path, _ in session_batch),
                    "vram_free_bytes": latest_vram_free,
                    "qkv_chunk_tokens": self.chunk_sizes["qkv"],
                    "attention_output_chunk_tokens": self.chunk_sizes["attention_output"],
                    "mlp_chunk_tokens": self.chunk_sizes["mlp"],
                    "chunk_io_binding": self.chunk_io_binding and self.runner.provider == "CUDAExecutionProvider",
                    "attention_buffer_dtype": "fp32_scaled" if self.fp16_attention_output else "fp32",
                    "qkv_buffer_dtype": "fp32" if self._streaming_qkv_dtype() == np.float32 else "fp16",
                    "l1_prefetch_enabled": l1_enabled,
                    "host_prefetch_budget_bytes": host_budget,
                    **self.runner.cache_stats(),
                }
                for graph_path, session in session_batch:
                    shard_current += 1
                    block = int(graph_path.name.split("_")[2]) + 1
                    if graph_path.name.endswith("_attention_qkv.onnx"):
                        self.report_activity(
                            "FL2VA",
                            "Attention QKV",
                            block,
                            50,
                            shard=shard_current,
                            shards=shard_total,
                            **l1_details,
                        )
                        attention_input = hidden
                        streaming_packed = self._run_streaming_qkv(
                            session,
                            hidden,
                            timestep_embedding,
                            mod_ids,
                            rotary,
                            self.chunk_sizes["qkv"],
                            silu_timestep_embedding,
                        )
                        _finite_guard(f"block {block} QKV", streaming_packed)
                    elif is_attention_output_graph(graph_path):
                        if attention_input is None or streaming_packed is None or attended is None:
                            raise RuntimeError("Streaming attention output encountered without QKV input")
                        self.report_activity(
                            "FL2VA",
                            "Attention output",
                            block,
                            50,
                            shard=shard_current,
                            shards=shard_total,
                            **l1_details,
                        )
                        hidden = self._run_streaming_output(
                            session,
                            attention_input,
                            attended,
                            timestep_embedding,
                            mod_ids,
                            self.chunk_sizes["attention_output"],
                            silu_timestep_embedding,
                        )
                        _finite_guard(f"block {block} attention output", hidden)
                        attention_input = None
                        streaming_packed = None
                        attended = None
                    elif graph_path.name.endswith("_attention.onnx"):
                        self.report_activity(
                            "FL2VA", "Attention", block, 50, shard=shard_current, shards=shard_total, **l1_details
                        )
                        hidden = session.run(
                            None,
                            {
                                "hidden_states": hidden.astype(np.float32, copy=False),
                                "timestep_embedding": timestep_embedding,
                                "modulation_ids": mod_ids,
                                "rotary_table": rotary,
                                **(
                                    {"silu_timestep_embedding": silu_timestep_embedding}
                                    if silu_timestep_embedding is not None
                                    else {}
                                ),
                            },
                        )[0]
                    elif graph_path.name.endswith("_mlp.onnx") and self.streaming_attention:
                        self.report_activity(
                            "FL2VA", "MLP", block, 50, shard=shard_current, shards=shard_total, **l1_details
                        )
                        hidden = self._run_chunked_graph(
                            session,
                            {
                                "hidden_states": hidden,
                                "timestep_embedding": timestep_embedding,
                                "modulation_ids": mod_ids,
                                **(
                                    {"silu_timestep_embedding": silu_timestep_embedding}
                                    if silu_timestep_embedding is not None
                                    else {}
                                ),
                            },
                            self.chunk_sizes["mlp"],
                        )
                        _finite_guard(f"block {block} MLP", hidden)
                    else:
                        self.report_activity(
                            "FL2VA", "MLP", block, 50, shard=shard_current, shards=shard_total, **l1_details
                        )
                        hidden = session.run(
                            None,
                            {
                                "hidden_states": hidden,
                                "timestep_embedding": timestep_embedding,
                                "modulation_ids": mod_ids,
                                **(
                                    {"silu_timestep_embedding": silu_timestep_embedding}
                                    if silu_timestep_embedding is not None
                                    else {}
                                ),
                            },
                        )[0]
                        _finite_guard(f"block {block} MLP", hidden)
            finally:
                # Drop the caller's references before the scheduler advances to
                # an attention memory barrier and starts the next CUDA session.
                session = None
                session_batch.clear()
                del session_batch
                gc.collect()

        text_count = text_hidden.shape[0]
        audio_start = text_count
        video_start = audio_start + self.profile.audio_tokens
        sigma_audio = time_shift_sigma(sigma_video, 12.0, 3.0)
        time_video, time_audio = 1.0 - sigma_video, 1.0 - sigma_audio
        video_t = timestep_embedding[int(np.argmin(np.abs(times - time_video))) :][:1]
        audio_t = timestep_embedding[int(np.argmin(np.abs(times - time_audio))) :][:1]
        video_silu_t = (
            silu_timestep_embedding[int(np.argmin(np.abs(times - time_video))) :][:1]
            if silu_timestep_embedding is not None
            else None
        )
        audio_silu_t = (
            silu_timestep_embedding[int(np.argmin(np.abs(times - time_audio))) :][:1]
            if silu_timestep_embedding is not None
            else None
        )
        self.report_activity("FL2VA", "Video/audio output head")
        video_out, audio_out = self.runner.run(
            self.directory / "main_head.onnx",
            {
                "video_hidden": hidden[video_start:].astype(np.float32, copy=False),
                "audio_hidden": hidden[audio_start:video_start].astype(np.float32, copy=False),
                "video_timestep_embedding": video_t,
                "audio_timestep_embedding": audio_t,
                **(
                    {
                        "video_silu_timestep_embedding": video_silu_t,
                        "audio_silu_timestep_embedding": audio_silu_t,
                    }
                    if video_silu_t is not None and audio_silu_t is not None
                    else {}
                ),
            },
        )
        _finite_guard("main head video patches", video_out)
        _finite_guard("main head audio patches", audio_out)
        video_velocity = -unpatchify_video(
            video_out,
            self.profile.video_latent_frames,
            self.profile.video_latent_height,
            self.profile.video_latent_width,
        )
        audio_velocity = -unpack_audio(audio_out)
        return video_velocity, audio_velocity

    def _streaming_qkv_dtype(self) -> np.dtype:
        # A long sequence can contain values outside FP16's finite range before
        # SDPA normalizes them. Turbo AdaLN increases that risk at the initial
        # high-noise step, so do not downcast this correctness-critical buffer.
        if self.turbo_adaln or self.profile.sequence_tokens > 4_096:
            return np.dtype(np.float32)
        return np.dtype(np.float16)

    def _run_streaming_qkv(
        self,
        session: ort.InferenceSession,
        hidden: np.ndarray,
        timestep_embedding: np.ndarray,
        modulation_ids: np.ndarray,
        rotary: np.ndarray,
        chunk_tokens: int,
        silu_timestep_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._run_chunked_session(
            session,
            hidden.shape[0],
            (56, 384),
            chunk_tokens,
            "qkv",
            lambda start, stop: {
                "hidden_states": hidden[start:stop],
                "timestep_embedding": timestep_embedding,
                "modulation_ids": modulation_ids[start:stop],
                "rotary_table": rotary[:, start:stop],
                **(
                    {"silu_timestep_embedding": silu_timestep_embedding}
                    if silu_timestep_embedding is not None
                    else {}
                ),
            },
            output_dtype=self._streaming_qkv_dtype(),
        )

    def _run_streaming_output(
        self,
        session: ort.InferenceSession,
        hidden: np.ndarray,
        attended: np.ndarray,
        timestep_embedding: np.ndarray,
        modulation_ids: np.ndarray,
        chunk_tokens: int,
        silu_timestep_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        attended_scratch = None
        if attended.dtype != np.dtype(np.float32) and not getattr(self, "fp16_attention_output", False):
            attended_scratch = np.empty(
                (min(hidden.shape[0], max(256, chunk_tokens)), attended.shape[1]),
                dtype=np.float32,
            )

        def chunk_inputs(start: int, stop: int) -> dict[str, np.ndarray]:
            attended_chunk = attended[start:stop]
            if attended_scratch is not None:
                attended_chunk = attended_scratch[: stop - start]
                np.copyto(attended_chunk, attended[start:stop], casting="unsafe")
            return {
                "hidden_states": hidden[start:stop],
                "attended": attended_chunk,
                "timestep_embedding": timestep_embedding,
                "modulation_ids": modulation_ids[start:stop],
                **(
                    {"silu_timestep_embedding": silu_timestep_embedding}
                    if silu_timestep_embedding is not None
                    else {}
                ),
            }

        return self._run_chunked_session(
            session,
            hidden.shape[0],
            (hidden.shape[1],),
            chunk_tokens,
            "attention_output",
            chunk_inputs,
        )

    def _run_chunked_graph(
        self,
        session: ort.InferenceSession,
        inputs: dict[str, np.ndarray],
        chunk_tokens: int,
    ) -> np.ndarray:
        hidden = inputs["hidden_states"]

        def chunk_inputs(start: int, stop: int) -> dict[str, np.ndarray]:
            chunk_inputs = dict(inputs)
            chunk_inputs["hidden_states"] = hidden[start:stop]
            chunk_inputs["modulation_ids"] = inputs["modulation_ids"][start:stop]
            return chunk_inputs

        return self._run_chunked_session(
            session,
            hidden.shape[0],
            (hidden.shape[1],),
            chunk_tokens,
            "mlp",
            chunk_inputs,
        )

    def _run_chunked_session(
        self,
        session: ort.InferenceSession,
        rows: int,
        output_tail: tuple[int, ...],
        chunk_tokens: int,
        chunk_key: str,
        inputs_for_chunk: Callable[[int, int], dict[str, np.ndarray]],
        output_dtype: np.dtype | type[np.floating] = np.float32,
    ) -> np.ndarray:
        current_chunk = max(256, chunk_tokens)
        while True:
            output = np.empty((rows, *output_tail), dtype=output_dtype)
            scratch = None
            if output.dtype != np.dtype(np.float32):
                scratch = np.empty((min(rows, current_chunk), *output_tail), dtype=np.float32)
            try:
                for start in range(0, rows, current_chunk):
                    stop = min(start + current_chunk, rows)
                    inputs = inputs_for_chunk(start, stop)
                    if self.chunk_io_binding and self.runner.provider == "CUDAExecutionProvider":
                        binding = session.io_binding()
                        for name, value in inputs.items():
                            binding.bind_cpu_input(name, value)
                        target = output[start:stop] if scratch is None else scratch[: stop - start]
                        binding.bind_output(
                            session.get_outputs()[0].name,
                            "cpu",
                            0,
                            np.float32,
                            target.shape,
                            target.ctypes.data,
                        )
                        session.run_with_iobinding(binding)
                        if scratch is not None:
                            output[start:stop] = target
                    else:
                        output[start:stop] = session.run(None, inputs)[0]
                return output
            except Exception as exc:
                if current_chunk <= 256 or not _is_cuda_oom(exc):
                    raise
                current_chunk = max(256, current_chunk // 2)
                self.chunk_sizes[chunk_key] = current_chunk
                gc.collect()


def initial_latents(profile: GenerationProfile, seed: int) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)
    video = random.standard_normal(
        (1, 24, profile.video_latent_frames, profile.video_latent_height, profile.video_latent_width),
        dtype=np.float32,
    )
    audio = random.standard_normal((1, 32, 2, profile.audio_latent_frames), dtype=np.float32)
    return video, audio


def sample_latents(
    runtime: H3MainRuntime,
    video: np.ndarray,
    audio: np.ndarray,
    text_states: np.ndarray,
    steps: int = 4,
    callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sigmas = shifted_flow_sigmas(steps)
    refined_text = runtime.prepare_text(text_states)
    for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
        runtime.sampling_step = index + 1
        runtime.sampling_steps = steps
        video_velocity, audio_velocity = runtime.denoise_step(
            video,
            audio,
            refined_text,
            sigma,
            text_is_refined=True,
        )
        if not np.isfinite(video_velocity).all():
            invalid = int((~np.isfinite(video_velocity)).sum())
            raise FloatingPointError(f"Non-finite FL2VA video velocity at step {index + 1}: {invalid} invalid values")
        audio_velocity_finite = np.isfinite(audio_velocity).all()
        if not audio_velocity_finite and runtime.audio_fallback_reason is None:
            invalid = int((~np.isfinite(audio_velocity)).sum())
            runtime.audio_fallback_reason = (
                f"Non-finite FL2VA audio velocity at step {index + 1}: {invalid} invalid values"
            )
        video_delta = sigma_next - sigma
        video = video + video_delta * video_velocity
        if audio_velocity_finite:
            audio_sigma = time_shift_sigma(sigma, 12.0, 3.0)
            audio_sigma_next = time_shift_sigma(sigma_next, 12.0, 3.0)
            next_audio = audio + (audio_sigma_next - audio_sigma) * audio_velocity
            if np.isfinite(next_audio).all():
                audio = next_audio
            elif runtime.audio_fallback_reason is None:
                invalid = int((~np.isfinite(next_audio)).sum())
                runtime.audio_fallback_reason = (
                    f"Non-finite FL2VA audio latent at step {index + 1}: {invalid} invalid values"
                )
        if not np.isfinite(video).all():
            invalid = int((~np.isfinite(video)).sum())
            raise FloatingPointError(f"Non-finite FL2VA video latent at step {index + 1}: {invalid} invalid values")
        if callback is not None:
            callback(index + 1, steps)
    return video, audio
