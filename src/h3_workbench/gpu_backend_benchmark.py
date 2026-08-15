from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from h3_workbench.gpu_toolchain import cuda_graph_eligibility, probe_gpu_toolchain
from h3_workbench.device_profile import selected_device_index
from h3_workbench.inference_runtime import _preload_cuda_dlls
from h3_workbench.main_benchmark import _gpu_sample


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    expected32 = expected.astype(np.float32, copy=False)
    actual32 = actual.astype(np.float32, copy=False)
    difference = actual32 - expected32
    denominator = max(float(np.linalg.norm(expected32.reshape(-1))), np.finfo(np.float32).tiny)
    return {
        "finite": bool(np.isfinite(actual32).all()),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / denominator),
    }


def _session(
    model: Path,
    provider: str,
    provider_options: dict[str, str] | None = None,
) -> tuple[ort.InferenceSession, float]:
    _preload_cuda_dlls()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    providers = [provider]
    provider_options_list = [provider_options or {}]
    if provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        provider_options_list[0].setdefault("device_id", str(selected_device_index()))
    if provider == "TensorrtExecutionProvider":
        providers.append("CUDAExecutionProvider")
        provider_options_list.append({"arena_extend_strategy": "kSameAsRequested"})
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=providers,
        provider_options=provider_options_list,
    )
    if provider not in session.get_providers():
        raise RuntimeError(f"ONNX Runtime rejected {provider}; active providers={session.get_providers()}")
    return session, time.perf_counter() - started


def _regular_cuda(
    model: Path,
    feeds: dict[str, np.ndarray],
    expected: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    session, build_seconds = _session(
        model,
        "CUDAExecutionProvider",
        {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "0"},
    )
    output = session.get_outputs()[0].name
    cold_started = time.perf_counter()
    actual = session.run([output], feeds)[0]
    cold_seconds = time.perf_counter() - cold_started
    warm_seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        actual = session.run([output], feeds)[0]
        warm_seconds.append(time.perf_counter() - started)
    return {
        "available": True,
        "providers": session.get_providers(),
        "build_seconds": round(build_seconds, 6),
        "cold_seconds": round(cold_seconds, 6),
        "warm_seconds": [round(value, 6) for value in warm_seconds],
        "warm_median_seconds": round(float(np.median(warm_seconds)), 6),
        "metrics": _metrics(expected, actual),
        "gpu": _gpu_sample(),
    }


def _cuda_graph(
    model: Path,
    feeds: dict[str, np.ndarray],
    expected: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    eligible, reason = cuda_graph_eligibility(stable_device_inputs=True, fixed_shapes=True)
    if not eligible:
        return {"available": False, "reason": reason}
    session, build_seconds = _session(
        model,
        "CUDAExecutionProvider",
        {
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": "1",
            "use_ep_level_unified_stream": "1",
        },
    )
    binding = session.io_binding()
    device_index = selected_device_index()
    device_inputs = {
        name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", device_index)
        for name, value in feeds.items()
    }
    for name, value in device_inputs.items():
        binding.bind_ortvalue_input(name, value)
    output_name = session.get_outputs()[0].name
    device_output = ort.OrtValue.ortvalue_from_shape_and_type(
        expected.shape, expected.dtype, "cuda", device_index
    )
    binding.bind_ortvalue_output(output_name, device_output)
    run_options = ort.RunOptions()
    run_options.add_run_config_entry("gpu_graph_id", "0")
    cold_started = time.perf_counter()
    session.run_with_iobinding(binding, run_options)
    cold_seconds = time.perf_counter() - cold_started
    warm_seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run_with_iobinding(binding, run_options)
        warm_seconds.append(time.perf_counter() - started)
    actual = device_output.numpy()
    return {
        "available": True,
        "providers": session.get_providers(),
        "build_seconds": round(build_seconds, 6),
        "capture_seconds": round(cold_seconds, 6),
        "replay_seconds": [round(value, 6) for value in warm_seconds],
        "replay_median_seconds": round(float(np.median(warm_seconds)), 6),
        "metrics": _metrics(expected, actual),
        "gpu": _gpu_sample(),
    }


def _tensorrt(
    model: Path,
    feeds: dict[str, np.ndarray],
    expected: np.ndarray,
    repeats: int,
    cache: Path,
    *,
    fp16: bool,
) -> dict[str, Any]:
    status = probe_gpu_toolchain()
    if not status.tensorrt_available:
        return {
            "available": False,
            "reason": "TensorRT runtime is not installed; install the tensorrt extra from an NVIDIA library source",
        }
    cache = cache / ("fp16" if fp16 else "fp32")
    cache.mkdir(parents=True, exist_ok=True)
    session, build_seconds = _session(
        model,
        "TensorrtExecutionProvider",
        {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(cache),
            "trt_fp16_enable": str(fp16),
            "trt_builder_optimization_level": "3",
        },
    )
    if "TensorrtExecutionProvider" not in session.get_providers():
        return {"available": False, "reason": "ONNX Runtime rejected the TensorRT provider"}
    output = session.get_outputs()[0].name
    cold_started = time.perf_counter()
    actual = session.run([output], feeds)[0]
    cold_seconds = time.perf_counter() - cold_started
    warm_seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        actual = session.run([output], feeds)[0]
        warm_seconds.append(time.perf_counter() - started)
    return {
        "available": True,
        "precision": "fp16" if fp16 else "fp32",
        "providers": session.get_providers(),
        "build_seconds": round(build_seconds, 6),
        "cold_seconds": round(cold_seconds, 6),
        "warm_median_seconds": round(float(np.median(warm_seconds)), 6),
        "metrics": _metrics(expected, actual),
        "gpu": _gpu_sample(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.inputs, allow_pickle=False) as archive:
        feeds = {name: archive[name].copy() for name in archive.files}
    with np.load(args.expected, allow_pickle=False) as archive:
        expected = archive[archive.files[0]].copy()
    result: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "inputs": str(args.inputs.resolve()),
        "expected": str(args.expected.resolve()),
        "toolchain": probe_gpu_toolchain().to_dict(),
        "cuda": _regular_cuda(args.model, feeds, expected, args.repeats),
    }
    for name, benchmark in (
        ("cuda_graph", lambda: _cuda_graph(args.model, feeds, expected, args.repeats)),
        (
            "tensorrt_fp16",
            lambda: _tensorrt(args.model, feeds, expected, args.repeats, args.cache, fp16=True),
        ),
        (
            "tensorrt_fp32",
            lambda: _tensorrt(args.model, feeds, expected, args.repeats, args.cache, fp16=False),
        ),
    ):
        try:
            result[name] = benchmark()
        except Exception as exc:  # noqa: BLE001 - preserve backend failure in benchmark output
            result[name] = {"available": False, "reason": str(exc)}
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one validated block across GPU backends")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".h3-workbench/cache/tensorrt"))
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    payload = run(args)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, args.output)
    print(serialized)


if __name__ == "__main__":
    main()
