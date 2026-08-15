"""Numerical validation for published h3-schedule-v2 If shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from h3_workbench.device_profile import selected_device_index
from h3_workbench.inference_runtime import streamed_attention
from h3_workbench.shard_planner import file_sha256, validate_schedule

_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}
_DIMS = {
    "sequence": 3,
    "video_sequence": 1,
    "audio_sequence": 1,
    "text_sequence": 1,
    "timestep_count": 2,
}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _shape(argument: Any, context: dict[str, int]) -> tuple[int, ...]:
    return tuple(
        int(dimension)
        if isinstance(dimension, int) and dimension > 0
        else int(context.get(str(dimension), _DIMS.get(str(dimension), 1)))
        for dimension in argument.shape
    )


def _value(argument: Any, random: np.random.Generator, context: dict[str, int]) -> np.ndarray:
    dtype = _DTYPES[argument.type]
    shape = _shape(argument, context)
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)
    if argument.name == "timesteps":
        return np.asarray([0.2, 0.8], dtype=dtype)[: shape[0]]
    if argument.name == "rotary_table" and shape[-2:] == (2, 2):
        result = np.zeros(shape, dtype=dtype)
        result[..., 0, 0] = 1
        result[..., 1, 1] = 1
        return result
    return (random.standard_normal(shape) * 0.01).astype(dtype)


def _context(arguments: list[Any], feeds: dict[str, np.ndarray]) -> dict[str, int]:
    result = dict(_DIMS)
    by_name = {argument.name: argument for argument in arguments}
    for name, value in feeds.items():
        argument = by_name.get(name)
        if argument is None:
            continue
        for dimension, actual in zip(argument.shape, value.shape, strict=False):
            if isinstance(dimension, str):
                result[dimension] = int(actual)
    return result


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected32 = np.asarray(expected, dtype=np.float32)
    actual32 = np.asarray(actual, dtype=np.float32)
    difference = actual32 - expected32
    denominator = max(float(np.linalg.norm(expected32.reshape(-1))), 1e-12)
    return {
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / denominator),
    }


def _run_branch(
    session: ort.InferenceSession,
    shard: dict,
    graph: str,
    inputs: dict[str, np.ndarray],
    output_ports: list[str],
) -> list[np.ndarray]:
    arguments = session.get_inputs()
    context = _context(arguments, inputs)
    feeds = dict(inputs)
    selected = shard["graphs"].index(graph)
    random = np.random.default_rng(0)
    for argument in arguments:
        if argument.name.startswith("selector_"):
            feeds[argument.name] = np.asarray(selected, dtype=np.int64)
        elif argument.name.startswith("run_"):
            feeds[argument.name] = np.asarray(
                int(argument.name.rsplit("_", 1)[1]) == selected,
                dtype=np.bool_,
            )
        elif argument.name not in feeds:
            feeds[argument.name] = _value(argument, random, context)
    return session.run([f"{graph}/{port}" for port in output_ports], feeds)


def _session(path: Path, provider: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    # Validation executes each graph once. Constant-folding the fp8 dequant
    # chain costs ~15 seconds per shard on CPU, versus ~0.6 seconds when the
    # chain is executed once by CUDA as part of the selected branch.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = True
    providers = ["CUDAExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    provider_options = (
        [
            {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_use_max_workspace": "0",
                "device_id": str(selected_device_index()),
            }
        ]
        if provider == "cuda"
        else [{}]
    )
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=providers,
        provider_options=provider_options,
    )


def validate_sharded_model(model_dir: Path, provider: str = "auto") -> dict:
    model_dir = model_dir.resolve()
    if provider not in {"auto", "cpu", "cuda"}:
        raise ValueError("provider must be auto, cpu, or cuda")
    if provider == "auto":
        provider = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if provider == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    schedule = json.loads((model_dir / "schedule.json").read_text(encoding="utf-8"))
    validate_schedule(schedule, model_dir)
    steps_by_graph: dict[str, dict] = {}
    for step in schedule["steps"]:
        if step["kind"] == "graph":
            steps_by_graph.setdefault(step["graph"], step)
    random = np.random.default_rng(20260814)
    graph_results: dict[str, dict[str, float]] = {}
    chain_hidden = random.standard_normal((3, 5376)).astype(np.float32) * 0.01
    chain_timestep = random.standard_normal((2, 8)).astype(np.float32) * 0.01
    chain_modulation = np.zeros(3, dtype=np.int64)
    chain_rotary = np.zeros((1, 3, 1, 48, 2, 2), dtype=np.float16)
    chain_rotary[..., 0, 0] = 1
    chain_rotary[..., 1, 1] = 1
    block_maxima: dict[str, float] = {}
    pending_attended: np.ndarray | None = None

    for shard_index, shard in enumerate(schedule["shards"], 1):
        print(f"[validate] shard {shard_index}/{len(schedule['shards'])}", flush=True)
        cases: list[tuple[str, dict[str, np.ndarray], list[str], list[np.ndarray]]] = []
        for graph in shard["graphs"]:
            source_session = _session(model_dir / f"{graph}.onnx", provider)
            source_arguments = source_session.get_inputs()
            output_ports = [output.name for output in source_session.get_outputs()]
            context = dict(_DIMS)
            if graph.startswith("main_block_"):
                feeds: dict[str, np.ndarray] = {}
                for argument in source_arguments:
                    if argument.name == "hidden_states":
                        feeds[argument.name] = chain_hidden
                    elif argument.name == "timestep_embedding":
                        feeds[argument.name] = chain_timestep
                    elif argument.name == "modulation_ids":
                        feeds[argument.name] = chain_modulation
                    elif argument.name == "rotary_table":
                        feeds[argument.name] = chain_rotary
                    elif argument.name == "attended":
                        if not graph.endswith("_attention_output") or pending_attended is None:
                            raise RuntimeError("Only attention output consumes attended")
                        feeds[argument.name] = pending_attended
                    else:
                        feeds[argument.name] = _value(argument, random, context)
                expected = source_session.run(None, feeds)
                if graph.endswith("_attention_qkv"):
                    pending_attended = streamed_attention(
                        expected[0],
                        use_cuda=False,
                        query_chunk_tokens=3,
                    )
                elif graph.endswith("_attention_output"):
                    chain_hidden = expected[0]
                    pending_attended = None
                else:
                    chain_hidden = expected[0]
                    block = graph.split("_")[2]
                    block_maxima[block] = float(np.max(np.abs(chain_hidden)))
                    if not np.isfinite(chain_hidden).all():
                        raise FloatingPointError(f"Non-finite chained hidden after block {block}")
            else:
                feeds = {argument.name: _value(argument, random, context) for argument in source_arguments}
                expected = source_session.run(None, feeds)
            cases.append((graph, feeds, output_ports, expected))
            del source_session
            gc.collect()

        shard_session = _session(model_dir / shard["file"], provider)
        for graph, feeds, output_ports, expected in cases:
            actual = _run_branch(shard_session, shard, graph, feeds, output_ports)
            per_output = [
                _metrics(left, right)
                for left, right in zip(expected, actual, strict=True)
            ]
            graph_results[graph] = {
                "max_abs": max(item["max_abs"] for item in per_output),
                "relative_l2": max(item["relative_l2"] for item in per_output),
            }
        del shard_session
        gc.collect()

    maximum = max(result["max_abs"] for result in graph_results.values())
    relative = max(result["relative_l2"] for result in graph_results.values())
    worst = sorted(
        graph_results.items(),
        key=lambda item: (item[1]["relative_l2"], item[1]["max_abs"]),
        reverse=True,
    )[:10]
    if maximum > 1e-5 or relative > 1e-6:
        failure = {
            "format": "h3-shard-validation-failure-v1",
            "provider": provider,
            "graphs_validated": len(graph_results),
            "max_abs": maximum,
            "max_relative_l2": relative,
            "worst_graphs": dict(worst),
        }
        _atomic_json(model_dir / "shard-validation-failure.json", failure)
        raise AssertionError(
            f"Shard branch mismatch: max_abs={maximum}, relative_l2={relative}, "
            f"worst={worst[:3]}"
        )
    summary = {
        "format": "h3-shard-validation-v1",
        "provider": provider,
        "graphs_validated": len(graph_results),
        "shards_validated": len(schedule["shards"]),
        "max_abs": maximum,
        "max_relative_l2": relative,
        "chain_finite": bool(np.isfinite(chain_hidden).all()),
        "block_04_max_abs": block_maxima.get("04"),
        "final_hidden_max_abs": float(np.max(np.abs(chain_hidden))),
        "graph_metrics": graph_results,
    }
    validation_path = model_dir / "shard-validation.json"
    _atomic_json(validation_path, summary)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shard_validation"] = {
        key: value for key, value in summary.items() if key != "graph_metrics"
    }
    manifest.setdefault("artifacts", {})[validation_path.name] = {
        "bytes": validation_path.stat().st_size,
        "sha256": file_sha256(validation_path),
    }
    _atomic_json(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all h3-schedule-v2 If branches")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.provider in {"auto", "cuda"}:
        try:
            from h3_workbench.inference_runtime import _preload_cuda_dlls

            _preload_cuda_dlls()
        except Exception:
            if args.provider == "cuda":
                raise
    summary = validate_sharded_model(args.model, args.provider)
    print(json.dumps({key: value for key, value in summary.items() if key != "graph_metrics"}, indent=2))


if __name__ == "__main__":
    main()
