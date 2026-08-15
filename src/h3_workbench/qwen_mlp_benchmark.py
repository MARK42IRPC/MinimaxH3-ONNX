from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from h3_workbench.exporter import _export_qwen_shard, _metrics
from h3_workbench.inference_runtime import ORTGraphRunner
from h3_workbench.main_benchmark import JsonlLogger
from h3_workbench.qwen_int8_graph import (
    build_int8_qdq_graph,
    build_weight_input_topology,
    initializer_inputs,
)


GIB = 1 << 30
EXPORT_BUDGET_BYTES = 2 * GIB


def _artifact_ready(path: Path) -> bool:
    return path.is_file() and path.with_name(f"{path.name}.data").is_file()


def _disk_guard(path: Path, min_free_gib: float, required_bytes: int = EXPORT_BUDGET_BYTES) -> dict[str, float]:
    free_bytes = shutil.disk_usage(path).free
    projected = free_bytes - required_bytes
    reserve = int(min_free_gib * GIB)
    if projected < reserve:
        raise RuntimeError(
            f"Disk guard stopped Qwen export: projected free {projected / GIB:.2f} GiB "
            f"is below reserve {min_free_gib:.2f} GiB"
        )
    return {"free_gib": free_bytes / GIB, "projected_free_gib": projected / GIB}


def _paths(output: Path, block: int) -> dict[str, Path]:
    prefix = f"qwen_layer_{block:02d}"
    return {kind: output / f"{prefix}_{kind}.onnx" for kind in ("gate", "up", "down", "mlp")}


def _prepare_candidates(
    source: Path,
    output: Path,
    block: int,
    tokens: int,
    min_free_gib: float,
    logger: JsonlLogger,
) -> tuple[dict[str, Path], dict[str, np.ndarray], np.ndarray]:
    output.mkdir(parents=True, exist_ok=True)
    paths = _paths(output, block)
    input_path = output / "inputs.npz"
    expected_path = output / "expected.npz"
    reused = all(_artifact_ready(path) for path in paths.values()) and input_path.is_file() and expected_path.is_file()
    if reused:
        with np.load(input_path, allow_pickle=False) as archive:
            feeds = {name: archive[name].copy() for name in archive.files}
        with np.load(expected_path, allow_pickle=False) as archive:
            expected = archive["hidden_states_out"].copy()
        logger.write("export_reused", durable=True, paths={kind: str(path) for kind, path in paths.items()})
    else:
        disk = _disk_guard(output, min_free_gib)
        logger.write("disk_guard", durable=True, reserve_gib=min_free_gib, **disk)
        random = np.random.default_rng(19 + block)
        hidden = (random.standard_normal((tokens, 5120), dtype=np.float32) * 0.25).astype(np.float32)
        started = time.perf_counter()
        normalized, gate = _export_qwen_shard(
            "gate", block, source, paths["gate"], {"hidden_states": hidden}
        )
        up = _export_qwen_shard(
            "up", block, source, paths["up"], {"normalized_states": normalized.numpy()}
        )[0]
        expected = _export_qwen_shard(
            "down",
            block,
            source,
            paths["down"],
            {"hidden_states": hidden, "gate": gate.numpy(), "up": up.numpy()},
        )[0].numpy()
        fused_expected = _export_qwen_shard(
            "mlp", block, source, paths["mlp"], {"hidden_states": hidden}
        )[0].numpy()
        source_agreement = _metrics(expected, fused_expected)
        if not np.isfinite(fused_expected).all() or source_agreement["relative_l2"] > 1e-7:
            raise RuntimeError(f"Fused Qwen MLP source mismatch: {source_agreement}")
        feeds = {"hidden_states": hidden}
        np.savez(input_path, **feeds)
        np.savez(expected_path, hidden_states_out=expected)
        logger.write(
            "export_completed",
            durable=True,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            source_agreement=source_agreement,
            storage_bytes={
                kind: path.stat().st_size + path.with_name(f"{path.name}.data").stat().st_size
                for kind, path in paths.items()
            },
        )
    qdq_path = output / paths["mlp"].name.replace(".onnx", "_int8_qdq_fp32scale.onnx")
    if not _artifact_ready(qdq_path):
        logger.write("qdq_build_started", durable=True)
        build_int8_qdq_graph(paths["mlp"], source, qdq_path, block=block)
        logger.write("qdq_build_completed", durable=True, path=str(qdq_path), bytes=qdq_path.with_name(f"{qdq_path.name}.data").stat().st_size)
    paths["qdq"] = qdq_path
    return paths, feeds, expected


def _timed_runs(run_once: Callable[[], np.ndarray], repeats: int) -> tuple[np.ndarray, dict[str, Any]]:
    run_once()
    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = run_once()
        samples.append(time.perf_counter() - started)
    assert output is not None
    return output, {
        "samples_seconds": samples,
        "min_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "max_seconds": max(samples),
    }


def _benchmark_baseline(
    paths: dict[str, Path], hidden: np.ndarray, expected: np.ndarray, repeats: int
) -> dict[str, Any]:
    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
    try:
        started = time.perf_counter()
        gate_session = runner.session(paths["gate"])
        up_session = runner.session(paths["up"])
        down_session = runner.session(paths["down"])
        build_seconds = time.perf_counter() - started

        def run_once() -> np.ndarray:
            normalized, gate = gate_session.run(None, {"hidden_states": hidden})
            up = up_session.run(None, {"normalized_states": normalized})[0]
            return down_session.run(None, {"hidden_states": hidden, "gate": gate, "up": up})[0]

        output, timing = _timed_runs(run_once, repeats)
        return {
            "provider": runner.provider,
            "build_seconds": build_seconds,
            "timing": timing,
            "metrics": _metrics(expected, output),
            "cache": runner.cache_stats(),
        }
    finally:
        runner.close()
        gc.collect()


def _benchmark_fused(
    path: Path, hidden: np.ndarray, expected: np.ndarray, repeats: int
) -> dict[str, Any]:
    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
    try:
        started = time.perf_counter()
        session = runner.session(path)
        build_seconds = time.perf_counter() - started

        def run_once() -> np.ndarray:
            return session.run(None, {"hidden_states": hidden})[0]

        output, timing = _timed_runs(run_once, repeats)
        return {
            "provider": runner.provider,
            "build_seconds": build_seconds,
            "timing": timing,
            "metrics": _metrics(expected, output),
            "cache": runner.cache_stats(),
        }
    finally:
        runner.close()
        gc.collect()


def _benchmark_weight_inputs(
    topology: Path,
    source_model: Path,
    input_names: set[str],
    hidden: np.ndarray,
    expected: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    weight_inputs = initializer_inputs(source_model, input_names)
    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
    try:
        started = time.perf_counter()
        session = runner.session(topology)
        build_seconds = time.perf_counter() - started
        feeds = {"hidden_states": hidden, **weight_inputs}

        def run_once() -> np.ndarray:
            return session.run(None, feeds)[0]

        first_started = time.perf_counter()
        output = run_once()
        first_run_seconds = time.perf_counter() - first_started
        output, timing = _timed_runs(run_once, repeats)
        return {
            "provider": runner.provider,
            "build_seconds": build_seconds,
            "first_run_seconds": first_run_seconds,
            "timing": timing,
            "metrics": _metrics(expected, output),
            "weight_input_bytes": sum(array.nbytes for array in weight_inputs.values()),
            "cache": runner.cache_stats(),
        }
    finally:
        runner.close()
        weight_inputs.clear()
        gc.collect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a fused Qwen MLP layer from an INT8 checkpoint")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0, choices=range(50))
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-free-gib", type=float, default=90.0)
    return parser


def run(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    try:
        logger.write(
            "started",
            durable=True,
            source=str(args.source.resolve()),
            output=str(args.output.resolve()),
            block=args.block,
            tokens=args.tokens,
            repeats=args.repeats,
        )
        paths, feeds, expected = _prepare_candidates(
            args.source.resolve(),
            args.output.resolve(),
            args.block,
            args.tokens,
            args.min_free_gib,
            logger,
        )
        baseline = _benchmark_baseline(paths, feeds["hidden_states"], expected, args.repeats)
        logger.write("baseline_completed", durable=True, **baseline)
        fused = _benchmark_fused(paths["mlp"], feeds["hidden_states"], expected, args.repeats)
        qdq = _benchmark_fused(paths["qdq"], feeds["hidden_states"], expected, args.repeats)
        fp16_input_names = {"norm", "gate.linear.weight", "up.linear.weight", "down.linear.weight"}
        int8_input_names = {
            "norm",
            "gate.linear.weight.int8",
            "gate.linear.weight.scale",
            "up.linear.weight.int8",
            "up.linear.weight.scale",
            "down.linear.weight.int8",
            "down.linear.weight.scale",
        }
        fp16_topology = args.output / "runtime_qwen_mlp_fp16.onnx"
        int8_topology = args.output / "runtime_qwen_mlp_int8_qdq_fp32scale.onnx"
        if not fp16_topology.is_file():
            build_weight_input_topology(paths["mlp"], fp16_topology, fp16_input_names)
        if not int8_topology.is_file():
            build_weight_input_topology(paths["qdq"], int8_topology, int8_input_names)
        persistent_fp16 = _benchmark_weight_inputs(
            fp16_topology,
            paths["mlp"],
            fp16_input_names,
            feeds["hidden_states"],
            expected,
            args.repeats,
        )
        logger.write("persistent_fp16_completed", durable=True, **persistent_fp16)
        persistent_int8 = _benchmark_weight_inputs(
            int8_topology,
            paths["qdq"],
            int8_input_names,
            feeds["hidden_states"],
            expected,
            args.repeats,
        )
        logger.write("persistent_int8_completed", durable=True, **persistent_int8)
        speedup = baseline["timing"]["median_seconds"] / fused["timing"]["median_seconds"]
        qdq_speedup = baseline["timing"]["median_seconds"] / qdq["timing"]["median_seconds"]
        persistent_speedup = (
            persistent_fp16["timing"]["median_seconds"]
            / persistent_int8["timing"]["median_seconds"]
        )
        result = {
            "baseline": baseline,
            "fused_fp16": fused,
            "fused_int8_qdq": qdq,
            "persistent_fp16_weights": persistent_fp16,
            "persistent_int8_weights": persistent_int8,
            "warm_speedup_fp16": speedup,
            "warm_speedup_int8_qdq": qdq_speedup,
            "persistent_int8_vs_fp16_speedup": persistent_speedup,
        }
        logger.write("completed", durable=True, **result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist benchmark failures
        logger.write(
            "failed",
            durable=True,
            error=str(exc),
            traceback="".join(traceback.format_exception(exc)),
        )
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        logger.close()


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
