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
ATTENTION_WEIGHT_TO_SOURCE = {
    "query.linear.weight": "self_attn.q_proj",
    "key.linear.weight": "self_attn.k_proj",
    "value.linear.weight": "self_attn.v_proj",
    "output.linear.weight": "self_attn.o_proj",
}


def _artifact_ready(path: Path) -> bool:
    return path.is_file() and path.with_name(f"{path.name}.data").is_file()


def _inputs(tokens: int, block: int) -> dict[str, np.ndarray]:
    random = np.random.default_rng(31 + block)
    hidden = (random.standard_normal((tokens, 5120), dtype=np.float32) * 0.25).astype(np.float32)
    positions = np.arange(tokens, dtype=np.float32)
    frequencies = 1.0 / (500_000.0 ** (np.arange(0, 128, 2, dtype=np.float32) / 128.0))
    angles = np.outer(positions, frequencies)
    angles = np.concatenate((angles, angles), axis=-1)
    mask = np.triu(np.full((1, 1, tokens, tokens), -10_000.0, dtype=np.float32), k=1)
    return {
        "hidden_states": hidden,
        "cosine": np.cos(angles).astype(np.float32),
        "sine": np.sin(angles).astype(np.float32),
        "attention_mask": mask,
    }


def _timed(call: Callable[[], np.ndarray], repeats: int) -> tuple[np.ndarray, dict[str, Any]]:
    first_started = time.perf_counter()
    output = call()
    first_seconds = time.perf_counter() - first_started
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = call()
        samples.append(time.perf_counter() - started)
    return output, {
        "first_run_seconds": first_seconds,
        "samples_seconds": samples,
        "min_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "max_seconds": max(samples),
    }


def _benchmark(
    topology: Path,
    feeds: dict[str, np.ndarray],
    expected: np.ndarray,
    repeats: int,
    source_model: Path | None = None,
    weight_names: set[str] | None = None,
) -> dict[str, Any]:
    weights = initializer_inputs(source_model, weight_names) if source_model is not None and weight_names else {}
    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
    try:
        started = time.perf_counter()
        session = runner.session(topology)
        build_seconds = time.perf_counter() - started
        all_feeds = {**feeds, **weights}
        output, timing = _timed(lambda: session.run(None, all_feeds)[0], repeats)
        return {
            "provider": runner.provider,
            "build_seconds": build_seconds,
            "timing": timing,
            "metrics": _metrics(expected, output),
            "weight_input_bytes": sum(array.nbytes for array in weights.values()),
        }
    finally:
        runner.close()
        weights.clear()
        gc.collect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark persistent INT8 weights for one Qwen attention layer")
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
        args.output.mkdir(parents=True, exist_ok=True)
        free_gib = shutil.disk_usage(args.output).free / GIB
        if free_gib - 0.5 < args.min_free_gib:
            raise RuntimeError(
                f"Disk guard stopped Qwen attention export: projected free {free_gib - 0.5:.2f} GiB"
            )
        feeds = _inputs(args.tokens, args.block)
        prefix = f"qwen_layer_{args.block:02d}_attention"
        fp16 = args.output / f"{prefix}.onnx"
        qdq = args.output / f"{prefix}_int8_qdq_fp32scale.onnx"
        inputs_path = args.output / "attention_inputs.npz"
        expected_path = args.output / "attention_expected.npz"
        logger.write(
            "started",
            durable=True,
            block=args.block,
            tokens=args.tokens,
            source=str(args.source.resolve()),
            free_gib=free_gib,
        )
        if not _artifact_ready(fp16) or not expected_path.is_file():
            expected = _export_qwen_shard("attention", args.block, args.source, fp16, feeds)[0].numpy()
            np.savez(inputs_path, **feeds)
            np.savez(expected_path, hidden_states_out=expected)
            logger.write("fp16_export_completed", durable=True)
        else:
            with np.load(inputs_path, allow_pickle=False) as archive:
                feeds = {name: archive[name].copy() for name in archive.files}
            with np.load(expected_path, allow_pickle=False) as archive:
                expected = archive["hidden_states_out"].copy()
        if not _artifact_ready(qdq):
            build_int8_qdq_graph(
                fp16,
                args.source,
                qdq,
                block=args.block,
                weight_to_source=ATTENTION_WEIGHT_TO_SOURCE,
            )
            logger.write("qdq_export_completed", durable=True)

        norms = {"input_norm", "q_norm", "k_norm"}
        fp16_weight_names = norms | set(ATTENTION_WEIGHT_TO_SOURCE)
        int8_weight_names = norms | {
            name
            for weight in ATTENTION_WEIGHT_TO_SOURCE
            for name in (f"{weight}.int8", f"{weight}.scale")
        }
        fp16_topology = args.output / "runtime_qwen_attention_fp16.onnx"
        int8_topology = args.output / "runtime_qwen_attention_int8_qdq_fp32scale.onnx"
        if not fp16_topology.is_file():
            build_weight_input_topology(fp16, fp16_topology, fp16_weight_names)
        if not int8_topology.is_file():
            build_weight_input_topology(qdq, int8_topology, int8_weight_names)

        embedded_fp16 = _benchmark(fp16, feeds, expected, args.repeats)
        embedded_int8 = _benchmark(qdq, feeds, expected, args.repeats)
        persistent_fp16 = _benchmark(
            fp16_topology, feeds, expected, args.repeats, fp16, fp16_weight_names
        )
        persistent_int8 = _benchmark(
            int8_topology, feeds, expected, args.repeats, qdq, int8_weight_names
        )
        result = {
            "embedded_fp16": embedded_fp16,
            "embedded_int8_qdq": embedded_int8,
            "persistent_fp16_weights": persistent_fp16,
            "persistent_int8_weights": persistent_int8,
            "persistent_int8_vs_fp16_speedup": (
                persistent_fp16["timing"]["median_seconds"]
                / persistent_int8["timing"]["median_seconds"]
            ),
        }
        logger.write("completed", durable=True, **result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist benchmark failures
        logger.write("failed", durable=True, error=str(exc), traceback="".join(traceback.format_exception(exc)))
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        logger.close()


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
