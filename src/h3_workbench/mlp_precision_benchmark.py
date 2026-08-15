from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from h3_workbench.inference_runtime import ORTGraphRunner
from h3_workbench.main_benchmark import JsonlLogger, _gpu_sample


PRECISION_FLAGS = {
    "fp32": (),
    "fp16": ("--gpu-native-fp16",),
    "scaled_fp16": ("--gpu-scaled-fp16",),
    "bf16": ("--gpu-native-bf16",),
}


def _graph_inventory(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    graphs = []

    def visit(graph: onnx.GraphProto) -> None:
        graphs.append(graph)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit(child)

    visit(model.graph)
    operators = Counter(node.op_type for graph in graphs for node in graph.node)
    dtypes = Counter(
        TensorProto.DataType.Name(initializer.data_type)
        for graph in graphs
        for initializer in graph.initializer
    )
    external = path.with_name(f"{path.name}.data")
    return {
        "operators": dict(operators),
        "initializer_dtypes": dict(dtypes),
        "storage_bytes": path.stat().st_size + (external.stat().st_size if external.is_file() else 0),
    }


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    expected32 = expected.astype(np.float32, copy=False)
    actual32 = actual.astype(np.float32, copy=False)
    if not np.isfinite(actual32).all():
        return {
            "finite": False,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "relative_l2": float("inf"),
            "cosine": float("nan"),
        }
    difference = actual32 - expected32
    expected_flat = expected32.reshape(-1)
    actual_flat = actual32.reshape(-1)
    denominator = np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
    expected_norm = np.linalg.norm(expected_flat)
    return {
        "finite": True,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / expected_norm),
        "cosine": float(np.dot(expected_flat, actual_flat) / denominator),
    }


def _conditioning(product: Path, rows: int) -> np.ndarray:
    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
    session = runner.session(product / "shard_006.onnx")
    try:
        return session.run(
            ["main_conditioning/timestep_embedding"],
            {
                "timesteps": np.asarray([0.2, 0.8], dtype=np.float32),
                "position_ids": np.zeros((rows, 3), dtype=np.float32),
                "selector_shard_006": np.asarray(0, dtype=np.int64),
            },
        )[0]
    finally:
        del session
        runner.close()


def _export_candidate(
    source: Path,
    block: int,
    mode: str,
    model_path: Path,
    input_path: Path,
    expected_path: Path,
    kind: str = "dit_mlp",
) -> float:
    command = [
        sys.executable,
        "-S",
        "-m",
        "h3_workbench.main_export_worker",
        kind,
        str(block),
        str(source),
        str(model_path),
        str(input_path),
        str(expected_path),
        *PRECISION_FLAGS[mode],
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        creationflags=flags,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(details)
    return time.perf_counter() - started


def _run_candidate(
    runner: ORTGraphRunner,
    model_path: Path,
    feeds: dict[str, np.ndarray],
    expected: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    session = runner.session(model_path)
    build_seconds = time.perf_counter() - started
    try:
        output_name = session.get_outputs()[0].name
        cold_started = time.perf_counter()
        actual = session.run([output_name], feeds)[0]
        cold_seconds = time.perf_counter() - cold_started
        warm_seconds = []
        for _ in range(repeats):
            run_started = time.perf_counter()
            actual = session.run([output_name], feeds)[0]
            warm_seconds.append(time.perf_counter() - run_started)
        return {
            "providers": session.get_providers(),
            "build_seconds": round(build_seconds, 6),
            "cold_seconds": round(cold_seconds, 6),
            "warm_seconds": [round(value, 6) for value in warm_seconds],
            "warm_median_seconds": round(float(np.median(warm_seconds)), 6),
            "gpu": _gpu_sample(),
            "metrics": _metrics(expected, actual),
        }
    finally:
        del session
        gc.collect()


def _product_model(product: Path, block: int) -> Path:
    return product / f"shard_{10 + block * 3:03d}.onnx"


def run(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.log)
    result: dict[str, Any] = {
        "source": str(args.source.resolve()),
        "product": str(args.product.resolve()),
        "rows": args.rows,
        "repeats": args.repeats,
        "blocks": {},
    }
    try:
        timestep_embedding = _conditioning(args.product, args.rows)
        for block in args.blocks:
            block_dir = args.output / f"block_{block:02d}"
            block_dir.mkdir(parents=True, exist_ok=True)
            random = np.random.default_rng(19 + block)
            inputs = {
                "hidden_states": random.standard_normal((args.rows, 5376), dtype=np.float32) * 0.25,
                "timestep_embedding": timestep_embedding,
                "modulation_ids": np.arange(args.rows, dtype=np.int64) % 6,
            }
            input_path = block_dir / "inputs.npz"
            np.savez(input_path, **inputs)
            block_result: dict[str, Any] = {}
            result["blocks"][str(block)] = block_result
            for mode in PRECISION_FLAGS:
                model_path = block_dir / f"{mode}.onnx"
                expected_path = block_dir / f"{mode}.expected.npz"
                logger.write("candidate_started", durable=True, block=block, mode=mode)
                try:
                    export_seconds = _export_candidate(
                        args.source,
                        block,
                        mode,
                        model_path,
                        input_path,
                        expected_path,
                    )
                    with np.load(expected_path) as archive:
                        expected = archive["output_0"].copy()
                    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
                    try:
                        run_result = _run_candidate(
                            runner,
                            model_path,
                            inputs,
                            expected,
                            args.repeats,
                        )
                    finally:
                        runner.close()
                    candidate = {
                        "status": "completed",
                        "export_seconds": round(export_seconds, 3),
                        **_graph_inventory(model_path),
                        **run_result,
                    }
                except Exception as exc:  # noqa: BLE001 - record unsupported candidates
                    candidate = {"status": "failed", "error": str(exc)}
                block_result[mode] = candidate
                logger.write("candidate_completed", durable=True, block=block, mode=mode, **candidate)

            product_path = _product_model(args.product, block)
            expected_path = block_dir / "fp32.expected.npz"
            with np.load(expected_path) as archive:
                expected = archive["output_0"].copy()
            product_feeds = {
                **inputs,
                f"selector_shard_{10 + block * 3:03d}": np.asarray(0, dtype=np.int64),
            }
            runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
            try:
                product_result = {
                    "status": "completed",
                    **_graph_inventory(product_path),
                    **_run_candidate(runner, product_path, product_feeds, expected, args.repeats),
                }
            finally:
                runner.close()
            block_result["product_fp32"] = product_result
            logger.write("product_completed", durable=True, block=block, **product_result)

        result_path = args.output / "results.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.write("completed", durable=True, result=str(result_path.resolve()))
        print(json.dumps({"status": "completed", "result": str(result_path.resolve())}), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist benchmark failure
        logger.write("failed", durable=True, error=str(exc))
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        logger.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and benchmark single-block MLP precision candidates")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 4, 49])
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
