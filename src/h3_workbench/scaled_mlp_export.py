from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from h3_workbench.inference_runtime import ORTGraphRunner
from h3_workbench.main_benchmark import JsonlLogger
from h3_workbench.mlp_precision_benchmark import (
    _conditioning,
    _export_candidate,
    _graph_inventory,
    _run_candidate,
)
from h3_workbench.shard_planner import file_sha256, validate_runtime_schedule, write_schedule


GIB = 1 << 30
DEFAULT_ESTIMATED_BLOCK_BYTES = 470_000_000
MLP_BLOCKS = 50


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _artifact_paths(block_dir: Path) -> list[Path]:
    graph = block_dir / "scaled_fp16.onnx"
    return [graph, graph.with_name(f"{graph.name}.data")]


def _file_record(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
    }


def _record_valid(record: dict[str, Any] | None, block_dir: Path) -> bool:
    if not record or record.get("status") != "completed":
        return False
    for name, details in record.get("files", {}).items():
        path = block_dir / name
        if not path.is_file() or path.stat().st_size != details.get("bytes"):
            return False
    return bool(record.get("files"))


def _space_projection(
    free_bytes: int,
    reserve_bytes: int,
    pending_blocks: int,
    estimated_block_bytes: int,
) -> int:
    projected_free = free_bytes - pending_blocks * estimated_block_bytes
    if projected_free < reserve_bytes:
        raise RuntimeError(
            f"Disk guard stopped export: projected free {projected_free / GIB:.2f} GiB "
            f"is below reserve {reserve_bytes / GIB:.2f} GiB"
        )
    return projected_free


def _validate_structure(model_path: Path) -> dict[str, Any]:
    inventory = _graph_inventory(model_path)
    operators = inventory["operators"]
    if operators.get("Gemm") != 3 or operators.get("Cast") != 4:
        raise RuntimeError(f"Unexpected scaled-FP16 graph structure: {operators}")
    model = onnx.load(str(model_path), load_external_data=False)
    large_initializers = [
        initializer
        for initializer in model.graph.initializer
        if int(np.prod(initializer.dims, dtype=np.int64)) >= 1_000_000
    ]
    if len(large_initializers) < 2 or any(
        initializer.data_type != TensorProto.FLOAT16 for initializer in large_initializers
    ):
        dtypes = [TensorProto.DataType.Name(item.data_type) for item in large_initializers]
        raise RuntimeError(f"Large MLP initializers are not FP16: {dtypes}")
    return inventory


def _validate_candidate(
    model_path: Path,
    inputs_path: Path,
    expected_path: Path,
    relative_l2_max: float,
) -> dict[str, Any]:
    inventory = _validate_structure(model_path)
    with np.load(inputs_path, allow_pickle=False) as archive:
        feeds = {name: archive[name].copy() for name in archive.files}
    with np.load(expected_path, allow_pickle=False) as archive:
        expected = archive["output_0"].copy()
    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
    try:
        result = _run_candidate(runner, model_path, feeds, expected, repeats=1)
    finally:
        runner.close()
    if "CUDAExecutionProvider" not in result["providers"]:
        raise RuntimeError(f"CUDA session was not built: {result['providers']}")
    metrics = result["metrics"]
    if not metrics["finite"] or metrics["relative_l2"] > relative_l2_max:
        raise RuntimeError(
            f"Scaled-FP16 numerical gate failed: relative_l2={metrics['relative_l2']}, "
            f"limit={relative_l2_max}"
        )
    return {**inventory, **result}


def _seed_block(seed_root: Path | None, block: int, block_dir: Path) -> bool:
    if seed_root is None:
        return False
    source = seed_root / f"block_{block:02d}"
    mapping = {
        source / "scaled_fp16.onnx": block_dir / "scaled_fp16.onnx",
        source / "scaled_fp16.onnx.data": block_dir / "scaled_fp16.onnx.data",
        source / "scaled_fp16.expected.npz": block_dir / "expected.npz",
        source / "inputs.npz": block_dir / "inputs.npz",
    }
    if not all(path.is_file() for path in mapping):
        return False
    block_dir.mkdir(parents=True, exist_ok=True)
    for source_path, target_path in mapping.items():
        if not target_path.exists():
            os.link(source_path, target_path)
    return True


def _block_inputs(timestep_embedding: np.ndarray, block: int, rows: int) -> dict[str, np.ndarray]:
    random = np.random.default_rng(19 + block)
    return {
        "hidden_states": random.standard_normal((rows, 5376), dtype=np.float32) * 0.25,
        "timestep_embedding": timestep_embedding,
        "modulation_ids": np.arange(rows, dtype=np.int64) % 6,
    }


def _export_block(
    source: Path,
    block: int,
    block_dir: Path,
    inputs: dict[str, np.ndarray],
    relative_l2_max: float,
) -> tuple[float, dict[str, Any]]:
    block_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = block_dir / "inputs.npz"
    if not inputs_path.is_file():
        np.savez(inputs_path, **inputs)
    with tempfile.TemporaryDirectory(prefix="scaled-fp16-", dir=block_dir) as temporary_name:
        temporary = Path(temporary_name)
        model_path = temporary / "scaled_fp16.onnx"
        expected_path = temporary / "expected.npz"
        export_seconds = _export_candidate(
            source,
            block,
            "scaled_fp16",
            model_path,
            inputs_path,
            expected_path,
        )
        validation = _validate_candidate(model_path, inputs_path, expected_path, relative_l2_max)
        for path in (model_path, model_path.with_name(f"{model_path.name}.data"), expected_path):
            os.replace(path, block_dir / path.name)
    return export_seconds, validation


def export_all(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "export-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"format": "h3-scaled-fp16-mlp-export-v1", "blocks": {}}
    )
    timestep_embedding = _conditioning(args.product, args.rows)
    for block in args.blocks:
        block_dir = args.output / f"block_{block:02d}"
        seeded = _seed_block(args.seed_dir, block, block_dir)
        record = state["blocks"].get(str(block))
        if _record_valid(record, block_dir):
            logger.write("block_skipped", durable=True, block=block, reason="valid_state")
            continue

        inputs = _block_inputs(timestep_embedding, block, args.rows)
        inputs_path = block_dir / "inputs.npz"
        expected_path = block_dir / "expected.npz"
        model_path = block_dir / "scaled_fp16.onnx"
        if model_path.is_file() and expected_path.is_file() and inputs_path.is_file():
            logger.write("block_validation_started", durable=True, block=block, seeded=seeded)
            validation = _validate_candidate(model_path, inputs_path, expected_path, args.relative_l2_max)
            export_seconds = 0.0
        else:
            pending = sum(
                not (args.output / f"block_{item:02d}" / "scaled_fp16.onnx.data").is_file()
                for item in args.blocks
                if item >= block
            )
            free_bytes = shutil.disk_usage(args.output).free
            projected_free = _space_projection(
                free_bytes,
                int(args.min_free_gib * GIB),
                pending,
                args.estimated_block_bytes,
            )
            logger.write(
                "disk_guard",
                durable=True,
                block=block,
                free_gib=round(free_bytes / GIB, 3),
                pending_blocks=pending,
                projected_free_gib=round(projected_free / GIB, 3),
                reserve_gib=args.min_free_gib,
            )
            logger.write("block_export_started", durable=True, block=block)
            export_seconds, validation = _export_block(
                args.source,
                block,
                block_dir,
                inputs,
                args.relative_l2_max,
            )

        files = _file_record(_artifact_paths(block_dir))
        record = {
            "status": "completed",
            "export_seconds": round(export_seconds, 3),
            "files": files,
            **validation,
        }
        state["blocks"][str(block)] = record
        _atomic_json(state_path, state)
        logger.write(
            "block_completed",
            durable=True,
            block=block,
            export_seconds=record["export_seconds"],
            relative_l2=record["metrics"]["relative_l2"],
            storage_bytes=record["storage_bytes"],
        )
        print(
            json.dumps(
                {
                    "block": block + 1,
                    "blocks": len(args.blocks),
                    "relative_l2": record["metrics"]["relative_l2"],
                    "free_gib": round(shutil.disk_usage(args.output).free / GIB, 2),
                }
            ),
            flush=True,
        )
    return state


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)


def build_hybrid(product: Path, candidates: Path, target: Path, state: dict[str, Any]) -> Path:
    product = product.resolve()
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Hybrid target already exists: {target}")
    staging = target.with_name(f"{target.name}.staging")
    if staging.exists():
        raise FileExistsError(f"Hybrid staging directory already exists: {staging}")
    missing = [block for block in range(MLP_BLOCKS) if not _record_valid(state["blocks"].get(str(block)), candidates / f"block_{block:02d}")]
    if missing:
        raise RuntimeError(f"Cannot build hybrid; scaled-FP16 blocks are incomplete: {missing}")
    staging.mkdir(parents=True)
    mlp_shard_names = {
        name
        for block in range(MLP_BLOCKS)
        for name in (f"shard_{10 + block * 3:03d}.onnx", f"shard_{10 + block * 3:03d}.onnx.data")
    }
    for source in product.iterdir():
        if source.is_file() and source.name not in {"manifest.json", "schedule.json", *mlp_shard_names}:
            _link(source, staging / source.name)

    schedule = json.loads((product / "schedule.json").read_text(encoding="utf-8"))
    artifacts = {
        name: details
        for name, details in json.loads((product / "manifest.json").read_text(encoding="utf-8"))["artifacts"].items()
        if name not in {"schedule.json", *mlp_shard_names}
    }
    by_id = {shard["id"]: shard for shard in schedule["shards"]}
    for block in range(MLP_BLOCKS):
        shard_id = f"shard_{10 + block * 3:03d}"
        relative = Path("scaled_mlp") / f"block_{block:02d}" / "scaled_fp16.onnx"
        source_dir = candidates / f"block_{block:02d}"
        _link(source_dir / "scaled_fp16.onnx", staging / relative)
        _link(source_dir / "scaled_fp16.onnx.data", staging / f"{relative}.data")
        record = state["blocks"][str(block)]
        by_id[shard_id]["file"] = relative.as_posix()
        by_id[shard_id]["storage_bytes"] = int(record["storage_bytes"])
        by_id[shard_id]["materialized_weight_bytes"] = int(record["storage_bytes"])
        by_id[shard_id]["session_peak_bytes"] = max(
            int(by_id[shard_id]["session_peak_bytes"]),
            725 * (1 << 20),
        )
        artifacts[relative.as_posix()] = record["files"]["scaled_fp16.onnx"]
        artifacts[f"{relative.as_posix()}.data"] = record["files"]["scaled_fp16.onnx.data"]
    schedule["model"] = target.name
    schedule["precision"].update(
        {"mlp_compute": "scaled_fp16_tensor_core", "mlp_storage": "fp16"}
    )
    schedule_path = write_schedule(schedule, staging / "schedule.json")
    artifacts["schedule.json"] = {
        "bytes": schedule_path.stat().st_size,
        "sha256": file_sha256(schedule_path),
    }
    product_manifest = json.loads((product / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        **product_manifest,
        "conversion": "scaled_fp16_tensor_core_mlp_hybrid",
        "activation_dtype": "fp32_residual_scaled_fp16_mlp",
        "validation_passed": False,
        "hybrid_chain_validation": "pending",
        "source_model_directory": str(product),
        "artifacts": artifacts,
    }
    _atomic_json(staging / "manifest.json", manifest)
    validate_runtime_schedule(schedule, staging)
    os.replace(staging, target)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumably export all scaled-FP16 MLP blocks")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--hybrid-output", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", default=list(range(MLP_BLOCKS)))
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--relative-l2-max", type=float, default=2e-3)
    parser.add_argument("--min-free-gib", type=float, default=90.0)
    parser.add_argument("--estimated-block-bytes", type=int, default=DEFAULT_ESTIMATED_BLOCK_BYTES)
    return parser


def run(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    started = time.perf_counter()
    try:
        state = export_all(args, logger)
        hybrid = None
        if args.hybrid_output is not None:
            hybrid = build_hybrid(args.product, args.output, args.hybrid_output, state)
        logger.write(
            "completed",
            durable=True,
            blocks=len(args.blocks),
            elapsed_seconds_total=round(time.perf_counter() - started, 3),
            hybrid=str(hybrid) if hybrid else None,
            free_gib=round(shutil.disk_usage(args.output).free / GIB, 3),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - durable batch failure is required
        logger.write("failed", durable=True, error=str(exc))
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        logger.close()


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
