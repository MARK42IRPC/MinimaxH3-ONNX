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
from h3_workbench.scaled_mlp_export import GIB, MLP_BLOCKS, _atomic_json, _file_record, _space_projection
from h3_workbench.shard_planner import file_sha256, validate_runtime_schedule, write_schedule


ESTIMATED_BLOCK_BYTES = 82_000_000


def _artifact_paths(block_dir: Path) -> list[Path]:
    graph = block_dir / "scaled_fp16.onnx"
    return [graph, graph.with_name(f"{graph.name}.data")]


def _record_valid(record: dict[str, Any] | None, block_dir: Path) -> bool:
    if not record or record.get("status") != "completed":
        return False
    files = record.get("files", {})
    return bool(files) and all(
        (block_dir / name).is_file() and (block_dir / name).stat().st_size == details.get("bytes")
        for name, details in files.items()
    )


def _inputs(timestep_embedding: np.ndarray, block: int, rows: int) -> dict[str, np.ndarray]:
    random = np.random.default_rng(71 + block)
    return {
        "hidden_states": random.standard_normal((rows, 5376), dtype=np.float32) * 0.25,
        "attended": random.standard_normal((rows, 7168), dtype=np.float32) * 0.25,
        "timestep_embedding": timestep_embedding,
        "modulation_ids": np.arange(rows, dtype=np.int64) % 6,
    }


def _validate_structure(model_path: Path) -> dict[str, Any]:
    inventory = _graph_inventory(model_path)
    operators = inventory["operators"]
    if operators.get("Gemm") != 2 or operators.get("Cast") != 2 or operators.get("Mul") != 2:
        raise RuntimeError(f"Unexpected scaled attention-output structure: {operators}")
    model = onnx.load(str(model_path), load_external_data=False)
    weights = [item for item in model.graph.initializer if list(item.dims) == [5376, 7168]]
    if len(weights) != 1 or weights[0].data_type != TensorProto.FLOAT16:
        raise RuntimeError("Attention output projection is not stored as one FP16 initializer")
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
    metrics = result["metrics"]
    if "CUDAExecutionProvider" not in result["providers"]:
        raise RuntimeError(f"CUDA session was not built: {result['providers']}")
    if not metrics["finite"] or metrics["relative_l2"] > relative_l2_max:
        raise RuntimeError(
            f"Scaled attention-output numerical gate failed: {metrics['relative_l2']} > {relative_l2_max}"
        )
    return {**inventory, **result}


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
    with tempfile.TemporaryDirectory(prefix="scaled-attention-", dir=block_dir) as temporary_name:
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
            kind="dit_attention_output",
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
        else {"format": "h3-scaled-fp16-attention-output-v1", "blocks": {}}
    )
    timestep_embedding = _conditioning(args.product, args.rows)
    for block in args.blocks:
        block_dir = args.output / f"block_{block:02d}"
        record = state["blocks"].get(str(block))
        if _record_valid(record, block_dir):
            logger.write("block_skipped", durable=True, block=block)
            continue
        pending = sum(
            not (args.output / f"block_{item:02d}" / "scaled_fp16.onnx.data").is_file()
            for item in args.blocks
            if item >= block
        )
        free_bytes = shutil.disk_usage(args.output).free
        projected = _space_projection(
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
            projected_free_gib=round(projected / GIB, 3),
        )
        logger.write("block_export_started", durable=True, block=block)
        export_seconds, validation = _export_block(
            args.source,
            block,
            block_dir,
            _inputs(timestep_embedding, block, args.rows),
            args.relative_l2_max,
        )
        record = {
            "status": "completed",
            "export_seconds": round(export_seconds, 3),
            "files": _file_record(_artifact_paths(block_dir)),
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
        )
        print(json.dumps({"block": block + 1, "blocks": len(args.blocks)}), flush=True)
    return state


def _link_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif relative.as_posix() not in {"manifest.json", "schedule.json"}:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)


def build_hybrid(source: Path, candidates: Path, target: Path, state: dict[str, Any]) -> Path:
    source = source.resolve()
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")
    staging = target.with_name(f"{target.name}.staging")
    if staging.exists():
        raise FileExistsError(f"Staging already exists: {staging}")
    missing = [
        block
        for block in range(MLP_BLOCKS)
        if not _record_valid(state["blocks"].get(str(block)), candidates / f"block_{block:02d}")
    ]
    if missing:
        raise RuntimeError(f"Scaled attention-output blocks are incomplete: {missing}")
    staging.mkdir(parents=True)
    _link_tree(source, staging)
    schedule = json.loads((source / "schedule.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    artifacts = dict(source_manifest["artifacts"])
    by_id = {shard["id"]: shard for shard in schedule["shards"]}
    for block in range(MLP_BLOCKS):
        shard_id = f"shard_{9 + block * 3:03d}"
        relative = Path("scaled_attention") / f"block_{block:02d}" / "scaled_fp16.onnx"
        source_dir = candidates / f"block_{block:02d}"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_dir / "scaled_fp16.onnx", destination)
        os.link(source_dir / "scaled_fp16.onnx.data", staging / f"{relative}.data")
        record = state["blocks"][str(block)]
        by_id[shard_id]["file"] = relative.as_posix()
        by_id[shard_id]["storage_bytes"] = int(record["storage_bytes"])
        artifacts[relative.as_posix()] = record["files"]["scaled_fp16.onnx"]
        artifacts[f"{relative.as_posix()}.data"] = record["files"]["scaled_fp16.onnx.data"]
    schedule["model"] = target.name
    schedule["precision"]["attention_output_compute"] = "scaled_fp16_tensor_core"
    schedule_path = write_schedule(schedule, staging / "schedule.json")
    artifacts["schedule.json"] = {
        "bytes": schedule_path.stat().st_size,
        "sha256": file_sha256(schedule_path),
    }
    manifest = {
        **source_manifest,
        "conversion": "scaled_fp16_tensor_core_mlp_and_attention_output",
        "validation_passed": False,
        "hybrid_chain_validation": "scaled_attention_pending",
        "source_model_directory": str(source),
        "artifacts": artifacts,
    }
    _atomic_json(staging / "manifest.json", manifest)
    validate_runtime_schedule(schedule, staging)
    os.replace(staging, target)
    return target


def prune_unreferenced_root_shards(model_dir: Path) -> dict[str, int]:
    """Remove obsolete packed root shards after direct replacements are published."""
    model_dir = model_dir.resolve()
    schedule = json.loads((model_dir / "schedule.json").read_text(encoding="utf-8"))
    referenced = {
        name
        for shard in schedule["shards"]
        for name in (str(shard["file"]), f"{shard['file']}.data")
    }
    removed: list[Path] = []
    removed_bytes = 0
    for path in model_dir.glob("shard_*.onnx*"):
        if path.name not in referenced:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed.append(path)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    for path in removed:
        artifacts.pop(path.name, None)
    manifest["pruned_unreferenced_root_shards"] = len(removed)
    _atomic_json(manifest_path, manifest)
    validate_runtime_schedule(schedule, model_dir)
    return {"files": len(removed), "logical_bytes": removed_bytes}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export scaled-FP16 attention-output graphs")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--hybrid-output", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", default=list(range(MLP_BLOCKS)))
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--relative-l2-max", type=float, default=2e-3)
    parser.add_argument("--min-free-gib", type=float, default=90.0)
    parser.add_argument("--estimated-block-bytes", type=int, default=ESTIMATED_BLOCK_BYTES)
    return parser


def run(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    started = time.perf_counter()
    try:
        state = export_all(args, logger)
        hybrid = build_hybrid(args.product, args.output, args.hybrid_output, state) if args.hybrid_output else None
        logger.write(
            "completed",
            durable=True,
            elapsed_seconds_total=round(time.perf_counter() - started, 3),
            hybrid=str(hybrid) if hybrid else None,
            free_gib=round(shutil.disk_usage(args.output).free / GIB, 3),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - persist batch failure
        logger.write("failed", durable=True, error=str(exc))
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        logger.close()


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
