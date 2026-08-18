"""Transactional P0 packer for h3-schedule-v2 model directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
from onnx import GraphProto, TensorProto, helper, numpy_helper

from h3_workbench.main_transformer import HIDDEN_SIZE, CheckpointReader
from h3_workbench.shard_planner import (
    SHARD_MATERIALIZED_TARGET_BYTES,
    SHARD_STORAGE_TARGET_BYTES,
    build_schedule,
    file_sha256,
    graph_inventory,
    validate_schedule,
    write_schedule,
)

FFN_SIZE = 14336
BUILD_STATE = "build-state.json"
MLP_FC_SHAPES = {"fc1": [2 * FFN_SIZE, HIDDEN_SIZE], "fc2": [HIDDEN_SIZE, FFN_SIZE]}
ProgressCallback = Callable[[str, int, int], None]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {"format": "h3-pack-state-v1", "graphs": {}, "shards": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _record_complete(state_path: Path, state: dict, section: str, name: str, paths: list[Path]) -> None:
    state[section][name] = {
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
            if path.is_file()
        }
    }
    _atomic_json(state_path, state)


def _record_valid(record: dict | None, directory: Path) -> bool:
    if not record:
        return False
    files = record.get("files", {})
    if not files:
        return False
    for name, details in files.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != details.get("bytes"):
            return False
        if file_sha256(path) != details.get("sha256"):
            return False
    return True


def _artifact_paths(directory: Path, graph_name: str) -> list[Path]:
    graph = directory / f"{graph_name}.onnx"
    data = graph.with_name(f"{graph.name}.data")
    return [graph, data] if data.is_file() else [graph]


def _atomic_copy_graph(source_dir: Path, staging_dir: Path, graph_name: str) -> list[Path]:
    source_graph = source_dir / f"{graph_name}.onnx"
    source_data = source_graph.with_name(f"{source_graph.name}.data")
    with tempfile.TemporaryDirectory(prefix="h3-copy-", dir=staging_dir) as temporary_name:
        temporary = Path(temporary_name)
        copied_graph = temporary / source_graph.name
        shutil.copy2(source_graph, copied_graph)
        copied_data = temporary / source_data.name
        if source_data.is_file():
            shutil.copy2(source_data, copied_data)
            os.replace(copied_data, staging_dir / source_data.name)
        os.replace(copied_graph, staging_dir / source_graph.name)
    return _artifact_paths(staging_dir, graph_name)


def _mlp_weight_initializers(graph: GraphProto) -> dict[str, str]:
    by_shape = {tuple(initializer.dims): initializer.name for initializer in graph.initializer}
    result: dict[str, str] = {}
    for role, shape in MLP_FC_SHAPES.items():
        name = by_shape.get(tuple(shape))
        if name is None:
            raise ValueError(f"Missing {role} weight initializer with shape {shape}")
        result[role] = name
    return result


def _weight_consumers(graph: GraphProto, initializer_name: str) -> list[onnx.NodeProto]:
    consumers = [node for node in graph.node if node.op_type == "Slice" and initializer_name in node.input]
    if not consumers:
        raise ValueError(f"No Slice consumes initializer {initializer_name}")
    return consumers


def fp8_quantize_mlp_graph(
    source_path: Path,
    output_dir: Path,
    reader: CheckpointReader,
    block: int,
) -> list[Path]:
    """Requantize one source MLP into output_dir without modifying source_path."""
    model = onnx.load(str(source_path), load_external_data=True)
    graph = model.graph
    for role, initializer_name in _mlp_weight_initializers(graph).items():
        prefix = f"blocks.{block}.mlp.{role}"
        if reader.lora_factors(prefix) is not None:
            continue
        consumers = _weight_consumers(graph, initializer_name)
        fp8_values = reader.raw_tensor(f"{prefix}.weight")
        scale = float(reader.raw_tensor(f"{prefix}.weight_scale"))
        fp8_init = numpy_helper.from_array(fp8_values, f"{initializer_name}_fp8")
        scale_init = numpy_helper.from_array(np.array(scale, dtype=np.float32), f"{initializer_name}_fp8_scale")
        cast_out = f"{initializer_name}_fp8_cast"
        w32 = f"{initializer_name}_fp8_f32"
        w16 = f"{initializer_name}_fp8_f16"
        dequant_nodes = [
            helper.make_node("Cast", [fp8_init.name], [cast_out], to=TensorProto.FLOAT),
            helper.make_node("Mul", [cast_out, scale_init.name], [w32]),
            helper.make_node("Cast", [w32], [w16], to=TensorProto.FLOAT16),
        ]
        for node in reversed(dequant_nodes):
            graph.node.insert(0, node)
        for consumer in consumers:
            for index, input_name in enumerate(consumer.input):
                if input_name == initializer_name:
                    consumer.input[index] = w16
        for index, initializer in enumerate(graph.initializer):
            if initializer.name == initializer_name:
                graph.initializer.pop(index)
                break
        graph.initializer.extend([fp8_init, scale_init])
    model.ir_version = max(model.ir_version, 9)
    onnx.checker.check_model(model, full_check=False)
    with tempfile.TemporaryDirectory(prefix="h3-requant-", dir=output_dir) as temporary_name:
        temporary = Path(temporary_name)
        graph_name = source_path.name
        graph_path = temporary / graph_name
        data_name = f"{graph_name}.data"
        onnx.save_model(
            model,
            str(graph_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )
        data_path = temporary / data_name
        os.replace(data_path, output_dir / data_name)
        os.replace(graph_path, output_dir / graph_name)
    return _artifact_paths(output_dir, source_path.stem)


def _captured_inputs(sources: list[GraphProto]) -> tuple[list[onnx.ValueInfoProto], dict[str, tuple[str, int]]]:
    seen: dict[str, onnx.ValueInfoProto] = {}
    dim_sources: dict[str, tuple[str, int]] = {}
    for graph in sources:
        for value_info in graph.input:
            seen.setdefault(value_info.name, value_info)
            for axis, dim in enumerate(value_info.type.tensor_type.shape.dim):
                if dim.HasField("dim_param"):
                    dim_sources.setdefault(dim.dim_param, (f"{value_info.name}_captured", axis))
    return list(seen.values()), dim_sources


def _passthrough_branch(
    outputs: list[onnx.ValueInfoProto],
    dim_sources: dict[str, tuple[str, int]],
) -> GraphProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    counter = 0
    for output in outputs:
        dimensions: list[str] = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                source = dim_sources.get(dim.dim_param)
                if source is None:
                    raise ValueError(f"No input dimension supplies {dim.dim_param!r} for {output.name}")
                captured, axis = source
                shape_name = f"else_shape_{counter}"
                index_name = f"else_index_{counter}"
                dimension_name = f"else_dimension_{counter}"
                counter += 1
                nodes.append(helper.make_node("Shape", [captured], [shape_name]))
                initializers.append(numpy_helper.from_array(np.array([axis], dtype=np.int64), index_name))
                nodes.append(helper.make_node("Gather", [shape_name, index_name], [dimension_name], axis=0))
                dimensions.append(dimension_name)
            else:
                name = f"else_constant_dimension_{counter}"
                counter += 1
                initializers.append(numpy_helper.from_array(np.array([dim.dim_value], dtype=np.int64), name))
                dimensions.append(name)
        shape_name = f"else_output_shape_{counter}"
        counter += 1
        nodes.append(helper.make_node("Concat", dimensions, [shape_name], axis=0))
        dtype = helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type)
        value = numpy_helper.from_array(np.zeros(1, dtype=dtype), f"else_value_{counter}")
        counter += 1
        nodes.append(helper.make_node("ConstantOfShape", [shape_name], [output.name], value=value))
    return helper.make_graph(nodes, "else_passthrough", [], list(outputs), initializers)


def _renamed_branch(
    source: GraphProto,
    namespace: str,
    input_renames: dict[str, str],
    output_names: list[str],
) -> tuple[GraphProto, list[onnx.TensorProto]]:
    """Namespace every branch-local value.

    ORT resolves sibling If subgraphs through a shared parent scope. Reusing
    exporter-generated names such as ``weight`` or ``add_17`` across branches
    can make a later branch consume values from an earlier branch even though
    the ONNX protobufs are individually valid. Only captured graph inputs and
    public If outputs remain unprefixed.
    """
    nodes: list[onnx.NodeProto] = []
    source_outputs = [value.name for value in source.output]
    output_renames = dict(zip(source_outputs, output_names, strict=True))
    local_names = {initializer.name for initializer in source.initializer}
    local_names.update(name for node in source.node for name in node.output if name)
    local_renames = {
        name: output_renames.get(name, f"{namespace}/{name}")
        for name in local_names
    }
    for node in source.node:
        clone = onnx.NodeProto()
        clone.CopyFrom(node)
        if clone.name:
            clone.name = f"{namespace}/{clone.name}"
        for index, input_name in enumerate(clone.input):
            if input_name in input_renames:
                clone.input[index] = input_renames[input_name]
            elif input_name in local_renames:
                clone.input[index] = local_renames[input_name]
        for index, output_name in enumerate(clone.output):
            if output_name in local_renames:
                clone.output[index] = local_renames[output_name]
        nodes.append(clone)
    initializers: list[onnx.TensorProto] = []
    for initializer in source.initializer:
        clone = onnx.TensorProto()
        clone.CopyFrom(initializer)
        clone.name = local_renames[initializer.name]
        initializers.append(clone)
    outputs: list[onnx.ValueInfoProto] = []
    for index, value_info in enumerate(source.output):
        output = onnx.ValueInfoProto()
        output.CopyFrom(value_info)
        output.name = output_names[index]
        outputs.append(output)
    # Keep weight initializers in the shard's parent graph. ORT 1.23 can
    # confuse host-prepacked initializers owned by sibling If subgraphs, and
    # CUDA FP8 branches may subsequently access invalid device memory.
    return helper.make_graph(nodes, f"{namespace}_branch", [], outputs), initializers


def pack_shard(model_dir: Path, shard_id: str, graph_names: list[str]) -> list[Path]:
    sources = [onnx.load(str(model_dir / f"{name}.onnx")) for name in graph_names]
    inputs, dim_sources = _captured_inputs([model.graph for model in sources])
    dim_sources = {
        dimension: (f"{shard_id}/{captured}", axis)
        for dimension, (captured, axis) in dim_sources.items()
    }
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    selector_name = f"selector_{shard_id}"
    input_renames: dict[str, str] = {}
    for value_info in inputs:
        captured = f"{shard_id}/{value_info.name}_captured"
        input_renames[value_info.name] = captured
        nodes.append(
            helper.make_node(
                "Identity",
                [value_info.name],
                [captured],
                name=f"{shard_id}/capture_{value_info.name}",
            )
        )
    for index, (name, model) in enumerate(zip(graph_names, sources, strict=True)):
        selector_index = f"{shard_id}/selector_index_{index}"
        condition = f"{shard_id}/condition_{index}"
        initializers.append(
            numpy_helper.from_array(np.asarray(index, dtype=np.int64), selector_index)
        )
        nodes.append(
            helper.make_node(
                "Equal",
                [selector_name, selector_index],
                [condition],
                name=f"{shard_id}/select_{index}",
            )
        )
        output_names = [f"{name}/{output.name}" for output in model.graph.output]
        then_branch, branch_initializers = _renamed_branch(
            model.graph,
            name,
            input_renames,
            output_names,
        )
        initializers.extend(branch_initializers)
        else_branch = _passthrough_branch(list(then_branch.output), dim_sources)
        nodes.append(
            helper.make_node(
                "If",
                [condition],
                output_names,
                then_branch=then_branch,
                else_branch=else_branch,
                name=f"if_{name}",
            )
        )
    graph_inputs = list(inputs) + [
        helper.make_tensor_value_info(selector_name, TensorProto.INT64, [])
    ]
    graph_outputs: list[onnx.ValueInfoProto] = []
    for name, model in zip(graph_names, sources, strict=True):
        for source_output in model.graph.output:
            output = onnx.ValueInfoProto()
            output.CopyFrom(source_output)
            output.name = f"{name}/{source_output.name}"
            graph_outputs.append(output)
    graph = helper.make_graph(nodes, shard_id, graph_inputs, graph_outputs, initializers)
    opsets = {opset.domain: opset.version for model in sources for opset in model.opset_import}
    packed = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain, version) for domain, version in opsets.items()],
    )
    packed.ir_version = max(model.ir_version for model in sources)
    onnx.checker.check_model(packed, full_check=False)
    graph_name = f"{shard_id}.onnx"
    data_name = f"{graph_name}.data"
    with tempfile.TemporaryDirectory(prefix="h3-shard-", dir=model_dir) as temporary_name:
        temporary = Path(temporary_name)
        graph_path = temporary / graph_name
        onnx.save_model(
            packed,
            str(graph_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )
        data_path = temporary / data_name
        if data_path.is_file():
            os.replace(data_path, model_dir / data_name)
        os.replace(graph_path, model_dir / graph_name)
    return _artifact_paths(model_dir, shard_id)


def _source_files(source_dir: Path) -> list[Path]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    files = [source_dir / "manifest.json"]
    for filename in manifest["graphs"]:
        graph = source_dir / filename
        files.append(graph)
        data = graph.with_name(f"{graph.name}.data")
        if data.is_file():
            files.append(data)
    return files


def _fingerprint(files: list[Path], root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    }


def _fingerprint_digest(fingerprint: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _preflight_space(source_dir: Path, target_parent: Path) -> None:
    source_bytes = sum(path.stat().st_size for path in _source_files(source_dir))
    free_bytes = shutil.disk_usage(target_parent).free
    required = int(source_bytes * 2.1) + (4 << 30)
    if free_bytes < required:
        raise RuntimeError(
            f"Insufficient disk space: need about {required / (1 << 30):.1f} GiB, "
            f"have {free_bytes / (1 << 30):.1f} GiB"
        )


def build_sharded_model(
    source_dir: Path,
    target_dir: Path,
    checkpoint: Path,
    blocks: int = 50,
    sdpa_scale: float = 0.125,
    turbo: bool = False,
    storage_target_bytes: int = SHARD_STORAGE_TARGET_BYTES,
    materialized_target_bytes: int = SHARD_MATERIALIZED_TARGET_BYTES,
    callback: ProgressCallback | None = None,
) -> dict:
    """Build target_dir transactionally while keeping source_dir immutable."""
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    if source_dir == target_dir:
        raise ValueError("source_dir and target_dir must differ")
    if target_dir.is_dir() and (target_dir / "manifest.json").is_file():
        manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("build_complete") and manifest.get("schedule") == "schedule.json":
            return json.loads((target_dir / "schedule.json").read_text(encoding="utf-8"))
        raise FileExistsError(f"Target directory already exists but is incomplete: {target_dir}")
    staging_dir = target_dir.with_name(f"{target_dir.name}.staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    _preflight_space(source_dir, staging_dir.parent)
    state_path = staging_dir / BUILD_STATE
    state = _load_state(state_path)
    source_files = _source_files(source_dir)
    if "source_fingerprint" not in state:
        state["source_fingerprint"] = _fingerprint(source_files, source_dir)
        _atomic_json(state_path, state)
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    # graph_inventory reads the graph list from manifest. This provisional
    # manifest is confined to the undiscoverable staging directory and is
    # replaced by the verified final manifest before publication.
    _atomic_json(staging_dir / "manifest.json", source_manifest)
    graph_names = [filename.removesuffix(".onnx") for filename in source_manifest["graphs"]]
    reader = CheckpointReader(checkpoint)
    virtual_source = str(source_manifest.get("weight_storage", "")).startswith(
        "source_safetensors"
    )
    total_graphs = len(graph_names)
    for index, graph_name in enumerate(graph_names, 1):
        if callback:
            callback("graphs", index, total_graphs)
        if _record_valid(state["graphs"].get(graph_name), staging_dir):
            continue
        source_graph = source_dir / f"{graph_name}.onnx"
        match = re_match_mlp(graph_name)
        if match is None or virtual_source:
            paths = _atomic_copy_graph(source_dir, staging_dir, graph_name)
        else:
            paths = fp8_quantize_mlp_graph(source_graph, staging_dir, reader, match)
        _record_complete(state_path, state, "graphs", graph_name, paths)

    inventory = graph_inventory(staging_dir)
    schedule = build_schedule(
        inventory,
        model_id=target_dir.name,
        blocks=blocks,
        sdpa_scale=sdpa_scale,
        turbo=turbo,
        storage_target_bytes=storage_target_bytes,
        materialized_target_bytes=materialized_target_bytes,
    )
    validate_schedule(schedule, staging_dir)
    total_shards = len(schedule["shards"])
    for index, shard in enumerate(schedule["shards"], 1):
        if callback:
            callback("shards", index, total_shards)
        shard_id = shard["id"]
        if _record_valid(state["shards"].get(shard_id), staging_dir):
            continue
        paths = pack_shard(staging_dir, shard_id, shard["graphs"])
        _record_complete(state_path, state, "shards", shard_id, paths)

    schedule_path = write_schedule(schedule, staging_dir / "schedule.json")
    artifacts = [schedule_path]
    for shard in schedule["shards"]:
        artifacts.extend(_artifact_paths(staging_dir, shard["id"]))
    final_manifest = {
        **source_manifest,
        "format": "h3-workbench-onnx-v2",
        "source_model_directory": str(source_dir),
        "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
        "source_tree_sha256": _fingerprint_digest(state["source_fingerprint"]),
        "schedule": "schedule.json",
        "schedule_format": schedule["format"],
        "shard_count": len(schedule["shards"]),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in artifacts
        },
        "build_complete": True,
    }
    _atomic_json(staging_dir / "manifest.json", final_manifest)
    after_fingerprint = _fingerprint(source_files, source_dir)
    if after_fingerprint != state["source_fingerprint"]:
        raise RuntimeError("Source model directory changed during packing")
    state_path.unlink(missing_ok=True)
    if target_dir.exists():
        raise FileExistsError(f"Cannot publish over existing target: {target_dir}")
    os.replace(staging_dir, target_dir)
    return schedule


def re_match_mlp(graph_name: str) -> int | None:
    prefix = "main_block_"
    suffix = "_mlp"
    if not graph_name.startswith(prefix) or not graph_name.endswith(suffix):
        return None
    return int(graph_name[len(prefix) : -len(suffix)])


def repack_published_shards(
    model_dir: Path,
    callback: ProgressCallback | None = None,
) -> dict:
    """Atomically rebuild shard containers from a published directory's fine graphs."""
    model_dir = model_dir.resolve()
    schedule = json.loads((model_dir / "schedule.json").read_text(encoding="utf-8"))
    validate_schedule(schedule, model_dir)
    for index, shard in enumerate(schedule["shards"], 1):
        if callback:
            callback("repack", index, len(schedule["shards"]))
        pack_shard(model_dir, shard["id"], shard["graphs"])
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    for shard in schedule["shards"]:
        for path in _artifact_paths(model_dir, shard["id"]):
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest.pop("shard_validation", None)
    for name in ("shard-validation.json", "shard-validation-failure.json"):
        (model_dir / name).unlink(missing_ok=True)
        artifacts.pop(name, None)
    _atomic_json(manifest_path, manifest)
    return schedule


def replan_published_shards(
    model_dir: Path,
    blocks: int = 50,
    callback: ProgressCallback | None = None,
) -> dict:
    """Rebuild a published shard layout without repeating source conversion."""
    model_dir = model_dir.resolve()
    old_schedule = json.loads((model_dir / "schedule.json").read_text(encoding="utf-8"))
    sdpa = next(step for step in old_schedule["steps"] if step.get("op") == "sdpa")
    turbo = any(
        "silu_timestep_embedding" in step.get("outputs", {})
        for step in old_schedule["steps"]
    )
    schedule = build_schedule(
        graph_inventory(model_dir),
        model_id=old_schedule["model"],
        blocks=blocks,
        sdpa_scale=float(sdpa["params"]["scale"]),
        turbo=turbo,
    )
    validate_schedule(schedule, model_dir)
    for index, shard in enumerate(schedule["shards"], 1):
        if callback:
            callback("replan", index, len(schedule["shards"]))
        pack_shard(model_dir, shard["id"], shard["graphs"])

    expected = {
        name
        for shard in schedule["shards"]
        for name in (shard["file"], f"{shard['file']}.data")
    }
    write_schedule(schedule, model_dir / "schedule.json")
    for path in model_dir.glob("shard_*.onnx*"):
        if path.name not in expected:
            path.unlink()

    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    for name in list(artifacts):
        if name.startswith("shard_") and name not in expected:
            del artifacts[name]
    for name in sorted(expected):
        path = model_dir / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    schedule_path = model_dir / "schedule.json"
    artifacts[schedule_path.name] = {
        "bytes": schedule_path.stat().st_size,
        "sha256": file_sha256(schedule_path),
    }
    manifest["shard_count"] = len(schedule["shards"])
    manifest.pop("shard_validation", None)
    for name in ("shard-validation.json", "shard-validation-failure.json"):
        (model_dir / name).unlink(missing_ok=True)
        artifacts.pop(name, None)
    _atomic_json(manifest_path, manifest)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Transactionally build h3-schedule-v2 shards")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--sdpa-scale", type=float, default=0.125)
    parser.add_argument("--turbo", action="store_true")
    args = parser.parse_args()

    def report(phase: str, current: int, total: int) -> None:
        print(f"[{phase}] {current}/{total}", flush=True)

    schedule = build_sharded_model(
        args.source,
        args.target,
        args.checkpoint,
        blocks=args.blocks,
        sdpa_scale=args.sdpa_scale,
        turbo=args.turbo,
        callback=report,
    )
    print(f"Published {len(schedule['shards'])} shards -> {args.target}")


if __name__ == "__main__":
    main()
