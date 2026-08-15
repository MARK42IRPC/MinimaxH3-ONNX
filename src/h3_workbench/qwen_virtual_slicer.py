from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from h3_workbench.main_transformer import _StreamingSafeTensorFile
from h3_workbench.qwen_persistent import (
    INT8_VIRTUAL_FORMAT,
    RUNTIME_INT8_MANIFEST,
    qwen_source_identity,
    runtime_file_identity,
)
from h3_workbench.qwen_transformer import QwenCheckpointReader, regular_hadamard


ATTENTION_SOURCES = {
    "input_norm": "model.layers.{layer}.input_layernorm.weight",
    "q_norm": "model.layers.{layer}.self_attn.q_norm.weight",
    "k_norm": "model.layers.{layer}.self_attn.k_norm.weight",
    "query.linear.weight.int8": "model.layers.{layer}.self_attn.q_proj.weight",
    "query.linear.weight.scale": "model.layers.{layer}.self_attn.q_proj.weight_scale",
    "key.linear.weight.int8": "model.layers.{layer}.self_attn.k_proj.weight",
    "key.linear.weight.scale": "model.layers.{layer}.self_attn.k_proj.weight_scale",
    "value.linear.weight.int8": "model.layers.{layer}.self_attn.v_proj.weight",
    "value.linear.weight.scale": "model.layers.{layer}.self_attn.v_proj.weight_scale",
    "output.linear.weight.int8": "model.layers.{layer}.self_attn.o_proj.weight",
    "output.linear.weight.scale": "model.layers.{layer}.self_attn.o_proj.weight_scale",
}
MLP_SOURCES = {
    "norm": "model.layers.{layer}.post_attention_layernorm.weight",
    "gate.linear.weight.int8": "model.layers.{layer}.mlp.gate_proj.weight",
    "gate.linear.weight.scale": "model.layers.{layer}.mlp.gate_proj.weight_scale",
    "up.linear.weight.int8": "model.layers.{layer}.mlp.up_proj.weight",
    "up.linear.weight.scale": "model.layers.{layer}.mlp.up_proj.weight_scale",
    "down.linear.weight.int8": "model.layers.{layer}.mlp.down_proj.weight",
    "down.linear.weight.scale": "model.layers.{layer}.mlp.down_proj.weight_scale",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _topology_specs(
    topology: Path,
    sources: dict[str, str],
    checkpoint: _StreamingSafeTensorFile,
) -> list[dict[str, Any]]:
    model = onnx.load(str(topology), load_external_data=False)
    graph_inputs = {value.name: value for value in model.graph.input}
    result: list[dict[str, Any]] = []
    for name, source_key in sources.items():
        try:
            value = graph_inputs[name]
        except KeyError as exc:
            raise ValueError(f"Topology {topology.name} is missing weight input {name}") from exc
        tensor = value.type.tensor_type
        target_shape = [int(dimension.dim_value) for dimension in tensor.shape.dim]
        for layer in range(50):
            source_shape = checkpoint.tensor_shape(source_key.format(layer=layer))
            source_non_unit = tuple(value for value in source_shape if value != 1)
            target_non_unit = tuple(value for value in target_shape if value != 1)
            if source_non_unit != target_non_unit:
                raise ValueError(
                    f"Qwen tensor shape mismatch for {name}, layer {layer}: "
                    f"source={source_shape}, target={target_shape}"
                )
        target_dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        result.append(
            {
                "name": name,
                "source_key": source_key,
                "target_dtype": target_dtype.str,
                "target_shape": target_shape,
            }
        )
    return result


def _validate_convrot_topology(topology: Path, sources: dict[str, str], group_size: int) -> None:
    model = onnx.load(str(topology), load_external_data=False)
    initializers = {value.name: value for value in model.graph.initializer}
    expected = regular_hadamard(group_size).numpy()
    node_inputs = {name for node in model.graph.node if node.op_type == "MatMul" for name in node.input}
    prefixes = {name.split(".linear.weight.int8", 1)[0] for name in sources if name.endswith(".int8")}
    for prefix in prefixes:
        name = f"{prefix}.convrot_hadamard"
        try:
            actual = numpy_helper.to_array(initializers[name])
        except KeyError as exc:
            raise ValueError(f"Topology {topology.name} is missing ConvRot initializer {name}") from exc
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f"Topology {topology.name} has an invalid ConvRot initializer {name}")
        if name not in node_inputs:
            raise ValueError(f"Topology {topology.name} does not apply ConvRot initializer {name}")


def build_virtual_qwen_product(
    source: Path,
    output: Path,
    attention_topology: Path,
    mlp_topology: Path,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_identity = qwen_source_identity(source)
    checkpoint = _StreamingSafeTensorFile(source)
    reader = QwenCheckpointReader(source)
    convrot_group_size: int | None = None
    if reader.source_quantization == "int8_tensorwise_convrot":
        group_sizes = {
            reader.convrot_group_size(template.format(layer=layer).removesuffix(".weight"))
            for layer in range(50)
            for name, template in {**ATTENTION_SOURCES, **MLP_SOURCES}.items()
            if name.endswith(".int8")
        }
        if len(group_sizes) != 1 or None in group_sizes:
            raise ValueError(f"Qwen ConvRot group sizes are inconsistent: {group_sizes}")
        convrot_group_size = int(next(iter(group_sizes)))
    attention_name = "runtime_qwen_attention_int8.onnx"
    mlp_name = "runtime_qwen_mlp_int8.onnx"
    # Rebuilding an existing product must revoke its old validation before any
    # topology is replaced. Each topology is then published atomically.
    manifest = {
        "format": "h3-workbench-onnx-v2",
        "source": str(source),
        "component": "text_encoder",
        "source_quantization": QwenCheckpointReader(source).source_quantization,
        "conversion": "virtual_int8_weight_slices_gpu_dequant_online_convrot_persistent_topologies",
        "activation_dtype": "float32",
        "validation_passed": False,
        "build_complete": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_copy(attention_topology.resolve(), output / attention_name)
    _atomic_copy(mlp_topology.resolve(), output / mlp_name)
    if convrot_group_size is not None:
        _validate_convrot_topology(output / attention_name, ATTENTION_SOURCES, convrot_group_size)
        _validate_convrot_topology(output / mlp_name, MLP_SOURCES, convrot_group_size)
    runtime = {
        "format": INT8_VIRTUAL_FORMAT,
        "source_checkpoint": str(source),
        "source_size": source.stat().st_size,
        "source_identity": source_identity,
        "layers": 50,
        "embedding_key": "model.embed_tokens.weight",
        "convrot": {
            "enabled": convrot_group_size is not None,
            "group_size": convrot_group_size,
            "transform": "normalized_regular_hadamard",
            "application": "online_activation",
        },
        "kinds": {
            "attention": {
                "graph": attention_name,
                "graph_identity": runtime_file_identity(output / attention_name),
                "inputs": _topology_specs(output / attention_name, ATTENTION_SOURCES, checkpoint),
            },
            "mlp": {
                "graph": mlp_name,
                "graph_identity": runtime_file_identity(output / mlp_name),
                "inputs": _topology_specs(output / mlp_name, MLP_SOURCES, checkpoint),
            },
        },
    }
    if qwen_source_identity(source) != source_identity:
        raise RuntimeError(f"Qwen source changed during virtual slicing: {source}")
    _atomic_json(output / RUNTIME_INT8_MANIFEST, runtime)
    manifest.update(
        {
            "build_complete": True,
            "architecture": {
                "hidden_size": 5120,
                "layers": 50,
                "heads": 64,
                "kv_heads": 8,
                "head_dim": 128,
                "intermediate_size": 25600,
            },
            "graphs": [attention_name, mlp_name],
            "weight_storage": "source_safetensors_zero_copy",
            "embedding": "source_bf16_cpu_row_gather",
            "convrot_group_size": convrot_group_size,
        }
    )
    _atomic_json(output / "manifest.json", manifest)
    return output / RUNTIME_INT8_MANIFEST


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build zero-copy Qwen INT8 virtual slices")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-topology", type=Path, required=True)
    parser.add_argument("--mlp-topology", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(build_virtual_qwen_product(args.source, args.output, args.attention_topology, args.mlp_topology))


if __name__ == "__main__":
    main()
