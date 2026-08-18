from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors.torch import save_file

from h3_workbench.ref2va_virtual_slicer import (
    Ref2VASourceWeights,
    build_virtual_ref2va_product,
    ref2va_virtual_ready,
)


def _write_source(path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(50):
        value = float(layer + 1)
        tensors.update(
            {
                f"blocks.{layer}.attn.qkv_proj.weight": torch.full((6, 4), value, dtype=torch.bfloat16),
                f"blocks.{layer}.attn.q_norm.weight": torch.full((2,), value, dtype=torch.bfloat16),
                f"blocks.{layer}.attn.k_norm.weight": torch.full((2,), value + 1, dtype=torch.bfloat16),
                f"blocks.{layer}.attn.out_proj.weight": torch.full((4, 2), value, dtype=torch.bfloat16),
                f"blocks.{layer}.mlp.fc1.weight": torch.full((6, 4), value, dtype=torch.bfloat16),
                f"blocks.{layer}.mlp.fc2.weight": torch.full((4, 3), value, dtype=torch.bfloat16),
                f"blocks.{layer}.norm1.weight": torch.full((4,), value, dtype=torch.bfloat16),
                f"blocks.{layer}.norm2.weight": torch.full((4,), value + 2, dtype=torch.bfloat16),
                f"blocks.{layer}.adaln_proj.linear.weight": torch.full((6, 1), value, dtype=torch.bfloat16),
                f"blocks.{layer}.adaln_proj.linear.bias": torch.full((6,), value, dtype=torch.bfloat16),
            }
        )
    save_file(tensors, path)


def _topology(path: Path, weights: list[tuple[str, list[int], int]]) -> None:
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    initializers = [
        numpy_helper.from_array(np.ones(shape, dtype=np.float16 if dtype == TensorProto.FLOAT16 else np.float32), name)
        for name, shape, dtype in weights
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["hidden"], ["output"])],
        path.stem,
        [hidden],
        [output],
        initializers,
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9),
        path,
    )


def _topologies(root: Path) -> dict[str, Path]:
    qkv = root / "attention_qkv.onnx"
    _topology(
        qkv,
        [
            ("attention.qkv.weight", [6, 4], TensorProto.FLOAT16),
            ("modulation.linear.weight", [6, 1], TensorProto.FLOAT),
            ("modulation.linear.bias", [6], TensorProto.FLOAT),
            ("_to_copy", [4], TensorProto.FLOAT),
            ("_to_copy_3", [2], TensorProto.FLOAT),
            ("_to_copy_4", [2], TensorProto.FLOAT),
        ],
    )
    output = root / "attention_output.onnx"
    _topology(
        output,
        [
            ("out.weight", [4, 2], TensorProto.FLOAT16),
            ("modulation.linear.weight", [6, 1], TensorProto.FLOAT),
            ("modulation.linear.bias", [6], TensorProto.FLOAT),
        ],
    )
    mlp = root / "mlp.onnx"
    _topology(
        mlp,
        [
            ("mlp.fc1.weight", [6, 4], TensorProto.FLOAT16),
            ("mlp.fc2.weight", [4, 3], TensorProto.FLOAT16),
            ("modulation.linear.weight", [6, 1], TensorProto.FLOAT),
            ("modulation.linear.bias", [6], TensorProto.FLOAT),
            ("_to_copy", [4], TensorProto.FLOAT),
        ],
    )
    return {
        "attention_qkv": qkv,
        "attention_output": output,
        "mlp": mlp,
    }


def test_ref2va_virtual_slicer_maps_generated_weight_inputs_and_reads_layer(tmp_path: Path) -> None:
    source = tmp_path / "ref2va.safetensors"
    _write_source(source)
    output = tmp_path / "ref2va_virtual"
    build_virtual_ref2va_product(source, output, _topologies(tmp_path))

    assert ref2va_virtual_ready(output)
    runtime = json.loads((output / "runtime_ref2va_manifest.json").read_text(encoding="utf-8"))
    qkv_inputs = {item["name"]: item for item in runtime["kinds"]["attention_qkv"]["inputs"]}
    assert qkv_inputs["_to_copy"]["source_key"] == "blocks.{layer}.norm1.weight"
    assert qkv_inputs["_to_copy_3"]["source_key"] == "blocks.{layer}.attn.q_norm.weight"
    assert qkv_inputs["_to_copy_4"]["source_key"] == "blocks.{layer}.attn.k_norm.weight"
    assert runtime["kinds"]["mlp"]["inputs"][-1]["source_key"] == "blocks.{layer}.norm2.weight"

    graph = "main_block_24_attention_qkv"
    weights = Ref2VASourceWeights(output, object(), {graph: output / f"{graph}.onnx"})
    feeds = weights.inputs(graph)
    np.testing.assert_array_equal(feeds["attention.qkv.weight"], np.full((6, 4), 25, dtype=np.float16))
    np.testing.assert_array_equal(feeds["_to_copy_4"], np.full((2,), 26, dtype=np.float32))
    assert weights.supports(graph)
    weights.close()


def test_ref2va_virtual_ready_rejects_changed_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "ref2va.safetensors"
    _write_source(source)
    output = tmp_path / "ref2va_virtual"
    build_virtual_ref2va_product(source, output, _topologies(tmp_path))
    assert ref2va_virtual_ready(output)

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert not ref2va_virtual_ready(output)
