from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file

from h3_workbench.jobs import _close_qwen_runtime_weights
from h3_workbench.qwen_persistent import (
    INT8_VIRTUAL_FORMAT,
    QwenInt8SourceWeights,
    int8_virtual_product_fingerprint,
    int8_virtual_qwen_ready,
    qwen_source_identity,
    resolve_qwen_directory,
    runtime_file_identity,
    validated_int8_virtual_qwen_ready,
)
from h3_workbench.qwen_int8_graph import install_split_qwen_attention
from h3_workbench.qwen_virtual_validation import run as run_virtual_validation
from h3_workbench.qwen_virtual_slicer import (
    ATTENTION_SOURCES,
    MLP_SOURCES,
    _validate_convrot_topology,
    build_virtual_qwen_product,
)
from h3_workbench.qwen_transformer import regular_hadamard


def _virtual_product(root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    directory = root / "qwen3vl_32b_minimax_h3_int8_virtual"
    directory.mkdir(parents=True)
    source = root / "qwen-int8.safetensors"
    embedding = np.arange(12, dtype=np.float32).reshape(4, 3)
    weight = np.asarray([[1, -2, 3], [-4, 5, -6]], dtype=np.int8)
    scale = np.asarray([[0.5], [0.25]], dtype=np.float32)
    save_file(
        {
            "embedding": embedding,
            "layer.0.weight": weight,
            "layer.0.scale": scale,
        },
        source,
    )
    for name in ("attention.onnx", "mlp.onnx"):
        (directory / name).write_bytes(b"graph")
    kind = {
        "graph": "attention.onnx",
        "inputs": [
            {
                "name": "weight",
                "source_key": "layer.{layer}.weight",
                "target_dtype": "|i1",
                "target_shape": [2, 3],
            },
            {
                "name": "scale",
                "source_key": "layer.{layer}.scale",
                "target_dtype": "<f2",
                "target_shape": [2],
            },
        ],
    }
    (directory / "runtime_int8_manifest.json").write_text(
        json.dumps(
            {
                "format": INT8_VIRTUAL_FORMAT,
                "source_checkpoint": str(source),
                "source_size": source.stat().st_size,
                "source_identity": qwen_source_identity(source),
                "layers": 50,
                "embedding_key": "embedding",
                "kinds": {
                    "attention": {
                        **kind,
                        "graph_identity": runtime_file_identity(directory / "attention.onnx"),
                    },
                    "mlp": {
                        **kind,
                        "graph": "mlp.onnx",
                        "graph_identity": runtime_file_identity(directory / "mlp.onnx"),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"component": "text_encoder", "validation_passed": False}),
        encoding="utf-8",
    )
    return directory, embedding, weight


def test_virtual_qwen_reads_source_offsets_and_converts_small_inputs(tmp_path: Path) -> None:
    directory, embedding, weight = _virtual_product(tmp_path)

    source = QwenInt8SourceWeights(directory)
    inputs = source.inputs("attention", 0)

    np.testing.assert_array_equal(source.embedding(np.asarray([3, 1])), embedding[[3, 1]])
    np.testing.assert_array_equal(inputs["weight"], weight)
    np.testing.assert_array_equal(inputs["scale"], np.asarray([0.5, 0.25], dtype=np.float16))
    assert isinstance(inputs["weight"], np.memmap)
    assert inputs["scale"].dtype == np.float16
    assert source.inputs("attention", 0) is inputs
    assert int8_virtual_qwen_ready(directory)
    assert not validated_int8_virtual_qwen_ready(directory)
    source.close()


def test_qwen_resolver_requires_validated_virtual_product(tmp_path: Path) -> None:
    directory, _, _ = _virtual_product(tmp_path)
    legacy = tmp_path / "qwen3vl_32b_minimax_h3_nvfp4_awq"

    assert resolve_qwen_directory(tmp_path) == legacy

    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "component": "text_encoder",
                "validation_passed": True,
                "validated_product_fingerprint": int8_virtual_product_fingerprint(directory),
                "validation": {"product_fingerprint": int8_virtual_product_fingerprint(directory)},
            }
        ),
        encoding="utf-8",
    )
    assert validated_int8_virtual_qwen_ready(directory)
    assert resolve_qwen_directory(tmp_path) == directory


def test_virtual_qwen_ready_rejects_changed_topology_and_source_identity(tmp_path: Path) -> None:
    directory, _, _ = _virtual_product(tmp_path)
    assert int8_virtual_qwen_ready(directory)

    graph = directory / "attention.onnx"
    graph.write_bytes(b"other")
    assert not int8_virtual_qwen_ready(directory)

    graph.write_bytes(b"graph")
    runtime = json.loads((directory / "runtime_int8_manifest.json").read_text(encoding="utf-8"))
    runtime["kinds"]["attention"]["graph_identity"] = runtime_file_identity(graph)
    (directory / "runtime_int8_manifest.json").write_text(json.dumps(runtime), encoding="utf-8")
    assert int8_virtual_qwen_ready(directory)

    source = Path(runtime["source_checkpoint"])
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert not int8_virtual_qwen_ready(directory)


def test_failed_validation_revokes_previous_pass(monkeypatch, tmp_path: Path) -> None:
    directory, _, _ = _virtual_product(tmp_path)
    fingerprint = int8_virtual_product_fingerprint(directory)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "component": "text_encoder",
                "validation_passed": True,
                "validated_product_fingerprint": fingerprint,
                "validation": {"product_fingerprint": fingerprint},
            }
        ),
        encoding="utf-8",
    )

    class FailingRunner:
        provider = "CPUExecutionProvider"

        def __init__(self, **_: object):
            pass

        def session(self, _: Path) -> object:
            raise RuntimeError("invalid topology")

        def close(self) -> None:
            pass

    monkeypatch.setattr("h3_workbench.qwen_virtual_validation.ORTGraphRunner", FailingRunner)
    args = Namespace(
        model=directory,
        log=tmp_path / "validation.jsonl",
        blocks=[0],
        tokens=4,
        relative_l2_max=1e-3,
        full_chain_output=None,
    )

    assert run_virtual_validation(args) == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_passed"] is False
    assert "validated_product_fingerprint" not in manifest
    assert manifest["validation"]["status"] == "failed"


def test_job_qwen_cleanup_closes_both_weight_sources() -> None:
    class Weights:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Runtime:
        int8_virtual = Weights()
        persistent = Weights()

    runtime = Runtime()
    _close_qwen_runtime_weights(runtime)

    assert runtime.int8_virtual.closed
    assert runtime.persistent.closed


def test_virtual_slicer_publishes_bound_graph_and_source_identities(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.arange(12, dtype=np.float32).reshape(4, 3)
    }
    for layer in range(50):
        for name, template in {**ATTENTION_SOURCES, **MLP_SOURCES}.items():
            key = template.format(layer=layer)
            if name.endswith(".int8"):
                tensors[key] = np.arange(6, dtype=np.int8).reshape(2, 3)
            elif name.endswith(".scale"):
                tensors[key] = np.ones((2, 1), dtype=np.float32)
            else:
                tensors[key] = np.ones(2, dtype=np.float32)
    save_file(tensors, source)

    def topology(path: Path, names: dict[str, str]) -> None:
        inputs = []
        for name in names:
            if name.endswith(".int8"):
                dtype, shape = TensorProto.INT8, [2, 3]
            else:
                dtype, shape = TensorProto.FLOAT, [2]
            inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
        graph = helper.make_graph(
            [helper.make_node("Identity", [inputs[0].name], ["output"])],
            path.stem,
            inputs,
            [output],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
        onnx.save_model(model, path)

    attention = tmp_path / "attention.onnx"
    mlp = tmp_path / "mlp.onnx"
    topology(attention, ATTENTION_SOURCES)
    topology(mlp, MLP_SOURCES)
    output = tmp_path / "qwen3vl_32b_minimax_h3_int8_virtual"

    build_virtual_qwen_product(source, output, attention, mlp)

    assert int8_virtual_qwen_ready(output)
    assert not validated_int8_virtual_qwen_ready(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_complete"] is True
    assert manifest["validation_passed"] is False


def test_virtual_slicer_requires_bound_convrot_hadamard(tmp_path: Path) -> None:
    path = tmp_path / "convrot.onnx"
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    hadamard = numpy_helper.from_array(regular_hadamard(4).numpy(), "query.convrot_hadamard")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["hidden", hadamard.name], ["output"])],
        "convrot",
        [hidden],
        [output],
        [hadamard],
    )
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), path)

    _validate_convrot_topology(
        path,
        {"query.linear.weight.int8": "model.layers.{layer}.self_attn.q_proj.weight"},
        4,
    )

    graph.ClearField("initializer")
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), path)
    try:
        _validate_convrot_topology(
            path,
            {"query.linear.weight.int8": "model.layers.{layer}.self_attn.q_proj.weight"},
            4,
        )
    except ValueError as exc:
        assert "missing ConvRot initializer" in str(exc)
    else:
        raise AssertionError("a ConvRot product must reject a topology without its Hadamard transform")


def test_install_split_attention_revokes_previous_validation(monkeypatch, tmp_path: Path) -> None:
    directory, _, _ = _virtual_product(tmp_path)
    manifest_path = directory / "manifest.json"
    fingerprint = int8_virtual_product_fingerprint(directory)
    manifest_path.write_text(
        json.dumps(
            {
                "component": "text_encoder",
                "validation_passed": True,
                "validated_product_fingerprint": fingerprint,
                "validation": {"status": "passed", "product_fingerprint": fingerprint},
            }
        ),
        encoding="utf-8",
    )

    def write_split_graph(path: Path, outputs: list[str], input_name: str) -> None:
        input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1])
        output_infos = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in outputs]
        nodes = [helper.make_node("Identity", [input_name], [name]) for name in outputs]
        graph = helper.make_graph(nodes, path.stem, [input_info], output_infos)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
        onnx.save_model(model, path)

    def fake_build(_fused: Path, qkv: Path, output: Path) -> tuple[Path, Path]:
        write_split_graph(qkv, ["add_154", "add_222", "view_8"], "hidden_states")
        write_split_graph(output, ["hidden_states_out"], "attended")
        return qkv, output

    monkeypatch.setattr("h3_workbench.qwen_int8_graph.build_split_qwen_attention_graphs", fake_build)

    install_split_qwen_attention(directory)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_passed"] is False
    assert "validated_product_fingerprint" not in manifest
    assert manifest["validation"]["status"] == "invalidated"
    assert int8_virtual_qwen_ready(directory)
    assert not validated_int8_virtual_qwen_ready(directory)
