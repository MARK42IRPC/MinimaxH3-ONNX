import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from safetensors.numpy import save_file

from h3_workbench.shard_packer import build_sharded_model, pack_shard


def _identity_model(path: Path, input_name: str, output_name: str, width: int = 2, bias: float = 0.0) -> None:
    source = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["sequence", width])
    output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["sequence", width])
    initializer = helper.make_tensor("bias", TensorProto.FLOAT, [1], [bias])
    graph = helper.make_graph(
        [helper.make_node("Add", [input_name, "bias"], [output_name], name="shared_node")],
        "shared_exporter_graph_name",
        [source],
        [output],
        [initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.save(model, path)


def test_if_shard_runs_only_selected_output(tmp_path: Path) -> None:
    _identity_model(tmp_path / "first.onnx", "first_input", "first_output", bias=1.0)
    _identity_model(tmp_path / "second.onnx", "second_input", "second_output", bias=2.0)
    pack_shard(tmp_path, "shard_000", ["first", "second"])
    packed = onnx.load(str(tmp_path / "shard_000.onnx"), load_external_data=False)
    assert {initializer.name for initializer in packed.graph.initializer} == {
        "first/bias",
        "second/bias",
        "shard_000/selector_index_0",
        "shard_000/selector_index_1",
    }
    assert all(
        not branch.g.initializer
        for node in packed.graph.node
        for branch in node.attribute
        if branch.name == "then_branch"
    )
    session = ort.InferenceSession(str(tmp_path / "shard_000.onnx"), providers=["CPUExecutionProvider"])
    value = np.arange(6, dtype=np.float32).reshape(3, 2)
    outputs = session.run(
        ["first/first_output"],
        {
                "first_input": value,
                "second_input": np.zeros_like(value),
                "selector_shard_000": np.array(0, dtype=np.int64),
        },
    )
    assert np.array_equal(outputs[0], value + 1.0)
    second = session.run(
        ["second/second_output"],
        {
                "first_input": np.zeros_like(value),
                "second_input": value,
                "selector_shard_000": np.array(1, dtype=np.int64),
        },
    )
    assert np.array_equal(second[0], value + 2.0)


def _constant_model(path: Path, inputs: dict[str, int], outputs: dict[str, int]) -> None:
    graph_inputs = [helper.make_tensor_value_info(name, dtype, ["sequence", 1]) for name, dtype in inputs.items()]
    graph_outputs = [helper.make_tensor_value_info(name, dtype, ["sequence", 1]) for name, dtype in outputs.items()]
    nodes = []
    for name, dtype in outputs.items():
        value = helper.make_tensor(f"{name}_value", dtype, [1, 1], [0])
        nodes.append(helper.make_node("Constant", [], [name], value=value))
    model = helper.make_model(helper.make_graph(nodes, path.stem, graph_inputs, graph_outputs))
    model.ir_version = 9
    onnx.save(model, path)


def test_transactional_build_publishes_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    ports = {
        "main_embeddings": (
            {"video_patches": TensorProto.FLOAT, "audio_patches": TensorProto.FLOAT, "text_states": TensorProto.FLOAT16},
            {"video_embeddings": TensorProto.FLOAT, "audio_embeddings": TensorProto.FLOAT, "text_embeddings": TensorProto.FLOAT},
        ),
        "main_token_refiner_block_00_attention": ({"hidden_states": TensorProto.FLOAT}, {"hidden_states_out": TensorProto.FLOAT}),
        "main_token_refiner_block_00_mlp": ({"hidden_states": TensorProto.FLOAT}, {"hidden_states_out": TensorProto.FLOAT}),
        "main_token_refiner_block_01_attention": ({"hidden_states": TensorProto.FLOAT}, {"hidden_states_out": TensorProto.FLOAT}),
        "main_token_refiner_block_01_mlp": ({"hidden_states": TensorProto.FLOAT}, {"hidden_states_out": TensorProto.FLOAT}),
        "main_token_refiner_norm": ({"hidden_states": TensorProto.FLOAT}, {"hidden_states_out": TensorProto.FLOAT}),
        "main_conditioning": (
            {"timesteps": TensorProto.FLOAT, "position_ids": TensorProto.FLOAT},
            {"timestep_embedding": TensorProto.FLOAT, "rotary_table": TensorProto.FLOAT16},
        ),
        "main_head": (
            {
                "video_hidden": TensorProto.FLOAT,
                "audio_hidden": TensorProto.FLOAT,
                "video_timestep_embedding": TensorProto.FLOAT,
                "audio_timestep_embedding": TensorProto.FLOAT,
            },
            {"video_patches": TensorProto.FLOAT, "audio_patches": TensorProto.FLOAT},
        ),
    }
    for name, (inputs, outputs) in ports.items():
        _constant_model(source / f"{name}.onnx", inputs, outputs)
    (source / "manifest.json").write_text(
        json.dumps({"format": "test", "graphs": [f"{name}.onnx" for name in ports]}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file({"unused": np.zeros(1, dtype=np.float32)}, checkpoint)
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    first = build_sharded_model(source, target, checkpoint, blocks=0)
    second = build_sharded_model(source, target, checkpoint, blocks=0)
    assert first == second
    assert (target / "schedule.json").is_file()
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_complete"] is True
    assert manifest["schedule_format"] == "h3-schedule-v2"
    assert not target.with_name("target.staging").exists()
    assert source_before == {path.name: path.read_bytes() for path in source.iterdir()}
