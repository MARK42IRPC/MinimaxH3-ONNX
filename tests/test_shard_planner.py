import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.shard_planner import (
    SCHEDULE_FORMAT,
    GraphInfo,
    build_schedule,
    graph_inventory,
    plan_shards,
    validate_schedule,
)

MIB = 1 << 20
GIB = 1 << 30


def make_inventory(blocks: int = 50) -> list[GraphInfo]:
    graphs = [
        GraphInfo("main_embeddings", "embeddings", None, 100 * MIB, 100 * MIB, "preamble"),
        GraphInfo("main_token_refiner_block_00_attention", "refiner_attention", 0, 100 * MIB, 100 * MIB, "preamble"),
        GraphInfo("main_token_refiner_block_00_mlp", "refiner_mlp", 0, 100 * MIB, 100 * MIB, "preamble"),
        GraphInfo("main_token_refiner_block_01_attention", "refiner_attention", 1, 100 * MIB, 100 * MIB, "preamble"),
        GraphInfo("main_token_refiner_block_01_mlp", "refiner_mlp", 1, 100 * MIB, 100 * MIB, "preamble"),
        GraphInfo("main_token_refiner_norm", "refiner_norm", None, 100 * MIB, 100 * MIB, "preamble"),
        GraphInfo("main_conditioning", "conditioning", None, 50 * MIB, 50 * MIB),
        GraphInfo("main_head", "head", None, 50 * MIB, 50 * MIB),
    ]
    for block in range(blocks):
        graphs.extend(
            [
                GraphInfo(f"main_block_{block:02d}_attention_qkv", "qkv", block, 300 * MIB, 300 * MIB),
                GraphInfo(f"main_block_{block:02d}_attention_output", "attn_out", block, 100 * MIB, 100 * MIB),
                GraphInfo(f"main_block_{block:02d}_mlp", "mlp", block, 225 * MIB, 450 * MIB),
            ]
        )
    return graphs


def test_each_graph_has_an_isolated_shard() -> None:
    shards = plan_shards(make_inventory())
    assert len(shards) == 158
    assert shards[0].resident
    assert all(len(shard.graphs) == 1 for shard in shards)
    for block in range(50):
        qkv = shards[8 + block * 3]
        attention = shards[9 + block * 3]
        mlp = shards[10 + block * 3]
        assert not qkv.resident and not attention.resident and not mlp.resident
        assert [qkv.graphs, attention.graphs, mlp.graphs] == [
            [f"main_block_{block:02d}_attention_qkv"],
            [f"main_block_{block:02d}_attention_output"],
            [f"main_block_{block:02d}_mlp"],
        ]
        assert qkv.storage_bytes == 300 * MIB
        assert attention.storage_bytes == 100 * MIB
        assert mlp.storage_bytes == 225 * MIB
        assert mlp.materialized_weight_bytes == 450 * MIB
        assert mlp.materialized_weight_bytes <= GIB


def test_qkv_attention_pair_is_atomic() -> None:
    graphs = [GraphInfo("main_block_00_attention_qkv", "qkv", 0, 1, 1)]
    with pytest.raises(ValueError, match="attn_out"):
        plan_shards(graphs)


def test_schedule_uses_named_realistic_bindings() -> None:
    schedule = build_schedule(make_inventory(2), "test", blocks=2)
    assert schedule["format"] == SCHEDULE_FORMAT
    assert schedule["resources"]["activation_peak_slots"] >= 3
    assert "session_slot_hints" in schedule["resources"]
    conditioning = next(step for step in schedule["steps"] if step.get("graph") == "main_conditioning")
    assert set(conditioning["inputs"]) == {"timesteps", "position_ids"}
    assert set(conditioning["outputs"]) == {"timestep_embedding", "rotary_table"}
    head = next(step for step in schedule["steps"] if step.get("graph") == "main_head")
    assert set(head["outputs"]) == {"video_patches", "audio_patches"}
    assert schedule["steps"][-1]["op"] == "unpack_velocity"
    assert set(schedule["steps"][-1]["outputs"]) == {"video_velocity", "audio_velocity"}


def test_schedule_is_deterministic() -> None:
    first = build_schedule(make_inventory(2), "test", blocks=2)
    second = build_schedule(make_inventory(2), "test", blocks=2)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


GRAPH_PORTS = {
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
    "main_block_00_attention_qkv": (
        {
            "hidden_states": TensorProto.FLOAT,
            "timestep_embedding": TensorProto.FLOAT,
            "modulation_ids": TensorProto.INT64,
            "rotary_table": TensorProto.FLOAT16,
        },
        {"hidden_states_out": TensorProto.FLOAT},
    ),
    "main_block_00_attention_output": (
        {
            "hidden_states": TensorProto.FLOAT,
            "attended": TensorProto.FLOAT,
            "timestep_embedding": TensorProto.FLOAT,
            "modulation_ids": TensorProto.INT64,
        },
        {"hidden_states_out": TensorProto.FLOAT},
    ),
    "main_block_00_mlp": (
        {"hidden_states": TensorProto.FLOAT, "timestep_embedding": TensorProto.FLOAT, "modulation_ids": TensorProto.INT64},
        {"hidden_states_out": TensorProto.FLOAT},
    ),
}


def _write_port_model(path: Path, inputs: dict[str, int], outputs: dict[str, int]) -> None:
    graph_inputs = [helper.make_tensor_value_info(name, dtype, ["sequence", 1]) for name, dtype in inputs.items()]
    graph_outputs = [helper.make_tensor_value_info(name, dtype, ["sequence", 1]) for name, dtype in outputs.items()]
    nodes = []
    for index, (name, dtype) in enumerate(outputs.items()):
        value = numpy_helper.from_array(
            np.zeros((1, 1), dtype=helper.tensor_dtype_to_np_dtype(dtype)),
            name=f"value_{index}",
        )
        nodes.append(helper.make_node("Constant", [], [name], value=value))
    model = helper.make_model(helper.make_graph(nodes, path.stem, graph_inputs, graph_outputs))
    onnx.save(model, path)


@pytest.fixture
def model_directory(tmp_path: Path) -> Path:
    names = list(GRAPH_PORTS)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"graphs": [f"{name}.onnx" for name in names]}),
        encoding="utf-8",
    )
    for name, (inputs, outputs) in GRAPH_PORTS.items():
        _write_port_model(tmp_path / f"{name}.onnx", inputs, outputs)
    return tmp_path


def test_validator_accepts_real_onnx_ports(model_directory: Path) -> None:
    inventory = graph_inventory(model_directory)
    schedule = build_schedule(inventory, "test", blocks=1)
    validate_schedule(schedule, model_directory)


def test_validator_rejects_missing_graph_port(model_directory: Path) -> None:
    inventory = graph_inventory(model_directory)
    schedule = build_schedule(inventory, "test", blocks=1)
    qkv = next(step for step in schedule["steps"] if step.get("graph") == "main_block_00_attention_qkv")
    del qkv["inputs"]["rotary_table"]
    with pytest.raises(ValueError, match="Graph port mismatch"):
        validate_schedule(schedule, model_directory)
