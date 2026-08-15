from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.video_vae_benchmark import (
    JsonlLogger,
    _metrics,
    block_schema_fingerprint,
    inspect_block_model,
    inspect_block_schemas,
    load_initializer_feeds,
    make_block_inputs,
    run_host_baseline,
    run_io_binding,
    run_persistent_io_binding,
    serialize_dynamic_batch_model,
    serialize_persistent_block_topology,
)


def _block_model() -> onnx.ModelProto:
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, "sequence", 4])
    rotary = helper.make_tensor_value_info("rotary_table", TensorProto.FLOAT, [1, "sequence", 1, 2, 2, 2])
    output = helper.make_tensor_value_info("hidden_states_out", TensorProto.FLOAT, [1, "sequence", 4])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["hidden_states"], ["hidden_states_out"])],
        "video_vae_block",
        [hidden, rotary],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def _weighted_block_model(weight: float = 1.0) -> onnx.ModelProto:
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, "sequence", 4])
    rotary = helper.make_tensor_value_info("rotary_table", TensorProto.FLOAT, [1, "sequence", 1, 2, 2, 2])
    output = helper.make_tensor_value_info("hidden_states_out", TensorProto.FLOAT, [1, "sequence", 4])
    bias = numpy_helper.from_array(np.full(4, weight, dtype=np.float32), "block.bias")
    graph = helper.make_graph(
        [helper.make_node("Add", ["hidden_states", "block.bias"], ["hidden_states_out"])],
        "weighted_video_vae_block",
        [hidden, rotary],
        [output],
        [bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)


def test_inspect_and_relax_video_block_batch_contract() -> None:
    model = _block_model()
    contract = inspect_block_model(model)

    assert contract["hidden_width"] == 4
    assert contract["rotary_tail"] == [1, 2, 2, 2]
    assert contract["native_dynamic_batch"] is False
    assert contract["operators"] == {"Identity": 1}

    dynamic = onnx.load_from_string(serialize_dynamic_batch_model(model))
    assert dynamic.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert dynamic.graph.input[1].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert dynamic.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"


def test_dynamic_batch_model_executes_multiple_tiles() -> None:
    model = _block_model()
    contract = inspect_block_model(model)
    feeds = make_block_inputs(contract, batch_size=2, sequence=3, seed=17)
    session = ort.InferenceSession(serialize_dynamic_batch_model(model), providers=["CPUExecutionProvider"])

    output = session.run(["hidden_states_out"], feeds)[0]

    assert output.shape == (2, 3, 4)
    np.testing.assert_array_equal(output, feeds["hidden_states"])


def test_host_and_io_binding_benchmarks_match_on_cpu() -> None:
    model = _block_model()
    contract = inspect_block_model(model)
    feeds = make_block_inputs(contract, batch_size=1, sequence=3, seed=19)
    session = _session(model)

    expected, baseline = run_host_baseline(session, feeds, warmup=1, repeats=2)
    actual, bound = run_io_binding(session, feeds, warmup=1, repeats=2, device_type="cpu")

    np.testing.assert_array_equal(actual, expected)
    assert baseline["timed_h2d_copies_per_call"] == 2
    assert bound["timed_h2d_copies_per_call"] == 0
    assert bound["setup_h2d_copies"] == 0
    assert _metrics(expected, actual)["max_abs"] == 0.0


def test_jsonl_logger_appends_durable_records(tmp_path: Path) -> None:
    path = tmp_path / "video-vae.jsonl"
    logger = JsonlLogger(path)
    logger.write("started", durable=True, block=0)
    logger.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "started"
    assert records[0]["block"] == 0


def test_persistent_topology_promotes_initializers_without_embedding_weights() -> None:
    model = _weighted_block_model()

    serialized, schema, fingerprint = serialize_persistent_block_topology(model, dynamic_batch=True)
    persistent = onnx.load_from_string(serialized)

    assert schema == [
        {
            "name": "block.bias",
            "dtype": TensorProto.FLOAT,
            "dtype_name": "FLOAT",
            "shape": [4],
            "nbytes": 16,
        }
    ]
    assert len(fingerprint) == 64
    assert not persistent.graph.initializer
    assert {value.name for value in persistent.graph.input} == {
        "hidden_states",
        "rotary_table",
        "block.bias",
    }
    assert persistent.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert len(serialized) < 4096


def test_persistent_topology_reuses_schema_with_different_block_weights(tmp_path: Path) -> None:
    topology_model = _weighted_block_model(weight=1.0)
    _, schema, fingerprint = serialize_persistent_block_topology(topology_model)
    selected_path = tmp_path / "video_decoder_block_01.onnx"
    onnx.save_model(_weighted_block_model(weight=2.0), selected_path)

    feeds, details = load_initializer_feeds(selected_path, schema, fingerprint)

    np.testing.assert_array_equal(feeds["block.bias"], np.full(4, 2.0, dtype=np.float32))
    assert details["total_bytes"] == 16
    assert details["fingerprint"] == fingerprint


def test_persistent_io_binding_matches_original_on_cpu() -> None:
    original_model = _weighted_block_model(weight=2.0)
    contract = inspect_block_model(original_model)
    feeds = make_block_inputs(contract, batch_size=1, sequence=3, seed=23)
    expected = _session(original_model).run(["hidden_states_out"], feeds)[0]

    topology_model = _weighted_block_model(weight=1.0)
    serialized, _, _ = serialize_persistent_block_topology(topology_model)
    persistent_session = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    actual, result = run_persistent_io_binding(
        persistent_session,
        feeds,
        {"block.bias": np.full(4, 2.0, dtype=np.float32)},
        warmup=1,
        repeats=2,
        device_type="cpu",
    )

    np.testing.assert_array_equal(actual, expected)
    assert result["weight_count"] == 1
    assert result["weight_bytes"] == 16
    assert result["weight_h2d_bytes"] == 0


def test_schema_inspection_accepts_weight_value_changes(tmp_path: Path) -> None:
    for index, weight in ((0, 1.0), (1, 2.0), (35, 3.0)):
        onnx.save_model(_weighted_block_model(weight), tmp_path / f"video_decoder_block_{index:02d}.onnx")

    result = inspect_block_schemas(tmp_path, [0, 1, 35])

    assert result["compatible"] is True
    assert len({record["fingerprint"] for record in result["blocks"].values()}) == 1
    assert block_schema_fingerprint(_weighted_block_model(1.0)) == block_schema_fingerprint(
        _weighted_block_model(2.0)
    )
