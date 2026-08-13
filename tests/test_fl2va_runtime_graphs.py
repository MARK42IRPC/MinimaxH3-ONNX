from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.fl2va_runtime_graphs import (
    build_fp16_attention_output_graphs,
    fp16_attention_output_ready,
    is_attention_output_graph,
    runtime_attention_output_graph,
    source_graph_name,
)


def _write_source_graph(path: Path, index: int) -> None:
    weight = numpy_helper.from_array(np.asarray([1.5], dtype=np.float32), name="weight")
    attended = helper.make_tensor_value_info("attended", TensorProto.FLOAT, ["sequence", 1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["sequence", 1])
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["attended"], ["attended_fp32"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["attended_fp32", "weight"], ["output"]),
        ],
        f"attention_output_{index}",
        [attended],
        [output],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=0,
    )


def test_fp16_attention_output_sidecars_reuse_original_external_weights(tmp_path) -> None:
    for index in range(50):
        _write_source_graph(tmp_path / f"main_block_{index:02d}_attention_output.onnx", index)

    manifest = build_fp16_attention_output_graphs(tmp_path)
    selected = runtime_attention_output_graph(tmp_path, "main_block_00_attention_output.onnx")
    model = onnx.load(str(selected), load_external_data=False)
    attended = next(item for item in model.graph.input if item.name == "attended")
    locations = {
        item.value
        for initializer in model.graph.initializer
        for item in initializer.external_data
        if item.key == "location"
    }

    assert manifest.is_file()
    assert fp16_attention_output_ready(tmp_path)
    assert selected.parent == tmp_path
    assert selected.name == "main_block_00_attention_output.fp16.onnx"
    assert is_attention_output_graph(selected)
    assert source_graph_name(selected) == "main_block_00_attention_output.onnx"
    assert attended.type.tensor_type.elem_type == TensorProto.FLOAT
    assert locations == {"main_block_00_attention_output.onnx.data"}
    assert not list(tmp_path.glob("*.fp16.onnx.data"))

    session = ort.InferenceSession(str(selected), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"attended": np.asarray([[2.0]], dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.asarray([[3.5]], dtype=np.float32))


def test_fp16_attention_output_scales_fp16_gemm_safely(tmp_path) -> None:
    weight = numpy_helper.from_array(np.asarray([[2.0]], dtype=np.float16), name="weight")
    attended = helper.make_tensor_value_info("attended", TensorProto.FLOAT, ["sequence", 1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["sequence", 1])
    nodes = [
        helper.make_node("Cast", ["attended"], ["attended_fp16"], to=TensorProto.FLOAT16),
        helper.make_node("Gemm", ["attended_fp16", "weight"], ["projected_fp16"]),
        helper.make_node("Cast", ["projected_fp16"], ["projected_fp32"], to=TensorProto.FLOAT),
        helper.make_node("Identity", ["projected_fp32"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "scaled_projection", [attended], [output], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    for index in range(50):
        path = tmp_path / f"main_block_{index:02d}_attention_output.onnx"
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}.data",
            size_threshold=0,
        )

    build_fp16_attention_output_graphs(tmp_path)
    selected = runtime_attention_output_graph(tmp_path, "main_block_00_attention_output.onnx")
    converted = onnx.load(str(selected), load_external_data=False)
    operations = [node.op_type for node in converted.graph.node]
    runtime_initializers = {item.name for item in converted.graph.initializer}

    assert operations.count("Mul") == 2
    assert {"runtime_attended_scale", "runtime_output_scale"} <= runtime_initializers
    session = ort.InferenceSession(str(selected), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"attended": np.asarray([[8.0]], dtype=np.float32)})[0]
    np.testing.assert_allclose(actual, np.asarray([[16.0]], dtype=np.float32))


def test_scaled_attention_output_accepts_values_above_fp16_range(tmp_path) -> None:
    weight = numpy_helper.from_array(np.asarray([[0.5]], dtype=np.float16), name="weight")
    attended = helper.make_tensor_value_info("attended", TensorProto.FLOAT, ["sequence", 1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["sequence", 1])
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["attended"], ["attended_fp16"], to=TensorProto.FLOAT16),
            helper.make_node("Gemm", ["attended_fp16", "weight"], ["projected_fp16"]),
            helper.make_node("Cast", ["projected_fp16"], ["projected_fp32"], to=TensorProto.FLOAT),
            helper.make_node("Identity", ["projected_fp32"], ["output"]),
        ],
        "overflowing_projection",
        [attended],
        [output],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    for index in range(50):
        path = tmp_path / f"main_block_{index:02d}_attention_output.onnx"
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}.data",
            size_threshold=0,
        )

    build_fp16_attention_output_graphs(tmp_path)
    selected = runtime_attention_output_graph(tmp_path, "main_block_00_attention_output.onnx")
    session = ort.InferenceSession(str(selected), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"attended": np.asarray([[100_000.0]], dtype=np.float32)})[0]

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.asarray([[50_000.0]], dtype=np.float32), rtol=1e-3)
