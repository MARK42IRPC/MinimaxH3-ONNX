from pathlib import Path
import threading

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.persistent_weights import (
    PERSISTENT_TOPOLOGIES,
    HostWeightPool,
    LoadedWeights,
    MappedWeights,
    PersistentWeightRuntime,
    _graph_kind,
    build_persistent_topology,
    device_prefetch_admitted,
    initializer_weight_inputs,
)


def test_dynamic_adapter_graph_kinds_cover_refiner_and_head() -> None:
    assert _graph_kind("main_token_refiner_block_00_attention") == "refiner_attention"
    assert _graph_kind("main_token_refiner_block_01_mlp") == "refiner_mlp"
    assert _graph_kind("main_head") == "head"
    assert _graph_kind("main_conditioning") is None


def _external_model(path: Path) -> np.ndarray:
    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    weight = numpy_helper.from_array(values, "weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "weighted",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 4])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=0,
    )
    return values


def test_persistent_topology_maps_external_initializers_as_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    expected = _external_model(source)
    topology = tmp_path / "topology.onnx"

    inputs = build_persistent_topology(source, topology)
    mapped = MappedWeights(source, inputs)
    persistent = onnx.load(str(topology), load_external_data=False)

    assert [item.name for item in inputs] == ["weight"]
    assert {item.name for item in persistent.graph.input} == {"input", "weight"}
    assert not persistent.graph.initializer
    np.testing.assert_array_equal(mapped.feeds()["weight"], expected)
    mapped.close()


def test_loaded_weights_bulk_reads_external_data(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    expected = _external_model(source)
    topology = tmp_path / "topology.onnx"
    inputs = build_persistent_topology(source, topology)

    loaded = LoadedWeights(source, inputs)

    np.testing.assert_array_equal(loaded.feeds()["weight"], expected)
    assert loaded.total_bytes == expected.nbytes
    assert loaded.host_bytes >= expected.nbytes
    assert loaded.load_seconds >= 0
    loaded.close()


def test_persistent_runtime_prefetches_and_releases_one_graph(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    expected = _external_model(source)
    topology = tmp_path / PERSISTENT_TOPOLOGIES["mlp"]
    build_persistent_topology(source, topology)
    runtime = PersistentWeightRuntime(
        tmp_path,
        object(),
        {"main_block_00_mlp": source},
        prefetch_depth=1,
    )

    assert runtime.prefetch("main_block_00_mlp")
    (
        feeds,
        weight_bytes,
        host_bytes,
        pinned_bytes,
        device_bytes,
        loaded,
        load_seconds,
        wait_seconds,
        upload_seconds,
        prefetched,
        device_resident,
    ) = runtime.weights("main_block_00_mlp")

    np.testing.assert_array_equal(feeds["weight"], expected)
    assert weight_bytes == expected.nbytes
    assert host_bytes >= weight_bytes
    assert pinned_bytes == 0
    assert device_bytes == 0
    assert loaded and prefetched
    assert load_seconds >= 0
    assert wait_seconds >= 0
    assert upload_seconds == 0
    assert not device_resident
    runtime.release("main_block_00_mlp")
    assert runtime.prefetch("main_block_00_mlp")
    runtime.close()


def test_adapter_topology_override_loads_inline_source_initializers(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    _external_model(source)
    model = onnx.load(str(source), load_external_data=False)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(7, dtype=np.int64), "inline_constant")
    )
    onnx.save_model(model, str(source))
    topology = tmp_path / "adapter_refiner_mlp.onnx"
    build_persistent_topology(source, topology, all_initializers=True)

    class Runner:
        def session(self, path: Path) -> object:
            assert Path(path) == topology
            return object()

    runtime = PersistentWeightRuntime(
        tmp_path,
        Runner(),
        {"main_token_refiner_block_00_mlp": source},
        topology_paths={"refiner_mlp": topology},
    )

    assert [item.name for item in runtime.topology_inputs["refiner_mlp"]] == [
        "weight",
        "inline_constant",
    ]
    _, built, _ = runtime.session("main_token_refiner_block_00_mlp")
    assert built
    runtime.close()


def test_persistent_runtime_can_recycle_one_topology_session(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    _external_model(source)
    topology = tmp_path / PERSISTENT_TOPOLOGIES["mlp"]
    build_persistent_topology(source, topology)

    class Runner:
        def __init__(self) -> None:
            self.builds = 0

        def session(self, path: Path) -> object:
            assert Path(path) == topology
            self.builds += 1
            return object()

    runner = Runner()
    runtime = PersistentWeightRuntime(
        tmp_path,
        runner,
        {"main_block_00_mlp": source},
    )

    assert runtime.session("main_block_00_mlp")[1]
    assert not runtime.session("main_block_00_mlp")[1]
    runtime.release_session("main_block_00_mlp")
    assert runtime.session("main_block_00_mlp")[1]
    assert runner.builds == 2
    runtime.close()


def test_device_prefetch_admission_preserves_reserve_and_inflight_bytes() -> None:
    mib = 1024**2

    assert device_prefetch_admitted(1200 * mib, 400 * mib, 384 * mib)
    assert not device_prefetch_admitted(900 * mib, 400 * mib, 384 * mib, 200 * mib)
    assert not device_prefetch_admitted(1200 * mib, 0, 384 * mib)


def test_persistent_runtime_prefetches_graph_kinds_in_parallel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.onnx"
    _external_model(source)
    for topology in PERSISTENT_TOPOLOGIES.values():
        build_persistent_topology(source, tmp_path / topology)
    graph_paths = {
        "main_block_00_attention_qkv": source,
        "main_block_00_attention_output": source,
        "main_block_00_mlp": source,
    }
    rendezvous = threading.Barrier(2, timeout=2)

    class ConcurrentLoadedWeights:
        total_bytes = 1
        host_bytes = 1
        pinned_bytes = 0
        device_bytes = 0
        load_seconds = 0.01
        upload_seconds = 0.0
        device_resident = False

        def __init__(self, *args, **kwargs) -> None:
            rendezvous.wait()

        def feeds(self) -> dict[str, np.ndarray]:
            return {}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "h3_workbench.persistent_weights.LoadedWeights",
        ConcurrentLoadedWeights,
    )
    runtime = PersistentWeightRuntime(
        tmp_path,
        object(),
        graph_paths,
        prefetch_depth=3,
    )

    concurrent_graphs = list(graph_paths)[:2]
    assert runtime.prefetch_workers == 2
    assert all(runtime.prefetch(graph) for graph in concurrent_graphs)
    for graph in concurrent_graphs:
        runtime.weights(graph)
        runtime.release(graph)
    runtime.close()


def test_loaded_weights_preserves_inline_scalar_rank(tmp_path: Path) -> None:
    scalar = numpy_helper.from_array(np.asarray(0, dtype=np.int64), "scalar")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "inline_scalar",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [scalar],
    )
    source = tmp_path / "inline.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9),
        str(source),
    )
    inputs = initializer_weight_inputs(source, include_inline=True)

    loaded = LoadedWeights(source, inputs)

    assert loaded.feeds()["scalar"].shape == ()
    loaded.close()


def test_host_weight_pool_reuses_released_buffer() -> None:
    pool = HostWeightPool(use_pinned=False)
    first = pool.acquire(("mlp", 0), 4096)
    first_address = first.array.ctypes.data
    first.release()

    second = pool.acquire(("mlp", 0), 4096)

    assert second.array.ctypes.data == first_address
    assert pool.allocations == 1
    second.release()
    pool.close()


def test_persistent_topology_extracts_single_graph_if_branch(tmp_path: Path) -> None:
    weight = numpy_helper.from_array(np.asarray([2.0, 3.0], dtype=np.float32), "weight")
    selector_index = numpy_helper.from_array(np.asarray(0, dtype=np.int64), "selector_index")
    then_branch = helper.make_graph(
        [helper.make_node("Add", ["captured", "weight"], ["block/output"])],
        "selected",
        [],
        [helper.make_tensor_value_info("block/output", TensorProto.FLOAT, [2])],
    )
    else_branch = helper.make_graph(
        [helper.make_node("Identity", ["captured"], ["block/output"])],
        "not_selected",
        [],
        [helper.make_tensor_value_info("block/output", TensorProto.FLOAT, [2])],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["captured"]),
            helper.make_node("Equal", ["selector", "selector_index"], ["condition"]),
            helper.make_node(
                "If",
                ["condition"],
                ["block/output"],
                then_branch=then_branch,
                else_branch=else_branch,
            ),
        ],
        "single_graph_shard",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("selector", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("block/output", TensorProto.FLOAT, [2])],
        [selector_index, weight],
    )
    source = tmp_path / "source.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9),
        str(source),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="source.onnx.data",
        size_threshold=0,
    )
    topology = tmp_path / "topology.onnx"

    build_persistent_topology(
        source,
        topology,
        canonical_outputs=True,
        all_initializers=True,
        selector_free_if=True,
    )
    model = onnx.load(str(topology), load_external_data=False)

    assert [node.op_type for node in model.graph.node] == ["Add"]
    assert {value.name for value in model.graph.input} == {
        "input",
        "selector_index",
        "weight",
    }
    assert [value.name for value in model.graph.output] == ["output"]
    onnx.checker.check_model(model, full_check=False)
