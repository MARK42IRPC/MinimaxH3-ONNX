from pathlib import Path

import onnx
from onnx import TensorProto, helper

from h3_workbench.shard_cache import ShardPrefetchCache, graph_files, graph_storage_bytes


def _make_graph(directory: Path, name: str, data: bytes) -> Path:
    graph = directory / name
    graph.write_bytes(b"onnx")
    graph.with_name(f"{graph.name}.data").write_bytes(data)
    return graph


def test_graph_storage_includes_external_data(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path, "block.onnx", b"weights")

    assert graph_files(graph) == (graph, tmp_path / "block.onnx.data")
    assert graph_storage_bytes(graph) == 11


def test_prefetch_cache_keeps_a_bounded_rolling_window(tmp_path: Path) -> None:
    first = _make_graph(tmp_path, "first.onnx", b"a" * 16)
    second = _make_graph(tmp_path, "second.onnx", b"b" * 16)
    third = _make_graph(tmp_path, "third.onnx", b"c" * 16)

    with ShardPrefetchCache(budget_bytes=40, prefetch_depth=3) as cache:
        cache.stage([first, second, third])
        cache.wait(first)
        initial = cache.snapshot()
        cache.stage([second, third])
        cache.wait(second)
        rolled = cache.snapshot()

    assert initial["l2_entries"] == 2
    assert initial["l2_staged_bytes"] == 40
    assert rolled["l2_entries"] == 2
    assert rolled["l2_staged_bytes"] == 40


def test_prefetch_cache_can_shrink_budget_without_replacing_the_worker(tmp_path: Path) -> None:
    first = _make_graph(tmp_path, "first.onnx", b"a" * 16)
    second = _make_graph(tmp_path, "second.onnx", b"b" * 16)

    with ShardPrefetchCache(budget_bytes=40, prefetch_depth=2) as cache:
        cache.stage([first, second])
        cache.wait(first)
        cache.set_budget(20)
        snapshot = cache.snapshot()

    assert snapshot["l2_budget_bytes"] == 20
    assert snapshot["l2_staged_bytes"] <= 20
    assert snapshot["l2_pressure_evictions"] >= 1


def test_graph_files_discovers_multiple_external_weight_files(tmp_path: Path) -> None:
    first = tmp_path / "first.data"
    second = tmp_path / "second.data"
    first.write_bytes(b"abc")
    second.write_bytes(b"defg")
    tensors = []
    for name, location in (("a", first.name), ("b", second.name)):
        tensor = helper.make_tensor(name, TensorProto.FLOAT, [1], [0.0])
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value=location)
        tensor.external_data.add(key="offset", value="0")
        tensor.external_data.add(key="length", value="4")
        tensors.append(tensor)
    graph = helper.make_graph([], "external", [], [], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    path = tmp_path / "merged.onnx"
    onnx.save_model(model, path)

    assert graph_files(path) == (path, first, second)
    assert graph_storage_bytes(path) == path.stat().st_size + 7
