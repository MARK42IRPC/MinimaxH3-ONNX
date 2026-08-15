from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.video_vae_persistent import (
    PERSISTENT_VIDEO_VAE_MANIFEST,
    PERSISTENT_VIDEO_VAE_TOPOLOGY,
    build_persistent_video_vae_topology,
    load_persistent_video_vae_manifest,
    load_video_vae_block_weights,
    preload_video_vae_block_weights,
    persistent_video_vae_ready,
    persistent_video_vae_schema,
    validate_video_vae_block_schemas,
    video_vae_block_indices,
)


def _block_model(weight: float, op_type: str = "Add") -> onnx.ModelProto:
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, "sequence", 4])
    rotary = helper.make_tensor_value_info("rotary_table", TensorProto.FLOAT, [1, "sequence", 1, 2, 2, 2])
    output = helper.make_tensor_value_info("hidden_states_out", TensorProto.FLOAT, [1, "sequence", 4])
    bias = numpy_helper.from_array(np.full(4, weight, dtype=np.float32), "block.bias")
    graph = helper.make_graph(
        [helper.make_node(op_type, ["hidden_states", "block.bias"], ["hidden_states_out"])],
        "video_vae_block",
        [hidden, rotary],
        [output],
        [bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)


def _product(directory: Path, blocks: dict[int, tuple[float, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for block, (weight, op_type) in blocks.items():
        onnx.save_model(_block_model(weight, op_type), directory / f"video_decoder_block_{block:02d}.onnx")
    (directory / "manifest.json").write_text(
        json.dumps({"component": "video_vae", "blocks": sorted(blocks)}),
        encoding="utf-8",
    )


def test_builds_small_atomic_topology_and_manifest_without_touching_sources(tmp_path: Path) -> None:
    _product(tmp_path, {0: (1.0, "Add"), 1: (2.0, "Add"), 35: (3.0, "Add")})
    source = tmp_path / "video_decoder_block_00.onnx"
    original = source.read_bytes()

    manifest_path = build_persistent_video_vae_topology(tmp_path, validate_blocks=None)

    topology_path = tmp_path / PERSISTENT_VIDEO_VAE_TOPOLOGY
    topology = onnx.load(topology_path, load_external_data=False)
    manifest = load_persistent_video_vae_manifest(tmp_path)
    assert manifest_path == tmp_path / PERSISTENT_VIDEO_VAE_MANIFEST
    assert not topology.graph.initializer
    assert {value.name for value in topology.graph.input} == {
        "hidden_states",
        "rotary_table",
        "block.bias",
    }
    assert topology_path.stat().st_size == manifest["topology"]["bytes"]
    assert topology_path.stat().st_size < 4096
    assert manifest["validation"]["validated_blocks"] == [0, 1, 35]
    assert manifest["weight_bytes"] == 16
    assert persistent_video_vae_schema(tmp_path)[0].name == "block.bias"
    assert persistent_video_vae_ready(tmp_path, (0, 1, 35), dynamic_batch=False)
    assert source.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_default_build_validates_every_declared_block(tmp_path: Path) -> None:
    _product(tmp_path, {0: (1.0, "Add"), 1: (2.0, "Add"), 2: (3.0, "Add")})

    build_persistent_video_vae_topology(tmp_path)

    manifest = load_persistent_video_vae_manifest(tmp_path)
    assert manifest["validation"]["validated_blocks"] == [0, 1, 2]
    assert persistent_video_vae_ready(tmp_path, range(3), dynamic_batch=False)


def test_block_weight_loader_is_suitable_for_threadpool_prefetch(tmp_path: Path) -> None:
    expected = {0: 1.0, 1: 2.0, 35: 3.0}
    _product(tmp_path, {block: (weight, "Add") for block, weight in expected.items()})
    build_persistent_video_vae_topology(tmp_path, validate_blocks=None)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            block: executor.submit(load_video_vae_block_weights, tmp_path, block)
            for block in expected
        }
        loaded = {block: future.result() for block, future in futures.items()}

    for block, item in loaded.items():
        np.testing.assert_array_equal(item.feeds()["block.bias"], np.full(4, expected[block], dtype=np.float32))
        assert item.block == block
        assert item.total_bytes == 16
        assert item.load_seconds >= 0
        item.close()
        assert not item.feeds()


def test_video_vae_ram_cache_loads_each_block_once(tmp_path: Path, monkeypatch) -> None:
    _product(tmp_path, {0: (1.0, "Add"), 1: (2.0, "Add")})
    build_persistent_video_vae_topology(tmp_path, validate_blocks=None)
    monkeypatch.setenv("H3_VIDEO_VAE_RAM_CACHE", "1")

    cache = preload_video_vae_block_weights(tmp_path, (0, 1))

    assert cache is not None
    assert cache.bytes == 32
    np.testing.assert_array_equal(cache.get(0).feeds()["block.bias"], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(cache.get(1).feeds()["block.bias"], np.full(4, 2.0, dtype=np.float32))
    cache.close()
    assert not cache._weights


def test_schema_validation_rejects_a_different_block_topology(tmp_path: Path) -> None:
    _product(tmp_path, {0: (1.0, "Add"), 1: (2.0, "Mul")})

    validation = validate_video_vae_block_schemas(tmp_path, blocks=None)

    assert validation["compatible"] is False
    assert validation["blocks"]["1"]["topology_matches"] is False
    with pytest.raises(ValueError, match="schemas are incompatible"):
        build_persistent_video_vae_topology(tmp_path, validate_blocks=None)
    assert not (tmp_path / PERSISTENT_VIDEO_VAE_MANIFEST).exists()


def test_ready_tracks_the_validated_source_files(tmp_path: Path) -> None:
    _product(tmp_path, {0: (1.0, "Add"), 1: (2.0, "Add")})
    build_persistent_video_vae_topology(tmp_path, validate_blocks=None)
    assert persistent_video_vae_ready(tmp_path, (0, 1))

    onnx.save_model(_block_model(4.0), tmp_path / "video_decoder_block_01.onnx")

    assert persistent_video_vae_ready(tmp_path, (0,))
    assert not persistent_video_vae_ready(tmp_path, (0, 1))
    with pytest.raises(RuntimeError, match="not ready"):
        load_video_vae_block_weights(tmp_path, 1)


def test_concurrent_builds_share_one_ready_product(tmp_path: Path) -> None:
    _product(tmp_path, {0: (1.0, "Add"), 1: (2.0, "Add")})

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: build_persistent_video_vae_topology(tmp_path, (0, 1)), range(4)))

    assert len(set(paths)) == 1
    assert persistent_video_vae_ready(tmp_path, (0, 1))
    assert not list(tmp_path.glob("*.tmp"))


def test_dynamic_batch_is_explicit_in_manifest_and_ready_check(tmp_path: Path) -> None:
    _product(tmp_path, {0: (1.0, "Add")})

    build_persistent_video_vae_topology(tmp_path, dynamic_batch=True)

    topology = onnx.load(tmp_path / PERSISTENT_VIDEO_VAE_TOPOLOGY, load_external_data=False)
    inputs = {value.name: value for value in topology.graph.input}
    assert inputs["hidden_states"].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert persistent_video_vae_ready(tmp_path, dynamic_batch=True)
    assert not persistent_video_vae_ready(tmp_path, dynamic_batch=False)
    assert video_vae_block_indices(tmp_path) == [0]
