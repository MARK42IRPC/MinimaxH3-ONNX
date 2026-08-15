from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
from safetensors.numpy import save_file

from h3_workbench.turbo_lora import (
    EXPECTED_FACTOR_SPECS,
    TURBO_LORA_PAIR_COUNT,
    TURBO_TOPOLOGIES,
    AssetIdentity,
    LoraFactorPair,
    SafeTensorFile,
    factor_feeds_for_graph,
    interpolate_silu_grid,
    publish_turbo_adapter,
    turbo_graph_kind,
    validate_asset_identity,
    validate_turbo_adapter,
    validate_turbo_assets,
)


WORKSPACE_LORA = Path(
    ".h3-workbench/sources/larryvrh--MiniMax-H3-Turbo-Lora/"
    "minimax_h3_turbo_v4_step600_ema.safetensors"
)
WORKSPACE_GRID = Path(
    ".h3-workbench/sources/Mark42IRPC--Minimax-H3-int8-fl2va-onnx-50CLIPS/"
    "export_support/h3_silu_temb_grid.safetensors"
)
WORKSPACE_BASE = Path("exported/minimax_h3_fl2va_scaled_fp16_tensor_core_v3")


def test_expected_turbo_v4_schema_has_all_259_pairs() -> None:
    assert len(EXPECTED_FACTOR_SPECS) == TURBO_LORA_PAIR_COUNT
    assert sum(prefix.startswith("blocks.") for prefix in EXPECTED_FACTOR_SPECS) == 250
    assert sum(prefix.startswith("token_refiner.") for prefix in EXPECTED_FACTOR_SPECS) == 8
    assert "final_layer.adaln_proj.linear" in EXPECTED_FACTOR_SPECS
    assert EXPECTED_FACTOR_SPECS["blocks.49.mlp.fc2"].a_shape == (64, 14336)
    assert EXPECTED_FACTOR_SPECS["blocks.49.mlp.fc2"].b_shape == (5376, 64)
    assert EXPECTED_FACTOR_SPECS["blocks.0.adaln_proj.linear"].cache_dtype == np.dtype(
        np.float32
    )


def test_silu_grid_interpolation_matches_endpoint_and_clamp_behavior() -> None:
    grid = np.arange(15, dtype=np.float32).reshape(5, 3)
    result = interpolate_silu_grid(
        grid,
        np.array([-1.0, 0.0, 0.125, 0.5, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(result[0], grid[0])
    np.testing.assert_array_equal(result[1], grid[0])
    np.testing.assert_allclose(result[2], (grid[0] + grid[1]) / 2)
    np.testing.assert_array_equal(result[3], grid[2])
    np.testing.assert_array_equal(result[4], grid[4])
    np.testing.assert_array_equal(result[5], grid[4])
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_graph_routing_and_factor_feeds_are_block_specific() -> None:
    pairs: dict[str, LoraFactorPair] = {}
    for prefix in (
        "blocks.24.mlp.fc1",
        "blocks.24.mlp.fc2",
        "blocks.24.adaln_proj.linear",
    ):
        value = np.array([len(pairs) + 1], dtype=np.float32)
        pairs[prefix] = LoraFactorPair(value, value + 10)

    feeds = factor_feeds_for_graph("main_block_24_mlp", pairs)
    assert turbo_graph_kind("main_block_24_mlp") == "mlp"
    assert set(feeds) == {
        "turbo_lora_fc1_A",
        "turbo_lora_fc1_B",
        "turbo_lora_fc2_A",
        "turbo_lora_fc2_B",
        "turbo_lora_adaln_A",
        "turbo_lora_adaln_B",
    }
    assert feeds["turbo_lora_fc1_A"] is pairs["blocks.24.mlp.fc1"].a
    assert feeds["turbo_lora_fc2_B"] is pairs["blocks.24.mlp.fc2"].b
    assert factor_feeds_for_graph("main_conditioning", pairs) == {}
    assert turbo_graph_kind("main_token_refiner_block_01_attention") == "refiner_attention"
    assert turbo_graph_kind("main_head") == "head"


def test_safe_tensor_reader_and_identity_are_offset_based(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    save_file({"values": values}, path, metadata={"kind": "test"})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    identity = validate_asset_identity(
        path,
        AssetIdentity(path.name, path.stat().st_size, digest),
    )
    reader = SafeTensorFile(path)
    assert identity.sha256 == digest
    assert reader.metadata == {"kind": "test"}
    assert reader.shape("values") == (3, 4)
    np.testing.assert_array_equal(reader.tensor("values", np.dtype(np.float32)), values)


@pytest.mark.skipif(
    not (WORKSPACE_LORA.is_file() and WORKSPACE_GRID.is_file() and WORKSPACE_BASE.is_dir()),
    reason="workspace Turbo assets and exported Base product are not available",
)
def test_workspace_assets_publish_graph_only_adapter(tmp_path: Path) -> None:
    assets = validate_turbo_assets(WORKSPACE_LORA, WORKSPACE_GRID)
    assert assets.factor_bytes == 779_792_384

    manifest = publish_turbo_adapter(
        WORKSPACE_BASE,
        tmp_path,
        WORKSPACE_LORA,
        WORKSPACE_GRID,
    )
    validated = validate_turbo_adapter(tmp_path, base_model_dir=WORKSPACE_BASE)
    assert validated == manifest
    assert set(manifest["topologies"]) == set(TURBO_TOPOLOGIES)
    assert sum(path.stat().st_size for path in tmp_path.glob("*.onnx")) < 1_000_000

    for kind, filename in TURBO_TOPOLOGIES.items():
        model = onnx.load(str(tmp_path / filename), load_external_data=False)
        inputs = {value.name for value in model.graph.input}
        assert any(name.startswith("turbo_lora_") for name in inputs), kind
        assert not model.graph.initializer
    qkv_inputs = {
        value.name
        for value in onnx.load(
            str(tmp_path / TURBO_TOPOLOGIES["attention_qkv"]),
            load_external_data=False,
        ).graph.input
    }
    assert "silu_timestep_embedding" in qkv_inputs
    assert "turbo_lora_qkv_A" in qkv_inputs
