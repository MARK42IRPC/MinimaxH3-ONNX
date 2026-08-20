from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from h3_workbench.ref2va_lora import (
    EXPECTED_REF2VA_LORA_SPECS,
    REF2VA_ADAPTER_VARIANT,
    REF2VA_LORA_MODULE_COUNT,
    REF2VA_LORA_TENSOR_COUNT,
    Ref2VALoraAdapter,
    Ref2VALoraFactorCache,
    _graph_spec,
)


def test_ref2va_schema_covers_main_and_token_refiner_modules() -> None:
    assert len(EXPECTED_REF2VA_LORA_SPECS) == REF2VA_LORA_MODULE_COUNT == 208
    assert sum(prefix.startswith("blocks.") for prefix in EXPECTED_REF2VA_LORA_SPECS) == 200
    assert sum(prefix.startswith("token_refiner.") for prefix in EXPECTED_REF2VA_LORA_SPECS) == 8
    assert REF2VA_LORA_TENSOR_COUNT == REF2VA_LORA_MODULE_COUNT * 3
    assert EXPECTED_REF2VA_LORA_SPECS["blocks.0.attn.qkv_proj"].a_shape == (384, 5376)
    assert EXPECTED_REF2VA_LORA_SPECS["blocks.49.mlp.fc2"].b_shape == (5376, 128)
    assert EXPECTED_REF2VA_LORA_SPECS["token_refiner.blocks.1.attn.out_proj"].feed_dtype == np.dtype(
        np.float16
    )


def test_ref2va_graph_routing_is_block_specific() -> None:
    assert _graph_spec("main_block_07_attention_qkv.onnx") == (
        "attention_qkv",
        "blocks.7.attn.qkv_proj",
    )
    assert _graph_spec("main_block_49_attention_output") == (
        "attention_output",
        "blocks.49.attn.out_proj",
    )
    assert _graph_spec("main_block_00_mlp.onnx") == ("mlp", "blocks.0.mlp")
    assert _graph_spec("main_token_refiner_block_01_attention.onnx") == (
        "refiner_attention",
        "token_refiner.blocks.1.attn",
    )
    assert _graph_spec("main_token_refiner_block_00_mlp") == (
        "refiner_mlp",
        "token_refiner.blocks.0.mlp",
    )
    assert _graph_spec("main_head") is None


def test_ref2va_factor_cache_applies_alpha_strength_and_rank(tmp_path: Path) -> None:
    prefix = "blocks.0.attn.qkv_proj"
    spec = EXPECTED_REF2VA_LORA_SPECS[prefix]
    path = tmp_path / "tiny-ref2va.safetensors"
    save_file(
        {
            spec.a_key: np.asarray([[2.0]], dtype=np.float16),
            spec.b_key: np.asarray([[3.0]], dtype=np.float32),
            spec.alpha_key: np.asarray(24.0, dtype=np.float32),
        },
        path,
    )

    cache = Ref2VALoraFactorCache(path, strength=0.5)
    factors = cache.factors(prefix)

    np.testing.assert_array_equal(factors.a, np.asarray([[2.0]], dtype=np.float16))
    np.testing.assert_allclose(factors.b, np.asarray([[0.09375]], dtype=np.float16), rtol=1e-3)
    assert cache.factors(prefix) is factors
    assert cache.total_bytes == factors.a.nbytes + factors.b.nbytes
    cache.close()
    assert cache.total_bytes == 0


def test_ref2va_adapter_routes_factors_to_runtime_graphs() -> None:
    class Factors:
        total_bytes = 0

        @staticmethod
        def factors(prefix: str):
            marker = np.asarray([[prefix]], dtype=object)
            return type("Pair", (), {"a": marker, "b": marker})()

    adapter = object.__new__(Ref2VALoraAdapter)
    adapter.factors = Factors()

    feeds = adapter.graph_feeds("main_token_refiner_block_01_attention")

    assert set(feeds) == {"lora_qkv_A", "lora_qkv_B", "lora_out_A", "lora_out_B"}
    assert feeds["lora_qkv_A"][0, 0] == "token_refiner.blocks.1.attn.qkv_proj"
    assert feeds["lora_out_B"][0, 0] == "token_refiner.blocks.1.attn.out_proj"
    assert adapter.graph_feeds("main_conditioning") == {}


def test_ref2va_adapter_declares_dynamic_variant_without_silu_grid() -> None:
    assert REF2VA_ADAPTER_VARIANT == "ref2v_turbo_v0.1_dynamic"
    assert Ref2VALoraAdapter.requires_silu is False
