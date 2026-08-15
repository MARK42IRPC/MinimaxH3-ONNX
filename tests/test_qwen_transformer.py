import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from h3_workbench.qwen_transformer import QwenCheckpointReader, SmoothedLinear, regular_hadamard


def test_qwen_reader_streams_and_dequantizes_per_channel_int8(tmp_path: Path) -> None:
    checkpoint = tmp_path / "qwen-int8.safetensors"
    weights = np.asarray([[1, -2, 3], [-4, 5, -6]], dtype=np.int8)
    scales = np.asarray([[0.5], [0.25]], dtype=np.float32)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": weights,
            "model.layers.0.self_attn.q_proj.weight_scale": scales,
        },
        checkpoint,
    )

    reader = QwenCheckpointReader(checkpoint)
    actual = reader.dequant_weight("model.layers.0.self_attn.q_proj").numpy()

    assert reader.source_quantization == "int8_per_channel"
    np.testing.assert_array_equal(actual, weights.astype(np.float16) * scales.astype(np.float16))


def test_qwen_reader_rejects_unknown_quantization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "qwen-unknown.safetensors"
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": np.ones((2, 3), dtype=np.float32)},
        checkpoint,
    )

    try:
        QwenCheckpointReader(checkpoint)
    except ValueError as exc:
        assert "Unsupported Qwen checkpoint quantization" in str(exc)
    else:
        raise AssertionError("unknown Qwen quantization should fail")


def _quant_marker(**overrides: object) -> np.ndarray:
    config = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 4,
        **overrides,
    }
    return np.frombuffer(json.dumps(config).encode("utf-8"), dtype=np.uint8)


def test_qwen_reader_detects_and_requires_convrot_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "qwen-int8-convrot.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": np.ones((2, 4), dtype=np.int8),
            "model.layers.0.self_attn.q_proj.weight_scale": np.ones((2, 1), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.comfy_quant": _quant_marker(),
        },
        checkpoint,
    )

    reader = QwenCheckpointReader(checkpoint)

    assert reader.source_quantization == "int8_tensorwise_convrot"
    assert reader.convrot_group_size("model.layers.0.self_attn.q_proj") == 4
    try:
        reader.convrot_group_size("model.layers.0.self_attn.k_proj")
    except ValueError as exc:
        assert "Missing INT8 ConvRot metadata" in str(exc)
    else:
        raise AssertionError("a ConvRot checkpoint must not silently accept an unmarked linear")


def test_smoothed_linear_applies_regular_hadamard_in_comfy_direction() -> None:
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]])
    inputs = torch.tensor([[0.25, -0.5, 1.5, 2.0]])
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    ) / 2.0
    layer = SmoothedLinear(weight, None, convrot_group_size=4)

    actual = layer(inputs)

    torch.testing.assert_close(regular_hadamard(4), h4)
    torch.testing.assert_close(actual, torch.nn.functional.linear(inputs @ h4, weight))
    torch.testing.assert_close(h4 @ h4.T, torch.eye(4))
