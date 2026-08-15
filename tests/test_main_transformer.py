from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from h3_workbench.main_transformer import (
    CheckpointReader,
    FrozenLinear,
    enable_gpu_native_bf16,
    enable_scaled_gpu_native_fp16,
)


def test_checkpoint_reader_streams_fp8_and_bfloat16(tmp_path: Path) -> None:
    path = tmp_path / "main.safetensors"
    source = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    save_file(
        {
            "test.weight": source.to(torch.float8_e4m3fn),
            "test.weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        },
        path,
    )

    reader = CheckpointReader(path)

    np.testing.assert_array_equal(
        reader.dequant_weight("test").numpy(),
        (source * 0.25).to(torch.float16).numpy(),
    )
    np.testing.assert_array_equal(
        reader.tensor("norm.weight").numpy(),
        np.asarray([1.0, 2.0], dtype=np.float16),
    )


def test_scaled_fp16_linear_preserves_large_finite_output() -> None:
    linear = FrozenLinear(torch.full((1, 2), 100.0), compute_fp32=True)
    inputs = torch.full((1, 2), 1000.0)
    expected = linear(inputs)

    enable_scaled_gpu_native_fp16(linear, fc1_weight_scale=16.0)
    actual = linear(inputs)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1.0)


def test_bf16_linear_preserves_fp32_range() -> None:
    linear = FrozenLinear(torch.full((1, 2), 100.0), compute_fp32=True)
    inputs = torch.full((1, 2), 1000.0)
    expected = linear(inputs)

    enable_gpu_native_bf16(linear)
    actual = linear(inputs)

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=512.0)
