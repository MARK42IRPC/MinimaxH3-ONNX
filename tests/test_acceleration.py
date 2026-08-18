import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from h3_workbench.acceleration import LoRAMerger, minimax_h3_euler_step, shifted_flow_sigmas


def test_six_step_shifted_flow_schedule() -> None:
    sigmas = shifted_flow_sigmas(6)
    assert len(sigmas) == 7
    assert sigmas[0] == 1.0
    assert sigmas[-1] == 0.0
    assert np.all(np.diff(sigmas) < 0)


def test_shifted_flow_schedule_can_start_from_conditioning_sigma() -> None:
    sigmas = shifted_flow_sigmas(4, start_sigma=0.35)

    assert sigmas[0] == 0.35
    assert sigmas[-1] == 0.0
    assert np.all(np.diff(sigmas) < 0)


def test_shifted_flow_schedule_rejects_invalid_start_sigma() -> None:
    with pytest.raises(ValueError, match="start_sigma"):
        shifted_flow_sigmas(4, start_sigma=1.1)


def test_minimax_h3_euler_step_matches_official_dataward_velocity_update() -> None:
    sample = np.asarray([2.0, -1.0], dtype=np.float16)
    velocity = np.asarray([3.0, 4.0], dtype=np.float16)

    actual = minimax_h3_euler_step(sample, velocity, sigma=0.8, sigma_next=0.2)

    np.testing.assert_allclose(actual, np.asarray([3.8, 1.4], dtype=np.float16), rtol=0, atol=1e-3)


def test_minimax_h3_euler_step_rejects_reverse_sigma() -> None:
    with pytest.raises(ValueError, match="sigma_next"):
        minimax_h3_euler_step(np.zeros(1), np.zeros(1), sigma=0.2, sigma_next=0.3)


def test_lora_merger_applies_b_times_a(tmp_path) -> None:
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.test.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            "diffusion_model.blocks.0.test.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16),
        },
        path,
    )
    weight = np.zeros((2, 2), dtype=np.float16)

    LoRAMerger(path, strength=0.5).merge("blocks.0.test", weight)

    np.testing.assert_array_equal(weight, np.asarray([[1.5, 3.0], [2.0, 4.0]], dtype=np.float16))


def test_lora_merger_accepts_original_unprefixed_keys(tmp_path) -> None:
    path = tmp_path / "original.safetensors"
    save_file(
        {
            "blocks.0.test.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            "blocks.0.test.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.bfloat16),
        },
        path,
    )
    weight = np.zeros((2, 2), dtype=np.float16)
    merger = LoRAMerger(path)

    merger.merge("blocks.0.test", weight)

    np.testing.assert_array_equal(weight, np.asarray([[3.0, 6.0], [4.0, 8.0]], dtype=np.float16))
    assert merger.merged == {"blocks.0.test"}
