import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from h3_workbench.acceleration import (
    LoRAMerger,
    minimax_h3_euler_step,
    minimax_h3_res_multistep_step,
    shifted_flow_sigmas,
)


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


def test_res_multistep_matches_official_first_and_terminal_boundaries() -> None:
    sample = np.asarray([2.0, -1.0], dtype=np.float32)
    velocity = np.asarray([3.0, 4.0], dtype=np.float32)

    first = minimax_h3_res_multistep_step(sample, velocity, sigma=0.8, sigma_next=0.2)
    terminal = minimax_h3_res_multistep_step(
        sample,
        velocity,
        sigma=0.8,
        sigma_next=0.0,
        previous_sigma=0.9,
        previous_sigma_down=0.8,
        previous_denoised=np.zeros_like(sample),
    )

    np.testing.assert_array_equal(first, minimax_h3_euler_step(sample, velocity, 0.8, 0.2))
    np.testing.assert_allclose(terminal, sample + 0.8 * velocity, rtol=0.0, atol=1e-6)


def test_res_multistep_matches_official_two_step_formula() -> None:
    sample = np.asarray([1.25, -0.5], dtype=np.float32)
    velocity = np.asarray([0.4, -0.8], dtype=np.float32)
    previous_denoised = np.asarray([0.9, 0.2], dtype=np.float32)
    sigma = np.float32(0.6)
    sigma_next = np.float32(0.3)
    previous_sigma = np.float32(0.9)
    previous_sigma_down = np.float32(0.6)

    t = -np.log(sigma)
    t_old = -np.log(previous_sigma_down)
    t_next = -np.log(sigma_next)
    t_prev = -np.log(previous_sigma)
    h = t_next - t
    c2 = (t_prev - t_old) / h
    phi1 = np.expm1(-h) / (-h)
    phi2 = (phi1 - 1.0) / (-h)
    b2 = phi2 / c2
    b1 = phi1 - b2
    current_denoised = sample + sigma * velocity
    expected = np.exp(-h) * sample + h * (b1 * current_denoised + b2 * previous_denoised)

    actual = minimax_h3_res_multistep_step(
        sample,
        velocity,
        sigma,
        sigma_next,
        previous_sigma,
        previous_sigma_down,
        previous_denoised,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


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
