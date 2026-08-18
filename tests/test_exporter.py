from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import torch
from torch import nn

from h3_workbench.exporter import AudioEncoder, _export_main_shard, _metrics, _parse_blocks, _require_close


def test_parse_blocks() -> None:
    assert _parse_blocks("all", 3) == [0, 1, 2]
    assert _parse_blocks("2,0,2", 3) == [0, 2]


def test_parse_blocks_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _parse_blocks("3", 3)


def test_validation_gate_rejects_bad_export() -> None:
    expected = np.array([1.0, 0.0], dtype=np.float32)
    actual = np.array([0.0, 1.0], dtype=np.float32)
    metrics = _metrics(expected, actual)

    with pytest.raises(RuntimeError):
        _require_close("test", metrics, max_abs=0.01)


def test_audio_encoder_wrapper_emits_channel_major_normalized_latents() -> None:
    class PreBlock(nn.Module):
        def forward(self, values):
            return values[..., :2]

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Conv1d(1, 3, 1, bias=False)
            self.pre_block = PreBlock()
            self.mean_proj = nn.Conv1d(2, 2, 1, bias=False)
            self.register_buffer("latents_mean", torch.tensor([1.0, 2.0]))
            self.register_buffer("latents_std", torch.tensor([2.0, 4.0]))
            nn.init.ones_(self.encoder.weight)
            nn.init.ones_(self.mean_proj.weight)

    waveform = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    output = AudioEncoder(Model())(waveform)  # type: ignore[arg-type]

    assert output.shape == (1, 2, 2, 4)
    torch.testing.assert_close(output[:, 0], (waveform * 2.0 - 1.0) / 2.0)
    torch.testing.assert_close(output[:, 1], (waveform * 2.0 - 2.0) / 4.0)


def test_attention_output_export_can_disable_gpu_native_fp16(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        np.savez(Path(command[-1]), output_0=np.zeros((1, 1), dtype=np.float32))

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    with patch("h3_workbench.exporter.subprocess.run", side_effect=fake_run):
        output = _export_main_shard(
            "dit_attention_output",
            44,
            tmp_path / "ref2va.safetensors",
            tmp_path / "attention_output.onnx",
            {"attended": np.zeros((1, 1), dtype=np.float32)},
            gpu_native_fp16=False,
        )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[3] == "dit_attention_output"
    assert "--gpu-native-fp16" not in command
    np.testing.assert_array_equal(output.numpy(), np.zeros((1, 1), dtype=np.float32))
