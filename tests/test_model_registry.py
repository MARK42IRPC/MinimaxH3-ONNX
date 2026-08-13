from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from h3_workbench.model_registry import inspect_checkpoint, read_safetensors_header, scan_models


def test_scans_audio_vae_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "minimax_h3_audio_vae_fp32.safetensors"
    metadata = {"minimax_h3_audio_vae": json.dumps({"sample_rate": 32000})}
    save_file({"weight": torch.ones(2, 3)}, checkpoint, metadata=metadata)

    records = scan_models(tmp_path)

    assert len(records) == 1
    assert records[0].component == "audio_vae"
    assert records[0].tensor_count == 1
    assert records[0].export_supported is True
    assert records[0].metadata["minimax_h3_audio_vae"]["sample_rate"] == 32000


def test_rejects_truncated_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"short")

    try:
        read_safetensors_header(path)
    except ValueError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("truncated checkpoint was accepted")


def test_inspect_uses_workspace_relative_id(tmp_path: Path) -> None:
    checkpoint = tmp_path / "minimax_h3_video_vae_fp16.safetensors"
    save_file({"weight": torch.ones(1, dtype=torch.float16)}, checkpoint)

    record = inspect_checkpoint(checkpoint, tmp_path)

    assert record.id == checkpoint.name
    assert record.component == "video_vae"


def test_classifies_acceleration_lora_from_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "minimax_h3_acceleration.safetensors"
    save_file(
        {"lora_A.weight": torch.ones(1, 1)},
        checkpoint,
        metadata={"conversion_type": "prefix_conversion_with_adaln_pairs_removed"},
    )

    record = inspect_checkpoint(checkpoint, tmp_path)

    assert record.component == "acceleration_lora"
    assert record.export_supported is False
