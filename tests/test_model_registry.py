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


def test_scans_validated_sliced_product_directory(tmp_path: Path) -> None:
    product = tmp_path / "exported" / "minimax_h3_fl2va_scaled_fp16_tensor_core_v3"
    product.mkdir(parents=True)
    (product / "schedule.json").write_text("{}", encoding="utf-8")
    (product / "shard_000.onnx").write_bytes(b"onnx")
    (product / "manifest.json").write_text(
        json.dumps(
            {
                "format": "h3-workbench-onnx-v2",
                "component": "fl2va_transformer",
                "validation_passed": True,
                "build_complete": True,
                "schedule_format": "h3-schedule-v2",
                "schedule": "schedule.json",
                "blocks": list(range(50)),
                "activation_dtype": "fp32_residual_scaled_fp16_mlp",
            }
        ),
        encoding="utf-8",
    )

    records = scan_models(tmp_path)

    assert len(records) == 1
    assert records[0].record_type == "product"
    assert records[0].ready is True
    assert records[0].id == "exported/minimax_h3_fl2va_scaled_fp16_tensor_core_v3"
    assert records[0].component == "fl2va_transformer"
    assert records[0].tensor_count == 50
    assert records[0].export_supported is False


def test_ignores_unvalidated_sliced_product_directory(tmp_path: Path) -> None:
    product = tmp_path / "onnx_models" / "partial"
    product.mkdir(parents=True)
    (product / "manifest.json").write_text(
        json.dumps({"component": "fl2va_transformer", "validation_passed": False}),
        encoding="utf-8",
    )

    assert scan_models(tmp_path) == []
