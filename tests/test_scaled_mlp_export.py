from pathlib import Path

import pytest

from h3_workbench.scaled_mlp_export import GIB, _record_valid, _space_projection


def test_space_projection_preserves_reserve() -> None:
    projected = _space_projection(150 * GIB, 90 * GIB, 50, 470_000_000)
    assert projected > 128 * GIB
    with pytest.raises(RuntimeError, match="Disk guard"):
        _space_projection(100 * GIB, 90 * GIB, 50, 470_000_000)


def test_completed_record_requires_matching_files(tmp_path: Path) -> None:
    model = tmp_path / "scaled_fp16.onnx"
    model.write_bytes(b"model")
    record = {
        "status": "completed",
        "files": {"scaled_fp16.onnx": {"bytes": 5, "sha256": "unused"}},
    }
    assert _record_valid(record, tmp_path)
    model.write_bytes(b"changed")
    assert not _record_valid(record, tmp_path)
