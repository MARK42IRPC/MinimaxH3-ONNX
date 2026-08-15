from pathlib import Path

import pytest

from h3_workbench.qwen_mlp_benchmark import GIB, _disk_guard


def test_qwen_disk_guard_preserves_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = type("Usage", (), {"free": 100 * GIB})()
    monkeypatch.setattr("h3_workbench.qwen_mlp_benchmark.shutil.disk_usage", lambda _: usage)

    result = _disk_guard(tmp_path, 90.0, 5 * GIB)
    assert result["projected_free_gib"] == 95.0

    with pytest.raises(RuntimeError, match="Disk guard"):
        _disk_guard(tmp_path, 90.0, 11 * GIB)
