from __future__ import annotations

import numpy as np
import pytest

from h3_workbench.exporter import _metrics, _parse_blocks, _require_close


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

