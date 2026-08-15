import numpy as np

from h3_workbench.mlp_precision_benchmark import _metrics


def test_metrics_report_finite_relative_error() -> None:
    expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
    actual = np.asarray([[1.0, 2.1]], dtype=np.float32)

    metrics = _metrics(expected, actual)

    assert metrics["finite"] is True
    assert 0.0 < metrics["relative_l2"] < 0.1


def test_metrics_isolate_nonfinite_candidate() -> None:
    metrics = _metrics(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[np.inf]], dtype=np.float32),
    )

    assert metrics["finite"] is False
    assert metrics["relative_l2"] == float("inf")
