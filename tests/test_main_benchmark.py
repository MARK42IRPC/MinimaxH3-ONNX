import json
from pathlib import Path

import numpy as np

from h3_workbench.main_benchmark import JsonlLogger, save_checkpoint


def test_jsonl_logger_flushes_durable_records(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    logger = JsonlLogger(path)
    logger.write("started", durable=True, step=0)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "started"
    assert records[0]["step"] == 0
    assert records[0]["elapsed_seconds"] >= 0
    logger.close()


def test_jsonl_logger_accepts_runtime_event_detail(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    logger = JsonlLogger(path)
    logger.write("runtime_activity", runtime_event="graph_complete")
    logger.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "runtime_activity"
    assert record["runtime_event"] == "graph_complete"


def test_save_checkpoint_atomically_replaces_arrays(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, np.zeros((2, 3), dtype=np.float32), np.ones((1, 2), dtype=np.float32))
    save_checkpoint(path, np.full((2, 3), 4, dtype=np.float32), np.full((1, 2), 5, dtype=np.float32))

    assert not (tmp_path / "checkpoint.tmp.npz").exists()
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["video"], np.full((2, 3), 4, dtype=np.float32))
        np.testing.assert_array_equal(archive["audio"], np.full((1, 2), 5, dtype=np.float32))
