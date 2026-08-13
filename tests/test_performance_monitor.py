import json
import time
from pathlib import Path

from h3_workbench.jobs import Job, JobManager
from h3_workbench.performance_monitor import PerformanceMonitor


def test_performance_monitor_writes_samples_and_terminal_state(tmp_path: Path) -> None:
    status = {"value": "running"}
    updates: list[dict[str, object]] = []
    log_path = tmp_path / "performance.jsonl"
    monitor = PerformanceMonitor(
        log_path,
        lambda: {"job_id": "job", "status": status["value"], "activity": {"module": "Qwen"}},
        updates.append,
        interval_seconds=0.01,
        sampler=lambda: {"system_cpu_percent": 12.5},
    )

    monitor.start()
    time.sleep(0.035)
    status["value"] = "completed"
    monitor.stop()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= 2
    assert records[0]["activity"]["module"] == "Qwen"
    assert records[-1]["status"] == "completed"
    assert records[-1]["final"] is True
    assert records[-1]["performance"]["system_cpu_percent"] == 12.5
    assert updates[-1] == records[-1]


def test_performance_log_path_rejects_path_outside_monitor_directory(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "onnx")
    manager._jobs["valid"] = Job(
        id="valid",
        model_id="model",
        performance_log=str(tmp_path / ".h3-workbench" / "performance" / "h3-valid.jsonl"),
    )
    manager._jobs["invalid"] = Job(
        id="invalid",
        model_id="model",
        performance_log=str(tmp_path / "outside.jsonl"),
    )

    assert manager.performance_log_path("valid") == (
        tmp_path / ".h3-workbench" / "performance" / "h3-valid.jsonl"
    ).resolve()
    assert manager.performance_log_path("invalid") is None
    assert manager.performance_log_path("missing") is None
