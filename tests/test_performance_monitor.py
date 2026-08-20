import json
import logging
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path

from h3_workbench.jobs import Job, JobManager
from h3_workbench.performance_monitor import LivePerformanceMonitor, PerformanceMonitor, PerformanceSampler, _gpu_sample


def test_job_performance_update_publishes_a_throttled_heartbeat(tmp_path: Path, caplog) -> None:
    manager = JobManager(tmp_path, tmp_path / "onnx")
    manager._jobs["job"] = Job(
        id="job",
        model_id="model",
        status="running",
        progress=0.35,
        activity={
            "phase": "text",
            "module": "Qwen",
            "operation": "MLP",
            "current": 21,
            "total": 50,
        },
    )
    record = {
        "sequence": 15,
        "elapsed_seconds": 42.0,
        "performance": {
            "gpu": {"utilization_percent": 97.0, "memory_used_mib": 3760.0},
            "process": {"cpu_percent": 88.0},
        },
    }

    with caplog.at_level(logging.INFO, logger="h3_workbench.jobs"):
        manager._update_performance("job", record)

    assert manager.get("job").performance == record
    assert "仍在运行" in manager.get("job").message
    assert "21/50" in manager.get("job").message
    assert "job=job heartbeat" in caplog.text
    assert "position=21/50" in caplog.text

    caplog.clear()
    manager._update_performance("job", {**record, "sequence": 16})
    assert not caplog.records


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


def test_gpu_sample_parses_devices_and_nullable_metrics(monkeypatch) -> None:
    from h3_workbench import performance_monitor as module

    output = (
        "0, RTX Test, 42, 8, 4096, 1024, 3072, 55.5, 1800, 61\n"
        "1, RTX Other, 7, 2, 8192, 512, 7680, N/A, [Not Supported], 45\n"
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))

    sample = _gpu_sample()

    assert sample["available"] is True
    assert sample["name"] == "RTX Test"
    assert sample["memory_total_mib"] == 4096
    assert sample["memory_total_bytes"] == 4 * 1024**3
    assert len(sample["devices"]) == 2
    assert sample["devices"][1]["power_watts"] is None
    assert sample["devices"][1]["sm_clock_mhz"] is None


def test_gpu_sample_handles_missing_and_failed_nvidia_smi(monkeypatch) -> None:
    from h3_workbench import performance_monitor as module

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(module.subprocess, "run", missing)
    assert _gpu_sample() == {
        "available": False,
        "backend": "nvidia-smi",
        "devices": [],
        "reason": "not_found",
    }

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(module.subprocess, "run", failed)
    assert _gpu_sample()["reason"] == "command_failed"


def test_performance_sampler_exposes_grouped_metrics_and_counter_rates(monkeypatch) -> None:
    from h3_workbench import performance_monitor as module

    disk_samples = iter(
        (
            SimpleNamespace(read_bytes=1000, write_bytes=2000, read_count=10, write_count=20),
            SimpleNamespace(read_bytes=1100, write_bytes=2200, read_count=11, write_count=22),
            SimpleNamespace(read_bytes=2100, write_bytes=3200, read_count=13, write_count=25),
        )
    )
    process_io = iter(
        (
            SimpleNamespace(read_bytes=100, write_bytes=200),
            SimpleNamespace(read_bytes=150, write_bytes=250),
            SimpleNamespace(read_bytes=350, write_bytes=550),
        )
    )

    class FakeProcess:
        pid = 123

        def cpu_percent(self, interval=None):
            return 12.5

        def io_counters(self):
            return next(process_io)

        def memory_info(self):
            return SimpleNamespace(rss=100, private=80, vms=200)

        def num_threads(self):
            return 4

    monotonic = iter((10.0, 11.0, 12.0))
    monkeypatch.setattr(module.psutil, "Process", lambda pid: FakeProcess())
    monkeypatch.setattr(module.psutil, "cpu_percent", lambda interval=None, percpu=False: [25.0, 50.0] if percpu else 37.5)
    monkeypatch.setattr(module.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(module.psutil, "disk_io_counters", lambda: next(disk_samples))
    monkeypatch.setattr(
        module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=1000, available=400, used=600, percent=60.0),
    )
    monkeypatch.setattr(
        module.psutil,
        "swap_memory",
        lambda: SimpleNamespace(total=500, used=100, percent=20.0),
    )
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(module, "probe_host_commit_memory", lambda: (700, 1500))
    monkeypatch.setattr(
        module,
        "_gpu_sample",
        lambda: {"available": False, "backend": "nvidia-smi", "devices": [], "reason": "not_found"},
    )

    sampler = PerformanceSampler()
    warmup = sampler.sample()
    sample = sampler.sample()

    assert warmup["disk"]["read_bytes_per_second"] == 0
    assert sample["cpu"] == {
        "system_percent": 37.5,
        "per_core_percent": [25.0, 50.0],
        "logical_cores": 8,
        "physical_cores": 4,
        "process_percent": 12.5,
    }
    assert sample["memory"]["commit_limit_bytes"] == 1500
    assert sample["disk"]["read_bytes_per_second"] == 1000
    assert sample["disk"]["write_iops"] == 3
    assert sample["process"]["read_bytes_per_second"] == 200
    assert sample["system_cpu_percent"] == sample["cpu"]["system_percent"]
    assert sample["gpu"]["available"] is False


def test_live_performance_monitor_shares_monotonic_sample_sequence() -> None:
    calls = {"value": 0}

    def sample() -> dict[str, object]:
        calls["value"] += 1
        return {"cpu": {"system_percent": float(calls["value"])}}

    monitor = LivePerformanceMonitor(interval_seconds=0.01, sampler=sample)
    try:
        first = monitor.snapshot(timeout=1)
        second = monitor.wait_for_sample(after_sequence=first["sequence"], timeout=1)
    finally:
        monitor.stop()

    assert first["schema"] == "h3-hardware-sample-v1"
    assert second is not None
    assert second["sequence"] > first["sequence"]
    assert second["cpu"]["system_percent"] > first["cpu"]["system_percent"]
