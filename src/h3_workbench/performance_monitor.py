from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from h3_workbench.inference_runtime import probe_host_commit_memory


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_float(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() == "N/A" or normalized.startswith("["):
        return None
    parsed = float(normalized)
    return parsed if math.isfinite(parsed) else None


def _gpu_sample() -> dict[str, Any]:
    flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,clocks.current.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            creationflags=flags,
            text=True,
            timeout=2,
        )
        rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
        if not rows:
            raise ValueError("nvidia-smi returned no devices")
        devices: list[dict[str, Any]] = []
        for row in rows:
            if len(row) != 10:
                raise ValueError(f"Unexpected nvidia-smi column count: {len(row)}")
            parts = [part.strip() for part in row]
            memory_total_mib = _optional_float(parts[4])
            memory_used_mib = _optional_float(parts[5])
            memory_free_mib = _optional_float(parts[6])
            device = {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_percent": _optional_float(parts[2]),
                "memory_utilization_percent": _optional_float(parts[3]),
                "memory_total_mib": memory_total_mib,
                "memory_used_mib": memory_used_mib,
                "memory_free_mib": memory_free_mib,
                "memory_total_bytes": int(memory_total_mib * 1024**2) if memory_total_mib is not None else None,
                "memory_used_bytes": int(memory_used_mib * 1024**2) if memory_used_mib is not None else None,
                "memory_free_bytes": int(memory_free_mib * 1024**2) if memory_free_mib is not None else None,
                "power_watts": _optional_float(parts[7]),
                "sm_clock_mhz": _optional_float(parts[8]),
                "temperature_c": _optional_float(parts[9]),
            }
            devices.append(device)
        primary = devices[0]
        return {
            "available": True,
            "backend": "nvidia-smi",
            "devices": devices,
            **primary,
        }
    except FileNotFoundError:
        reason = "not_found"
    except subprocess.TimeoutExpired:
        reason = "timeout"
    except subprocess.CalledProcessError:
        reason = "command_failed"
    except (OSError, ValueError, subprocess.SubprocessError, IndexError, csv.Error):
        reason = "malformed_output"
    return {
        "available": False,
        "backend": "nvidia-smi",
        "devices": [],
        "reason": reason,
    }


class PerformanceSampler:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._last_time = time.monotonic()
        self._last_disk = psutil.disk_io_counters()
        try:
            self._last_process_io = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            self._last_process_io = None
        self._samples = 0
        self._lock = threading.Lock()

    def sample(self) -> dict[str, Any]:
        with self._lock:
            return self._sample_locked()

    def _sample_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(1e-6, now - self._last_time)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process_memory = self.process.memory_info()
        disk = psutil.disk_io_counters()
        try:
            process_io = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            process_io = None
        committed, commit_limit = probe_host_commit_memory()

        disk_read = 0.0
        disk_write = 0.0
        disk_read_iops = 0.0
        disk_write_iops = 0.0
        if self._samples > 0 and disk is not None and self._last_disk is not None:
            disk_read = max(0, disk.read_bytes - self._last_disk.read_bytes) / elapsed
            disk_write = max(0, disk.write_bytes - self._last_disk.write_bytes) / elapsed
            disk_read_iops = max(0, disk.read_count - self._last_disk.read_count) / elapsed
            disk_write_iops = max(0, disk.write_count - self._last_disk.write_count) / elapsed
        process_read = 0.0
        process_write = 0.0
        if self._samples > 0 and process_io is not None and self._last_process_io is not None:
            process_read = max(0, process_io.read_bytes - self._last_process_io.read_bytes) / elapsed
            process_write = max(0, process_io.write_bytes - self._last_process_io.write_bytes) / elapsed

        self._last_time = now
        self._last_disk = disk
        self._last_process_io = process_io
        self._samples += 1
        system_cpu = psutil.cpu_percent(None)
        per_core = psutil.cpu_percent(None, percpu=True)
        process_cpu = self.process.cpu_percent(None)
        private_bytes = getattr(process_memory, "private", process_memory.vms)
        gpu = _gpu_sample()
        cpu = {
            "system_percent": system_cpu,
            "per_core_percent": per_core,
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "process_percent": process_cpu,
        }
        process = {
            "pid": self.process.pid,
            "cpu_percent": process_cpu,
            "threads": self.process.num_threads(),
            "rss_bytes": process_memory.rss,
            "private_bytes": private_bytes,
            "virtual_bytes": process_memory.vms,
            "read_bytes_per_second": process_read,
            "write_bytes_per_second": process_write,
        }
        memory_group = {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "percent": memory.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "swap_percent": swap.percent,
            "commit_used_bytes": committed,
            "commit_limit_bytes": commit_limit,
        }
        disk_group = {
            "available": disk is not None,
            "scope": "system_aggregate",
            "read_bytes_per_second": disk_read,
            "write_bytes_per_second": disk_write,
            "read_iops": disk_read_iops,
            "write_iops": disk_write_iops,
            "process_read_bytes_per_second": process_read,
            "process_write_bytes_per_second": process_write,
        }
        return {
            "interval_seconds": elapsed,
            "cpu": cpu,
            "memory": memory_group,
            "gpu": gpu,
            "disk": disk_group,
            "process": process,
            # Preserve the flat job-log contract while the WebUI migrates to grouped metrics.
            "system_cpu_percent": system_cpu,
            "system_cpu_per_core_percent": per_core,
            "process_cpu_percent": process_cpu,
            "process_threads": process["threads"],
            "process_rss_bytes": process_memory.rss,
            "process_private_bytes": private_bytes,
            "memory_total_bytes": memory.total,
            "memory_available_bytes": memory.available,
            "memory_percent": memory.percent,
            "pagefile_used_bytes": swap.used,
            "commit_used_bytes": committed,
            "commit_limit_bytes": commit_limit,
            "disk_read_bytes_per_second": disk_read,
            "disk_write_bytes_per_second": disk_write,
            "process_read_bytes_per_second": process_read,
            "process_write_bytes_per_second": process_write,
        }


class LivePerformanceMonitor:
    """One sampler thread shared by snapshot, SSE, and WebSocket clients."""

    def __init__(
        self,
        interval_seconds: float = 1.0,
        sampler: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self._sampler_instance = None if sampler is not None else PerformanceSampler()
        self.sampler = sampler or self._sampler_instance.sample  # type: ignore[union-attr]
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._sequence = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                name="h3-live-performance-monitor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
            self._condition.notify_all()
        thread.join(timeout=max(3.0, self.interval_seconds * 2))
        with self._condition:
            if self._thread is thread and not thread.is_alive():
                self._thread = None

    def snapshot(self, timeout: float = 3.0) -> dict[str, Any]:
        self.start()
        with self._condition:
            ready = self._condition.wait_for(lambda: self._latest is not None, timeout=max(0.0, timeout))
            if not ready or self._latest is None:
                raise TimeoutError("Timed out waiting for a hardware performance sample")
            return copy.deepcopy(self._latest)

    def wait_for_sample(
        self,
        after_sequence: int = -1,
        timeout: float = 3.0,
    ) -> dict[str, Any] | None:
        self.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._latest is not None
                    and int(self._latest["sequence"]) > after_sequence
                )
                or self._stop.is_set(),
                timeout=max(0.0, timeout),
            )
            if not ready or self._latest is None or int(self._latest["sequence"]) <= after_sequence:
                return None
            return copy.deepcopy(self._latest)

    def _publish(self) -> None:
        try:
            metrics = self.sampler()
            record = {
                "schema": "h3-hardware-sample-v1",
                "timestamp": _now(),
                "sequence": self._sequence,
                **metrics,
            }
        except Exception as exc:  # noqa: BLE001 - live monitoring must survive transient probe failures
            record = {
                "schema": "h3-hardware-sample-v1",
                "timestamp": _now(),
                "sequence": self._sequence,
                "monitor_error": str(exc),
            }
        with self._condition:
            self._latest = record
            self._sequence += 1
            self._condition.notify_all()

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            self._publish()
            next_sample += self.interval_seconds
            delay = next_sample - time.monotonic()
            if delay <= 0:
                next_sample = time.monotonic()
                continue
            self._stop.wait(delay)


class PerformanceMonitor:
    def __init__(
        self,
        path: Path,
        state_callback: Callable[[], dict[str, Any]],
        update_callback: Callable[[dict[str, Any]], None],
        interval_seconds: float = 1.0,
        sampler: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.state_callback = state_callback
        self.update_callback = update_callback
        self.interval_seconds = interval_seconds
        self._sampler_instance = None if sampler is not None else PerformanceSampler()
        self.sampler = sampler or self._sampler_instance.sample  # type: ignore[union-attr]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="h3-performance-monitor", daemon=True)
        self._started = False
        self._started_at = time.monotonic()
        self._sequence = 0

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = time.monotonic()
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=max(3.0, self.interval_seconds * 2))
        if not self._thread.is_alive():
            self._write_sample(final=True)

    def _write_sample(self, final: bool = False) -> None:
        try:
            state = self.state_callback()
            record = {
                "timestamp": _now(),
                "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
                "sequence": self._sequence,
                "final": final,
                **state,
                "performance": self.sampler(),
            }
        except Exception as exc:
            record = {
                "timestamp": _now(),
                "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
                "sequence": self._sequence,
                "final": final,
                "monitor_error": str(exc),
            }
        with self.path.open("a", encoding="utf-8", buffering=1) as output:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._sequence += 1
        if "performance" in record:
            self.update_callback(record)

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            self._write_sample()
            next_sample += self.interval_seconds
            delay = next_sample - time.monotonic()
            if delay <= 0:
                next_sample = time.monotonic()
                continue
            self._stop.wait(delay)
