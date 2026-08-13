from __future__ import annotations

import json
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


def _gpu_sample() -> dict[str, float | int | str] | None:
    flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.current.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            creationflags=flags,
            text=True,
            timeout=2,
        )
        parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "utilization_percent": float(parts[1]),
            "memory_utilization_percent": float(parts[2]),
            "memory_used_mib": float(parts[3]),
            "memory_free_mib": float(parts[4]),
            "power_watts": float(parts[5]),
            "sm_clock_mhz": float(parts[6]),
            "temperature_c": float(parts[7]),
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


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

    def sample(self) -> dict[str, Any]:
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
        if disk is not None and self._last_disk is not None:
            disk_read = max(0, disk.read_bytes - self._last_disk.read_bytes) / elapsed
            disk_write = max(0, disk.write_bytes - self._last_disk.write_bytes) / elapsed
        process_read = 0.0
        process_write = 0.0
        if process_io is not None and self._last_process_io is not None:
            process_read = max(0, process_io.read_bytes - self._last_process_io.read_bytes) / elapsed
            process_write = max(0, process_io.write_bytes - self._last_process_io.write_bytes) / elapsed

        self._last_time = now
        self._last_disk = disk
        self._last_process_io = process_io
        return {
            "system_cpu_percent": psutil.cpu_percent(None),
            "system_cpu_per_core_percent": psutil.cpu_percent(None, percpu=True),
            "process_cpu_percent": self.process.cpu_percent(None),
            "process_threads": self.process.num_threads(),
            "process_rss_bytes": process_memory.rss,
            "process_private_bytes": getattr(process_memory, "private", process_memory.vms),
            "memory_available_bytes": memory.available,
            "memory_percent": memory.percent,
            "pagefile_used_bytes": swap.used,
            "commit_used_bytes": committed,
            "commit_limit_bytes": commit_limit,
            "disk_read_bytes_per_second": disk_read,
            "disk_write_bytes_per_second": disk_write,
            "process_read_bytes_per_second": process_read,
            "process_write_bytes_per_second": process_write,
            "gpu": _gpu_sample(),
        }


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
