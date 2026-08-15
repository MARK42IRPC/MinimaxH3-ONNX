from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from pathlib import Path

"""Cross-process VRAM backpressure for concurrent workbench instances.

Each reservation is one small JSON file owned by a single process, so writers
never contend and readers only glob a directory. A dead PID or a stale
timestamp (crashed writer, no heartbeat) invalidates an entry.
"""

RESERVATION_TTL_SECONDS = 600.0
_READ_TTL_SECONDS = 5.0

_registry_dir: Path | None = None
_lock = threading.Lock()
_last_read_at = 0.0
_last_read_total = 0
_own_records: dict[str, tuple[int, str | None]] = {}


def configure_reservations(directory: Path) -> None:
    global _registry_dir
    directory.mkdir(parents=True, exist_ok=True)
    with _lock:
        _registry_dir = directory


def _reservation_path(token: str) -> Path | None:
    if _registry_dir is None:
        return None
    return _registry_dir / f"{token}.json"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_api = ctypes.windll.kernel32
        handle = process_api.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not process_api.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            process_api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError):
        return False
    return True


def acquire_reservation(token: str, reserved_bytes: int, device: str | None = None) -> bool:
    path = _reservation_path(token)
    if path is None or reserved_bytes <= 0:
        return False
    with _lock:
        _own_records[token] = (int(reserved_bytes), device)
    return refresh_reservation(token)


def refresh_reservation(token: str) -> bool:
    path = _reservation_path(token)
    if path is None:
        return False
    with _lock:
        record = _own_records.get(token)
    if record is None:
        return False
    reserved_bytes, device = record
    payload = {"pid": os.getpid(), "bytes": int(reserved_bytes), "updated_at": time.time()}
    if device is not None:
        payload["device"] = device
    try:
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)
        return True
    except OSError:
        return False


def release_reservation(token: str) -> None:
    with _lock:
        _own_records.pop(token, None)
    path = _reservation_path(token)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def other_reserved_bytes(
    ttl_seconds: float = RESERVATION_TTL_SECONDS,
    device: str | None = None,
) -> int:
    """Sum reservations of alive, recently refreshed foreign processes."""
    global _last_read_at, _last_read_total
    now = time.monotonic()
    with _lock:
        if device is None and now - _last_read_at < _READ_TTL_SECONDS:
            return _last_read_total
        directory = _registry_dir
        own_pid = os.getpid()
        total = 0
        wall_clock = time.time()
        if directory is not None:
            try:
                for path in directory.glob("*.json"):
                    try:
                        raw = json.loads(path.read_text(encoding="utf-8"))
                        pid = int(raw["pid"])
                        reserved = int(raw["bytes"])
                        updated_at = float(raw["updated_at"])
                    except (OSError, KeyError, TypeError, ValueError):
                        continue
                    if pid == own_pid:
                        continue
                    record_device = raw.get("device")
                    # Old records predate per-device reservations. Count them
                    # conservatively for every selected device.
                    if device is not None and record_device not in (None, "default", device):
                        continue
                    if not _pid_alive(pid):
                        continue
                    if wall_clock - updated_at > ttl_seconds:
                        continue
                    total += reserved
            except OSError:
                total = 0
        if device is None:
            _last_read_at = now
            _last_read_total = total
        return total
