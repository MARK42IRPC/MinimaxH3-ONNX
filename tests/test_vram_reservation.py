import json
import os
import subprocess
import sys
import time

import pytest

from h3_workbench import vram_reservation as vr


@pytest.fixture
def registry(tmp_path, monkeypatch):
    directory = tmp_path / "reservations"
    vr.configure_reservations(directory)
    vr._own_records.clear()
    monkeypatch.setattr(vr, "_last_read_at", 0.0)
    monkeypatch.setattr(vr, "_last_read_total", 0)
    yield directory
    vr._own_records.clear()
    monkeypatch.setattr(vr, "_registry_dir", None)
    monkeypatch.setattr(vr, "_last_read_at", 0.0)
    monkeypatch.setattr(vr, "_last_read_total", 0)


def _write_entry(
    directory,
    token: str,
    pid: int,
    reserved_bytes: int,
    updated_at: float | None = None,
    device: str | None = None,
) -> None:
    payload = {"pid": pid, "bytes": reserved_bytes, "updated_at": updated_at or time.time()}
    if device is not None:
        payload["device"] = device
    (directory / f"{token}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _flush(monkeypatch) -> None:
    monkeypatch.setattr(vr, "_last_read_at", 0.0)
    monkeypatch.setattr(vr, "_last_read_total", 0)


def _sleeping_process() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def test_acquire_writes_and_release_removes_own_reservation(registry) -> None:
    assert vr.acquire_reservation("test-token", 7 * 1024**3)
    record = json.loads((registry / "test-token.json").read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid()
    assert record["bytes"] == 7 * 1024**3

    vr.release_reservation("test-token")
    assert not (registry / "test-token.json").exists()


def test_own_reservation_never_counts_as_foreign(registry) -> None:
    assert vr.acquire_reservation("own-token", 7 * 1024**3)
    try:
        assert vr.other_reserved_bytes() == 0
    finally:
        vr.release_reservation("own-token")


def test_alive_foreign_process_is_counted(registry, monkeypatch) -> None:
    process = _sleeping_process()
    try:
        _write_entry(registry, "foreign", process.pid, 5 * 1024**3)
        _flush(monkeypatch)
        assert vr.other_reserved_bytes() == 5 * 1024**3
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_foreign_reservations_are_scoped_to_the_selected_device(registry, monkeypatch) -> None:
    process = _sleeping_process()
    try:
        _write_entry(registry, "gpu-a", process.pid, 5 * 1024**3, device="GPU-a")
        _flush(monkeypatch)
        assert vr.other_reserved_bytes(device="GPU-a") == 5 * 1024**3
        assert vr.other_reserved_bytes(device="GPU-b") == 0
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_dead_pid_is_ignored(registry, monkeypatch) -> None:
    process = _sleeping_process()
    process.terminate()
    process.wait(timeout=10)
    _write_entry(registry, "dead", process.pid, 5 * 1024**3)
    _flush(monkeypatch)
    assert vr.other_reserved_bytes() == 0


def test_stale_timestamp_is_ignored(registry, monkeypatch) -> None:
    process = _sleeping_process()
    try:
        _write_entry(registry, "stale", process.pid, 5 * 1024**3, updated_at=time.time() - 700.0)
        _flush(monkeypatch)
        assert vr.other_reserved_bytes() == 0
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_heartbeat_refresh_updates_timestamp(registry) -> None:
    assert vr.acquire_reservation("heartbeat", 3 * 1024**3)
    try:
        before = json.loads((registry / "heartbeat.json").read_text(encoding="utf-8"))["updated_at"]
        time.sleep(0.01)
        assert vr.refresh_reservation("heartbeat")
        after = json.loads((registry / "heartbeat.json").read_text(encoding="utf-8"))["updated_at"]
        assert after >= before
    finally:
        vr.release_reservation("heartbeat")


def test_refresh_without_acquire_is_a_noop(registry) -> None:
    assert vr.refresh_reservation("never-acquired") is False


def test_unconfigured_registry_is_a_noop(monkeypatch) -> None:
    monkeypatch.setattr(vr, "_registry_dir", None)
    _flush(monkeypatch)
    assert vr.other_reserved_bytes() == 0
    assert vr.acquire_reservation("nowhere", 1024) is False
    assert vr.refresh_reservation("nowhere") is False
    vr.release_reservation("nowhere")
