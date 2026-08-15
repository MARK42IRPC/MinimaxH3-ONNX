from __future__ import annotations

import mmap
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import psutil
import onnx
from google.protobuf.message import DecodeError

GIB = 1024**3
DEFAULT_PREFETCH_DEPTH = 16
READ_CHUNK_BYTES = 8 * 1024**2


@dataclass
class _MappedFile:
    handle: object
    mapping: mmap.mmap

    def close(self) -> None:
        self.mapping.close()
        self.handle.close()  # type: ignore[attr-defined]


@dataclass
class _CacheEntry:
    path: Path
    size_bytes: int
    future: Future[list[_MappedFile]]


def graph_files(path: Path) -> tuple[Path, ...]:
    if path.name.endswith(".onnx.data") or path.suffix == ".data":
        return (path,)
    external = path.with_name(f"{path.name}.data")
    if external.is_file():
        return (path, external)
    try:
        model = onnx.load(str(path), load_external_data=False)
    except (OSError, ValueError, DecodeError):
        return (path,)
    locations: list[Path] = []
    seen: set[Path] = set()
    for initializer in model.graph.initializer:
        location = next((item.value for item in initializer.external_data if item.key == "location"), None)
        if not location:
            continue
        item = (path.parent / location).resolve()
        if item.is_file() and item not in seen:
            seen.add(item)
            locations.append(item)
    return (path, *locations)


def graph_storage_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in graph_files(path))


def default_l2_cache_bytes() -> int:
    override = os.environ.get("H3_L2_CACHE_GIB")
    if override:
        try:
            return max(0, int(float(override) * GIB))
        except ValueError:
            pass
    available = psutil.virtual_memory().available
    # Keep a generous OS/application reserve while allowing the 32 GB-class
    # edge host to read ahead enough 0.2-0.5 GB shards to hide SSD latency.
    reserve = 8 * GIB
    return min(12 * GIB, max(1 * GIB, available - reserve))


def default_prefetch_depth() -> int:
    try:
        return max(1, int(os.environ.get("H3_PREFETCH_SHARDS", DEFAULT_PREFETCH_DEPTH)))
    except ValueError:
        return DEFAULT_PREFETCH_DEPTH


class ShardPrefetchCache:
    """Bounded SSD-to-RAM read-ahead backed by the operating system page cache."""

    def __init__(self, budget_bytes: int | None = None, prefetch_depth: int = DEFAULT_PREFETCH_DEPTH):
        self._auto_budget = budget_bytes is None
        self._budget_cap = default_l2_cache_bytes() if budget_bytes is None else max(0, budget_bytes)
        self.budget_bytes = self._budget_cap
        self.prefetch_depth = max(1, prefetch_depth)
        self._entries: OrderedDict[Path, _CacheEntry] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-shard-prefetch")
        self._lock = threading.Lock()
        self._closed = False
        self._hits = 0
        self._waits = 0
        self._wait_seconds = 0.0
        self._budget_adjustments = 0
        self._pressure_evictions = 0
        try:
            self._memory_reserve_bytes = max(
                1 * GIB,
                int(float(os.environ.get("H3_L2_CACHE_RESERVE_GIB", "8")) * GIB),
            )
        except ValueError:
            self._memory_reserve_bytes = 8 * GIB

    @staticmethod
    def _warm(path: Path) -> list[_MappedFile]:
        mapped: list[_MappedFile] = []
        try:
            for item in graph_files(path):
                handle = item.open("rb")
                if item.stat().st_size == 0:
                    handle.close()
                    continue
                mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                mapped.append(_MappedFile(handle, mapping))
                mapping.seek(0)
                while mapping.read(READ_CHUNK_BYTES):
                    pass
                mapping.seek(0)
            return mapped
        except Exception:
            for item in mapped:
                item.close()
            raise

    def stage(self, paths: list[Path]) -> None:
        with self._lock:
            self._refresh_budget_locked()
            budget_bytes = self.budget_bytes
        if budget_bytes <= 0:
            return
        desired: list[tuple[Path, int]] = []
        desired_bytes = 0
        for raw_path in paths[: self.prefetch_depth]:
            path = raw_path.resolve()
            size = graph_storage_bytes(path)
            if desired and desired_bytes + size > budget_bytes:
                break
            desired.append((path, size))
            desired_bytes += size

        with self._lock:
            if self._closed:
                return
            desired_paths = {path for path, _ in desired}
            for path in list(self._entries):
                if path not in desired_paths:
                    self._remove_locked(path)
            for path, size in desired:
                if path in self._entries:
                    self._entries.move_to_end(path)
                    continue
                future = self._executor.submit(self._warm, path)
                self._entries[path] = _CacheEntry(path, size, future)

    def set_budget(self, budget_bytes: int) -> None:
        """Lower or raise the active budget while retaining the cache object."""
        with self._lock:
            target = max(0, int(budget_bytes))
            if self._auto_budget:
                self._budget_cap = min(self._budget_cap, target)
                target = min(self.budget_bytes, self._budget_cap)
            else:
                self._budget_cap = target
            if target == self.budget_bytes:
                return
            self.budget_bytes = target
            self._budget_adjustments += 1
            while self._entries and sum(item.size_bytes for item in self._entries.values()) > target:
                self._pressure_evictions += 1
                self._remove_locked(next(iter(self._entries)))

    def _refresh_budget_locked(self) -> None:
        if not self._auto_budget:
            return
        available = int(psutil.virtual_memory().available)
        target = min(self._budget_cap, max(0, available - self._memory_reserve_bytes))
        if target >= self.budget_bytes:
            return
        self.budget_bytes = target
        self._budget_adjustments += 1
        while self._entries and sum(item.size_bytes for item in self._entries.values()) > target:
            self._pressure_evictions += 1
            self._remove_locked(next(iter(self._entries)))

    def wait(self, path: Path) -> None:
        if self.budget_bytes <= 0:
            return
        resolved = path.resolve()
        with self._lock:
            entry = self._entries.get(resolved)
        if entry is None:
            self.stage([resolved])
            with self._lock:
                entry = self._entries.get(resolved)
        if entry is None:
            return
        ready = entry.future.done()
        started = time.perf_counter()
        try:
            entry.future.result()
        except CancelledError:
            # A concurrently advancing prefetch window can evict an entry just
            # before its consumer waits. Restage it instead of failing the job.
            self.stage([resolved])
            with self._lock:
                replacement = self._entries.get(resolved)
            if replacement is None:
                return
            replacement.future.result()
        elapsed = time.perf_counter() - started
        with self._lock:
            if ready:
                self._hits += 1
            else:
                self._waits += 1
                self._wait_seconds += elapsed

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            entries = list(self._entries.values())
            return {
                "l2_budget_bytes": self.budget_bytes,
                "l2_staged_bytes": sum(entry.size_bytes for entry in entries),
                "l2_entries": len(entries),
                "l2_ready": sum(entry.future.done() and entry.future.exception() is None for entry in entries),
                "l2_hits": self._hits,
                "l2_waits": self._waits,
                "l2_wait_seconds": round(self._wait_seconds, 3),
                "l2_budget_cap_bytes": self._budget_cap,
                "l2_budget_adjustments": self._budget_adjustments,
                "l2_pressure_evictions": self._pressure_evictions,
            }

    def _remove_locked(self, path: Path) -> None:
        entry = self._entries.pop(path)
        if entry.future.cancel():
            return
        try:
            mapped = entry.future.result()
        except Exception:
            return
        for item in mapped:
            item.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            paths = list(self._entries)
            for path in paths:
                self._remove_locked(path)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "ShardPrefetchCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
