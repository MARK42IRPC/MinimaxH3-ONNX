from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from h3_workbench.device_profile import probe_device_profiles, select_device_profile
from h3_workbench.profiles import GenerationProfile

MIB = 1024**2
DEFAULT_FIXED_RESERVE = 768 * MIB
DEFAULT_WEIGHT_FACTOR = 1.20
STREAMING_HEAD_DIM = 128


@dataclass(frozen=True)
class MemorySnapshot:
    provider: str
    total_bytes: int
    free_bytes: int
    device: str
    index: int = -1
    uuid: str | None = None
    compute_capability: str | None = None
    driver: str | None = None
    tier: str = "cpu"

    @property
    def device_key(self) -> str | None:
        if self.index < 0:
            return None
        return self.uuid or f"index:{self.index}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["device_key"] = self.device_key
        return result


@dataclass(frozen=True)
class ShardEstimate:
    graph: str
    weight_bytes: int
    estimated_resident_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ShardBatch:
    index: int
    shards: tuple[ShardEstimate, ...]
    estimated_resident_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "shards": [item.to_dict() for item in self.shards],
            "estimated_resident_bytes": self.estimated_resident_bytes,
        }


_PROBE_TTL_SECONDS = 2.0
_last_probe_at = 0.0
_last_probe_snapshot: MemorySnapshot | None = None
_last_probe_selector: str | None = None


def probe_gpu_memory(device_index: int | None = None) -> MemorySnapshot:
    """Return the selected device's live memory and compatibility identity."""
    global _last_probe_at, _last_probe_snapshot, _last_probe_selector
    now = time.monotonic()
    selector = str(device_index) if device_index is not None else os.environ.get("H3_CUDA_DEVICE", "0")
    if (
        device_index is None
        and _last_probe_snapshot is not None
        and _last_probe_selector == selector
        and now - _last_probe_at < _PROBE_TTL_SECONDS
    ):
        return _last_probe_snapshot
    profiles = probe_device_profiles()
    snapshot_profile = select_device_profile(
        profiles,
        selector=str(device_index) if device_index is not None else None,
    )
    runtime_memory_available = snapshot_profile.provider == "cuda"
    snapshot = MemorySnapshot(
        snapshot_profile.provider,
        snapshot_profile.total_bytes if runtime_memory_available else 0,
        snapshot_profile.free_bytes if runtime_memory_available else 0,
        snapshot_profile.name,
        snapshot_profile.index,
        snapshot_profile.uuid,
        snapshot_profile.compute_capability,
        snapshot_profile.driver,
        snapshot_profile.tier,
    )
    if device_index is None:
        _last_probe_at = now
        _last_probe_snapshot = snapshot
        _last_probe_selector = selector
    return snapshot


def _graph_weight_bytes(directory: Path, graph: str) -> int:
    graph_path = directory / graph
    external_path = graph_path.with_name(f"{graph_path.name}.data")
    if external_path.is_file():
        return external_path.stat().st_size
    return graph_path.stat().st_size


def main_model_shards(directory: Path) -> list[tuple[str, int]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return [
        (graph, _graph_weight_bytes(directory, graph))
        for graph in manifest["graphs"]
        if graph.startswith("main_block_")
    ]


def plan_shard_batches(
    shards: list[tuple[str, int]],
    profile: GenerationProfile,
    free_bytes: int,
    fixed_reserve_bytes: int = DEFAULT_FIXED_RESERVE,
    weight_factor: float = DEFAULT_WEIGHT_FACTOR,
) -> list[ShardBatch]:
    if not shards:
        return []
    workspace = profile.attention_workspace_bytes
    usable = max(1, free_bytes - fixed_reserve_bytes - workspace)
    estimates = [
        ShardEstimate(name, size, max(1, int(size * weight_factor)))
        for name, size in shards
    ]

    batches: list[ShardBatch] = []
    current: list[ShardEstimate] = []
    current_bytes = 0
    for shard in estimates:
        if current and current_bytes + shard.estimated_resident_bytes > usable:
            batches.append(ShardBatch(len(batches), tuple(current), current_bytes + workspace))
            current = []
            current_bytes = 0
        current.append(shard)
        current_bytes += shard.estimated_resident_bytes
    if current:
        batches.append(ShardBatch(len(batches), tuple(current), current_bytes + workspace))
    return batches


def streaming_kv_bytes(profile: GenerationProfile, element_bytes: int = 2) -> int:
    return profile.sequence_tokens * profile.main_attention_heads * STREAMING_HEAD_DIM * 2 * element_bytes


def plan_streaming_shard_batches(
    shards: list[tuple[str, int]],
    profile: GenerationProfile,
    free_bytes: int,
    fixed_reserve_bytes: int = DEFAULT_FIXED_RESERVE,
    weight_factor: float = DEFAULT_WEIGHT_FACTOR,
    max_sessions: int = 3,
) -> list[ShardBatch]:
    """Plan dependency-safe L1 batches around the streaming-attention workspace."""
    if not shards:
        return []
    reserve = fixed_reserve_bytes + streaming_kv_bytes(profile)
    usable = max(1, free_bytes - reserve)
    estimates = [ShardEstimate(name, size, max(1, int(size * weight_factor))) for name, size in shards]
    batches: list[ShardBatch] = []
    current: list[ShardEstimate] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            batches.append(ShardBatch(len(batches), tuple(current), current_bytes + reserve))
            current = []
            current_bytes = 0

    for shard in estimates:
        if shard.graph.endswith("_attention_output.onnx"):
            flush()
            current.append(shard)
            current_bytes += shard.estimated_resident_bytes
            flush()
            continue
        if current and (len(current) >= max_sessions or current_bytes + shard.estimated_resident_bytes > usable):
            flush()
        current.append(shard)
        current_bytes += shard.estimated_resident_bytes
        if shard.graph.endswith("_attention_qkv.onnx"):
            flush()
    flush()
    return batches


def halve_batch(batch: ShardBatch) -> tuple[ShardBatch, ShardBatch | None]:
    if len(batch.shards) <= 1:
        return batch, None
    split = max(1, len(batch.shards) // 2)
    left = batch.shards[:split]
    right = batch.shards[split:]
    first = ShardBatch(batch.index, left, sum(item.estimated_resident_bytes for item in left))
    second = ShardBatch(batch.index + 1, right, sum(item.estimated_resident_bytes for item in right))
    return first, second
