from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


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


def probe_gpu_memory() -> MemorySnapshot:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        name, total_mib, free_mib = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return MemorySnapshot("cuda", int(total_mib) * MIB, int(free_mib) * MIB, name)
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return MemorySnapshot("cpu", 0, 0, "CPU")


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
