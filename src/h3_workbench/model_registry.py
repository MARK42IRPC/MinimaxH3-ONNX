from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SAFETENSORS_HEADER_LIMIT = 32 * 1024 * 1024


@dataclass(frozen=True)
class ModelRecord:
    id: str
    name: str
    path: str
    component: str
    dtype: str
    size_bytes: int
    tensor_count: int
    metadata: dict[str, Any]
    export_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("file is too short to be a Safetensors checkpoint")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > SAFETENSORS_HEADER_LIMIT:
            raise ValueError(f"Safetensors header is unexpectedly large: {header_length} bytes")
        payload = handle.read(header_length)
        if len(payload) != header_length:
            raise ValueError("Safetensors header is truncated")
    return json.loads(payload)


def _decode_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            try:
                result[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        result[key] = value
    return result


def _classify(name: str, metadata: dict[str, Any]) -> str:
    lower = name.lower()
    if "minimax_h3_audio_vae" in metadata or "audio_vae" in lower:
        return "audio_vae"
    if "minimax_h3_video_vae" in metadata or "video_vae" in lower:
        return "video_vae"
    if "text_encoder" in lower or "qwen3vl" in lower:
        return "text_encoder"
    if "lora" in lower or "conversion_type" in metadata or "target_model" in metadata:
        return "acceleration_lora"
    if "fl2va" in lower:
        return "fl2va_transformer"
    if "ref2va" in lower:
        return "ref2va_transformer"
    return "unknown"


def inspect_checkpoint(path: Path, workspace: Path) -> ModelRecord:
    header = read_safetensors_header(path)
    raw_metadata = header.pop("__metadata__", {})
    metadata = _decode_metadata(raw_metadata)
    tensors = list(header.values())
    dtypes = sorted({entry.get("dtype", "unknown") for entry in tensors})
    component = _classify(path.name, metadata)
    relative = path.relative_to(workspace).as_posix()
    return ModelRecord(
        id=relative,
        name=path.name,
        path=str(path),
        component=component,
        dtype=", ".join(dtypes),
        size_bytes=path.stat().st_size,
        tensor_count=len(tensors),
        metadata=metadata,
        export_supported=component in {
            "audio_vae",
            "video_vae",
            "fl2va_transformer",
            "ref2va_transformer",
            "text_encoder",
        },
    )


def scan_models(workspace: Path) -> list[ModelRecord]:
    records: list[ModelRecord] = []
    for path in workspace.glob("*.safetensors"):
        try:
            records.append(inspect_checkpoint(path.resolve(), workspace))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item.name.lower())
