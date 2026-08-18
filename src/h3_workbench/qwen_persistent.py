from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

from h3_workbench.main_transformer import SAFETENSORS_HEADER_LIMIT, _StreamingSafeTensorFile

RUNTIME_MANIFEST = "runtime_persistent_manifest.json"
RUNTIME_INT8_MANIFEST = "runtime_int8_manifest.json"
RUNTIME_KINDS = ("embedding", "attention", "gate", "up", "down")
INT8_VIRTUAL_FORMAT = "h3-qwen-int8-virtual-v1"
INT8_VIRTUAL_KINDS = ("attention", "mlp")
INT8_SPLIT_KINDS = ("qkv", "output")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_file_identity(path: Path) -> dict[str, int | str]:
    path = path.resolve()
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def qwen_source_identity(path: Path) -> dict[str, int | str]:
    """Fingerprint a large checkpoint without reading its tensor payload."""
    path = path.resolve()
    stat = path.stat()
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Qwen source is not a Safetensors checkpoint: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > SAFETENSORS_HEADER_LIMIT:
            raise ValueError(f"Qwen Safetensors header is unexpectedly large: {header_length} bytes")
        header = handle.read(header_length)
    if len(header) != header_length:
        raise ValueError(f"Qwen Safetensors header is truncated: {path}")
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def int8_virtual_product_fingerprint(directory: Path) -> str:
    raw = json.loads((directory.resolve() / RUNTIME_INT8_MANIFEST).read_text(encoding="utf-8"))
    return _canonical_digest(raw)


def _close_memmaps(values: object) -> None:
    mappings: dict[int, np.memmap] = {}
    pending = list(values) if isinstance(values, (list, tuple)) else [values]
    for value in pending:
        current = value
        while isinstance(current, np.ndarray):
            if isinstance(current, np.memmap):
                mappings[id(current)] = current
            current = current.base
    for mapping in mappings.values():
        mmap = getattr(mapping, "_mmap", None)
        if mmap is not None:
            mmap.close()


@dataclass(frozen=True)
class ExternalInput:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExternalInput":
        return cls(
            name=str(value["name"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),  # type: ignore[arg-type]
            offset=int(value["offset"]),
            length=int(value["length"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "length": self.length,
        }


def _source_graph(directory: Path, kind: str) -> Path:
    if kind == "embedding":
        return directory / "qwen_embedding.onnx"
    return directory / f"qwen_layer_00_{kind}.onnx"


def _external_fields(initializer: onnx.TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


_BUILD_LOCK = threading.Lock()


def build_persistent_qwen_graphs(directory: Path) -> Path:
    """Build the persistent runtime graphs, skipping work when already ready.

    Concurrent job workers may race on the first build, so the whole check
    and build sequence runs under a lock and temporary files carry the PID
    so distinct processes cannot collide.
    """
    with _BUILD_LOCK:
        manifest_path = directory / RUNTIME_MANIFEST
        if persistent_qwen_ready(directory):
            return manifest_path
        return _build_persistent_qwen_graphs_locked(directory)


def _build_persistent_qwen_graphs_locked(directory: Path) -> Path:
    """Promote external initializers to inputs without copying any model weights."""
    directory = directory.resolve()
    manifest: dict[str, object] = {"format": "h3-qwen-persistent-v1", "kinds": {}}
    kinds = manifest["kinds"]
    assert isinstance(kinds, dict)
    for kind in RUNTIME_KINDS:
        source = _source_graph(directory, kind)
        model = onnx.load(str(source), load_external_data=False)
        external: list[ExternalInput] = []
        retained: list[onnx.TensorProto] = []
        for initializer in model.graph.initializer:
            fields = _external_fields(initializer)
            if "location" not in fields:
                retained.append(initializer)
                continue
            dtype = np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type))
            item = ExternalInput(
                initializer.name,
                dtype.str,
                tuple(initializer.dims),
                int(fields.get("offset", "0")),
                int(fields.get("length", str(int(np.prod(initializer.dims)) * dtype.itemsize))),
            )
            external.append(item)
            model.graph.input.append(
                helper.make_tensor_value_info(initializer.name, initializer.data_type, list(initializer.dims))
            )
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
        graph_name = f"runtime_qwen_{kind}.onnx"
        model.graph.name = f"h3_persistent_qwen_{kind}"
        onnx.checker.check_model(model)
        temporary = directory / f"{graph_name}.{os.getpid()}.tmp"
        onnx.save(model, temporary)
        os.replace(temporary, directory / graph_name)
        kinds[kind] = {"graph": graph_name, "external_inputs": [item.to_dict() for item in external]}
    path = directory / RUNTIME_MANIFEST
    temporary_manifest = directory / f"{RUNTIME_MANIFEST}.{os.getpid()}.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, path)
    return path


class QwenWeightInputs:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        raw = json.loads((self.directory / RUNTIME_MANIFEST).read_text(encoding="utf-8"))
        self.kinds = raw["kinds"]
        # Read-only memmaps stay valid for the lifetime of the data files, so
        # build each (kind, layer) feed dict once instead of reopening
        # hundreds of file mappings per encode.
        self._input_cache: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}

    def close(self) -> None:
        arrays = [array for feeds in self._input_cache.values() for array in feeds.values()]
        self._input_cache.clear()
        _close_memmaps(arrays)

    def graph(self, kind: str) -> Path:
        return self.directory / str(self.kinds[kind]["graph"])

    def data_path(self, kind: str, layer: int | None = None) -> Path:
        if kind == "embedding":
            return self.directory / "qwen_embedding.onnx.data"
        if layer is None:
            raise ValueError(f"A layer index is required for Qwen {kind} weights")
        return self.directory / f"qwen_layer_{layer:02d}_{kind}.onnx.data"

    def inputs(self, kind: str, layer: int | None = None) -> dict[str, np.ndarray]:
        key = (kind, layer)
        cached = self._input_cache.get(key)
        if cached is not None:
            return cached
        data_path = self.data_path(kind, layer)
        result: dict[str, np.ndarray] = {}
        for raw in self.kinds[kind]["external_inputs"]:
            spec = ExternalInput.from_dict(raw)
            result[spec.name] = np.memmap(
                data_path,
                dtype=np.dtype(spec.dtype),
                mode="r",
                offset=spec.offset,
                shape=spec.shape,
                order="C",
            )
        self._input_cache[key] = result
        return result


class QwenInt8SourceWeights:
    """Resolve virtual per-layer shards directly from the original checkpoint."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        raw = json.loads((self.directory / RUNTIME_INT8_MANIFEST).read_text(encoding="utf-8"))
        source = Path(str(raw["source_checkpoint"]))
        if not source.is_absolute():
            source = self.directory / source
        self.source = source.resolve()
        if not int8_virtual_qwen_ready(self.directory):
            raise ValueError(f"Qwen INT8 virtual product is stale or incomplete: {self.directory}")
        self.reader = _StreamingSafeTensorFile(self.source)
        self.raw = raw
        self.kinds = raw["kinds"]
        self.attention_split = raw.get("attention_split")
        self.embedding_key = str(raw["embedding_key"])
        self._embedding: np.ndarray | None = self.reader.memmap_tensor(self.embedding_key)
        self._input_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def close(self) -> None:
        arrays = [self._embedding, *(array for feeds in self._input_cache.values() for array in feeds.values())]
        self._input_cache.clear()
        self._embedding = None
        _close_memmaps(arrays)

    def graph(self, kind: str) -> Path:
        if kind in {"attention_qkv", "attention_output"}:
            split_kind = "qkv" if kind == "attention_qkv" else "output"
            if not isinstance(self.attention_split, dict) or split_kind not in self.attention_split:
                raise KeyError(f"Qwen split attention graph is not installed: {kind}")
            return self.directory / str(self.attention_split[split_kind]["graph"])
        return self.directory / str(self.kinds[kind]["graph"])

    def embedding(self, token_ids: np.ndarray) -> np.ndarray:
        if self._embedding is None:
            raise RuntimeError("Qwen INT8 source weights are closed")
        return np.asarray(self._embedding[token_ids.astype(np.int64)], dtype=np.float32)

    def inputs(self, kind: str, layer: int) -> dict[str, np.ndarray]:
        key = (kind, layer)
        cached = self._input_cache.get(key)
        if cached is not None:
            return cached
        result: dict[str, np.ndarray] = {}
        if kind in {"attention_qkv", "attention_output"}:
            split_kind = "qkv" if kind == "attention_qkv" else "output"
            if not isinstance(self.attention_split, dict) or split_kind not in self.attention_split:
                raise KeyError(f"Qwen split attention graph is not installed: {kind}")
            specs = self.attention_split[split_kind]["inputs"]
        else:
            specs = self.kinds[kind]["inputs"]
        for spec in specs:
            source_key = str(spec["source_key"]).format(layer=layer)
            source = self.reader.memmap_tensor(source_key)
            target_dtype = np.dtype(str(spec["target_dtype"]))
            target_shape = tuple(int(value) for value in spec["target_shape"])
            if source.dtype == target_dtype and source.shape == target_shape:
                value = source
            else:
                value = np.asarray(source, dtype=target_dtype).reshape(target_shape)
            result[str(spec["name"])] = np.ascontiguousarray(value) if not isinstance(value, np.memmap) else value
        self._input_cache[key] = result
        return result


def persistent_qwen_ready(directory: Path) -> bool:
    manifest = directory / RUNTIME_MANIFEST
    if not manifest.is_file():
        return False
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        return all((directory / raw["kinds"][kind]["graph"]).is_file() for kind in RUNTIME_KINDS)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def int8_virtual_qwen_ready(directory: Path) -> bool:
    manifest = directory / RUNTIME_INT8_MANIFEST
    if not manifest.is_file():
        return False
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if raw.get("format") != INT8_VIRTUAL_FORMAT or int(raw.get("layers", 0)) != 50:
            return False
        source = Path(str(raw["source_checkpoint"]))
        if not source.is_absolute():
            source = directory / source
        if not source.is_file() or qwen_source_identity(source) != raw["source_identity"]:
            return False
        for kind in INT8_VIRTUAL_KINDS:
            graph = directory / str(raw["kinds"][kind]["graph"])
            if not graph.is_file() or runtime_file_identity(graph) != raw["kinds"][kind]["graph_identity"]:
                return False
        split = raw.get("attention_split")
        if split is not None:
            if not isinstance(split, dict):
                return False
            for kind in INT8_SPLIT_KINDS:
                entry = split.get(kind)
                if not isinstance(entry, dict):
                    return False
                graph = directory / str(entry["graph"])
                if not graph.is_file() or runtime_file_identity(graph) != entry["graph_identity"]:
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def validated_int8_virtual_qwen_ready(directory: Path) -> bool:
    if not int8_virtual_qwen_ready(directory):
        return False
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        fingerprint = int8_virtual_product_fingerprint(directory)
        return (
            manifest.get("validation_passed") is True
            and manifest.get("validated_product_fingerprint") == fingerprint
            and manifest.get("validation", {}).get("product_fingerprint") == fingerprint
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def resolve_qwen_directory(output_root: Path) -> Path:
    virtual = output_root / "qwen3vl_32b_minimax_h3_int8_virtual"
    if validated_int8_virtual_qwen_ready(virtual):
        return virtual
    return output_root / "qwen3vl_32b_minimax_h3_nvfp4_awq"
