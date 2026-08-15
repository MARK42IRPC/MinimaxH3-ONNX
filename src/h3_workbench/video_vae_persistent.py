from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import psutil
from onnx import TensorProto, helper, numpy_helper


PERSISTENT_VIDEO_VAE_FORMAT = "h3-video-vae-persistent-v1"
PERSISTENT_VIDEO_VAE_TOPOLOGY = "runtime_persistent_video_decoder_block.onnx"
PERSISTENT_VIDEO_VAE_MANIFEST = "runtime_persistent_video_decoder_manifest.json"

_BLOCK_PATTERN = re.compile(r"^video_decoder_block_(\d+)\.onnx$")
_ACTIVATION_INPUTS = ("hidden_states", "rotary_table")
_ACTIVATION_OUTPUT = "hidden_states_out"
_BUILD_LOCK = threading.Lock()


@dataclass(frozen=True)
class VideoVAEWeightInput:
    name: str
    dtype: int
    shape: tuple[int, ...]

    @classmethod
    def from_initializer(cls, initializer: onnx.TensorProto) -> VideoVAEWeightInput:
        return cls(
            name=initializer.name,
            dtype=initializer.data_type,
            shape=tuple(int(value) for value in initializer.dims),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VideoVAEWeightInput:
        return cls(
            name=str(value["name"]),
            dtype=int(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
        )

    @property
    def nbytes(self) -> int:
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(self.dtype))
        return int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize

    @property
    def signature(self) -> tuple[str, int, tuple[int, ...]]:
        return self.name, self.dtype, self.shape

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "dtype_name": TensorProto.DataType.Name(self.dtype),
            "numpy_dtype": np.dtype(helper.tensor_dtype_to_np_dtype(self.dtype)).str,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
        }


@dataclass
class LoadedVideoVAEBlockWeights:
    block: int
    source: Path
    arrays: dict[str, np.ndarray]
    total_bytes: int
    load_seconds: float
    fingerprint: str

    def feeds(self) -> dict[str, np.ndarray]:
        return self.arrays

    def close(self) -> None:
        self.arrays.clear()


class VideoVAEWeightCache:
    """Keep the complete decoder block working set in host RAM when it fits."""

    def __init__(self, weights: dict[int, LoadedVideoVAEBlockWeights], load_seconds: float) -> None:
        self._weights = weights
        self.load_seconds = load_seconds
        self.bytes = sum(item.total_bytes for item in weights.values())

    def get(self, block: int) -> LoadedVideoVAEBlockWeights:
        return self._weights[block]

    def close(self) -> None:
        for item in self._weights.values():
            item.close()
        self._weights.clear()


def video_vae_ram_cache_budget_bytes(directory: Path, block_count: int) -> int:
    """Return a safe full-cache budget, or zero when the working set should stream."""
    setting = os.environ.get("H3_VIDEO_VAE_RAM_CACHE", "auto").strip().lower()
    if setting not in {"auto", "0", "1", "true", "false", "on", "off", "yes", "no"}:
        raise ValueError("H3_VIDEO_VAE_RAM_CACHE must be auto, 0, or 1")
    if setting in {"0", "false", "off", "no"}:
        return 0
    manifest = load_persistent_video_vae_manifest(directory)
    required_bytes = int(manifest["weight_bytes"]) * max(0, int(block_count))
    if setting in {"1", "true", "on", "yes"}:
        return required_bytes
    try:
        reserve_gib = max(1.0, float(os.environ.get("H3_VIDEO_VAE_RAM_RESERVE_GIB", "6")))
    except ValueError:
        reserve_gib = 6.0
    available = int(psutil.virtual_memory().available)
    return required_bytes if available >= required_bytes + int(reserve_gib * 1024**3) else 0


def preload_video_vae_block_weights(directory: Path, blocks: Iterable[int]) -> VideoVAEWeightCache | None:
    """Load all decoder block weights once so tile decoding never rereads the SSD."""
    requested = tuple(int(block) for block in blocks)
    budget = video_vae_ram_cache_budget_bytes(directory, len(requested))
    if budget <= 0 or not requested:
        return None
    try:
        workers = max(1, min(4, int(os.environ.get("H3_VIDEO_VAE_RAM_WORKERS", "2"))))
    except ValueError:
        workers = 2
    started = time.perf_counter()
    loaded: dict[int, LoadedVideoVAEBlockWeights] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(requested)), thread_name_prefix="video-vae-ram-cache") as executor:
            for item in executor.map(lambda block: load_video_vae_block_weights(directory, block), requested):
                loaded[item.block] = item
    except MemoryError:
        for item in loaded.values():
            item.close()
        if os.environ.get("H3_VIDEO_VAE_RAM_CACHE", "auto").strip().lower() == "auto":
            return None
        raise
    cache = VideoVAEWeightCache(loaded, time.perf_counter() - started)
    if cache.bytes > budget:
        cache.close()
        raise RuntimeError(
            f"Video VAE RAM cache exceeded its budget: {cache.bytes} > {budget} bytes"
        )
    return cache


def _value_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_param:
            result.append(dimension.dim_param)
        elif dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        else:
            result.append(None)
    return result


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": tensor.elem_type,
        "shape": _value_shape(value),
    }


def _weight_inputs(model: onnx.ModelProto) -> list[VideoVAEWeightInput]:
    return [VideoVAEWeightInput.from_initializer(initializer) for initializer in model.graph.initializer]


def video_vae_block_fingerprint(model: onnx.ModelProto) -> str:
    topology = {
        "ir_version": model.ir_version,
        "opsets": [(item.domain, item.version) for item in model.opset_import],
        "inputs": [_value_contract(value) for value in model.graph.input],
        "outputs": [_value_contract(value) for value in model.graph.output],
        "value_info": [_value_contract(value) for value in model.graph.value_info],
        "initializers": [item.signature for item in _weight_inputs(model)],
        "nodes": [
            {
                "op_type": node.op_type,
                "domain": node.domain,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": [hashlib.sha256(attribute.SerializeToString()).hexdigest() for attribute in node.attribute],
            }
            for node in model.graph.node
        ],
        "functions": [hashlib.sha256(function.SerializeToString()).hexdigest() for function in model.functions],
    }
    encoded = json.dumps(topology, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _block_path(directory: Path, block: int) -> Path:
    if block < 0:
        raise ValueError("Video VAE block index must be non-negative")
    path = directory / f"video_decoder_block_{block:02d}.onnx"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def video_vae_block_indices(directory: Path) -> list[int]:
    directory = directory.resolve()
    source_manifest = directory / "manifest.json"
    if source_manifest.is_file():
        try:
            raw = json.loads(source_manifest.read_text(encoding="utf-8"))
            declared = sorted({int(value) for value in raw.get("blocks", [])})
            if declared and all(_block_path(directory, block).is_file() for block in declared):
                return declared
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    discovered = sorted(
        int(match.group(1))
        for path in directory.glob("video_decoder_block_*.onnx")
        if (match := _BLOCK_PATTERN.match(path.name)) is not None
    )
    if not discovered:
        raise FileNotFoundError(f"No Video VAE decoder blocks found in {directory}")
    return discovered


def _requested_blocks(directory: Path, blocks: Iterable[int] | None) -> list[int]:
    requested = video_vae_block_indices(directory) if blocks is None else sorted({int(value) for value in blocks})
    if not requested:
        raise ValueError("At least one Video VAE block must be selected")
    requested = sorted({0, *requested})
    for block in requested:
        _block_path(directory, block)
    return requested


def validate_video_vae_block_schemas(
    directory: Path,
    blocks: Iterable[int] | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    requested = _requested_blocks(directory, blocks)
    started = time.perf_counter()
    reference_fingerprint: str | None = None
    reference_weights: tuple[tuple[str, int, tuple[int, ...]], ...] | None = None
    records: dict[str, Any] = {}
    for block in requested:
        path = _block_path(directory, block)
        model = onnx.load(path, load_external_data=False)
        try:
            fingerprint = video_vae_block_fingerprint(model)
            weights = _weight_inputs(model)
            signature = tuple(item.signature for item in weights)
            if reference_fingerprint is None:
                reference_fingerprint = fingerprint
                reference_weights = signature
            records[str(block)] = {
                "source": path.name,
                "source_stat": _file_stat(path),
                "fingerprint": fingerprint,
                "topology_matches": fingerprint == reference_fingerprint,
                "weight_schema_matches": signature == reference_weights,
                "initializer_count": len(weights),
                "initializer_bytes": sum(item.nbytes for item in weights),
                "nodes": len(model.graph.node),
            }
        finally:
            del model
            gc.collect()
    compatible = all(
        bool(record["topology_matches"] and record["weight_schema_matches"])
        for record in records.values()
    )
    return {
        "reference_block": 0,
        "fingerprint": reference_fingerprint,
        "validated_blocks": requested,
        "compatible": compatible,
        "blocks": records,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _set_dynamic_batch(value: onnx.ValueInfoProto) -> None:
    dimensions = value.type.tensor_type.shape.dim
    if not dimensions:
        raise ValueError(f"Tensor {value.name} has no batch dimension")
    dimensions[0].ClearField("dim_value")
    dimensions[0].dim_param = "batch"


def _promote_initializers(model: onnx.ModelProto, dynamic_batch: bool) -> list[VideoVAEWeightInput]:
    if model.graph.sparse_initializer:
        raise ValueError("Sparse initializers are not supported by the persistent Video VAE topology")
    weights = _weight_inputs(model)
    if not weights:
        raise ValueError("Video VAE block has no initializers to promote")
    existing_inputs = {value.name for value in model.graph.input}
    for initializer, weight in zip(model.graph.initializer, weights, strict=True):
        if weight.name in existing_inputs:
            raise ValueError(f"Initializer already appears as a graph input: {weight.name}")
        model.graph.input.append(
            helper.make_tensor_value_info(weight.name, weight.dtype, list(weight.shape))
        )
    del model.graph.initializer[:]
    if dynamic_batch:
        inputs = {value.name: value for value in model.graph.input}
        outputs = {value.name: value for value in model.graph.output}
        for name in _ACTIVATION_INPUTS:
            if name not in inputs:
                raise ValueError(f"Video VAE topology is missing input {name}")
            _set_dynamic_batch(inputs[name])
        if _ACTIVATION_OUTPUT not in outputs:
            raise ValueError(f"Video VAE topology is missing output {_ACTIVATION_OUTPUT}")
        _set_dynamic_batch(outputs[_ACTIVATION_OUTPUT])
    model.graph.name = "h3_persistent_video_vae_decoder_block"
    onnx.checker.check_model(model, full_check=False)
    return weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def _flush_file(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_persistent_video_vae_manifest(directory: Path) -> dict[str, Any]:
    path = directory.resolve() / PERSISTENT_VIDEO_VAE_MANIFEST
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != PERSISTENT_VIDEO_VAE_FORMAT:
        raise ValueError(f"Unsupported persistent Video VAE manifest: {path}")
    return raw


def persistent_video_vae_schema(directory: Path) -> list[VideoVAEWeightInput]:
    raw = load_persistent_video_vae_manifest(directory)
    return [VideoVAEWeightInput.from_dict(item) for item in raw["weight_inputs"]]


def persistent_video_vae_ready(
    directory: Path,
    required_blocks: Iterable[int] = (0,),
    dynamic_batch: bool | None = None,
) -> bool:
    directory = directory.resolve()
    topology_path = directory / PERSISTENT_VIDEO_VAE_TOPOLOGY
    manifest_path = directory / PERSISTENT_VIDEO_VAE_MANIFEST
    if not topology_path.is_file() or not manifest_path.is_file():
        return False
    try:
        raw = load_persistent_video_vae_manifest(directory)
        if dynamic_batch is not None and bool(raw.get("dynamic_batch")) != dynamic_batch:
            return False
        validation = raw["validation"]
        if not validation.get("compatible"):
            return False
        if raw.get("fingerprint") != validation.get("fingerprint"):
            return False
        validated = {int(value) for value in validation["validated_blocks"]}
        required = {0, *(int(value) for value in required_blocks)}
        if not required.issubset(validated):
            return False
        for block in required:
            path = _block_path(directory, block)
            expected = validation["blocks"][str(block)]["source_stat"]
            if _file_stat(path) != {"size": int(expected["size"]), "mtime_ns": int(expected["mtime_ns"])}:
                return False
        topology = raw["topology"]
        if topology.get("file") != PERSISTENT_VIDEO_VAE_TOPOLOGY:
            return False
        if topology_path.stat().st_size != int(topology["bytes"]):
            return False
        if _sha256(topology_path) != topology["sha256"]:
            return False
        weights = [VideoVAEWeightInput.from_dict(item) for item in raw["weight_inputs"]]
        if sum(weight.nbytes for weight in weights) != int(raw["weight_bytes"]):
            return False
        model = onnx.load(topology_path, load_external_data=False)
        try:
            if model.graph.initializer or model.graph.sparse_initializer:
                return False
            inputs = {value.name: value for value in model.graph.input}
            expected_inputs = {*_ACTIVATION_INPUTS, *(weight.name for weight in weights)}
            if set(inputs) != expected_inputs:
                return False
            for weight in weights:
                value = inputs.get(weight.name)
                if value is None:
                    return False
                if value.type.tensor_type.elem_type != weight.dtype or tuple(_value_shape(value)) != weight.shape:
                    return False
            dynamic = bool(raw.get("dynamic_batch"))
            for name in _ACTIVATION_INPUTS:
                shape = _value_shape(inputs[name])
                if not shape or (shape[0] == "batch") != dynamic:
                    return False
            outputs = {value.name for value in model.graph.output}
            return outputs == {_ACTIVATION_OUTPUT}
        finally:
            del model
    except Exception:  # noqa: BLE001 - ready checks treat every malformed artifact as not ready
        return False


def build_persistent_video_vae_topology(
    directory: Path,
    validate_blocks: Iterable[int] | None = None,
    dynamic_batch: bool = False,
) -> Path:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    requested = _requested_blocks(directory, validate_blocks)
    manifest_path = directory / PERSISTENT_VIDEO_VAE_MANIFEST
    with _BUILD_LOCK:
        if persistent_video_vae_ready(directory, requested, dynamic_batch):
            return manifest_path

        validation = validate_video_vae_block_schemas(directory, requested)
        if not validation["compatible"]:
            raise ValueError(f"Video VAE block schemas are incompatible in {directory}")
        source_path = _block_path(directory, 0)
        model = onnx.load(source_path, load_external_data=False)
        try:
            fingerprint = video_vae_block_fingerprint(model)
            if fingerprint != validation["fingerprint"]:
                raise RuntimeError("Video VAE block0 changed during persistent topology construction")
            weights = _promote_initializers(model, dynamic_batch)
            topology_path = directory / PERSISTENT_VIDEO_VAE_TOPOLOGY
            temporary_topology = _temporary_path(topology_path)
            try:
                onnx.save_model(model, temporary_topology)
                _flush_file(temporary_topology)
                topology_digest = _sha256(temporary_topology)
                topology_bytes = temporary_topology.stat().st_size
                os.replace(temporary_topology, topology_path)
            finally:
                temporary_topology.unlink(missing_ok=True)
        finally:
            del model
            gc.collect()

        manifest = {
            "format": PERSISTENT_VIDEO_VAE_FORMAT,
            "topology": {
                "file": PERSISTENT_VIDEO_VAE_TOPOLOGY,
                "bytes": topology_bytes,
                "sha256": topology_digest,
                "source": source_path.name,
            },
            "dynamic_batch": dynamic_batch,
            "fingerprint": fingerprint,
            "weight_inputs": [weight.to_dict() for weight in weights],
            "weight_bytes": sum(weight.nbytes for weight in weights),
            "validation": validation,
        }
        _atomic_write_json(manifest_path, manifest)
        if not persistent_video_vae_ready(directory, requested, dynamic_batch):
            raise RuntimeError("Persistent Video VAE topology failed its ready check")
        return manifest_path


def load_video_vae_block_weights(directory: Path, block: int) -> LoadedVideoVAEBlockWeights:
    directory = directory.resolve()
    manifest = load_persistent_video_vae_manifest(directory)
    if not persistent_video_vae_ready(directory, (block,), bool(manifest.get("dynamic_batch"))):
        raise RuntimeError(f"Persistent Video VAE topology is not ready in {directory}")
    expected = [VideoVAEWeightInput.from_dict(item) for item in manifest["weight_inputs"]]
    expected_signature = tuple(item.signature for item in expected)
    path = _block_path(directory, block)
    started = time.perf_counter()
    model = onnx.load(path, load_external_data=True)
    try:
        actual = _weight_inputs(model)
        if tuple(item.signature for item in actual) != expected_signature:
            raise ValueError(f"Persistent Video VAE weight schema mismatch in {path}")
        fingerprint = video_vae_block_fingerprint(model)
        if fingerprint != manifest["fingerprint"]:
            raise ValueError(f"Persistent Video VAE topology mismatch in {path}")
        arrays: dict[str, np.ndarray] = {}
        for initializer in model.graph.initializer:
            value = np.asarray(numpy_helper.to_array(initializer, base_dir=str(path.parent)))
            arrays[initializer.name] = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    finally:
        del model
        gc.collect()
    total_bytes = sum(array.nbytes for array in arrays.values())
    return LoadedVideoVAEBlockWeights(
        block=block,
        source=path,
        arrays=arrays,
        total_bytes=total_bytes,
        load_seconds=time.perf_counter() - started,
        fingerprint=fingerprint,
    )
