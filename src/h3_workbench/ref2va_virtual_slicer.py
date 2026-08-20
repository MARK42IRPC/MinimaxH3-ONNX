from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper

from h3_workbench.main_transformer import SAFETENSORS_HEADER_LIMIT, _StreamingSafeTensorFile


RUNTIME_MANIFEST = "runtime_ref2va_manifest.json"
VIRTUAL_FORMAT = "h3-ref2va-bf16-virtual-v1"
VIRTUAL_KINDS = ("attention_qkv", "attention_output", "mlp")
LAYERS = 50

_KIND_SUFFIXES = {
    "attention_qkv": "attention_qkv",
    "attention_output": "attention_output",
    "mlp": "mlp",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_file_identity(path: Path) -> dict[str, int | str]:
    path = path.resolve()
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def source_identity(path: Path) -> dict[str, int | str]:
    """Fingerprint a large SafeTensors source without reading its payload."""
    path = path.resolve()
    stat = path.stat()
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Ref2VA source is not a SafeTensors checkpoint: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > SAFETENSORS_HEADER_LIMIT:
            raise ValueError(f"Ref2VA SafeTensors header is unexpectedly large: {header_length} bytes")
        header = handle.read(header_length)
    if len(header) != header_length:
        raise ValueError(f"Ref2VA SafeTensors header is truncated: {path}")
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def virtual_product_fingerprint(directory: Path) -> str:
    raw = json.loads((directory.resolve() / RUNTIME_MANIFEST).read_text(encoding="utf-8"))
    return _canonical_digest(raw)


def _close_memmaps(values: object) -> None:
    pending = list(values) if isinstance(values, (list, tuple)) else [values]
    mappings: dict[int, np.memmap] = {}
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


def _external_fields(initializer: onnx.TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


def _short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _source_key(kind: str, name: str, layer: int) -> str:
    short = _short_name(name)
    root = f"blocks.{layer}"
    mapping = {
        "attention.qkv.weight": f"{root}.attn.qkv_proj.weight",
        "attention.q_weight": f"{root}.attn.q_norm.weight",
        "attention.k_weight": f"{root}.attn.k_norm.weight",
        "out.weight": f"{root}.attn.out_proj.weight",
        "mlp.fc1.weight": f"{root}.mlp.fc1.weight",
        "mlp.fc2.weight": f"{root}.mlp.fc2.weight",
        "modulation.linear.weight": f"{root}.adaln_proj.linear.weight",
        "modulation.linear.bias": f"{root}.adaln_proj.linear.bias",
    }
    if kind == "attention_qkv":
        generated_names = {
            "_to_copy": f"{root}.norm1.weight",
            "_to_copy_3": f"{root}.attn.q_norm.weight",
            "_to_copy_4": f"{root}.attn.k_norm.weight",
        }
        if short in generated_names:
            return generated_names[short]
    elif kind == "mlp" and short == "_to_copy":
        return f"{root}.norm2.weight"
    if short == "norm_weight":
        suffix = "norm1" if kind in {"attention_qkv", "attention_output"} else "norm2"
        return f"{root}.{suffix}.weight"
    try:
        return mapping[short]
    except KeyError as exc:
        raise ValueError(f"Unsupported {kind} topology weight input: {name}") from exc


def _shape_from_value(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"Virtual weight input {value.name} must have a static shape")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def _promote_topology(
    source: Path,
    destination: Path,
    kind: str,
    checkpoint: _StreamingSafeTensorFile,
) -> list[dict[str, Any]]:
    model = onnx.load(str(source), load_external_data=False)
    retained: list[onnx.TensorProto] = []
    specs: list[dict[str, Any]] = []
    existing_inputs = {value.name for value in model.graph.input}
    for initializer in model.graph.initializer:
        try:
            source_key = _source_key(kind, initializer.name, 0)
        except ValueError:
            source_key = None
        if source_key is None:
            retained.append(initializer)
            continue
        if initializer.name in existing_inputs:
            raise ValueError(f"Topology weight is already a graph input: {initializer.name}")
        target_shape = tuple(int(value) for value in initializer.dims)
        source_shape = checkpoint.tensor_shape(source_key)
        if source_shape != target_shape:
            raise ValueError(
                f"Ref2VA topology shape mismatch for {initializer.name}: "
                f"source={source_shape}, target={target_shape}"
            )
        target_dtype = np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type))
        model.graph.input.append(
            helper.make_tensor_value_info(initializer.name, initializer.data_type, list(target_shape))
        )
        specs.append(
            {
                "name": initializer.name,
                "source_key": _source_key(kind, initializer.name, 0).replace("blocks.0", "blocks.{layer}"),
                "target_dtype": target_dtype.str,
                "target_shape": list(target_shape),
            }
        )
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    model.graph.name = f"h3_ref2va_virtual_{kind}"
    onnx.checker.check_model(model, full_check=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    onnx.save_model(model, str(temporary), save_as_external_data=False)
    os.replace(temporary, destination)
    return specs


def _source_dtype(checkpoint: _StreamingSafeTensorFile) -> str:
    dtypes = sorted({str(value["dtype"]) for value in checkpoint._entries.values()})
    return ", ".join(dtypes)


def build_virtual_ref2va_product(
    source: Path,
    output: Path,
    topologies: dict[str, Path],
) -> Path:
    """Publish reusable block topologies backed by the original BF16 source."""
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if set(topologies) != set(VIRTUAL_KINDS):
        raise ValueError(f"Ref2VA virtual topologies must contain {VIRTUAL_KINDS}")
    checkpoint = _StreamingSafeTensorFile(source)
    identity = source_identity(source)
    for layer in range(LAYERS):
        for kind in VIRTUAL_KINDS:
            checkpoint.tensor_shape(f"blocks.{layer}.adaln_proj.linear.weight")

    manifest: dict[str, Any] = {
        "format": "h3-workbench-onnx-v2",
        "source": str(source),
        "component": "ref2va_transformer",
        "source_quantization": "bfloat16",
        "conversion": "bf16_source_virtual_weight_slices_mixed_precision_runtime_topologies",
        "activation_dtype": "fp16_attention_qkv_fp32_attention_output_mlp",
        "validation_passed": False,
        "build_complete": False,
    }
    _atomic_json(output / "manifest.json", manifest)

    runtime_kinds: dict[str, Any] = {}
    graph_names: list[str] = []
    for kind in VIRTUAL_KINDS:
        graph_name = f"main_block_00_{_KIND_SUFFIXES[kind]}.onnx"
        destination = output / graph_name
        specs = _promote_topology(topologies[kind].resolve(), destination, kind, checkpoint)
        for layer in range(LAYERS):
            name = f"main_block_{layer:02d}_{_KIND_SUFFIXES[kind]}.onnx"
            if layer:
                _atomic_copy(destination, output / name)
            graph_names.append(name)
        runtime_kinds[kind] = {
            "graph": graph_name,
            "graph_identity": runtime_file_identity(destination),
            "inputs": [
                {
                    **spec,
                    "source_key": str(spec["source_key"]),
                }
                for spec in specs
            ],
        }

    runtime = {
        "format": VIRTUAL_FORMAT,
        "source_checkpoint": str(source),
        "source_size": source.stat().st_size,
        "source_identity": identity,
        "source_dtype": _source_dtype(checkpoint),
        "layers": LAYERS,
        "kinds": runtime_kinds,
        "weight_storage": "source_safetensors_memmap_per_block",
        "target_storage": "onnx_topology_inputs",
    }
    _atomic_json(output / RUNTIME_MANIFEST, runtime)
    manifest.update(
        {
            "build_complete": True,
            "architecture": {
                "hidden_size": 5376,
                "layers": LAYERS,
                "heads": 56,
                "head_dim": 128,
                "ffn_size": 14336,
                "curve_dim": 8,
            },
            "blocks": list(range(LAYERS)),
            "graphs": graph_names,
            "weight_storage": "source_safetensors_zero_copy_with_runtime_dtype_conversion",
            "runtime_manifest": RUNTIME_MANIFEST,
            "virtual_slices": {
                "kinds": list(VIRTUAL_KINDS),
                "representative_layers": [0, 24, 49],
                "source_dtype": _source_dtype(checkpoint),
            },
            "capabilities": ["t2va", "fl2va", "ref2va"],
        }
    )
    _atomic_json(output / "manifest.json", manifest)
    return output / RUNTIME_MANIFEST


def append_graphs(directory: Path, graphs: list[str]) -> None:
    path = directory.resolve() / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current = list(manifest.get("graphs", []))
    manifest["graphs"] = [*graphs, *(item for item in current if item not in graphs)]
    _atomic_json(path, manifest)


def ref2va_virtual_ready(directory: Path) -> bool:
    directory = directory.resolve()
    path = directory / RUNTIME_MANIFEST
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != VIRTUAL_FORMAT or int(raw.get("layers", 0)) != LAYERS:
            return False
        source = Path(str(raw["source_checkpoint"]))
        if not source.is_absolute():
            source = directory / source
        if not source.is_file() or source_identity(source) != raw["source_identity"]:
            return False
        for kind in VIRTUAL_KINDS:
            item = raw["kinds"][kind]
            graph = directory / str(item["graph"])
            if not graph.is_file() or runtime_file_identity(graph) != item["graph_identity"]:
                return False
            if not item.get("inputs"):
                return False
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return manifest.get("runtime_manifest") == RUNTIME_MANIFEST
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validated_ref2va_virtual_ready(directory: Path) -> bool:
    if not ref2va_virtual_ready(directory):
        return False
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        fingerprint = virtual_product_fingerprint(directory)
        return (
            manifest.get("validation_passed") is True
            and manifest.get("validated_product_fingerprint") == fingerprint
            and manifest.get("validation", {}).get("product_fingerprint") == fingerprint
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _logical_graph_name(graph: str) -> str:
    return Path(str(graph)).name.removesuffix(".onnx")


def _graph_kind(graph: str) -> str | None:
    graph = _logical_graph_name(graph)
    if graph.startswith("main_block_") and graph.endswith("_attention_qkv"):
        return "attention_qkv"
    if graph.startswith("main_block_") and graph.endswith("_attention_output"):
        return "attention_output"
    if graph.startswith("main_block_") and graph.endswith("_mlp"):
        return "mlp"
    return None


class Ref2VASourceWeights:
    """Runtime weight provider for BF16 Ref2VA virtual block slices."""

    def __init__(
        self,
        directory: Path,
        runner: Any,
        graph_paths: dict[str, Path],
        *,
        topology_paths: dict[str, Path] | None = None,
    ) -> None:
        self.directory = directory.resolve()
        if not ref2va_virtual_ready(self.directory):
            raise ValueError(f"Ref2VA virtual product is stale or incomplete: {self.directory}")
        self.runner = runner
        raw = json.loads((self.directory / RUNTIME_MANIFEST).read_text(encoding="utf-8"))
        source = Path(str(raw["source_checkpoint"]))
        if not source.is_absolute():
            source = self.directory / source
        self.source = source.resolve()
        self.reader = _StreamingSafeTensorFile(self.source)
        self.kinds = raw["kinds"]
        self.graph_paths = {
            _logical_graph_name(graph): path.resolve()
            for graph, path in graph_paths.items()
            if _graph_kind(graph) is not None
        }
        self._topology_paths = {
            kind: self.directory / str(self.kinds[kind]["graph"])
            for kind in VIRTUAL_KINDS
        }
        if topology_paths is not None:
            unexpected = set(topology_paths) - set(VIRTUAL_KINDS)
            if unexpected:
                raise ValueError(f"Unsupported Ref2VA topology overrides: {sorted(unexpected)}")
            for kind, path in topology_paths.items():
                resolved = Path(path).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                self._topology_paths[kind] = resolved
        self._sessions: dict[str, Any] = {}
        self._active: dict[str, dict[str, np.ndarray]] = {}
        self._load_seconds: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return all(path.is_file() for path in self._topology_paths.values())

    @property
    def prefetch_depth(self) -> int:
        return 0

    @property
    def prefetch_workers(self) -> int:
        return 0

    @property
    def pinned_enabled(self) -> bool:
        return False

    @property
    def pinned_fallback_reason(self) -> str | None:
        return "virtual_safetensors_source"

    @property
    def host_pool_allocations(self) -> int:
        return 0

    @property
    def device_metrics(self) -> dict[str, int | float | bool]:
        return {"persistent_virtual_source": True}

    def prime_ram_cache(self) -> dict[str, int | float | bool]:
        return {
            "ram_cache_enabled": False,
            "ram_cache_budget_bytes": 0,
            "ram_cache_scheduled": 0,
            "virtual_source": True,
        }

    def prefetch_metrics(self) -> dict[str, int | float]:
        return {"prefetch_queue_depth": 0, "prefetch_reserved_bytes": 0, "prefetch_budget_bytes": 0}

    def supports(self, graph: str) -> bool:
        logical = _logical_graph_name(graph)
        return _graph_kind(logical) is not None and logical in self.graph_paths

    def graph(self, kind: str) -> Path:
        if kind not in VIRTUAL_KINDS:
            raise KeyError(kind)
        return self._topology_paths[kind]

    def _specs(self, graph: str) -> tuple[str, list[dict[str, Any]]]:
        kind = _graph_kind(graph)
        logical = _logical_graph_name(graph)
        if kind is None or logical not in self.graph_paths:
            raise KeyError(graph)
        block = int(logical.split("_")[2])
        if not 0 <= block < LAYERS:
            raise ValueError(f"Invalid Ref2VA block graph: {graph}")
        return kind, self.kinds[kind]["inputs"]

    def _load(self, graph: str) -> dict[str, np.ndarray]:
        kind, specs = self._specs(graph)
        block = int(_logical_graph_name(graph).split("_")[2])
        started = time.perf_counter()
        result: dict[str, np.ndarray] = {}
        try:
            for spec in specs:
                source_key = str(spec["source_key"]).format(layer=block)
                source = self.reader.memmap_tensor(source_key)
                target_dtype = np.dtype(str(spec["target_dtype"]))
                target_shape = tuple(int(value) for value in spec["target_shape"])
                if source.shape != target_shape:
                    raise ValueError(
                        f"Ref2VA source shape changed for {source_key}: "
                        f"{source.shape} != {target_shape}"
                    )
                if source.dtype == target_dtype:
                    value = source
                else:
                    value = np.asarray(source, dtype=target_dtype).reshape(target_shape)
                result[str(spec["name"])] = value if isinstance(value, np.memmap) else np.ascontiguousarray(value)
            self._load_seconds[graph] = time.perf_counter() - started
            return result
        except Exception:
            _close_memmaps(result.values())
            raise

    def inputs(self, graph: str) -> dict[str, np.ndarray]:
        cached = self._active.get(graph)
        if cached is None:
            cached = self._load(graph)
            self._active[graph] = cached
        return cached

    def session(self, graph: str) -> tuple[Any, bool, float]:
        kind, _ = self._specs(graph)
        cached = self._sessions.get(kind)
        if cached is not None:
            return cached, False, 0.0
        started = time.perf_counter()
        session = self.runner.session(self._topology_paths[kind])
        elapsed = time.perf_counter() - started
        self._sessions[kind] = session
        return session, True, elapsed

    def release_session(self, graph: str) -> None:
        kind = _graph_kind(graph)
        if kind is not None:
            self._sessions.pop(kind, None)

    def weights(
        self, graph: str
    ) -> tuple[dict[str, Any], int, int, int, int, bool, float, float, float, bool, bool]:
        started = time.perf_counter()
        feeds = self.inputs(graph)
        elapsed = self._load_seconds.get(graph, time.perf_counter() - started)
        total = sum(int(value.nbytes) for value in feeds.values())
        return feeds, total, total, 0, 0, True, elapsed, 0.0, 0.0, False, False

    def release(self, graph: str) -> None:
        feeds = self._active.pop(graph, None)
        if feeds is not None:
            _close_memmaps(feeds.values())

    def prefetch(self, graph: str, _stage: bool = True) -> bool:
        return False

    def prefetch_many(self, graphs: list[str]) -> int:
        return 0

    def close(self) -> None:
        for feeds in self._active.values():
            _close_memmaps(feeds.values())
        self._active.clear()
        self._sessions.clear()
