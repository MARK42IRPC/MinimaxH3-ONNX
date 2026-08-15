from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper

from h3_workbench.persistent_weights import build_persistent_topology


TURBO_ADAPTER_FORMAT = "h3-turbo-lora-adapter-v1"
TURBO_ADAPTER_VARIANT = "turbo_v4_dynamic"
TURBO_LORA_PAIR_COUNT = 259
SAFETENSORS_HEADER_LIMIT = 32 * 1024 * 1024


@dataclass(frozen=True)
class AssetIdentity:
    filename: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
        }


TURBO_LORA_IDENTITY = AssetIdentity(
    filename="minimax_h3_turbo_v4_step600_ema.safetensors",
    size=779_849_816,
    sha256="5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
)
TURBO_SILU_GRID_IDENTITY = AssetIdentity(
    filename="h3_silu_temb_grid.safetensors",
    size=5_510_600,
    sha256="30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8",
)

TURBO_TOPOLOGIES = {
    "attention_qkv": "runtime_turbo_attention_qkv.onnx",
    "attention_output": "runtime_turbo_scaled_attention_output.onnx",
    "mlp": "runtime_turbo_scaled_mlp.onnx",
    "refiner_attention": "runtime_turbo_refiner_attention.onnx",
    "refiner_mlp": "runtime_turbo_refiner_mlp.onnx",
    "head": "runtime_turbo_head.onnx",
}


@dataclass(frozen=True)
class LoraFactorSpec:
    prefix: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    cache_dtype: np.dtype[Any]

    @property
    def a_key(self) -> str:
        return f"{self.prefix}.lora_A.weight"

    @property
    def b_key(self) -> str:
        return f"{self.prefix}.lora_B.weight"


@dataclass(frozen=True)
class LoraFactorPair:
    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class ValidatedTurboAssets:
    lora_path: Path
    grid_path: Path
    lora_identity: AssetIdentity
    grid_identity: AssetIdentity
    factor_bytes: int


def expected_factor_specs() -> dict[str, LoraFactorSpec]:
    fp16 = np.dtype(np.float16)
    fp32 = np.dtype(np.float32)
    specs: dict[str, LoraFactorSpec] = {}

    def add(
        prefix: str,
        a_shape: tuple[int, int],
        b_shape: tuple[int, int],
        dtype: np.dtype[Any],
    ) -> None:
        specs[prefix] = LoraFactorSpec(prefix, a_shape, b_shape, dtype)

    for block in range(50):
        root = f"blocks.{block}"
        add(f"{root}.adaln_proj.linear", (16, 2688), (96768, 16), fp32)
        add(f"{root}.attn.out_proj", (64, 7168), (5376, 64), fp16)
        add(f"{root}.attn.qkv_proj", (64, 5376), (21504, 64), fp16)
        add(f"{root}.mlp.fc1", (64, 5376), (28672, 64), fp16)
        add(f"{root}.mlp.fc2", (64, 14336), (5376, 64), fp16)
    for block in range(2):
        root = f"token_refiner.blocks.{block}"
        add(f"{root}.attn.out_proj", (64, 7168), (5376, 64), fp16)
        add(f"{root}.attn.qkv_proj", (64, 5376), (21504, 64), fp16)
        add(f"{root}.mlp.fc1", (64, 5376), (28672, 64), fp16)
        add(f"{root}.mlp.fc2", (64, 14336), (5376, 64), fp16)
    add("final_layer.adaln_proj.linear", (16, 2688), (10752, 16), fp32)
    if len(specs) != TURBO_LORA_PAIR_COUNT:
        raise AssertionError(f"Internal Turbo LoRA schema has {len(specs)} pairs")
    return specs


EXPECTED_FACTOR_SPECS = expected_factor_specs()


class SafeTensorFile:
    """Small offset reader that preserves BF16 without mapping the whole file."""

    _DTYPES: dict[str, Any] = {
        "BOOL": np.bool_,
        "U8": np.uint8,
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "F16": np.float16,
        "BF16": ml_dtypes.bfloat16,
        "F32": np.float32,
        "F64": np.float64,
    }

    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.file_size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Safetensors file is too short: {self.path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 1 or header_length > SAFETENSORS_HEADER_LIMIT:
                raise ValueError(f"Invalid Safetensors header length: {header_length}")
            payload = handle.read(header_length)
        if len(payload) != header_length:
            raise ValueError(f"Safetensors header is truncated: {self.path}")
        try:
            header = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Safetensors header: {self.path}") from exc
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("Safetensors metadata must contain string values")
        self.metadata: dict[str, str] = metadata
        self.entries: dict[str, dict[str, Any]] = {}
        self.data_start = 8 + header_length
        ranges: list[tuple[int, int, str]] = []
        for key, value in header.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("Invalid Safetensors tensor entry")
            dtype_name = value.get("dtype")
            shape = value.get("shape")
            offsets = value.get("data_offsets")
            if dtype_name not in self._DTYPES:
                raise ValueError(f"Unsupported Safetensors dtype for {key}: {dtype_name}")
            if not isinstance(shape, list) or not all(
                isinstance(dimension, int) and dimension >= 0 for dimension in shape
            ):
                raise ValueError(f"Invalid Safetensors shape for {key}")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(offset, int) for offset in offsets)
            ):
                raise ValueError(f"Invalid Safetensors offsets for {key}")
            start, stop = offsets
            dtype = np.dtype(self._DTYPES[dtype_name])
            expected = math.prod(shape) * dtype.itemsize
            if start < 0 or stop < start or stop - start != expected:
                raise ValueError(f"Invalid Safetensors bounds for {key}: {start}:{stop}")
            if self.data_start + stop > self.file_size:
                raise ValueError(f"Safetensors tensor is truncated: {key}")
            entry = {
                "dtype": dtype_name,
                "shape": tuple(shape),
                "data_offsets": (start, stop),
            }
            self.entries[key] = entry
            ranges.append((start, stop, key))
        expected_start = 0
        for start, stop, key in sorted(ranges):
            if start != expected_start:
                raise ValueError(f"Non-contiguous Safetensors payload before {key}")
            expected_start = stop
        if self.data_start + expected_start != self.file_size:
            raise ValueError("Safetensors payload has trailing or missing bytes")

    def keys(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def dtype_name(self, key: str) -> str:
        return str(self.entries[key]["dtype"])

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.entries[key]["shape"])

    def tensor(self, key: str, dtype: np.dtype[Any]) -> np.ndarray:
        try:
            entry = self.entries[key]
        except KeyError as exc:
            raise KeyError(f"Tensor not found in {self.path.name}: {key}") from exc
        start, _ = entry["data_offsets"]
        source = np.memmap(
            self.path,
            dtype=np.dtype(self._DTYPES[entry["dtype"]]),
            mode="r",
            offset=self.data_start + start,
            shape=entry["shape"],
            order="C",
        )
        try:
            return np.asarray(source, dtype=dtype).copy(order="C")
        finally:
            del source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_asset_identity(path: Path, expected: AssetIdentity) -> AssetIdentity:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    size = resolved.stat().st_size
    if size != expected.size:
        raise ValueError(
            f"Unexpected size for {resolved.name}: {size} bytes, expected {expected.size}"
        )
    digest = _sha256(resolved)
    if digest != expected.sha256:
        raise ValueError(
            f"Unexpected SHA-256 for {resolved.name}: {digest}, expected {expected.sha256}"
        )
    return AssetIdentity(resolved.name, size, digest)


def _validate_lora_schema(reader: SafeTensorFile) -> int:
    expected_keys = {
        key
        for spec in EXPECTED_FACTOR_SPECS.values()
        for key in (spec.a_key, spec.b_key)
    }
    actual_keys = set(reader.keys())
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} (first: {missing[0]})")
        if extra:
            details.append(f"unexpected {len(extra)} (first: {extra[0]})")
        raise ValueError("Turbo LoRA tensor set mismatch: " + "; ".join(details))
    if reader.metadata.get("application") != "W_eff = W + lora_B @ lora_A":
        raise ValueError("Turbo LoRA application metadata is missing or incompatible")
    if reader.metadata.get("base_model") != "MiniMax-H3":
        raise ValueError("Turbo LoRA base_model metadata is missing or incompatible")
    factor_bytes = 0
    for spec in EXPECTED_FACTOR_SPECS.values():
        for key, shape in ((spec.a_key, spec.a_shape), (spec.b_key, spec.b_shape)):
            if reader.dtype_name(key) != "BF16":
                raise ValueError(f"Turbo LoRA tensor must be BF16: {key}")
            actual_shape = reader.shape(key)
            if actual_shape != shape:
                raise ValueError(
                    f"Turbo LoRA shape mismatch for {key}: {actual_shape}, expected {shape}"
                )
            factor_bytes += math.prod(shape) * 2
    return factor_bytes


def _validate_grid_schema(reader: SafeTensorFile) -> None:
    if set(reader.keys()) != {"silu_t_emb_grid"}:
        raise ValueError("SiLU grid must contain only silu_t_emb_grid")
    if reader.dtype_name("silu_t_emb_grid") != "BF16":
        raise ValueError("SiLU grid must be BF16")
    if reader.shape("silu_t_emb_grid") != (1025, 2688):
        raise ValueError(
            "SiLU grid shape must be (1025, 2688), got "
            f"{reader.shape('silu_t_emb_grid')}"
        )
    if reader.metadata.get("grid") != "linspace(0,1,1025)":
        raise ValueError("SiLU grid metadata is missing or incompatible")


def validate_turbo_assets(
    lora_path: Path,
    grid_path: Path,
    *,
    verify_identity: bool = True,
) -> ValidatedTurboAssets:
    lora_path = lora_path.resolve()
    grid_path = grid_path.resolve()
    if verify_identity:
        lora_identity = validate_asset_identity(lora_path, TURBO_LORA_IDENTITY)
        grid_identity = validate_asset_identity(grid_path, TURBO_SILU_GRID_IDENTITY)
    else:
        if not lora_path.is_file() or not grid_path.is_file():
            raise FileNotFoundError(lora_path if not lora_path.is_file() else grid_path)
        lora_identity = AssetIdentity(lora_path.name, lora_path.stat().st_size, _sha256(lora_path))
        grid_identity = AssetIdentity(grid_path.name, grid_path.stat().st_size, _sha256(grid_path))
    lora_reader = SafeTensorFile(lora_path)
    factor_bytes = _validate_lora_schema(lora_reader)
    _validate_grid_schema(SafeTensorFile(grid_path))
    return ValidatedTurboAssets(
        lora_path=lora_path,
        grid_path=grid_path,
        lora_identity=lora_identity,
        grid_identity=grid_identity,
        factor_bytes=factor_bytes,
    )


def interpolate_silu_grid(grid: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    times = np.asarray(timesteps, dtype=np.float32)
    if grid.ndim != 2 or grid.shape[0] < 2:
        raise ValueError("SiLU grid must be a two-dimensional array with at least two rows")
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim != 1:
        raise ValueError("timesteps must be a scalar or one-dimensional array")
    position = np.clip(times, 0.0, 1.0) * np.float32(grid.shape[0] - 1)
    lower = np.floor(position).astype(np.int64)
    lower = np.minimum(lower, grid.shape[0] - 2)
    fraction = (position - lower.astype(np.float32))[:, None]
    result = grid[lower] + (grid[lower + 1] - grid[lower]) * fraction
    return np.ascontiguousarray(result, dtype=np.float32)


_BLOCK_GRAPH = re.compile(r"^main_block_(\d{2})_(attention_qkv|attention_output|mlp)$")
_REFINER_GRAPH = re.compile(
    r"^main_token_refiner_block_(\d{2})_(attention|mlp)$"
)


def turbo_graph_kind(graph: str) -> str | None:
    block = _BLOCK_GRAPH.fullmatch(graph)
    if block is not None:
        return block.group(2)
    refiner = _REFINER_GRAPH.fullmatch(graph)
    if refiner is not None:
        return f"refiner_{refiner.group(2)}"
    return "head" if graph == "main_head" else None


def _factor_feed_schema(graph: str) -> tuple[tuple[str, str, str], ...]:
    block = _BLOCK_GRAPH.fullmatch(graph)
    if block is not None:
        index = int(block.group(1))
        if not 0 <= index < 50:
            raise ValueError(f"Invalid DiT block in graph name: {graph}")
        root = f"blocks.{index}"
        kind = block.group(2)
        adaln = (
            ("turbo_lora_adaln_A", f"{root}.adaln_proj.linear", "a"),
            ("turbo_lora_adaln_B", f"{root}.adaln_proj.linear", "b"),
        )
        if kind == "attention_qkv":
            return (
                ("turbo_lora_qkv_A", f"{root}.attn.qkv_proj", "a"),
                ("turbo_lora_qkv_B", f"{root}.attn.qkv_proj", "b"),
                *adaln,
            )
        if kind == "attention_output":
            return (
                ("turbo_lora_out_A", f"{root}.attn.out_proj", "a"),
                ("turbo_lora_out_B", f"{root}.attn.out_proj", "b"),
                *adaln,
            )
        return (
            ("turbo_lora_fc1_A", f"{root}.mlp.fc1", "a"),
            ("turbo_lora_fc1_B", f"{root}.mlp.fc1", "b"),
            ("turbo_lora_fc2_A", f"{root}.mlp.fc2", "a"),
            ("turbo_lora_fc2_B", f"{root}.mlp.fc2", "b"),
            *adaln,
        )
    refiner = _REFINER_GRAPH.fullmatch(graph)
    if refiner is not None:
        index = int(refiner.group(1))
        if not 0 <= index < 2:
            raise ValueError(f"Invalid token refiner block in graph name: {graph}")
        root = f"token_refiner.blocks.{index}"
        if refiner.group(2) == "attention":
            return (
                ("turbo_lora_qkv_A", f"{root}.attn.qkv_proj", "a"),
                ("turbo_lora_qkv_B", f"{root}.attn.qkv_proj", "b"),
                ("turbo_lora_out_A", f"{root}.attn.out_proj", "a"),
                ("turbo_lora_out_B", f"{root}.attn.out_proj", "b"),
            )
        return (
            ("turbo_lora_fc1_A", f"{root}.mlp.fc1", "a"),
            ("turbo_lora_fc1_B", f"{root}.mlp.fc1", "b"),
            ("turbo_lora_fc2_A", f"{root}.mlp.fc2", "a"),
            ("turbo_lora_fc2_B", f"{root}.mlp.fc2", "b"),
        )
    if graph == "main_head":
        return (
            ("turbo_lora_adaln_A", "final_layer.adaln_proj.linear", "a"),
            ("turbo_lora_adaln_B", "final_layer.adaln_proj.linear", "b"),
        )
    return ()


def factor_feeds_for_graph(
    graph: str,
    factors: Mapping[str, LoraFactorPair],
) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for feed_name, prefix, part in _factor_feed_schema(graph):
        try:
            pair = factors[prefix]
        except KeyError as exc:
            raise KeyError(f"Turbo LoRA cache has no factors for {prefix}") from exc
        feeds[feed_name] = pair.a if part == "a" else pair.b
    return feeds


class TurboLoraFactorCache:
    def __init__(self, lora_path: Path, *, strength: float = 1.0):
        if not math.isfinite(strength):
            raise ValueError("Turbo LoRA strength must be finite")
        started = time.perf_counter()
        self.path = lora_path.resolve()
        self.strength = float(strength)
        reader = SafeTensorFile(self.path)
        _validate_lora_schema(reader)
        self.factors: dict[str, LoraFactorPair] = {}
        loaded: dict[str, dict[str, np.ndarray]] = {}
        entries: list[tuple[int, LoraFactorSpec, str, str]] = []
        for spec in EXPECTED_FACTOR_SPECS.values():
            for part, key in (("a", spec.a_key), ("b", spec.b_key)):
                start = int(reader.entries[key]["data_offsets"][0])
                entries.append((start, spec, part, key))
        for _, spec, part, key in sorted(entries, key=lambda item: item[0]):
            value = reader.tensor(key, spec.cache_dtype)
            if part == "b" and self.strength != 1.0:
                np.multiply(value, self.strength, out=value, casting="unsafe")
            loaded.setdefault(spec.prefix, {})[part] = value
        for prefix, parts in loaded.items():
            self.factors[prefix] = LoraFactorPair(parts["a"], parts["b"])
        self.total_bytes = sum(
            pair.a.nbytes + pair.b.nbytes for pair in self.factors.values()
        )
        self.load_seconds = time.perf_counter() - started

    def graph_feeds(self, graph: str) -> dict[str, np.ndarray]:
        return factor_feeds_for_graph(graph, self.factors)

    def close(self) -> None:
        self.factors.clear()
        self.total_bytes = 0


def _append_input(
    model: onnx.ModelProto,
    name: str,
    dtype: int,
    shape: tuple[int | str, ...],
) -> None:
    if any(value.name == name for value in model.graph.input):
        raise ValueError(f"Topology input already exists: {name}")
    model.graph.input.append(helper.make_tensor_value_info(name, dtype, list(shape)))


def _node_index(model: onnx.ModelProto, node_name: str) -> int:
    matches = [index for index, node in enumerate(model.graph.node) if node.name == node_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one node named {node_name}, found {len(matches)}")
    return matches[0]


def _gemms_by_weight(model: onnx.ModelProto, suffix: str) -> list[onnx.NodeProto]:
    return [
        node
        for node in model.graph.node
        if node.op_type == "Gemm" and len(node.input) >= 2 and node.input[1].endswith(suffix)
    ]


def _one_gemm_by_weight(model: onnx.ModelProto, suffix: str) -> onnx.NodeProto:
    matches = _gemms_by_weight(model, suffix)
    if len(matches) != 1:
        raise ValueError(f"Expected one Gemm using *{suffix}, found {len(matches)}")
    return matches[0]


def _consumers(model: onnx.ModelProto, value: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if value in node.input]


def _one_consumer(
    model: onnx.ModelProto,
    value: str,
    expected_op: str,
) -> onnx.NodeProto:
    matches = [node for node in _consumers(model, value) if node.op_type == expected_op]
    if len(matches) != 1 or len(_consumers(model, value)) != 1:
        raise ValueError(
            f"Expected {value} to have one {expected_op} consumer, found {len(matches)}"
        )
    return matches[0]


def _insert_nodes(
    model: onnx.ModelProto,
    after_node: onnx.NodeProto,
    nodes: list[onnx.NodeProto],
) -> None:
    index = _node_index(model, after_node.name)
    for offset, node in enumerate(nodes, start=1):
        model.graph.node.insert(index + offset, node)


def _low_rank_nodes(
    source: str,
    a_name: str,
    b_name: str,
    stem: str,
) -> tuple[list[onnx.NodeProto], str]:
    down = f"{stem}.down"
    delta = f"{stem}.delta"
    return (
        [
            helper.make_node(
                "Gemm",
                [source, a_name],
                [down],
                name=f"{stem}/A",
                transB=1,
            ),
            helper.make_node(
                "Gemm",
                [down, b_name],
                [delta],
                name=f"{stem}/B",
                transB=1,
            ),
        ],
        delta,
    )


def _inject_after_gemm(
    model: onnx.ModelProto,
    gemm: onnx.NodeProto,
    source: str,
    a_name: str,
    b_name: str,
    stem: str,
) -> None:
    output = gemm.output[0]
    base_output = f"{output}.base"
    gemm.output[0] = base_output
    nodes, delta = _low_rank_nodes(source, a_name, b_name, stem)
    nodes.append(
        helper.make_node(
            "Add",
            [base_output, delta],
            [output],
            name=f"{stem}/Add",
        )
    )
    _insert_nodes(model, gemm, nodes)


def _inject_after_casted_gemm(
    model: onnx.ModelProto,
    gemm: onnx.NodeProto,
    source: str,
    a_name: str,
    b_name: str,
    stem: str,
) -> None:
    cast = _one_consumer(model, gemm.output[0], "Cast")
    output = cast.output[0]
    base_output = f"{output}.base"
    cast.output[0] = base_output
    nodes, delta = _low_rank_nodes(source, a_name, b_name, stem)
    delta_fp32 = f"{stem}.delta_fp32"
    nodes.extend(
        (
            helper.make_node(
                "Cast",
                [delta],
                [delta_fp32],
                name=f"{stem}/Cast",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Add",
                [base_output, delta_fp32],
                [output],
                name=f"{stem}/Add",
            ),
        )
    )
    _insert_nodes(model, cast, nodes)


def _inject_after_restored_output(
    model: onnx.ModelProto,
    weight_suffix: str,
    a_name: str,
    b_name: str,
    stem: str,
    *,
    delta_scale: str | None = None,
) -> None:
    gemm = _one_gemm_by_weight(model, weight_suffix)
    cast = _one_consumer(model, gemm.output[0], "Cast")
    restore = _one_consumer(model, cast.output[0], "Mul")
    output = restore.output[0]
    base_output = f"{output}.base"
    restore.output[0] = base_output
    nodes, delta = _low_rank_nodes(gemm.input[0], a_name, b_name, stem)
    delta_fp32 = f"{stem}.delta_fp32"
    nodes.append(
        helper.make_node(
            "Cast",
            [delta],
            [delta_fp32],
            name=f"{stem}/Cast",
            to=TensorProto.FLOAT,
        )
    )
    if delta_scale is not None:
        scaled = f"{stem}.delta_scaled"
        nodes.append(
            helper.make_node(
                "Mul",
                [delta_fp32, delta_scale],
                [scaled],
                name=f"{stem}/RestoreScale",
            )
        )
        delta_fp32 = scaled
    nodes.append(
        helper.make_node(
            "Add",
            [base_output, delta_fp32],
            [output],
            name=f"{stem}/Add",
        )
    )
    _insert_nodes(model, restore, nodes)


def _add_factor_inputs(
    model: onnx.ModelProto,
    feeds: tuple[tuple[str, tuple[int, int], int], ...],
) -> None:
    for name, shape, dtype in feeds:
        _append_input(model, name, dtype, shape)


def _inject_adaln(model: onnx.ModelProto) -> None:
    _append_input(
        model,
        "silu_timestep_embedding",
        TensorProto.FLOAT,
        ("timestep_count", 2688),
    )
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_adaln_A", (16, 2688), TensorProto.FLOAT),
            ("turbo_lora_adaln_B", (96768, 16), TensorProto.FLOAT),
        ),
    )
    gemm = _one_gemm_by_weight(model, "modulation.linear.weight")
    _inject_after_gemm(
        model,
        gemm,
        "silu_timestep_embedding",
        "turbo_lora_adaln_A",
        "turbo_lora_adaln_B",
        "turbo_lora/adaln",
    )


def _patch_attention_qkv(model: onnx.ModelProto) -> None:
    _inject_adaln(model)
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_qkv_A", (64, 5376), TensorProto.FLOAT16),
            ("turbo_lora_qkv_B", (21504, 64), TensorProto.FLOAT16),
        ),
    )
    gemm = _one_gemm_by_weight(model, "attention.qkv.weight")
    _inject_after_casted_gemm(
        model,
        gemm,
        gemm.input[0],
        "turbo_lora_qkv_A",
        "turbo_lora_qkv_B",
        "turbo_lora/qkv",
    )


def _patch_attention_output(model: onnx.ModelProto) -> None:
    _inject_adaln(model)
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_out_A", (64, 7168), TensorProto.FLOAT16),
            ("turbo_lora_out_B", (5376, 64), TensorProto.FLOAT16),
        ),
    )
    _inject_after_restored_output(
        model,
        "out.weight",
        "turbo_lora_out_A",
        "turbo_lora_out_B",
        "turbo_lora/out",
    )


def _patch_mlp(model: onnx.ModelProto) -> None:
    _inject_adaln(model)
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_fc1_A", (64, 5376), TensorProto.FLOAT16),
            ("turbo_lora_fc1_B", (28672, 64), TensorProto.FLOAT16),
            ("turbo_lora_fc2_A", (64, 14336), TensorProto.FLOAT16),
            ("turbo_lora_fc2_B", (5376, 64), TensorProto.FLOAT16),
        ),
    )
    _inject_after_restored_output(
        model,
        "mlp.fc1.weight",
        "turbo_lora_fc1_A",
        "turbo_lora_fc1_B",
        "turbo_lora/fc1",
    )
    fc2 = _one_gemm_by_weight(model, "mlp.fc2.weight")
    div_nodes = [node for node in model.graph.node if node.output and node.output[0] == fc2.input[0]]
    if len(div_nodes) != 1 or div_nodes[0].op_type != "Cast":
        raise ValueError("Scaled MLP FC2 input Cast was not found")
    div = [
        node
        for node in model.graph.node
        if node.output and node.output[0] == div_nodes[0].input[0]
    ]
    if len(div) != 1 or div[0].op_type != "Div" or len(div[0].input) != 2:
        raise ValueError("Scaled MLP FC2 input divisor was not found")
    _inject_after_restored_output(
        model,
        "mlp.fc2.weight",
        "turbo_lora_fc2_A",
        "turbo_lora_fc2_B",
        "turbo_lora/fc2",
        delta_scale=div[0].input[1],
    )


def _patch_refiner_attention(model: onnx.ModelProto) -> None:
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_qkv_A", (64, 5376), TensorProto.FLOAT16),
            ("turbo_lora_qkv_B", (21504, 64), TensorProto.FLOAT16),
            ("turbo_lora_out_A", (64, 7168), TensorProto.FLOAT16),
            ("turbo_lora_out_B", (5376, 64), TensorProto.FLOAT16),
        ),
    )
    qkv = _one_gemm_by_weight(model, "attention.qkv.weight")
    _inject_after_casted_gemm(
        model,
        qkv,
        qkv.input[0],
        "turbo_lora_qkv_A",
        "turbo_lora_qkv_B",
        "turbo_lora/qkv",
    )
    out = _one_gemm_by_weight(model, "attention.out.weight")
    _inject_after_casted_gemm(
        model,
        out,
        out.input[0],
        "turbo_lora_out_A",
        "turbo_lora_out_B",
        "turbo_lora/out",
    )


def _patch_refiner_mlp(model: onnx.ModelProto) -> None:
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_fc1_A", (64, 5376), TensorProto.FLOAT16),
            ("turbo_lora_fc1_B", (28672, 64), TensorProto.FLOAT16),
            ("turbo_lora_fc2_A", (64, 14336), TensorProto.FLOAT16),
            ("turbo_lora_fc2_B", (5376, 64), TensorProto.FLOAT16),
        ),
    )
    fc1 = _one_gemm_by_weight(model, "mlp.fc1.weight")
    _inject_after_casted_gemm(
        model,
        fc1,
        fc1.input[0],
        "turbo_lora_fc1_A",
        "turbo_lora_fc1_B",
        "turbo_lora/fc1",
    )
    fc2 = _one_gemm_by_weight(model, "mlp.fc2.weight")
    _inject_after_casted_gemm(
        model,
        fc2,
        fc2.input[0],
        "turbo_lora_fc2_A",
        "turbo_lora_fc2_B",
        "turbo_lora/fc2",
    )


def _patch_head(model: onnx.ModelProto) -> None:
    _append_input(
        model,
        "video_silu_timestep_embedding",
        TensorProto.FLOAT,
        (1, 2688),
    )
    _append_input(
        model,
        "audio_silu_timestep_embedding",
        TensorProto.FLOAT,
        (1, 2688),
    )
    _add_factor_inputs(
        model,
        (
            ("turbo_lora_adaln_A", (16, 2688), TensorProto.FLOAT),
            ("turbo_lora_adaln_B", (10752, 16), TensorProto.FLOAT),
        ),
    )
    gemms = _gemms_by_weight(model, "adaln.weight")
    if len(gemms) != 2:
        raise ValueError(f"Expected two head AdaLN Gemms, found {len(gemms)}")
    branches = (
        (gemms[0].name, "video_silu_timestep_embedding", "turbo_lora/head_video"),
        (gemms[1].name, "audio_silu_timestep_embedding", "turbo_lora/head_audio"),
    )
    for name, source, stem in branches:
        node = model.graph.node[_node_index(model, name)]
        _inject_after_gemm(
            model,
            node,
            source,
            "turbo_lora_adaln_A",
            "turbo_lora_adaln_B",
            stem,
        )


_TOPOLOGY_PATCHERS = {
    "attention_qkv": _patch_attention_qkv,
    "attention_output": _patch_attention_output,
    "mlp": _patch_mlp,
    "refiner_attention": _patch_refiner_attention,
    "refiner_mlp": _patch_refiner_mlp,
    "head": _patch_head,
}


def _representative_sources(base_model_dir: Path) -> dict[str, tuple[Path, bool]]:
    return {
        "attention_qkv": (base_model_dir / "shard_008.onnx", True),
        "attention_output": (
            base_model_dir / "scaled_attention" / "block_00" / "scaled_fp16.onnx",
            False,
        ),
        "mlp": (
            base_model_dir / "scaled_mlp" / "block_00" / "scaled_fp16.onnx",
            False,
        ),
        "refiner_attention": (base_model_dir / "shard_001.onnx", True),
        "refiner_mlp": (base_model_dir / "shard_002.onnx", True),
        "head": (base_model_dir / "shard_007.onnx", True),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def publish_turbo_adapter(
    base_model_dir: Path,
    output_dir: Path,
    lora_path: Path,
    grid_path: Path,
    *,
    verify_identity: bool = True,
) -> dict[str, Any]:
    base_model_dir = base_model_dir.resolve()
    output_dir = output_dir.resolve()
    assets = validate_turbo_assets(
        lora_path,
        grid_path,
        verify_identity=verify_identity,
    )
    sources = _representative_sources(base_model_dir)
    missing = [str(path) for path, _ in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Base topology sources: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    topology_metadata: dict[str, Any] = {}
    built_files: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix=".turbo-build-", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        for kind, filename in TURBO_TOPOLOGIES.items():
            source, selector_free = sources[kind]
            base_topology = temporary_dir / f"{kind}.base.onnx"
            final_topology = temporary_dir / filename
            weights = build_persistent_topology(
                source,
                base_topology,
                canonical_outputs=True,
                all_initializers=True,
                selector_free_if=selector_free,
            )
            model = onnx.load(str(base_topology), load_external_data=False)
            _TOPOLOGY_PATCHERS[kind](model)
            model.graph.name = f"{model.graph.name}_turbo_v4_dynamic"
            onnx.checker.check_model(model, full_check=False)
            onnx.save_model(model, str(final_topology))
            adapter_inputs = [
                value.name
                for value in model.graph.input
                if value.name.startswith("turbo_lora_")
                or "silu_timestep_embedding" in value.name
            ]
            topology_metadata[kind] = {
                "file": filename,
                "sha256": _sha256(final_topology),
                "source": str(source.relative_to(base_model_dir)).replace("\\", "/"),
                "source_sha256": _sha256(source),
                "base_weight_inputs": len(weights),
                "adapter_inputs": adapter_inputs,
            }
            built_files[kind] = final_topology
        for kind, filename in TURBO_TOPOLOGIES.items():
            os.replace(built_files[kind], output_dir / filename)

    base_manifest = base_model_dir / "manifest.json"
    payload: dict[str, Any] = {
        "format": TURBO_ADAPTER_FORMAT,
        "variant": TURBO_ADAPTER_VARIANT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": {
            "directory": base_model_dir.name,
            "manifest_sha256": _sha256(base_manifest) if base_manifest.is_file() else None,
        },
        "assets": {
            "lora": assets.lora_identity.as_dict(),
            "silu_grid": assets.grid_identity.as_dict(),
        },
        "factor_pairs": TURBO_LORA_PAIR_COUNT,
        "factor_storage_bytes": assets.factor_bytes,
        "topologies": topology_metadata,
    }
    _atomic_json(output_dir / "adapter.json", payload)
    return payload


def validate_turbo_adapter(
    adapter_dir: Path,
    *,
    base_model_dir: Path | None = None,
) -> dict[str, Any]:
    adapter_dir = adapter_dir.resolve()
    manifest_path = adapter_dir / "adapter.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Turbo adapter manifest: {manifest_path}") from exc
    if payload.get("format") != TURBO_ADAPTER_FORMAT:
        raise ValueError("Unsupported Turbo adapter format")
    if payload.get("variant") != TURBO_ADAPTER_VARIANT:
        raise ValueError("Unsupported Turbo adapter variant")
    if payload.get("factor_pairs") != TURBO_LORA_PAIR_COUNT:
        raise ValueError("Turbo adapter factor-pair count is incompatible")
    assets = payload.get("assets", {})
    for key, identity in (
        ("lora", TURBO_LORA_IDENTITY),
        ("silu_grid", TURBO_SILU_GRID_IDENTITY),
    ):
        declared = assets.get(key)
        if not isinstance(declared, dict):
            raise ValueError(f"Turbo adapter has no {key} identity")
        if declared.get("size") != identity.size or declared.get("sha256") != identity.sha256:
            raise ValueError(f"Turbo adapter {key} identity is incompatible")
    topologies = payload.get("topologies")
    if not isinstance(topologies, dict) or set(topologies) != set(TURBO_TOPOLOGIES):
        raise ValueError("Turbo adapter topology set is incomplete")
    for kind, expected_filename in TURBO_TOPOLOGIES.items():
        record = topologies[kind]
        if not isinstance(record, dict) or record.get("file") != expected_filename:
            raise ValueError(f"Invalid topology record for {kind}")
        path = (adapter_dir / expected_filename).resolve()
        if path.parent != adapter_dir or not path.is_file():
            raise ValueError(f"Turbo adapter topology is missing: {expected_filename}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"Turbo adapter topology hash mismatch: {expected_filename}")
        onnx.checker.check_model(onnx.load(str(path), load_external_data=False), full_check=False)
    if base_model_dir is not None:
        base_model_dir = base_model_dir.resolve()
        base_record = payload.get("base", {})
        base_manifest = base_model_dir / "manifest.json"
        declared_manifest = base_record.get("manifest_sha256")
        if declared_manifest is not None:
            if not base_manifest.is_file() or _sha256(base_manifest) != declared_manifest:
                raise ValueError("Turbo adapter Base manifest identity mismatch")
        sources = _representative_sources(base_model_dir)
        for kind, (source, _) in sources.items():
            if not source.is_file() or _sha256(source) != topologies[kind].get("source_sha256"):
                raise ValueError(f"Turbo adapter Base topology mismatch: {kind}")
    return payload


class TurboLoraAdapter:
    """Resident factor/grid cache plus graph-only topology routing for runtime use."""

    def __init__(
        self,
        adapter_dir: Path,
        lora_path: Path,
        grid_path: Path,
        *,
        strength: float = 1.0,
        base_model_dir: Path | None = None,
        verify_identity: bool = True,
    ):
        started = time.perf_counter()
        self.adapter_dir = adapter_dir.resolve()
        self.manifest = validate_turbo_adapter(
            self.adapter_dir,
            base_model_dir=base_model_dir,
        )
        self.assets = validate_turbo_assets(
            lora_path,
            grid_path,
            verify_identity=verify_identity,
        )
        manifest_assets = self.manifest["assets"]
        if self.assets.lora_identity.sha256 != manifest_assets["lora"]["sha256"]:
            raise ValueError("Turbo LoRA does not match the published adapter")
        if self.assets.grid_identity.sha256 != manifest_assets["silu_grid"]["sha256"]:
            raise ValueError("Turbo SiLU grid does not match the published adapter")
        self.strength = float(strength)
        self._factor_cache = TurboLoraFactorCache(
            self.assets.lora_path,
            strength=self.strength,
        )
        grid_reader = SafeTensorFile(self.assets.grid_path)
        self._silu_grid = grid_reader.tensor("silu_t_emb_grid", np.dtype(np.float32))
        self.topology_paths = {
            kind: self.adapter_dir / filename
            for kind, filename in TURBO_TOPOLOGIES.items()
        }
        self.load_seconds = time.perf_counter() - started
        self._closed = False

    def graph_feeds(self, graph: str) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("Turbo LoRA adapter is closed")
        return self._factor_cache.graph_feeds(graph)

    def silu_timestep_embeddings(self, timesteps: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Turbo LoRA adapter is closed")
        return interpolate_silu_grid(self._silu_grid, timesteps)

    def metrics(self) -> dict[str, Any]:
        return {
            "variant": TURBO_ADAPTER_VARIANT,
            "strength": self.strength,
            "factor_pairs": len(self._factor_cache.factors),
            "factor_cache_bytes": self._factor_cache.total_bytes,
            "grid_cache_bytes": self._silu_grid.nbytes,
            "load_seconds": self.load_seconds,
            "lora_sha256": self.assets.lora_identity.sha256,
            "silu_grid_sha256": self.assets.grid_identity.sha256,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._factor_cache.close()
        self._silu_grid = np.empty((0, 2688), dtype=np.float32)

    def __enter__(self) -> TurboLoraAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
