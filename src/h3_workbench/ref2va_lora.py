from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper
from safetensors import safe_open


REF2VA_ADAPTER_FORMAT = "h3-ref2va-lora-adapter-v1"
REF2VA_ADAPTER_VARIANT = "ref2v_turbo_v0.1_dynamic"
REF2VA_LORA_FILENAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REF2VA_LORA_SIZE = 1_956_193_000
REF2VA_LORA_SHA256 = "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"
REF2VA_LORA_TENSOR_COUNT = 624
REF2VA_LORA_MODULE_COUNT = 208

REF2VA_TOPOLOGIES = {
    "attention_qkv": "runtime_ref2va_lora_attention_qkv.onnx",
    "attention_output": "runtime_ref2va_lora_attention_output.onnx",
    "mlp": "runtime_ref2va_lora_mlp.onnx",
    "refiner_00_attention": "runtime_ref2va_lora_refiner_00_attention.onnx",
    "refiner_01_attention": "runtime_ref2va_lora_refiner_01_attention.onnx",
    "refiner_00_mlp": "runtime_ref2va_lora_refiner_00_mlp.onnx",
    "refiner_01_mlp": "runtime_ref2va_lora_refiner_01_mlp.onnx",
}


@dataclass(frozen=True)
class Ref2VALoraIdentity:
    filename: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
        }


REF2VA_LORA_IDENTITY = Ref2VALoraIdentity(
    REF2VA_LORA_FILENAME,
    REF2VA_LORA_SIZE,
    REF2VA_LORA_SHA256,
)


@dataclass(frozen=True)
class Ref2VALoraSpec:
    prefix: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    alpha: float
    feed_dtype: np.dtype[Any]

    @property
    def source_prefix(self) -> str:
        return f"diffusion_model.{self.prefix}"

    @property
    def a_key(self) -> str:
        return f"{self.source_prefix}.lora_A.weight"

    @property
    def b_key(self) -> str:
        return f"{self.source_prefix}.lora_B.weight"

    @property
    def alpha_key(self) -> str:
        return f"{self.source_prefix}.alpha"


@dataclass(frozen=True)
class Ref2VALoraFactors:
    a: np.ndarray
    b: np.ndarray


def _specs() -> dict[str, Ref2VALoraSpec]:
    specs: dict[str, Ref2VALoraSpec] = {}
    fp16 = np.dtype(np.float16)
    fp32 = np.dtype(np.float32)

    def add(prefix: str, a_shape: tuple[int, int], b_shape: tuple[int, int], alpha: float, dtype: np.dtype[Any]) -> None:
        specs[prefix] = Ref2VALoraSpec(prefix, a_shape, b_shape, alpha, dtype)

    for block in range(50):
        root = f"blocks.{block}"
        add(f"{root}.attn.qkv_proj", (384, 5376), (21504, 384), 24.0, fp16)
        add(f"{root}.attn.out_proj", (128, 7168), (5376, 128), 8.0, fp32)
        add(f"{root}.mlp.fc1", (128, 5376), (28672, 128), 8.0, fp32)
        add(f"{root}.mlp.fc2", (128, 14336), (5376, 128), 8.0, fp32)
    for block in range(2):
        root = f"token_refiner.blocks.{block}"
        add(f"{root}.attn.qkv_proj", (384, 5376), (21504, 384), 24.0, fp16)
        add(f"{root}.attn.out_proj", (128, 7168), (5376, 128), 8.0, fp16)
        add(f"{root}.mlp.fc1", (128, 5376), (28672, 128), 8.0, fp32)
        add(f"{root}.mlp.fc2", (128, 14336), (5376, 128), 8.0, fp32)
    if len(specs) != REF2VA_LORA_MODULE_COUNT:
        raise AssertionError(f"Ref2VA LoRA schema has {len(specs)} modules")
    return specs


EXPECTED_REF2VA_LORA_SPECS = _specs()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(checkpoint: Any, key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in checkpoint.get_slice(key).get_shape())


def validate_ref2va_lora_file(
    path: Path,
    *,
    verify_identity: bool = True,
) -> Ref2VALoraIdentity:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if verify_identity and size != REF2VA_LORA_SIZE:
        raise ValueError(f"Unexpected Ref2VA LoRA size: {size}, expected {REF2VA_LORA_SIZE}")
    digest = _sha256(path)
    if verify_identity and digest != REF2VA_LORA_SHA256:
        raise ValueError(f"Unexpected Ref2VA LoRA SHA-256: {digest}, expected {REF2VA_LORA_SHA256}")
    with safe_open(str(path), framework="numpy") as checkpoint:
        keys = set(checkpoint.keys())
        expected = {
            key
            for spec in EXPECTED_REF2VA_LORA_SPECS.values()
            for key in (spec.a_key, spec.b_key, spec.alpha_key)
        }
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing {len(missing)} ({missing[0]})")
            if extra:
                details.append(f"unexpected {len(extra)} ({extra[0]})")
            raise ValueError("Ref2VA LoRA tensor set mismatch: " + "; ".join(details))
        for spec in EXPECTED_REF2VA_LORA_SPECS.values():
            if _shape(checkpoint, spec.a_key) != spec.a_shape:
                raise ValueError(f"Ref2VA LoRA A shape mismatch: {spec.a_key}")
            if _shape(checkpoint, spec.b_key) != spec.b_shape:
                raise ValueError(f"Ref2VA LoRA B shape mismatch: {spec.b_key}")
            if _shape(checkpoint, spec.alpha_key) != ():
                raise ValueError(f"Ref2VA LoRA alpha must be scalar: {spec.alpha_key}")
    return Ref2VALoraIdentity(path.name, size, digest)


def inspect_ref2va_lora(path: Path, strength: float = 1.0) -> dict[str, object]:
    identity = validate_ref2va_lora_file(path, verify_identity=False)
    return {
        **identity.as_dict(),
        "variant": REF2VA_ADAPTER_VARIANT,
        "strength": float(strength),
        "tensor_count": REF2VA_LORA_TENSOR_COUNT,
        "module_count": REF2VA_LORA_MODULE_COUNT,
        "rank": 128,
        "recommended_steps": [4],
    }


class Ref2VALoraFactorCache:
    def __init__(self, path: Path, *, strength: float = 1.0) -> None:
        if not math.isfinite(strength):
            raise ValueError("Ref2VA LoRA strength must be finite")
        self.path = path.resolve()
        self.strength = float(strength)
        self._file = safe_open(str(self.path), framework="numpy")
        self._factors: dict[str, Ref2VALoraFactors] = {}
        self.total_bytes = 0

    def factors(self, prefix: str) -> Ref2VALoraFactors:
        cached = self._factors.get(prefix)
        if cached is not None:
            return cached
        try:
            spec = EXPECTED_REF2VA_LORA_SPECS[prefix]
        except KeyError as exc:
            raise KeyError(f"Unknown Ref2VA LoRA prefix: {prefix}") from exc
        a = np.asarray(self._file.get_tensor(spec.a_key), dtype=spec.feed_dtype)
        b = np.asarray(self._file.get_tensor(spec.b_key), dtype=np.float32)
        alpha = float(np.asarray(self._file.get_tensor(spec.alpha_key), dtype=np.float32))
        scale = np.float32(self.strength * alpha / spec.a_shape[0])
        b = np.asarray(b * scale, dtype=spec.feed_dtype)
        result = Ref2VALoraFactors(
            np.ascontiguousarray(a),
            np.ascontiguousarray(b),
        )
        self._factors[prefix] = result
        self.total_bytes += result.a.nbytes + result.b.nbytes
        return result

    def close(self) -> None:
        self._factors.clear()
        self.total_bytes = 0
        self._file = None


def _graph_spec(graph: str) -> tuple[str, str] | None:
    name = Path(graph).name.removesuffix(".onnx")
    match = re.fullmatch(r"main_block_(\d{2})_(attention_qkv|attention_output|mlp)", name)
    if match:
        block = int(match.group(1))
        kind = match.group(2)
        prefix = {
            "attention_qkv": f"blocks.{block}.attn.qkv_proj",
            "attention_output": f"blocks.{block}.attn.out_proj",
            "mlp": f"blocks.{block}.mlp",
        }[kind]
        return kind, prefix
    match = re.fullmatch(r"main_token_refiner_block_(\d{2})_(attention|mlp)", name)
    if match:
        block = int(match.group(1))
        kind = match.group(2)
        if kind == "attention":
            return "refiner_attention", f"token_refiner.blocks.{block}.attn"
        return "refiner_mlp", f"token_refiner.blocks.{block}.mlp"
    return None


def _feed_specs(kind: str, prefix: str) -> tuple[tuple[str, str], ...]:
    if kind == "attention_qkv":
        return (("lora_qkv_A", prefix), ("lora_qkv_B", prefix))
    if kind == "attention_output":
        return (("lora_out_A", f"{prefix}"), ("lora_out_B", f"{prefix}"))
    if kind == "refiner_attention":
        return (
            ("lora_qkv_A", f"{prefix}.qkv_proj"),
            ("lora_qkv_B", f"{prefix}.qkv_proj"),
            ("lora_out_A", f"{prefix}.out_proj"),
            ("lora_out_B", f"{prefix}.out_proj"),
        )
    return (
        ("lora_fc1_A", f"{prefix}.fc1"),
        ("lora_fc1_B", f"{prefix}.fc1"),
        ("lora_fc2_A", f"{prefix}.fc2"),
        ("lora_fc2_B", f"{prefix}.fc2"),
    )


def _append_input(model: onnx.ModelProto, name: str, dtype: int, shape: tuple[int | str, ...]) -> None:
    if any(value.name == name for value in model.graph.input):
        raise ValueError(f"Topology input already exists: {name}")
    model.graph.input.append(helper.make_tensor_value_info(name, dtype, list(shape)))


def _insert_after(model: onnx.ModelProto, node: onnx.NodeProto, additions: list[onnx.NodeProto]) -> None:
    index = next((i for i, item in enumerate(model.graph.node) if item.name == node.name), None)
    if index is None:
        raise ValueError(f"Node not found: {node.name}")
    for offset, item in enumerate(additions, start=1):
        model.graph.node.insert(index + offset, item)


def _low_rank_nodes(source: str, a_name: str, b_name: str, stem: str) -> tuple[list[onnx.NodeProto], str]:
    down = f"{stem}.down"
    delta = f"{stem}.delta"
    return [
        helper.make_node("Gemm", [source, a_name], [down], name=f"{stem}/A", transB=1),
        helper.make_node("Gemm", [down, b_name], [delta], name=f"{stem}/B", transB=1),
    ], delta


def _one_gemm_by_weight(model: onnx.ModelProto, suffix: str) -> onnx.NodeProto:
    matches = [
        node
        for node in model.graph.node
        if node.op_type == "Gemm" and len(node.input) >= 2 and node.input[1].endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Gemm using *{suffix}, found {len(matches)}")
    return matches[0]


def _inject_after_casted_gemm(
    model: onnx.ModelProto,
    gemm: onnx.NodeProto,
    source: str,
    a_name: str,
    b_name: str,
    stem: str,
) -> None:
    casts = [node for node in model.graph.node if gemm.output[0] in node.input and node.op_type == "Cast"]
    consumers = [node for node in model.graph.node if gemm.output[0] in node.input]
    if len(casts) != 1 or len(consumers) != 1:
        raise ValueError(f"Expected one Cast after {gemm.name}")
    cast = casts[0]
    output = cast.output[0]
    base_output = f"{output}.base"
    cast.output[0] = base_output
    nodes, delta = _low_rank_nodes(source, a_name, b_name, stem)
    delta_cast = f"{stem}.delta_cast"
    nodes.extend(
        [
            helper.make_node("Cast", [delta], [delta_cast], name=f"{stem}/Cast", to=TensorProto.FLOAT),
            helper.make_node("Add", [base_output, delta_cast], [output], name=f"{stem}/Add"),
        ]
    )
    _insert_after(model, cast, nodes)


def _inject_at_value(
    model: onnx.ModelProto,
    target: str,
    source: str,
    a_name: str,
    b_name: str,
    stem: str,
    output_dtype: int,
) -> None:
    producers = [node for node in model.graph.node if target in node.output]
    if len(producers) != 1:
        raise ValueError(f"Expected one producer for {target}, found {len(producers)}")
    producer = producers[0]
    output_index = list(producer.output).index(target)
    base_output = f"{target}.base"
    producer.output[output_index] = base_output
    nodes, delta = _low_rank_nodes(source, a_name, b_name, stem)
    delta_cast = f"{stem}.delta_cast"
    nodes.extend(
        [
            helper.make_node("Cast", [delta], [delta_cast], name=f"{stem}/Cast", to=output_dtype),
            helper.make_node("Add", [base_output, delta_cast], [target], name=f"{stem}/Add"),
        ]
    )
    _insert_after(model, producer, nodes)


def _add_inputs(model: onnx.ModelProto, specs: tuple[tuple[str, tuple[int, int], int], ...]) -> None:
    for name, shape, dtype in specs:
        _append_input(model, name, dtype, shape)


def _patch_qkv(model: onnx.ModelProto, *, refiner: bool) -> None:
    _add_inputs(
        model,
        (
            ("lora_qkv_A", (384, 5376), TensorProto.FLOAT16),
            ("lora_qkv_B", (21504, 384), TensorProto.FLOAT16),
        ),
    )
    suffix = "attention.qkv.weight"
    gemm = _one_gemm_by_weight(model, suffix)
    _inject_after_casted_gemm(model, gemm, gemm.input[0], "lora_qkv_A", "lora_qkv_B", "ref2va_lora/qkv")


def _patch_main_attention_output(model: onnx.ModelProto) -> None:
    _add_inputs(
        model,
        (
            ("lora_out_A", (128, 7168), TensorProto.FLOAT),
            ("lora_out_B", (5376, 128), TensorProto.FLOAT),
        ),
    )
    _inject_at_value(
        model,
        "cat",
        "attended",
        "lora_out_A",
        "lora_out_B",
        "ref2va_lora/out",
        TensorProto.FLOAT,
    )


def _patch_mlp(model: onnx.ModelProto, *, refiner: bool) -> None:
    _add_inputs(
        model,
        (
            ("lora_fc1_A", (128, 5376), TensorProto.FLOAT),
            ("lora_fc1_B", (28672, 128), TensorProto.FLOAT),
            ("lora_fc2_A", (128, 14336), TensorProto.FLOAT),
            ("lora_fc2_B", (5376, 128), TensorProto.FLOAT),
        ),
    )
    source_fc1 = "mul_8" if refiner else "add_59"
    source_fc2 = "mul_243" if refiner else "mul_273"
    _inject_at_value(model, "cat", source_fc1, "lora_fc1_A", "lora_fc1_B", "ref2va_lora/fc1", TensorProto.FLOAT)
    _inject_at_value(model, "cat_1", source_fc2, "lora_fc2_A", "lora_fc2_B", "ref2va_lora/fc2", TensorProto.FLOAT)


def _patch_refiner_attention(model: onnx.ModelProto) -> None:
    _patch_qkv(model, refiner=True)
    _add_inputs(
        model,
        (
            ("lora_out_A", (128, 7168), TensorProto.FLOAT16),
            ("lora_out_B", (5376, 128), TensorProto.FLOAT16),
        ),
    )
    gemm = _one_gemm_by_weight(model, "attention.out.weight")
    _inject_after_casted_gemm(model, gemm, gemm.input[0], "lora_out_A", "lora_out_B", "ref2va_lora/out")


def _patch_graph(kind: str, model: onnx.ModelProto) -> None:
    if kind == "attention_qkv":
        _patch_qkv(model, refiner=False)
    elif kind == "attention_output":
        _patch_main_attention_output(model)
    elif kind == "mlp":
        _patch_mlp(model, refiner=False)
    elif kind.endswith("_attention"):
        _patch_refiner_attention(model)
    elif kind.endswith("_mlp"):
        _patch_mlp(model, refiner=True)
    else:
        raise ValueError(f"Unsupported Ref2VA LoRA graph kind: {kind}")
    model.graph.name = f"{model.graph.name}_{REF2VA_ADAPTER_VARIANT}"


def _copy_external_data(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    # ONNX refuses to validate external tensors stored in hard-linked files.
    # A regular adjacent copy also keeps the published adapter self-contained.
    shutil.copy2(source, destination)


def _save_patched_graph(source: Path, destination: Path, kind: str) -> None:
    model = onnx.load(str(source), load_external_data=False)
    _patch_graph(kind, model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination), save_as_external_data=False)
    locations = {
        item.value
        for initializer in model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL
        for item in initializer.external_data
        if item.key == "location"
    }
    for location in locations:
        source_data = source.parent / location
        if not source_data.is_file():
            raise FileNotFoundError(source_data)
        _copy_external_data(source_data, destination.parent / location)
    # Token-refiner graphs retain external initializers.  Checking by file path
    # lets ONNX resolve those adjacent .onnx.data files after they are linked.
    onnx.checker.check_model(str(destination), full_check=False)


def _manifest_hash(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def publish_ref2va_adapter(
    base_model_dir: Path,
    output_dir: Path,
    lora_path: Path,
    *,
    verify_identity: bool = True,
) -> dict[str, Any]:
    from h3_workbench.ref2va_virtual_slicer import (
        validated_ref2va_virtual_ready,
        virtual_product_fingerprint,
    )

    base_model_dir = base_model_dir.resolve()
    output_dir = output_dir.resolve()
    if not validated_ref2va_virtual_ready(base_model_dir):
        raise ValueError(f"Ref2VA virtual base is not validated: {base_model_dir}")
    identity = validate_ref2va_lora_file(lora_path, verify_identity=verify_identity)
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest_path = output_dir / "adapter.json"
        if manifest_path.is_file():
            return validate_ref2va_adapter(output_dir, base_model_dir=base_model_dir)
        raise FileExistsError(f"Ref2VA adapter target is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "attention_qkv": base_model_dir / "main_block_00_attention_qkv.onnx",
        "attention_output": base_model_dir / "main_block_00_attention_output.onnx",
        "mlp": base_model_dir / "main_block_00_mlp.onnx",
        "refiner_00_attention": base_model_dir / "main_token_refiner_block_00_attention.onnx",
        "refiner_01_attention": base_model_dir / "main_token_refiner_block_01_attention.onnx",
        "refiner_00_mlp": base_model_dir / "main_token_refiner_block_00_mlp.onnx",
        "refiner_01_mlp": base_model_dir / "main_token_refiner_block_01_mlp.onnx",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Ref2VA topology sources: " + ", ".join(missing))
    topology_metadata: dict[str, Any] = {}
    for kind, source in sources.items():
        filename = REF2VA_TOPOLOGIES[kind]
        destination = output_dir / filename
        _save_patched_graph(source, destination, kind)
        topology_metadata[kind] = {
            "file": filename,
            "sha256": _sha256(destination),
            "source": source.name,
            "source_sha256": _sha256(source),
        }
    base_manifest = base_model_dir / "manifest.json"
    payload: dict[str, Any] = {
        "format": REF2VA_ADAPTER_FORMAT,
        "variant": REF2VA_ADAPTER_VARIANT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": {
            "directory": base_model_dir.name,
            "manifest_sha256": _manifest_hash(base_manifest),
            "virtual_product_fingerprint": virtual_product_fingerprint(base_model_dir),
        },
        "assets": {"lora": identity.as_dict()},
        "tensor_count": REF2VA_LORA_TENSOR_COUNT,
        "factor_pairs": REF2VA_LORA_MODULE_COUNT,
        "rank": 128,
        "strength": 1.0,
        "recommended_steps": [4],
        "topologies": topology_metadata,
    }
    temporary = output_dir / f"adapter.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "adapter.json")
    return payload


def validate_ref2va_adapter(
    adapter_dir: Path,
    *,
    base_model_dir: Path | None = None,
) -> dict[str, Any]:
    adapter_dir = adapter_dir.resolve()
    payload = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
    if payload.get("format") != REF2VA_ADAPTER_FORMAT:
        raise ValueError("Unsupported Ref2VA adapter format")
    if payload.get("variant") != REF2VA_ADAPTER_VARIANT:
        raise ValueError("Unsupported Ref2VA adapter variant")
    if payload.get("factor_pairs") != REF2VA_LORA_MODULE_COUNT:
        raise ValueError("Ref2VA adapter factor count is incompatible")
    declared = payload.get("assets", {}).get("lora")
    if not isinstance(declared, dict) or declared.get("size") != REF2VA_LORA_SIZE or declared.get("sha256") != REF2VA_LORA_SHA256:
        raise ValueError("Ref2VA adapter LoRA identity is incompatible")
    topologies = payload.get("topologies")
    if not isinstance(topologies, dict) or set(topologies) != set(REF2VA_TOPOLOGIES):
        raise ValueError("Ref2VA adapter topology set is incomplete")
    for kind, filename in REF2VA_TOPOLOGIES.items():
        record = topologies[kind]
        path = (adapter_dir / filename).resolve()
        if path.parent != adapter_dir or not path.is_file():
            raise ValueError(f"Ref2VA adapter topology is missing: {filename}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"Ref2VA adapter topology hash mismatch: {filename}")
        onnx.checker.check_model(str(path), full_check=False)
    if base_model_dir is not None:
        from h3_workbench.ref2va_virtual_slicer import virtual_product_fingerprint

        base_model_dir = base_model_dir.resolve()
        base = payload.get("base", {})
        manifest = base_model_dir / "manifest.json"
        if base.get("manifest_sha256") is not None and _manifest_hash(manifest) != base.get("manifest_sha256"):
            raise ValueError("Ref2VA adapter base manifest mismatch")
        if virtual_product_fingerprint(base_model_dir) != base.get("virtual_product_fingerprint"):
            raise ValueError("Ref2VA adapter virtual product mismatch")
    return payload


class Ref2VALoraAdapter:
    """Resident low-rank factors and patched topologies for the Ref2VA base."""

    base_component = "ref2va_transformer"
    requires_silu = False

    def __init__(
        self,
        adapter_dir: Path,
        lora_path: Path,
        *,
        strength: float = 1.0,
        base_model_dir: Path | None = None,
        verify_identity: bool = True,
    ) -> None:
        self.adapter_dir = adapter_dir.resolve()
        self.manifest = validate_ref2va_adapter(self.adapter_dir, base_model_dir=base_model_dir)
        self.identity = validate_ref2va_lora_file(lora_path, verify_identity=verify_identity)
        declared = self.manifest["assets"]["lora"]
        if self.identity.sha256 != declared["sha256"]:
            raise ValueError("Ref2VA LoRA does not match the published adapter")
        self.factors = Ref2VALoraFactorCache(lora_path, strength=strength)
        self.strength = float(strength)
        self.topology_paths = {
            kind: self.adapter_dir / filename
            for kind, filename in REF2VA_TOPOLOGIES.items()
            if kind in {"attention_qkv", "attention_output", "mlp"}
        }
        self.session_paths = {
            "shard_001": self.adapter_dir / REF2VA_TOPOLOGIES["refiner_00_attention"],
            "shard_002": self.adapter_dir / REF2VA_TOPOLOGIES["refiner_00_mlp"],
            "shard_003": self.adapter_dir / REF2VA_TOPOLOGIES["refiner_01_attention"],
            "shard_004": self.adapter_dir / REF2VA_TOPOLOGIES["refiner_01_mlp"],
        }

    def graph_feeds(self, graph: str) -> dict[str, np.ndarray]:
        parsed = _graph_spec(graph)
        if parsed is None:
            return {}
        kind, prefix = parsed
        feeds: dict[str, np.ndarray] = {}
        for name, factor_prefix in _feed_specs(kind, prefix):
            pair = self.factors.factors(factor_prefix)
            feeds[name] = pair.a if name.endswith("_A") else pair.b
        return feeds

    def uses_embedded_weights(self, graph: str) -> bool:
        return Path(graph).name.removesuffix(".onnx") in {
            "main_token_refiner_block_00_attention",
            "main_token_refiner_block_01_attention",
            "main_token_refiner_block_00_mlp",
            "main_token_refiner_block_01_mlp",
        }

    def metrics(self) -> dict[str, object]:
        return {
            "variant": REF2VA_ADAPTER_VARIANT,
            "strength": self.strength,
            "factor_pairs": REF2VA_LORA_MODULE_COUNT,
            "factor_cache_bytes": self.factors.total_bytes,
            "lora_sha256": self.identity.sha256,
        }

    def close(self) -> None:
        self.factors.close()
