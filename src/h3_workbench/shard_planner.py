"""Dependency-safe shard planning and h3-schedule-v2 generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

SCHEDULE_FORMAT = "h3-schedule-v2"
SHARD_STORAGE_TARGET_BYTES = 1 << 30
SHARD_MATERIALIZED_TARGET_BYTES = 1 << 30
DEFAULT_SESSION_SLOT_HINTS = {"vram_4gib": 1, "vram_24gib": 8}

_BLOCK_PATTERN = re.compile(r"^main_block_(\d+)_(attention_qkv|attention_output|mlp)$")
_VALID_SOURCES = {"external", "const", "buffer"}
_VALID_TARGETS = {"external", "const", "buffer", "discard"}


@dataclass(frozen=True)
class GraphInfo:
    name: str
    kind: str
    block: int | None = None
    storage_bytes: int = 0
    materialized_weight_bytes: int = 0
    phase: str = "denoise"

    @property
    def weight_bytes(self) -> int:
        """Compatibility alias for callers that only need L3 bytes."""
        return self.storage_bytes


@dataclass
class ShardPlan:
    id: str
    graphs: list[str] = field(default_factory=list)
    storage_bytes: int = 0
    materialized_weight_bytes: int = 0
    session_peak_bytes: int = 0
    resident: bool = False
    oversize: bool = False


def _binding(source: str, name: str) -> dict[str, str]:
    return {"source": source, "name": name}


def _target(target: str, name: str | None = None) -> dict[str, str]:
    result = {"target": target}
    if name is not None:
        result["name"] = name
    return result


def _graph_step(
    phase: str,
    shard: str,
    graph: str,
    inputs: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "kind": "graph",
        "shard": shard,
        "graph": graph,
        "inputs": inputs,
        "outputs": outputs,
        "release": [],
        "barrier_after": False,
    }


def _op_step(
    phase: str,
    op: str,
    inputs: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, str]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "phase": phase,
        "kind": "op",
        "op": op,
        "inputs": inputs,
        "outputs": outputs,
        "release": [],
        "barrier_after": False,
    }
    if params:
        result["params"] = params
    return result


OP_PORTS: dict[str, tuple[set[str], set[str]]] = {
    "preamble_inputs": (
        {"raw_text_states"},
        {"video_patches", "audio_patches", "text_states"},
    ),
    "denoise_inputs": (
        {"video_latent", "audio_latent", "sigma_video", "text_states"},
        {
            "video_patches",
            "audio_patches",
            "embedding_text_padding",
            "timesteps",
            "position_ids",
            "modulation_ids",
            "sigma_audio",
        },
    ),
    "concat_hidden": (
        {"text_states", "audio_embeddings", "video_embeddings"},
        {"hidden"},
    ),
    "sdpa": ({"qkv_packed"}, {"attended"}),
    "split_hidden": ({"hidden", "text_states"}, {"audio_hidden", "video_hidden"}),
    "select_head_timestep": (
        {"timesteps", "timestep_embedding", "sigma_video", "sigma_audio"},
        {"video_timestep_embedding", "audio_timestep_embedding"},
    ),
    "select_head_timestep_turbo": (
        {"timesteps", "timestep_embedding", "silu_timestep_embedding", "sigma_video", "sigma_audio"},
        {
            "video_timestep_embedding",
            "audio_timestep_embedding",
            "video_silu_timestep_embedding",
            "audio_silu_timestep_embedding",
        },
    ),
    "unpack_velocity": (
        {"video_patches", "audio_patches"},
        {"video_velocity", "audio_velocity"},
    ),
}


def _shard_id(index: int) -> str:
    return f"shard_{index:03d}"


def _denoise_units(graphs: list[GraphInfo]) -> list[list[GraphInfo]]:
    by_block: dict[int, dict[str, GraphInfo]] = {}
    for graph in graphs:
        if graph.block is not None:
            by_block.setdefault(graph.block, {})[graph.kind] = graph
    units: list[list[GraphInfo]] = []
    for block in sorted(by_block):
        kinds = by_block[block]
        qkv = kinds.get("qkv")
        attn_out = kinds.get("attn_out")
        if (qkv is None) != (attn_out is None):
            missing = "attn_out" if qkv is not None else "qkv"
            raise ValueError(f"Block {block} is missing {missing}; qkv and attention output are atomic")
        if qkv is not None:
            units.append([qkv])
            units.append([attn_out])  # type: ignore[list-item]
        if "mlp" in kinds:
            units.append([kinds["mlp"]])
    return units


def plan_shards(
    graphs: list[GraphInfo],
    storage_target_bytes: int = SHARD_STORAGE_TARGET_BYTES,
    materialized_target_bytes: int = SHARD_MATERIALIZED_TARGET_BYTES,
) -> list[ShardPlan]:
    resident = [g for g in graphs if g.phase == "preamble" or g.kind in {"embeddings", "conditioning", "head"}]
    denoise = [g for g in graphs if g not in resident]
    units: list[tuple[bool, list[GraphInfo]]] = [(True, [g]) for g in resident]
    units.extend((False, unit) for unit in _denoise_units(denoise))
    shards: list[ShardPlan] = []
    current = ShardPlan(id=_shard_id(0), resident=True)
    in_resident = True

    def flush() -> None:
        nonlocal current
        if not current.graphs:
            return
        current.session_peak_bytes = max(
            current.materialized_weight_bytes,
            int(current.materialized_weight_bytes * 1.25),
        )
        shards.append(current)
        current = ShardPlan(id=_shard_id(len(shards)), resident=in_resident)

    for is_resident, unit in units:
        if in_resident and not is_resident:
            flush()
            in_resident = False
            current.resident = False
        if current.graphs:
            flush()
        storage = sum(g.storage_bytes for g in unit)
        materialized = sum(g.materialized_weight_bytes for g in unit)
        oversize = storage > storage_target_bytes or materialized > materialized_target_bytes
        exceeds = (
            current.graphs
            and (
                current.storage_bytes + storage > storage_target_bytes
                or current.materialized_weight_bytes + materialized > materialized_target_bytes
            )
        )
        if exceeds:
            flush()
        if oversize and current.graphs:
            flush()
        current.graphs.extend(g.name for g in unit)
        current.storage_bytes += storage
        current.materialized_weight_bytes += materialized
        current.oversize = current.oversize or oversize
        flush()
    flush()
    return shards


def _classify_graph(stem: str) -> tuple[str, int | None, str] | None:
    if stem == "main_embeddings":
        return "embeddings", None, "preamble"
    if stem == "main_conditioning":
        return "conditioning", None, "denoise"
    if stem == "main_head":
        return "head", None, "denoise"
    if stem == "main_token_refiner_norm":
        return "refiner_norm", None, "preamble"
    if stem.startswith("main_token_refiner_block_"):
        tail = stem.removeprefix("main_token_refiner_block_")
        block_text, kind = tail.split("_", 1)
        return f"refiner_{kind}", int(block_text), "preamble"
    match = _BLOCK_PATTERN.match(stem)
    if match is None:
        return None
    return {
        "attention_qkv": "qkv",
        "attention_output": "attn_out",
        "mlp": "mlp",
    }[match.group(2)], int(match.group(1)), "denoise"


def _tensor_bytes(tensor: onnx.TensorProto) -> int:
    item_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.BFLOAT16: 2,
        TensorProto.FLOAT8E4M3FN: 1,
        TensorProto.FLOAT8E4M3FNUZ: 1,
        TensorProto.FLOAT8E5M2: 1,
        TensorProto.FLOAT8E5M2FNUZ: 1,
    }
    size = item_sizes.get(tensor.data_type, 0)
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    return count * size


def graph_inventory(directory: Path) -> list[GraphInfo]:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    graphs: list[GraphInfo] = []
    for filename in manifest.get("graphs", []):
        stem = filename.removesuffix(".onnx")
        classified = _classify_graph(stem)
        if classified is None:
            continue
        kind, block, phase = classified
        graph_path = directory / filename
        external_path = graph_path.with_name(f"{graph_path.name}.data")
        storage = graph_path.stat().st_size + (external_path.stat().st_size if external_path.is_file() else 0)
        model = onnx.load(str(graph_path), load_external_data=False)
        fp8_bytes = sum(
            _tensor_bytes(tensor)
            for tensor in model.graph.initializer
            if tensor.data_type
            in {
                TensorProto.FLOAT8E4M3FN,
                TensorProto.FLOAT8E4M3FNUZ,
                TensorProto.FLOAT8E5M2,
                TensorProto.FLOAT8E5M2FNUZ,
            }
        )
        graphs.append(GraphInfo(stem, kind, block, storage, storage + fp8_bytes, phase))
    return graphs


def _graph_to_shard(shards: list[ShardPlan]) -> dict[str, str]:
    return {graph: shard.id for shard in shards for graph in shard.graphs}


def _build_steps(graphs: list[GraphInfo], shards: list[ShardPlan], blocks: int, sdpa_scale: float, turbo: bool) -> list[dict]:
    graph_by_kind = {g.kind: g for g in graphs if g.block is None}
    graph_by_block = {(g.block, g.kind): g for g in graphs if g.block is not None}
    shard_of = _graph_to_shard(shards)
    steps: list[dict] = []

    steps.append(
        _op_step(
            "preamble",
            "preamble_inputs",
            {"raw_text_states": _binding("external", "raw_text_states")},
            {
                "video_patches": _target("buffer", "preamble_video_patches"),
                "audio_patches": _target("buffer", "preamble_audio_patches"),
                "text_states": _target("buffer", "preamble_text_states"),
            },
        )
    )
    embeddings = graph_by_kind["embeddings"]
    steps.append(
        _graph_step(
            "preamble",
            shard_of[embeddings.name],
            embeddings.name,
            {
                "video_patches": _binding("buffer", "preamble_video_patches"),
                "audio_patches": _binding("buffer", "preamble_audio_patches"),
                "text_states": _binding("buffer", "preamble_text_states"),
            },
            {
                "video_embeddings": _target("discard"),
                "audio_embeddings": _target("discard"),
                "text_embeddings": _target("buffer", "refined_text"),
            },
        )
    )
    for block in range(2):
        for kind in ("refiner_attention", "refiner_mlp"):
            graph = graph_by_block[(block, kind)]
            steps.append(
                _graph_step(
                    "preamble",
                    shard_of[graph.name],
                    graph.name,
                    {"hidden_states": _binding("buffer", "refined_text")},
                    {"hidden_states_out": _target("buffer", "refined_text")},
                )
            )
    refiner_norm = graph_by_kind["refiner_norm"]
    steps.append(
        _graph_step(
            "preamble",
            shard_of[refiner_norm.name],
            refiner_norm.name,
            {"hidden_states": _binding("buffer", "refined_text")},
            {"hidden_states_out": _target("const", "text_states")},
        )
    )

    steps.append(
        _op_step(
            "denoise",
            "denoise_inputs",
            {
                "video_latent": _binding("external", "video_latent"),
                "audio_latent": _binding("external", "audio_latent"),
                "sigma_video": _binding("external", "sigma_video"),
                "text_states": _binding("const", "text_states"),
            },
            {
                "video_patches": _target("buffer", "video_patches"),
                "audio_patches": _target("buffer", "audio_patches"),
                "embedding_text_padding": _target("const", "embedding_text_padding"),
                "timesteps": _target("const", "timesteps"),
                "position_ids": _target("const", "position_ids"),
                "modulation_ids": _target("const", "modulation_ids"),
                "sigma_audio": _target("const", "sigma_audio"),
            },
        )
    )
    steps.append(
        _graph_step(
            "denoise",
            shard_of[embeddings.name],
            embeddings.name,
            {
                "video_patches": _binding("buffer", "video_patches"),
                "audio_patches": _binding("buffer", "audio_patches"),
                "text_states": _binding("const", "embedding_text_padding"),
            },
            {
                "video_embeddings": _target("buffer", "video_embeddings"),
                "audio_embeddings": _target("buffer", "audio_embeddings"),
                "text_embeddings": _target("discard"),
            },
        )
    )
    steps.append(
        _op_step(
            "denoise",
            "concat_hidden",
            {
                "text_states": _binding("const", "text_states"),
                "audio_embeddings": _binding("buffer", "audio_embeddings"),
                "video_embeddings": _binding("buffer", "video_embeddings"),
            },
            {"hidden": _target("buffer", "hidden")},
        )
    )
    conditioning = graph_by_kind["conditioning"]
    conditioning_outputs = {
        "timestep_embedding": _target("const", "timestep_embedding"),
        "rotary_table": _target("const", "rotary_table"),
    }
    if turbo:
        conditioning_outputs["silu_timestep_embedding"] = _target("const", "silu_timestep_embedding")
    steps.append(
        _graph_step(
            "denoise",
            shard_of[conditioning.name],
            conditioning.name,
            {
                "timesteps": _binding("const", "timesteps"),
                "position_ids": _binding("const", "position_ids"),
            },
            conditioning_outputs,
        )
    )

    for block in range(blocks):
        qkv = graph_by_block[(block, "qkv")]
        attn_out = graph_by_block[(block, "attn_out")]
        mlp = graph_by_block[(block, "mlp")]
        block_consts = {
            "timestep_embedding": _binding("const", "timestep_embedding"),
            "modulation_ids": _binding("const", "modulation_ids"),
        }
        qkv_inputs = {
            "hidden_states": _binding("buffer", "hidden"),
            **block_consts,
            "rotary_table": _binding("const", "rotary_table"),
        }
        mlp_inputs = {"hidden_states": _binding("buffer", "hidden"), **block_consts}
        if turbo:
            qkv_inputs["silu_timestep_embedding"] = _binding("const", "silu_timestep_embedding")
            mlp_inputs["silu_timestep_embedding"] = _binding("const", "silu_timestep_embedding")
        steps.append(
            _graph_step(
                "denoise",
                shard_of[qkv.name],
                qkv.name,
                qkv_inputs,
                {"hidden_states_out": _target("buffer", "qkv_packed")},
            )
        )
        steps.append(
            _op_step(
                "denoise",
                "sdpa",
                {"qkv_packed": _binding("buffer", "qkv_packed")},
                {"attended": _target("buffer", "attended")},
                {"scale": sdpa_scale},
            )
        )
        attn_inputs = {
            "hidden_states": _binding("buffer", "hidden"),
            "attended": _binding("buffer", "attended"),
            **block_consts,
        }
        steps.append(
            _graph_step(
                "denoise",
                shard_of[attn_out.name],
                attn_out.name,
                attn_inputs,
                {"hidden_states_out": _target("buffer", "hidden")},
            )
        )
        steps.append(
            _graph_step(
                "denoise",
                shard_of[mlp.name],
                mlp.name,
                mlp_inputs,
                {"hidden_states_out": _target("buffer", "hidden")},
            )
        )

    steps.append(
        _op_step(
            "denoise",
            "split_hidden",
            {
                "hidden": _binding("buffer", "hidden"),
                "text_states": _binding("const", "text_states"),
            },
            {
                "audio_hidden": _target("buffer", "audio_hidden"),
                "video_hidden": _target("buffer", "video_hidden"),
            },
        )
    )
    select_op = "select_head_timestep_turbo" if turbo else "select_head_timestep"
    select_inputs = {
        "timesteps": _binding("const", "timesteps"),
        "timestep_embedding": _binding("const", "timestep_embedding"),
        "sigma_video": _binding("external", "sigma_video"),
        "sigma_audio": _binding("const", "sigma_audio"),
    }
    select_outputs = {
        "video_timestep_embedding": _target("const", "video_timestep_embedding"),
        "audio_timestep_embedding": _target("const", "audio_timestep_embedding"),
    }
    if turbo:
        select_inputs["silu_timestep_embedding"] = _binding("const", "silu_timestep_embedding")
        select_outputs.update(
            {
                "video_silu_timestep_embedding": _target("const", "video_silu_timestep_embedding"),
                "audio_silu_timestep_embedding": _target("const", "audio_silu_timestep_embedding"),
            }
        )
    steps.append(_op_step("denoise", select_op, select_inputs, select_outputs))
    head = graph_by_kind["head"]
    head_inputs = {
        "video_hidden": _binding("buffer", "video_hidden"),
        "audio_hidden": _binding("buffer", "audio_hidden"),
        "video_timestep_embedding": _binding("const", "video_timestep_embedding"),
        "audio_timestep_embedding": _binding("const", "audio_timestep_embedding"),
    }
    if turbo:
        head_inputs.update(
            {
                "video_silu_timestep_embedding": _binding("const", "video_silu_timestep_embedding"),
                "audio_silu_timestep_embedding": _binding("const", "audio_silu_timestep_embedding"),
            }
        )
    steps.append(
        _graph_step(
            "denoise",
            shard_of[head.name],
            head.name,
            head_inputs,
            {
                "video_patches": _target("buffer", "video_velocity_patches"),
                "audio_patches": _target("buffer", "audio_velocity_patches"),
            },
        )
    )
    steps.append(
        _op_step(
            "denoise",
            "unpack_velocity",
            {
                "video_patches": _binding("buffer", "video_velocity_patches"),
                "audio_patches": _binding("buffer", "audio_velocity_patches"),
            },
            {
                "video_velocity": _target("external", "video_velocity"),
                "audio_velocity": _target("external", "audio_velocity"),
            },
        )
    )
    for index, step in enumerate(steps):
        step["id"] = index
    _annotate_lifetimes(steps)
    _annotate_barriers(steps)
    return steps


def _annotate_lifetimes(steps: list[dict]) -> None:
    for phase in ("preamble", "denoise"):
        phase_steps = [step for step in steps if step["phase"] == phase]
        last_reads: dict[str, int] = {}
        for step in phase_steps:
            for binding in step["inputs"].values():
                if binding["source"] == "buffer":
                    last_reads[binding["name"]] = step["id"]
        for step in phase_steps:
            step["release"] = sorted(name for name, last in last_reads.items() if last == step["id"])


def _activation_slots(steps: list[dict]) -> tuple[dict[str, dict[str, int]], int]:
    result: dict[str, dict[str, int]] = {}
    peak = 0
    for phase in ("preamble", "denoise"):
        phase_steps = [step for step in steps if step["phase"] == phase]
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        for step in phase_steps:
            for binding in step["outputs"].values():
                if binding["target"] == "buffer":
                    first.setdefault(binding["name"], step["id"])
                    last.setdefault(binding["name"], step["id"])
            for binding in step["inputs"].values():
                if binding["source"] == "buffer":
                    last[binding["name"]] = step["id"]
        assignments: dict[str, int] = {}
        active: list[tuple[int, int]] = []
        for name in sorted(first, key=lambda item: (first[item], item)):
            active = [(end, slot) for end, slot in active if end >= first[name]]
            used = {slot for _, slot in active}
            slot = next(index for index in range(len(used) + 1) if index not in used)
            assignments[name] = slot
            active.append((last[name], slot))
            peak = max(peak, slot + 1)
        result[phase] = assignments
    return result, peak


def _annotate_barriers(steps: list[dict]) -> None:
    for phase in ("preamble", "denoise"):
        graph_steps = [step for step in steps if step["phase"] == phase and step["kind"] == "graph"]
        for current, following in zip(graph_steps, graph_steps[1:]):
            current["barrier_after"] = current["shard"] != following["shard"]
        if phase == "denoise" and graph_steps:
            graph_steps[-1]["barrier_after"] = graph_steps[-1]["shard"] != graph_steps[0]["shard"]


def build_schedule(
    graphs: list[GraphInfo],
    model_id: str,
    blocks: int,
    sdpa_scale: float = 0.125,
    turbo: bool = False,
    storage_target_bytes: int = SHARD_STORAGE_TARGET_BYTES,
    materialized_target_bytes: int = SHARD_MATERIALIZED_TARGET_BYTES,
    session_slot_hints: dict[str, int] | None = None,
) -> dict:
    shards = plan_shards(graphs, storage_target_bytes, materialized_target_bytes)
    steps = _build_steps(graphs, shards, blocks, sdpa_scale, turbo)
    activation_slots, peak_slots = _activation_slots(steps)
    return {
        "format": SCHEDULE_FORMAT,
        "model": model_id,
        "precision": {
            "activation": "fp32",
            "attention_gemm": "fp16",
            "mlp_compute": "fp32",
            "mlp_storage": "fp8_e4m3fn_scaled",
        },
        "resources": {
            "activation_peak_slots": peak_slots,
            "activation_slots": activation_slots,
            "session_slot_hints": dict(session_slot_hints or DEFAULT_SESSION_SLOT_HINTS),
        },
        "shards": [
            {
                "id": shard.id,
                "file": f"{shard.id}.onnx",
                "storage_bytes": shard.storage_bytes,
                "materialized_weight_bytes": shard.materialized_weight_bytes,
                "session_peak_bytes": shard.session_peak_bytes,
                "resident": shard.resident,
                "graphs": list(shard.graphs),
                **({"oversize": True} if shard.oversize else {}),
            }
            for shard in shards
        ],
        "steps": steps,
    }


def _model_ports(path: Path) -> tuple[set[str], set[str]]:
    model = onnx.load(str(path), load_external_data=False)
    return {item.name for item in model.graph.input}, {item.name for item in model.graph.output}


def validate_schedule(schedule: dict, model_dir: Path) -> None:
    if schedule.get("format") != SCHEDULE_FORMAT:
        raise ValueError(f"Unsupported schedule format: {schedule.get('format')!r}")
    shards = schedule.get("shards")
    steps = schedule.get("steps")
    if not isinstance(shards, list) or not isinstance(steps, list):
        raise ValueError("schedule shards and steps must be arrays")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    virtual_source = str(manifest.get("weight_storage", "")).startswith("source_safetensors")
    graph_to_shard: dict[str, str] = {}
    for shard in shards:
        for graph in shard.get("graphs", []):
            if graph in graph_to_shard:
                raise ValueError(f"Graph appears in multiple shards: {graph}")
            graph_to_shard[graph] = shard["id"]
    inventory_names = {graph.name for graph in graph_inventory(model_dir)}
    if set(graph_to_shard) != inventory_names:
        missing = sorted(inventory_names - set(graph_to_shard))
        extra = sorted(set(graph_to_shard) - inventory_names)
        raise ValueError(f"Shard graph coverage mismatch; missing={missing}, extra={extra}")
    for expected_id, step in enumerate(steps):
        if step.get("id") != expected_id:
            raise ValueError(f"Step id mismatch at index {expected_id}")
        if step.get("phase") not in {"preamble", "denoise"}:
            raise ValueError(f"Invalid phase at step {expected_id}")
        inputs = step.get("inputs")
        outputs = step.get("outputs")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError(f"Step {expected_id} must have named inputs and outputs")
        for port, binding in inputs.items():
            if binding.get("source") not in _VALID_SOURCES or not binding.get("name"):
                raise ValueError(f"Invalid input binding {port!r} at step {expected_id}")
        for port, binding in outputs.items():
            if binding.get("target") not in _VALID_TARGETS:
                raise ValueError(f"Invalid output binding {port!r} at step {expected_id}")
            if binding.get("target") != "discard" and not binding.get("name"):
                raise ValueError(f"Output binding {port!r} needs a name at step {expected_id}")
        if step.get("kind") == "graph":
            graph = step.get("graph")
            if graph not in graph_to_shard or step.get("shard") != graph_to_shard[graph]:
                raise ValueError(f"Invalid graph/shard reference at step {expected_id}")
            expected_inputs, expected_outputs = _model_ports(model_dir / f"{graph}.onnx")
            schedule_inputs = set(inputs)
            inputs_match = schedule_inputs == expected_inputs
            if virtual_source and str(graph).startswith("main_block_"):
                # Ref2VA virtual topologies expose source-backed weights as
                # graph inputs; the persistent weight provider supplies them
                # alongside the schedule's activation feeds.
                graph_name = str(graph)
                if graph_name.endswith("_attention_qkv"):
                    functional_inputs = {
                        "hidden_states",
                        "timestep_embedding",
                        "modulation_ids",
                        "rotary_table",
                    }
                elif graph_name.endswith("_attention_output"):
                    functional_inputs = {
                        "hidden_states",
                        "attended",
                        "timestep_embedding",
                        "modulation_ids",
                    }
                elif graph_name.endswith("_mlp"):
                    functional_inputs = {"hidden_states", "timestep_embedding", "modulation_ids"}
                else:
                    functional_inputs = set()
                if "silu_timestep_embedding" in expected_inputs:
                    functional_inputs.add("silu_timestep_embedding")
                virtual_weight_inputs = expected_inputs - functional_inputs
                if schedule_inputs != functional_inputs or not virtual_weight_inputs:
                    raise ValueError(
                        f"Virtual graph port mismatch for {graph}: schedule inputs "
                        f"{sorted(schedule_inputs)} != functional inputs {sorted(functional_inputs)}; "
                        f"weight inputs={sorted(virtual_weight_inputs)}"
                    )
                inputs_match = True
            if not inputs_match or set(outputs) != expected_outputs:
                raise ValueError(
                    f"Graph port mismatch for {graph}: inputs {sorted(inputs)} != {sorted(expected_inputs)}, "
                    f"outputs {sorted(outputs)} != {sorted(expected_outputs)}"
                )
        elif step.get("kind") == "op":
            op = step.get("op")
            if op not in OP_PORTS:
                raise ValueError(f"Unknown op {op!r} at step {expected_id}")
            expected_inputs, expected_outputs = OP_PORTS[op]
            if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
                raise ValueError(f"Op port mismatch for {op} at step {expected_id}")
        else:
            raise ValueError(f"Invalid step kind at step {expected_id}")


def validate_runtime_schedule(schedule: dict, model_dir: Path) -> None:
    """Validate a published schedule without requiring source fine graphs."""
    if schedule.get("format") != SCHEDULE_FORMAT:
        raise ValueError(f"Unsupported schedule format: {schedule.get('format')!r}")
    shards = schedule.get("shards")
    steps = schedule.get("steps")
    if not isinstance(shards, list) or not isinstance(steps, list):
        raise ValueError("schedule shards and steps must be arrays")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    graph_to_shard: dict[str, str] = {}
    shard_ids: set[str] = set()
    for shard in shards:
        shard_id = shard.get("id")
        graphs = shard.get("graphs")
        if not isinstance(shard_id, str) or shard_id in shard_ids:
            raise ValueError(f"Invalid or duplicate shard id: {shard_id!r}")
        if not isinstance(graphs, list) or len(graphs) != 1:
            raise ValueError(f"Runtime shard {shard_id} must contain exactly one graph")
        shard_ids.add(shard_id)
        graph = graphs[0]
        if graph in graph_to_shard:
            raise ValueError(f"Graph appears in multiple shards: {graph}")
        graph_to_shard[graph] = shard_id
        filename = shard.get("file")
        path = model_dir / str(filename)
        artifact = artifacts.get(str(filename), {})
        if not path.is_file() or artifact.get("bytes") != path.stat().st_size:
            raise ValueError(f"Missing or size-mismatched shard artifact: {filename}")
        data_name = f"{filename}.data"
        data_path = model_dir / data_name
        data_artifact = artifacts.get(data_name)
        if data_artifact is not None and (
            not data_path.is_file() or data_artifact.get("bytes") != data_path.stat().st_size
        ):
            raise ValueError(f"Missing or size-mismatched shard artifact: {data_name}")

    referenced_graphs: set[str] = set()
    for expected_id, step in enumerate(steps):
        if step.get("id") != expected_id or step.get("phase") not in {"preamble", "denoise"}:
            raise ValueError(f"Invalid runtime step at index {expected_id}")
        inputs = step.get("inputs")
        outputs = step.get("outputs")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError(f"Step {expected_id} must have named inputs and outputs")
        if step.get("kind") == "graph":
            graph = step.get("graph")
            if graph not in graph_to_shard or step.get("shard") != graph_to_shard[graph]:
                raise ValueError(f"Invalid graph/shard reference at step {expected_id}")
            referenced_graphs.add(graph)
        elif step.get("kind") == "op":
            op = step.get("op")
            if op not in OP_PORTS:
                raise ValueError(f"Unknown op {op!r} at step {expected_id}")
            expected_inputs, expected_outputs = OP_PORTS[op]
            if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
                raise ValueError(f"Op port mismatch for {op} at step {expected_id}")
        else:
            raise ValueError(f"Invalid step kind at step {expected_id}")
    if referenced_graphs != set(graph_to_shard):
        raise ValueError("Runtime schedule contains unreferenced shard graphs")


def write_schedule(schedule: dict, path: Path) -> Path:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate an h3-schedule-v2 schedule")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--sdpa-scale", type=float, default=0.125)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--storage-target-bytes", type=int, default=SHARD_STORAGE_TARGET_BYTES)
    parser.add_argument("--materialized-target-bytes", type=int, default=SHARD_MATERIALIZED_TARGET_BYTES)
    args = parser.parse_args()
    inventory = graph_inventory(args.model)
    schedule = build_schedule(
        inventory,
        model_id=args.model.name,
        blocks=args.blocks,
        sdpa_scale=args.sdpa_scale,
        turbo=args.turbo,
        storage_target_bytes=args.storage_target_bytes,
        materialized_target_bytes=args.materialized_target_bytes,
    )
    validate_schedule(schedule, args.model)
    output = args.output or (args.model / "schedule.json")
    write_schedule(schedule, output)
    print(
        f"{len(inventory)} graphs, {len(schedule['shards'])} shards, "
        f"{schedule['resources']['activation_peak_slots']} activation slots -> {output}"
    )


if __name__ == "__main__":
    main()
