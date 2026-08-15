from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.device_profile import selected_device_index
from h3_workbench.inference_runtime import ORTGraphRunner


_INPUT_HIDDEN = "hidden_states"
_INPUT_ROTARY = "rotary_table"
_OUTPUT_HIDDEN = "hidden_states_out"
_CUDA_PROVIDER = "CUDAExecutionProvider"


def _resolve_device_id(device_id: int | None) -> int:
    return selected_device_index() if device_id is None else int(device_id)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, durable: bool = False, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 6),
            "event": event,
            **details,
        }
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
            if durable:
                os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()


def _shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_param:
            dimensions.append(dimension.dim_param)
        elif dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        else:
            dimensions.append(None)
    return dimensions


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor.elem_type),
        "shape": _shape(value),
    }


def inspect_block_model(model: onnx.ModelProto, path: Path | None = None) -> dict[str, Any]:
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    missing = [name for name in (_INPUT_HIDDEN, _INPUT_ROTARY) if name not in inputs]
    if _OUTPUT_HIDDEN not in outputs:
        missing.append(_OUTPUT_HIDDEN)
    if missing:
        raise ValueError(f"Video VAE block contract is missing: {', '.join(missing)}")

    hidden = inputs[_INPUT_HIDDEN]
    rotary = inputs[_INPUT_ROTARY]
    output = outputs[_OUTPUT_HIDDEN]
    hidden_shape = _shape(hidden)
    rotary_shape = _shape(rotary)
    output_shape = _shape(output)
    if len(hidden_shape) != 3 or not isinstance(hidden_shape[-1], int):
        raise ValueError(f"Unsupported hidden_states shape: {hidden_shape}")
    if len(rotary_shape) != 6 or not all(isinstance(value, int) for value in rotary_shape[2:]):
        raise ValueError(f"Unsupported rotary_table shape: {rotary_shape}")
    if len(output_shape) != 3:
        raise ValueError(f"Unsupported hidden_states_out shape: {output_shape}")

    input_batch_axes = (hidden_shape[0], rotary_shape[0])
    native_dynamic_batch = all(not isinstance(value, int) for value in input_batch_axes)
    return {
        "path": str(path.resolve()) if path is not None else None,
        "inputs": [_value_contract(hidden), _value_contract(rotary)],
        "outputs": [_value_contract(output)],
        "hidden_width": hidden_shape[-1],
        "rotary_tail": rotary_shape[2:],
        "native_dynamic_batch": native_dynamic_batch,
        "operators": dict(Counter(node.op_type for node in model.graph.node)),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "storage_bytes": path.stat().st_size if path is not None else None,
    }


def load_block_contract(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    try:
        return inspect_block_model(model, path)
    finally:
        del model
        gc.collect()


def _initializer_schema(model: onnx.ModelProto) -> list[dict[str, Any]]:
    return [
        {
            "name": initializer.name,
            "dtype": initializer.data_type,
            "dtype_name": TensorProto.DataType.Name(initializer.data_type),
            "shape": [int(value) for value in initializer.dims],
            "nbytes": int(np.prod(initializer.dims, dtype=np.int64))
            * np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type)).itemsize,
        }
        for initializer in model.graph.initializer
    ]


def block_schema_fingerprint(model: onnx.ModelProto) -> str:
    topology = {
        "inputs": [_value_contract(value) for value in model.graph.input],
        "outputs": [_value_contract(value) for value in model.graph.output],
        "initializers": [
            (item["name"], item["dtype"], item["shape"])
            for item in _initializer_schema(model)
        ],
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
    }
    encoded = json.dumps(topology, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_block_schemas(directory: Path, blocks: list[int]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("At least one schema block is required")
    records: dict[str, Any] = {}
    reference: str | None = None
    started = time.perf_counter()
    for index in blocks:
        path = directory / f"video_decoder_block_{index:02d}.onnx"
        if not path.is_file():
            raise FileNotFoundError(path)
        model = onnx.load(path, load_external_data=False)
        try:
            fingerprint = block_schema_fingerprint(model)
            initializers = _initializer_schema(model)
            if reference is None:
                reference = fingerprint
            records[str(index)] = {
                "fingerprint": fingerprint,
                "matches_reference": fingerprint == reference,
                "initializer_count": len(initializers),
                "initializer_bytes": sum(int(item["nbytes"]) for item in initializers),
                "nodes": len(model.graph.node),
            }
        finally:
            del model
            gc.collect()
    return {
        "blocks": records,
        "compatible": all(bool(record["matches_reference"]) for record in records.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def serialize_persistent_block_topology(
    model: onnx.ModelProto,
    dynamic_batch: bool = False,
) -> tuple[bytes, list[dict[str, Any]], str]:
    schema = _initializer_schema(model)
    if not schema:
        raise ValueError("Video VAE block has no initializers to promote")
    fingerprint = block_schema_fingerprint(model)
    existing_inputs = {value.name for value in model.graph.input}
    for initializer in model.graph.initializer:
        if initializer.name in existing_inputs:
            raise ValueError(f"Initializer already appears as a graph input: {initializer.name}")
        model.graph.input.append(
            helper.make_tensor_value_info(initializer.name, initializer.data_type, list(initializer.dims))
        )
    del model.graph.initializer[:]
    if model.graph.sparse_initializer:
        raise ValueError("Sparse initializers are not supported by the persistent Video VAE topology")
    if dynamic_batch:
        inputs = {value.name: value for value in model.graph.input}
        outputs = {value.name: value for value in model.graph.output}
        _set_symbolic_batch(inputs[_INPUT_HIDDEN])
        _set_symbolic_batch(inputs[_INPUT_ROTARY])
        _set_symbolic_batch(outputs[_OUTPUT_HIDDEN])
    model.graph.name = "h3_persistent_video_vae_block"
    onnx.checker.check_model(model, full_check=False)
    return model.SerializeToString(), schema, fingerprint


def load_initializer_feeds(
    path: Path,
    expected_schema: list[dict[str, Any]],
    expected_fingerprint: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    model = onnx.load(path, load_external_data=True)
    try:
        actual_schema = _initializer_schema(model)
        actual_signature = [
            (item["name"], item["dtype"], item["shape"])
            for item in actual_schema
        ]
        expected_signature = [
            (item["name"], item["dtype"], item["shape"])
            for item in expected_schema
        ]
        if actual_signature != expected_signature:
            raise ValueError(f"Persistent Video VAE weight schema mismatch in {path}")
        fingerprint = block_schema_fingerprint(model)
        if fingerprint != expected_fingerprint:
            raise ValueError(f"Persistent Video VAE topology mismatch in {path}")
        feeds = {
            initializer.name: np.array(
                numpy_helper.to_array(initializer, base_dir=str(path.parent)),
                copy=True,
                order="C",
            )
            for initializer in model.graph.initializer
        }
    finally:
        del model
        gc.collect()
    total_bytes = sum(array.nbytes for array in feeds.values())
    return feeds, {
        "load_seconds": round(time.perf_counter() - started, 6),
        "count": len(feeds),
        "total_bytes": total_bytes,
        "fingerprint": fingerprint,
    }


def _set_symbolic_batch(value: onnx.ValueInfoProto, symbol: str = "batch") -> None:
    dimensions = value.type.tensor_type.shape.dim
    if not dimensions:
        raise ValueError(f"Tensor {value.name} has no batch axis")
    dimensions[0].ClearField("dim_value")
    dimensions[0].dim_param = symbol


def serialize_dynamic_batch_model(model: onnx.ModelProto) -> bytes:
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    for name in (_INPUT_HIDDEN, _INPUT_ROTARY):
        if name not in inputs:
            raise ValueError(f"Video VAE block contract is missing input {name}")
        _set_symbolic_batch(inputs[name])
    if _OUTPUT_HIDDEN not in outputs:
        raise ValueError(f"Video VAE block contract is missing output {_OUTPUT_HIDDEN}")
    _set_symbolic_batch(outputs[_OUTPUT_HIDDEN])
    return model.SerializeToString()


def _numpy_dtype(dtype_name: str) -> type[np.generic]:
    mapping: dict[str, type[np.generic]] = {
        "FLOAT16": np.float16,
        "FLOAT": np.float32,
        "DOUBLE": np.float64,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark dtype: {dtype_name}") from exc


def make_block_inputs(
    contract: dict[str, Any],
    batch_size: int,
    sequence: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sequence < 1:
        raise ValueError("sequence must be positive")
    hidden_dtype = _numpy_dtype(contract["inputs"][0]["dtype"])
    rotary_dtype = _numpy_dtype(contract["inputs"][1]["dtype"])
    hidden_width = int(contract["hidden_width"])
    rotary_tail = tuple(int(value) for value in contract["rotary_tail"])
    random = np.random.default_rng(seed)
    hidden = (random.standard_normal((batch_size, sequence, hidden_width), dtype=np.float32) * 0.25).astype(
        hidden_dtype
    )
    angles = random.uniform(-np.pi, np.pi, size=(batch_size, sequence, 1, rotary_tail[1])).astype(np.float32)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotary = np.empty((batch_size, sequence, *rotary_tail), dtype=np.float32)
    rotary[..., 0, 0] = cosine
    rotary[..., 0, 1] = -sine
    rotary[..., 1, 0] = sine
    rotary[..., 1, 1] = cosine
    return {
        _INPUT_HIDDEN: np.ascontiguousarray(hidden),
        _INPUT_ROTARY: np.ascontiguousarray(rotary.astype(rotary_dtype)),
    }


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    expected32 = expected.astype(np.float32, copy=False)
    actual32 = actual.astype(np.float32, copy=False)
    if expected32.shape != actual32.shape or not np.isfinite(actual32).all():
        return {
            "finite": bool(np.isfinite(actual32).all()),
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "relative_l2": float("inf"),
            "cosine": float("nan"),
        }
    difference = actual32 - expected32
    expected_flat = expected32.reshape(-1)
    actual_flat = actual32.reshape(-1)
    expected_norm = float(np.linalg.norm(expected_flat))
    denominator = expected_norm * float(np.linalg.norm(actual_flat))
    return {
        "finite": True,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / max(expected_norm, np.finfo(np.float32).tiny)),
        "cosine": float(np.dot(expected_flat, actual_flat) / max(denominator, np.finfo(np.float32).tiny)),
    }


def _timed_runs(
    call: Callable[[], np.ndarray],
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, list[float], float, float]:
    cold_seconds: float | None = None
    for _ in range(warmup):
        started = time.perf_counter()
        call()
        if cold_seconds is None:
            cold_seconds = time.perf_counter() - started
    output: np.ndarray | None = None
    durations: list[float] = []
    cpu_started = time.process_time()
    for _ in range(repeats):
        started = time.perf_counter()
        output = call()
        duration = time.perf_counter() - started
        durations.append(duration)
        if cold_seconds is None:
            cold_seconds = duration
    cpu_seconds = time.process_time() - cpu_started
    if output is None or cold_seconds is None:
        raise ValueError("repeats must be positive")
    return output, durations, cpu_seconds, cold_seconds


def _timing_result(
    durations: list[float],
    cpu_seconds: float,
    batch_size: int,
    cold_seconds: float,
) -> dict[str, Any]:
    total = sum(durations)
    median = float(np.median(durations))
    return {
        "cold_seconds": round(cold_seconds, 6),
        "warm_seconds": [round(value, 6) for value in durations],
        "warm_median_seconds": round(median, 6),
        "warm_p90_seconds": round(float(np.percentile(durations, 90)), 6),
        "timed_wall_seconds": round(total, 6),
        "timed_process_cpu_seconds": round(cpu_seconds, 6),
        "process_cpu_to_wall_ratio": round(cpu_seconds / max(total, 1e-12), 4),
        "tiles_per_second": round(batch_size / max(median, 1e-12), 4),
        "median_seconds_per_tile": round(median / batch_size, 6),
    }


def run_host_baseline(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    def call() -> np.ndarray:
        return session.run([_OUTPUT_HIDDEN], feeds)[0]

    output, durations, cpu_seconds, cold_seconds = _timed_runs(call, warmup, repeats)
    result = {
        "status": "completed",
        "mode": "session_run_host",
        "timed_h2d_copies_per_call": 2,
        "timed_d2h_copies_per_call": 1,
        **_timing_result(durations, cpu_seconds, int(feeds[_INPUT_HIDDEN].shape[0]), cold_seconds),
        "output_shape": list(output.shape),
        "output_finite": bool(np.isfinite(output).all()),
    }
    return output, result


def run_io_binding(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    warmup: int,
    repeats: int,
    device_type: str = "cuda",
    device_id: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    device_id = _resolve_device_id(device_id)
    hidden = feeds[_INPUT_HIDDEN]
    rotary = feeds[_INPUT_ROTARY]
    setup_started = time.perf_counter()
    hidden_device = ort.OrtValue.ortvalue_from_numpy(hidden, device_type, device_id)
    rotary_device = ort.OrtValue.ortvalue_from_numpy(rotary, device_type, device_id)
    output_device = ort.OrtValue.ortvalue_from_shape_and_type(hidden.shape, hidden.dtype, device_type, device_id)
    binding = session.io_binding()
    binding.bind_ortvalue_input(_INPUT_HIDDEN, hidden_device)
    binding.bind_ortvalue_input(_INPUT_ROTARY, rotary_device)
    binding.bind_ortvalue_output(_OUTPUT_HIDDEN, output_device)
    binding.synchronize_inputs()
    setup_seconds = time.perf_counter() - setup_started

    def call() -> np.ndarray:
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        # Timed calls deliberately leave the activation on the selected device.
        return hidden

    _, durations, cpu_seconds, cold_seconds = _timed_runs(call, warmup, repeats)
    download_started = time.perf_counter()
    output = output_device.numpy()
    download_seconds = time.perf_counter() - download_started
    result = {
        "status": "completed",
        "mode": "io_binding_device_resident",
        "device_type": device_type,
        "device_id": device_id,
        "device_setup_seconds": round(setup_seconds, 6),
        "final_download_seconds": round(download_seconds, 6),
        "timed_h2d_copies_per_call": 0,
        "timed_d2h_copies_per_call": 0,
        "setup_h2d_copies": 2 if device_type != "cpu" else 0,
        "final_d2h_copies": 1 if device_type != "cpu" else 0,
        **_timing_result(durations, cpu_seconds, int(hidden.shape[0]), cold_seconds),
        "output_shape": list(output.shape),
        "output_finite": bool(np.isfinite(output).all()),
    }
    return output, result


def run_persistent_io_binding(
    session: ort.InferenceSession,
    activation_feeds: dict[str, np.ndarray],
    weight_feeds: dict[str, np.ndarray],
    warmup: int,
    repeats: int,
    device_type: str = "cuda",
    device_id: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    device_id = _resolve_device_id(device_id)
    hidden = activation_feeds[_INPUT_HIDDEN]
    rotary = activation_feeds[_INPUT_ROTARY]
    binding = session.io_binding()

    weight_upload_started = time.perf_counter()
    weight_devices: dict[str, ort.OrtValue] = {}
    for name, array in weight_feeds.items():
        weight_devices[name] = ort.OrtValue.ortvalue_from_numpy(array, device_type, device_id)
        binding.bind_ortvalue_input(name, weight_devices[name])
    binding.synchronize_inputs()
    weight_upload_seconds = time.perf_counter() - weight_upload_started

    activation_setup_started = time.perf_counter()
    hidden_device = ort.OrtValue.ortvalue_from_numpy(hidden, device_type, device_id)
    rotary_device = ort.OrtValue.ortvalue_from_numpy(rotary, device_type, device_id)
    output_device = ort.OrtValue.ortvalue_from_shape_and_type(hidden.shape, hidden.dtype, device_type, device_id)
    binding.bind_ortvalue_input(_INPUT_HIDDEN, hidden_device)
    binding.bind_ortvalue_input(_INPUT_ROTARY, rotary_device)
    binding.bind_ortvalue_output(_OUTPUT_HIDDEN, output_device)
    binding.synchronize_inputs()
    activation_setup_seconds = time.perf_counter() - activation_setup_started

    def call() -> np.ndarray:
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return hidden

    _, durations, cpu_seconds, cold_seconds = _timed_runs(call, warmup, repeats)
    download_started = time.perf_counter()
    output = output_device.numpy()
    download_seconds = time.perf_counter() - download_started
    weight_bytes = sum(array.nbytes for array in weight_feeds.values())
    result = {
        "status": "completed",
        "mode": "persistent_topology_device_weights",
        "device_type": device_type,
        "device_id": device_id,
        "weight_count": len(weight_feeds),
        "weight_bytes": weight_bytes,
        "weight_upload_seconds": round(weight_upload_seconds, 6),
        "weight_h2d_bytes": weight_bytes if device_type != "cpu" else 0,
        "activation_setup_seconds": round(activation_setup_seconds, 6),
        "activation_h2d_bytes": hidden.nbytes + rotary.nbytes if device_type != "cpu" else 0,
        "final_download_seconds": round(download_seconds, 6),
        "timed_h2d_copies_per_call": 0,
        "timed_d2h_copies_per_call": 0,
        **_timing_result(durations, cpu_seconds, int(hidden.shape[0]), cold_seconds),
        "output_shape": list(output.shape),
        "output_finite": bool(np.isfinite(output).all()),
    }
    return output, result


def _gpu_sample(device_id: int) -> dict[str, float | int] | None:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,power.draw",
                "--format=csv,noheader,nounits",
            ],
            creationflags=flags,
            text=True,
            timeout=3,
        )
        values = [part.strip() for part in output.strip().splitlines()[0].split(",")]
        return {
            "utilization_percent": int(values[0]),
            "memory_utilization_percent": int(values[1]),
            "memory_used_mib": int(values[2]),
            "memory_free_mib": int(values[3]),
            "power_watts": float(values[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _profile_sequence(directory: Path) -> int:
    path = directory / "video_decoder_prelude.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot infer the tile sequence without {path}")
    model = onnx.load(path, load_external_data=False)
    try:
        outputs = {value.name: value for value in model.graph.output}
        if _INPUT_HIDDEN not in outputs:
            raise ValueError(f"Prelude {path} does not expose {_INPUT_HIDDEN}")
        sequence = _shape(outputs[_INPUT_HIDDEN])[1]
        if not isinstance(sequence, int) or sequence < 1:
            raise ValueError(f"Prelude sequence is not static: {sequence}")
        return sequence
    finally:
        del model
        gc.collect()


def _validate_args(args: argparse.Namespace) -> None:
    if args.block < 0:
        raise ValueError("block must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not args.batch_sizes or any(batch < 1 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("batch sizes must be unique")
    if any(block < 0 for block in args.schema_blocks):
        raise ValueError("schema blocks must be non-negative")
    if len(set(args.schema_blocks)) != len(args.schema_blocks):
        raise ValueError("schema blocks must be unique")


def _c_drive_free_gib() -> float | None:
    if os.name != "nt":
        return None
    system_drive = os.environ.get("SystemDrive", "C:")
    return shutil.disk_usage(f"{system_drive}\\").free / 2**30


def run(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    runner: ORTGraphRunner | None = None
    persistent_session: ort.InferenceSession | None = None
    persistent_weights: dict[str, np.ndarray] | None = None
    process = psutil.Process()
    try:
        if args.device_id is None:
            args.device_id = selected_device_index()
        _validate_args(args)
        c_free_gib = _c_drive_free_gib()
        if c_free_gib is not None and c_free_gib < args.min_c_free_gib:
            raise RuntimeError(
                f"C: free space {c_free_gib:.2f} GiB is below the {args.min_c_free_gib:.2f} GiB guard"
            )
        directory = args.model.resolve()
        block_path = directory / f"video_decoder_block_{args.block:02d}.onnx"
        if not block_path.is_file():
            raise FileNotFoundError(block_path)
        sequence = args.sequence if args.sequence is not None else _profile_sequence(directory)
        logger.write(
            "preparing",
            durable=True,
            model=str(directory),
            block=args.block,
            persistent_topology=args.persistent_topology,
            schema_blocks=args.schema_blocks,
        )

        schema_check: dict[str, Any] | None = None
        if args.schema_blocks:
            schema_check = inspect_block_schemas(directory, args.schema_blocks)
            logger.write("schema_check", durable=True, **schema_check)
            if not schema_check["compatible"]:
                raise ValueError(f"Video VAE block schemas are incompatible: {args.schema_blocks}")

        model_load_started = time.perf_counter()
        model = onnx.load(block_path, load_external_data=False)
        contract = inspect_block_model(model, block_path)
        model_load_seconds = time.perf_counter() - model_load_started
        dynamic_model: bytes | None = None
        batch_patch_seconds: float | None = None
        if any(batch > 1 for batch in args.batch_sizes) and not contract["native_dynamic_batch"]:
            patch_started = time.perf_counter()
            dynamic_model = serialize_dynamic_batch_model(model)
            batch_patch_seconds = time.perf_counter() - patch_started
        del model
        gc.collect()

        persistent_topology: bytes | None = None
        persistent_schema: list[dict[str, Any]] | None = None
        persistent_fingerprint: str | None = None
        persistent_prepare_seconds: float | None = None
        if args.persistent_topology:
            persistent_prepare_started = time.perf_counter()
            topology_path = directory / "video_decoder_block_00.onnx"
            topology_model = onnx.load(topology_path, load_external_data=False)
            try:
                persistent_topology, persistent_schema, persistent_fingerprint = serialize_persistent_block_topology(
                    topology_model,
                    dynamic_batch=any(batch > 1 for batch in args.batch_sizes),
                )
            finally:
                del topology_model
                gc.collect()
            persistent_prepare_seconds = time.perf_counter() - persistent_prepare_started

        runner = ORTGraphRunner(prefer_cuda=not args.cpu, prefetch_depth=0)
        if not args.cpu and runner.provider != _CUDA_PROVIDER:
            raise RuntimeError("CUDAExecutionProvider is required unless --cpu is specified")

        persistent_session_build_seconds: float | None = None
        persistent_weight_load: dict[str, Any] | None = None
        if persistent_topology is not None:
            persistent_build_started = time.perf_counter()
            persistent_session = runner.session(serialized_model=persistent_topology)
            persistent_session_build_seconds = time.perf_counter() - persistent_build_started
            if persistent_schema is None or persistent_fingerprint is None:
                raise RuntimeError("Persistent topology metadata was not prepared")
            persistent_weights, persistent_weight_load = load_initializer_feeds(
                block_path,
                persistent_schema,
                persistent_fingerprint,
            )
            logger.write(
                "persistent_topology_ready",
                durable=True,
                topology_bytes=len(persistent_topology),
                topology_prepare_seconds=round(persistent_prepare_seconds or 0.0, 6),
                session_build_seconds=round(persistent_session_build_seconds, 6),
                weight_load=persistent_weight_load,
            )
        logger.write(
            "started",
            durable=True,
            pid=os.getpid(),
            model=str(directory),
            block=args.block,
            block_path=str(block_path),
            provider=runner.provider,
            sequence=sequence,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            contract=contract,
            model_load_seconds=round(model_load_seconds, 6),
            dynamic_batch_patch_seconds=round(batch_patch_seconds, 6) if batch_patch_seconds is not None else None,
            dynamic_batch_model_bytes=len(dynamic_model) if dynamic_model is not None else 0,
            persistent_topology=args.persistent_topology,
            persistent_topology_bytes=len(persistent_topology) if persistent_topology is not None else 0,
            persistent_topology_prepare_seconds=(
                round(persistent_prepare_seconds, 6) if persistent_prepare_seconds is not None else None
            ),
            persistent_session_build_seconds=(
                round(persistent_session_build_seconds, 6)
                if persistent_session_build_seconds is not None
                else None
            ),
            persistent_weight_load=persistent_weight_load,
            schema_check=schema_check,
            c_free_gib=round(c_free_gib, 3) if c_free_gib is not None else None,
        )

        completed: dict[str, Any] = {}
        for batch_index, batch_size in enumerate(args.batch_sizes):
            strategy = "native" if batch_size == 1 or contract["native_dynamic_batch"] else "in_memory_batch_axis_relaxation"
            logger.write("batch_started", durable=True, batch_size=batch_size, batch_strategy=strategy)
            feeds = make_block_inputs(contract, batch_size, sequence, args.seed + batch_size)
            rss_before = process.memory_info().rss
            gpu_before = _gpu_sample(args.device_id) if runner.provider == _CUDA_PROVIDER else None
            session: ort.InferenceSession | None = None
            try:
                build_started = time.perf_counter()
                if strategy == "in_memory_batch_axis_relaxation":
                    if dynamic_model is None:
                        raise RuntimeError("Dynamic-batch model was not prepared")
                    session = runner.session(serialized_model=dynamic_model)
                else:
                    session = runner.session(block_path)
                build_seconds = time.perf_counter() - build_started
                baseline_output, baseline = run_host_baseline(session, feeds, args.warmup, args.repeats)
                baseline["session_build_plus_cold_run_seconds"] = round(
                    build_seconds + float(baseline["cold_seconds"]),
                    6,
                )
                if runner.provider == _CUDA_PROVIDER:
                    bound_output, io_binding = run_io_binding(
                        session,
                        feeds,
                        args.warmup,
                        args.repeats,
                        device_type="cuda",
                        device_id=args.device_id,
                    )
                    io_binding["metrics_vs_session_run"] = _metrics(baseline_output, bound_output)
                    io_binding["session_build_device_setup_cold_run_seconds"] = round(
                        build_seconds
                        + float(io_binding["device_setup_seconds"])
                        + float(io_binding["cold_seconds"]),
                        6,
                    )
                else:
                    io_binding = {
                        "status": "skipped",
                        "mode": "io_binding_device_resident",
                        "reason": "CUDAExecutionProvider is not active",
                    }
                original_providers = session.get_providers()
                del session
                session = None
                gc.collect()

                persistent: dict[str, Any]
                if persistent_session is None or persistent_weights is None:
                    persistent = {
                        "status": "disabled",
                        "mode": "persistent_topology_device_weights",
                    }
                else:
                    try:
                        persistent_output, persistent = run_persistent_io_binding(
                            persistent_session,
                            feeds,
                            persistent_weights,
                            args.warmup,
                            args.repeats,
                            device_type="cuda" if runner.provider == _CUDA_PROVIDER else "cpu",
                            device_id=args.device_id,
                        )
                        metrics = _metrics(baseline_output, persistent_output)
                        validation_passed = bool(
                            metrics["finite"]
                            and float(metrics["relative_l2"]) <= 1e-3
                            and float(metrics["cosine"]) >= 0.9999
                        )
                        persistent["metrics_vs_original"] = metrics
                        persistent["validation_passed"] = validation_passed
                        persistent["topology_session_built_once"] = True
                        persistent["topology_session_reused"] = batch_index > 0
                        persistent["topology_session_build_seconds"] = round(
                            persistent_session_build_seconds or 0.0,
                            6,
                        )
                        persistent["host_weight_load"] = persistent_weight_load
                        weight_upload_plus_run = (
                            float(persistent["weight_upload_seconds"])
                            + float(persistent["cold_seconds"])
                        )
                        persistent["steady_weight_upload_plus_cold_run_seconds"] = round(
                            weight_upload_plus_run,
                            6,
                        )
                        standalone_device_ready = weight_upload_plus_run + float(
                            persistent["activation_setup_seconds"]
                        )
                        persistent["standalone_activation_weight_upload_cold_run_seconds"] = round(
                            standalone_device_ready,
                            6,
                        )
                        persistent["first_session_build_activation_weight_upload_cold_run_seconds"] = round(
                            (persistent_session_build_seconds or 0.0) + standalone_device_ready,
                            6,
                        )
                        persistent["first_validated_pipeline_seconds"] = round(
                            (persistent_session_build_seconds or 0.0)
                            + standalone_device_ready
                            + float(persistent["final_download_seconds"]),
                            6,
                        )
                        if not validation_passed:
                            persistent["status"] = "validation_failed"
                    except Exception as exc:  # noqa: BLE001 - persistent topology is an experimental candidate
                        persistent = {
                            "status": "unsupported",
                            "mode": "persistent_topology_device_weights",
                            "error": str(exc),
                            "traceback": "".join(traceback.format_exception(exc)),
                        }
                result = {
                    "status": "completed",
                    "batch_size": batch_size,
                    "batch_strategy": strategy,
                    "runtime_batch_verified": list(baseline_output.shape)[0] == batch_size,
                    "session_build_seconds": round(build_seconds, 6),
                    "providers": original_providers,
                    "input_bytes": sum(array.nbytes for array in feeds.values()),
                    "output_bytes": baseline_output.nbytes,
                    "baseline": baseline,
                    "io_binding": io_binding,
                    "persistent_topology": persistent,
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": process.memory_info().rss,
                    "gpu_before": gpu_before,
                    "gpu_after": _gpu_sample(args.device_id) if runner.provider == _CUDA_PROVIDER else None,
                }
            except Exception as exc:  # noqa: BLE001 - unsupported batch candidates are benchmark results
                result = {
                    "status": "unsupported",
                    "batch_size": batch_size,
                    "batch_strategy": strategy,
                    "error": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": process.memory_info().rss,
                    "gpu_before": gpu_before,
                    "gpu_after": _gpu_sample(args.device_id) if runner.provider == _CUDA_PROVIDER else None,
                }
            finally:
                if session is not None:
                    del session
                del feeds
                gc.collect()
            completed[str(batch_size)] = result
            logger.write("batch_completed", durable=True, **result)

        logger.write("completed", durable=True, batches=completed)
        print(json.dumps({"status": "completed", "batches": completed}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - every benchmark failure must reach durable storage
        logger.write(
            "failed",
            durable=True,
            error=str(exc),
            traceback="".join(traceback.format_exception(exc)),
        )
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        if persistent_session is not None:
            del persistent_session
        if persistent_weights is not None:
            persistent_weights.clear()
        if runner is not None:
            runner.close()
        logger.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one Video VAE decoder block and device-resident I/O")
    parser.add_argument("--model", type=Path, required=True, help="Video VAE ONNX product directory")
    parser.add_argument("--log", type=Path, required=True, help="Append-only JSONL result path")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--sequence", type=int, help="Defaults to video_decoder_prelude.onnx output sequence")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="CUDA device index; defaults to H3_CUDA_DEVICE selection",
    )
    parser.add_argument("--persistent-topology", action="store_true")
    parser.add_argument(
        "--schema-blocks",
        type=int,
        nargs="+",
        default=[],
        help="Optionally verify reusable topology fingerprints, for example 0 1 35",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPUExecutionProvider for functional smoke tests")
    parser.add_argument("--min-c-free-gib", type=float, default=90.0)
    return parser


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
