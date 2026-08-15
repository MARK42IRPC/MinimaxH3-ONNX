from __future__ import annotations

import os
import json
import time
import ctypes
import ctypes.util
import copy
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PERSISTENT_TOPOLOGIES = {
    "attention_qkv": "runtime_persistent_attention_qkv.onnx",
    "mlp": "runtime_persistent_scaled_mlp.onnx",
    "attention_output": "runtime_persistent_scaled_attention_output.onnx",
}


def device_prefetch_admitted(
    free_bytes: int,
    weight_bytes: int,
    reserve_bytes: int,
    inflight_bytes: int = 0,
) -> bool:
    return weight_bytes > 0 and free_bytes >= weight_bytes + reserve_bytes + inflight_bytes


@dataclass(frozen=True)
class WeightInput:
    name: str
    dtype: int
    shape: tuple[int, ...]
    offset: int
    length: int
    location: str | None
    inline_proto: bytes | None = None

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(
            helper.tensor_dtype_to_np_dtype(self.dtype)
        ).itemsize


def _external_fields(initializer: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def initializer_weight_inputs(model_path: Path, include_inline: bool = False) -> list[WeightInput]:
    model_path = model_path.resolve()
    model = onnx.load(str(model_path), load_external_data=False)
    result: list[WeightInput] = []
    for initializer in model.graph.initializer:
        external = initializer.data_location == TensorProto.EXTERNAL
        if not external and not include_inline:
            continue
        fields = _external_fields(initializer) if external else {}
        item = WeightInput(
            name=initializer.name,
            dtype=initializer.data_type,
            shape=tuple(int(value) for value in initializer.dims),
            offset=int(fields.get("offset", 0)),
            length=(
                int(fields["length"])
                if external
                else int(np.prod(initializer.dims, dtype=np.int64))
                * np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type)).itemsize
            ),
            location=fields.get("location"),
            inline_proto=None if external else initializer.SerializeToString(),
        )
        if item.length != item.nbytes:
            raise ValueError(
                f"External initializer length mismatch for {item.name}: {item.length} != {item.nbytes}"
            )
        result.append(item)
    return result


def external_weight_inputs(model_path: Path) -> list[WeightInput]:
    result = initializer_weight_inputs(model_path)
    if not result:
        raise ValueError(f"No external weight initializers in {model_path}")
    return result


def _rename_graph_values(model: onnx.ModelProto, names: dict[str, str]) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = names.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = names.get(name, name)
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            value.name = names.get(value.name, value.name)


def _extract_selected_if_branch(model: onnx.ModelProto) -> None:
    if_nodes = [node for node in model.graph.node if node.op_type == "If"]
    if len(if_nodes) != 1:
        raise ValueError(f"Expected exactly one If node, found {len(if_nodes)}")
    if_node = if_nodes[0]
    branches = {
        attribute.name: attribute.g
        for attribute in if_node.attribute
        if attribute.type == onnx.AttributeProto.GRAPH
    }
    branch = branches.get("then_branch")
    if branch is None:
        raise ValueError("Single-graph shard If node has no then_branch")

    captures = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
    }
    nodes = copy.deepcopy(list(branch.node))
    for node in nodes:
        for index, name in enumerate(node.input):
            node.input[index] = captures.get(name, name)

    graph_input_names = {value.name for value in model.graph.input}
    selector_inputs = {
        name
        for node in model.graph.node
        if node.op_type == "Equal"
        for name in node.input
        if name in graph_input_names
    }
    inputs = [
        copy.deepcopy(value)
        for value in model.graph.input
        if value.name not in selector_inputs
    ]
    graph = helper.make_graph(
        nodes,
        f"{branch.name}_selector_free",
        inputs,
        copy.deepcopy(list(branch.output)),
        copy.deepcopy(list(model.graph.initializer)),
        doc_string=branch.doc_string,
        value_info=copy.deepcopy(list(branch.value_info)),
    )
    model.graph.CopyFrom(graph)


def build_persistent_topology(
    source_path: Path,
    output_path: Path,
    canonical_outputs: bool = False,
    all_initializers: bool = False,
    selector_free_if: bool = False,
) -> list[WeightInput]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    model = onnx.load(str(source_path), load_external_data=False)
    weights = initializer_weight_inputs(source_path, include_inline=all_initializers)
    if not weights:
        raise ValueError(f"No weight initializers in {source_path}")
    if selector_free_if:
        _extract_selected_if_branch(model)
    weight_names = {item.name for item in weights}
    retained = [item for item in model.graph.initializer if item.name not in weight_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    existing_inputs = {item.name for item in model.graph.input}
    for item in weights:
        if item.name in existing_inputs:
            raise ValueError(f"Initializer already appears as graph input: {item.name}")
        model.graph.input.append(
            helper.make_tensor_value_info(item.name, item.dtype, list(item.shape))
        )
    if canonical_outputs:
        output_names = {
            output.name: output.name.split("/", 1)[1]
            for output in model.graph.output
            if "/" in output.name
        }
        _rename_graph_values(model, output_names)
    model.graph.name = f"{model.graph.name}_persistent_weights"
    onnx.checker.check_model(model, full_check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    onnx.save_model(model, str(temporary))
    os.replace(temporary, output_path)
    return weights


class MappedWeights:
    def __init__(self, model_path: Path, topology_inputs: list[WeightInput]) -> None:
        self.model_path = model_path.resolve()
        actual = external_weight_inputs(self.model_path)
        expected_signatures = [(item.dtype, item.shape) for item in topology_inputs]
        actual_signatures = [(item.dtype, item.shape) for item in actual]
        if actual_signatures != expected_signatures:
            raise ValueError(f"Weight topology mismatch in {self.model_path}")
        self.arrays: dict[str, np.memmap[Any, Any]] = {}
        for expected, item in zip(topology_inputs, actual, strict=True):
            if item.location is None:
                raise ValueError(f"Expected external initializer for {item.name}")
            data_path = (self.model_path.parent / item.location).resolve()
            if data_path.parent != self.model_path.parent:
                raise ValueError(f"External data escapes model directory: {item.location}")
            if not data_path.is_file() or data_path.stat().st_size < item.offset + item.length:
                raise ValueError(f"External data range is missing for {item.name}: {data_path}")
            self.arrays[expected.name] = np.memmap(
                data_path,
                dtype=helper.tensor_dtype_to_np_dtype(item.dtype),
                mode="r",
                offset=item.offset,
                shape=item.shape,
                order="C",
            )
        self.total_bytes = sum(item.length for item in actual)

    def feeds(self) -> dict[str, np.ndarray]:
        return dict(self.arrays)

    def close(self) -> None:
        self.arrays.clear()


class _HostBuffer:
    def __init__(self, nbytes: int, cuda_runtime: Any | None) -> None:
        self.nbytes = nbytes
        self.cuda_runtime = cuda_runtime
        self.pointer = ctypes.c_void_p()
        self.raw: Any | None = None
        if cuda_runtime is not None:
            error = cuda_runtime.cudaHostAlloc(
                ctypes.byref(self.pointer),
                nbytes,
                1,  # cudaHostAllocPortable: the prefetch and ORT threads share it.
            )
            if error:
                raise OSError(f"cudaHostAlloc failed with CUDA error {error}")
            self.raw = (ctypes.c_ubyte * nbytes).from_address(self.pointer.value)
            self.array = np.ctypeslib.as_array(self.raw)
        else:
            self.array = np.empty(nbytes, dtype=np.uint8)

    @property
    def pinned(self) -> bool:
        return self.cuda_runtime is not None

    def close(self) -> None:
        if self.cuda_runtime is not None and self.pointer.value:
            error = self.cuda_runtime.cudaFreeHost(self.pointer)
            if error:
                raise OSError(f"cudaFreeHost failed with CUDA error {error}")
            self.pointer = ctypes.c_void_p()
        self.array = np.empty(0, dtype=np.uint8)
        self.raw = None


class DeviceWeightFeed:
    def __init__(self, pointer: int, dtype: np.dtype, shape: tuple[int, ...]) -> None:
        self.pointer = pointer
        self.dtype = dtype
        self.shape = shape

    def bind_input(self, binding: Any, name: str) -> None:
        binding.bind_input(name, "cuda", 0, self.dtype, self.shape, self.pointer)


class _CudaDeviceBuffer:
    def __init__(self, cuda_runtime: Any, nbytes: int) -> None:
        self.cuda_runtime = cuda_runtime
        self.nbytes = nbytes
        self.pointer = ctypes.c_void_p()
        error = cuda_runtime.cudaMalloc(ctypes.byref(self.pointer), nbytes)
        if error:
            raise OSError(f"cudaMalloc failed with CUDA error {error}")

    def copy_from(self, offset: int, array: np.ndarray) -> None:
        error = self.cuda_runtime.cudaMemcpy(
            ctypes.c_void_p(int(self.pointer.value) + offset),
            ctypes.c_void_p(array.ctypes.data),
            array.nbytes,
            1,  # cudaMemcpyHostToDevice
        )
        if error:
            raise OSError(f"cudaMemcpy failed with CUDA error {error}")

    def close(self) -> None:
        if self.pointer.value:
            error = self.cuda_runtime.cudaFree(self.pointer)
            if error:
                raise OSError(f"cudaFree failed with CUDA error {error}")
            self.pointer = ctypes.c_void_p()


class _HostBufferLease:
    def __init__(self, pool: HostWeightPool, key: tuple[str, int], buffer: _HostBuffer) -> None:
        self.pool = pool
        self.key = key
        self.buffer = buffer
        self.released = False

    @property
    def array(self) -> np.ndarray:
        return self.buffer.array

    @property
    def pinned(self) -> bool:
        return self.buffer.pinned

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.pool.release(self.key)


def _load_cuda_runtime() -> Any | None:
    candidates: list[str] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(str(Path(cuda_path) / "bin" / "cudart64_12.dll"))
    candidates.append(
        str(
            Path(sys.prefix)
            / "Lib"
            / "site-packages"
            / "nvidia"
            / "cuda_runtime"
            / "bin"
            / "cudart64_12.dll"
        )
    )
    discovered = ctypes.util.find_library("cudart64_12")
    if discovered:
        candidates.append(discovered)
    candidates.append("cudart64_12.dll")
    for candidate in dict.fromkeys(candidates):
        try:
            runtime = ctypes.WinDLL(candidate)
        except OSError:
            continue
        runtime.cudaHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        runtime.cudaHostAlloc.restype = ctypes.c_int
        runtime.cudaFreeHost.argtypes = [ctypes.c_void_p]
        runtime.cudaFreeHost.restype = ctypes.c_int
        runtime.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        runtime.cudaMemGetInfo.restype = ctypes.c_int
        runtime.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        runtime.cudaMalloc.restype = ctypes.c_int
        runtime.cudaFree.argtypes = [ctypes.c_void_p]
        runtime.cudaFree.restype = ctypes.c_int
        runtime.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        runtime.cudaMemcpy.restype = ctypes.c_int
        return runtime
    return None


class HostWeightPool:
    """One reusable sequential-read buffer per persistent graph kind."""

    def __init__(self, use_pinned: bool) -> None:
        self._condition = threading.Condition()
        self._buffers: dict[tuple[str, int], _HostBuffer] = {}
        self._in_use: set[tuple[str, int]] = set()
        self._cuda_runtime = _load_cuda_runtime() if use_pinned and os.name == "nt" else None
        self.pinned_requested = use_pinned
        self.fallback_reason = (
            "CUDA runtime not found" if use_pinned and self._cuda_runtime is None else None
        )
        self.allocations = 0

    @property
    def pinned_enabled(self) -> bool:
        return self._cuda_runtime is not None

    def acquire(self, key: tuple[str, int], nbytes: int) -> _HostBufferLease:
        with self._condition:
            while key in self._in_use:
                self._condition.wait()
            buffer = self._buffers.get(key)
            if buffer is not None and buffer.nbytes != nbytes:
                buffer.close()
                del self._buffers[key]
                buffer = None
            if buffer is None:
                try:
                    buffer = _HostBuffer(nbytes, self._cuda_runtime)
                except OSError as exc:
                    self.fallback_reason = str(exc)
                    self._cuda_runtime = None
                    buffer = _HostBuffer(nbytes, None)
                self._buffers[key] = buffer
                self.allocations += 1
            self._in_use.add(key)
            return _HostBufferLease(self, key, buffer)

    def release(self, key: tuple[str, int]) -> None:
        with self._condition:
            self._in_use.discard(key)
            self._condition.notify_all()

    def device_memory_info(self) -> tuple[int, int] | None:
        if self._cuda_runtime is None:
            return None
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        error = self._cuda_runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return None if error else (int(free.value), int(total.value))

    @property
    def cuda_runtime(self) -> Any | None:
        return self._cuda_runtime

    def close(self) -> None:
        with self._condition:
            if self._in_use:
                raise RuntimeError("Cannot close host weight pool while buffers are in use")
            buffers = list(self._buffers.values())
            self._buffers.clear()
        for buffer in buffers:
            buffer.close()


class LoadedWeights:
    """Resident, sequentially-loaded views over one graph's external data."""

    def __init__(
        self,
        model_path: Path,
        topology_inputs: list[WeightInput],
        host_pool: HostWeightPool | None = None,
        pool_key: str | None = None,
    ) -> None:
        started = time.perf_counter()
        self.model_path = model_path.resolve()
        actual = initializer_weight_inputs(
            self.model_path,
            include_inline=any(item.inline_proto is not None for item in topology_inputs),
        )
        expected_signatures = [(item.dtype, item.shape) for item in topology_inputs]
        actual_signatures = [(item.dtype, item.shape) for item in actual]
        if actual_signatures != expected_signatures:
            raise ValueError(f"Weight topology mismatch in {self.model_path}")

        self._storage: dict[str, np.ndarray] = {}
        self._leases: list[_HostBufferLease] = []
        self._device_arrays: dict[str, DeviceWeightFeed] = {}
        self._device_buffer: _CudaDeviceBuffer | None = None
        self.device_bytes = 0
        self.upload_seconds = 0.0
        locations = list(
            dict.fromkeys(item.location for item in actual if item.location is not None)
        )
        for location_index, location in enumerate(locations):
            assert location is not None
            data_path = (self.model_path.parent / location).resolve()
            if data_path.parent != self.model_path.parent:
                raise ValueError(f"External data escapes model directory: {location}")
            required = max(
                item.offset + item.length for item in actual if item.location == location
            )
            if not data_path.is_file() or data_path.stat().st_size < required:
                raise ValueError(f"External data is missing or truncated: {data_path}")
            if host_pool is None:
                # A bulk read faults pages in sequentially before ORT begins H2D copies.
                self._storage[location] = np.fromfile(data_path, dtype=np.uint8)
            else:
                lease = host_pool.acquire(
                    (pool_key or str(self.model_path), location_index),
                    data_path.stat().st_size,
                )
                try:
                    view = memoryview(lease.array).cast("B")
                    offset = 0
                    with data_path.open("rb", buffering=0) as handle:
                        while offset < len(view):
                            read = handle.readinto(view[offset:])
                            if not read:
                                raise OSError(f"Short read from external data: {data_path}")
                            offset += read
                except Exception:
                    lease.release()
                    for retained in self._leases:
                        retained.release()
                    self._leases.clear()
                    self._storage.clear()
                    raise
                self._leases.append(lease)
                self._storage[location] = lease.array

        self.arrays: dict[str, np.ndarray] = {}
        try:
            for expected, item in zip(topology_inputs, actual, strict=True):
                if item.location is not None:
                    value = np.ndarray(
                        item.shape,
                        dtype=helper.tensor_dtype_to_np_dtype(item.dtype),
                        buffer=self._storage[item.location],
                        offset=item.offset,
                        order="C",
                    )
                else:
                    if item.inline_proto is None:
                        raise ValueError(f"Missing inline initializer data for {item.name}")
                    value = np.array(
                        numpy_helper.to_array(TensorProto.FromString(item.inline_proto)),
                        copy=True,
                        order="C",
                    )
                self.arrays[expected.name] = value
        except Exception:
            self.arrays.clear()
            self._storage.clear()
            for lease in self._leases:
                lease.release()
            self._leases.clear()
            raise
        self.total_bytes = sum(item.length for item in actual)
        self.host_bytes = sum(array.nbytes for array in self._storage.values()) + sum(
            item.length for item in actual if item.location is None
        )
        self.pinned_bytes = sum(
            lease.array.nbytes for lease in self._leases if lease.pinned
        )
        self.load_seconds = time.perf_counter() - started

    @property
    def device_resident(self) -> bool:
        return bool(self._device_arrays)

    def promote_to_device(self, cuda_runtime: Any) -> None:
        if self._device_arrays:
            return
        started = time.perf_counter()
        alignment = 256
        offsets: dict[str, int] = {}
        total = 0
        for name, array in self.arrays.items():
            total = (total + alignment - 1) // alignment * alignment
            offsets[name] = total
            total += array.nbytes
        buffer = _CudaDeviceBuffer(cuda_runtime, total)
        uploaded: dict[str, DeviceWeightFeed] = {}
        try:
            for name, array in self.arrays.items():
                contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
                offset = offsets[name]
                buffer.copy_from(offset, contiguous)
                uploaded[name] = DeviceWeightFeed(
                    int(buffer.pointer.value) + offset,
                    contiguous.dtype,
                    tuple(contiguous.shape),
                )
        except Exception:
            uploaded.clear()
            buffer.close()
            raise
        self._device_arrays = uploaded
        self._device_buffer = buffer
        self.device_bytes = sum(array.nbytes for array in self.arrays.values())
        self.upload_seconds = time.perf_counter() - started
        self.arrays.clear()
        self._storage.clear()
        for lease in self._leases:
            lease.release()
        self._leases.clear()

    def feeds(self) -> dict[str, Any]:
        return dict(self._device_arrays or self.arrays)

    def close(self) -> None:
        self._device_arrays.clear()
        if self._device_buffer is not None:
            self._device_buffer.close()
            self._device_buffer = None
        self.arrays.clear()
        self._storage.clear()
        for lease in self._leases:
            lease.release()
        self._leases.clear()


def _graph_kind(graph: str) -> str | None:
    if graph.startswith("main_token_refiner_block_") and graph.endswith("_attention"):
        return "refiner_attention"
    if graph.startswith("main_token_refiner_block_") and graph.endswith("_mlp"):
        return "refiner_mlp"
    if graph == "main_head":
        return "head"
    if graph.startswith("main_block_") and graph.endswith("_attention_qkv"):
        return "attention_qkv"
    if graph.startswith("main_block_") and graph.endswith("_attention_output"):
        return "attention_output"
    if graph.startswith("main_block_") and graph.endswith("_mlp"):
        return "mlp"
    return None


class PersistentWeightRuntime:
    def __init__(
        self,
        directory: Path,
        runner: Any,
        graph_paths: dict[str, Path],
        prefetch_depth: int = 2,
        prefetch_workers: int | None = None,
        topology_paths: dict[str, Path] | None = None,
    ) -> None:
        self.directory = directory.resolve()
        self.runner = runner
        self.graph_paths = {
            graph: path.resolve()
            for graph, path in graph_paths.items()
            if _graph_kind(graph) is not None
        }
        representatives: dict[str, Path] = {}
        for graph, path in self.graph_paths.items():
            representatives.setdefault(str(_graph_kind(graph)), path)
        overrides = {
            kind: Path(path).resolve()
            for kind, path in (topology_paths or {}).items()
        }
        self._topology_paths: dict[str, Path] = {}
        for kind in representatives:
            topology = overrides.get(kind)
            if topology is None and kind in PERSISTENT_TOPOLOGIES:
                topology = self.directory / PERSISTENT_TOPOLOGIES[kind]
            if topology is not None and topology.is_file():
                self._topology_paths[kind] = topology
        self.topology_inputs = {
            kind: initializer_weight_inputs(
                path,
                include_inline=kind == "attention_qkv" or kind in overrides,
            )
            for kind, path in representatives.items()
            if kind in self._topology_paths
        }
        self._sessions: dict[str, Any] = {}
        self._weights: dict[str, LoadedWeights] = {}
        self._pending: dict[str, Future[LoadedWeights]] = {}
        self._prefetch_depth = max(0, prefetch_depth)
        requested_workers = 2 if prefetch_workers is None else prefetch_workers
        self._prefetch_workers = max(
            1,
            min(requested_workers, self._prefetch_depth or 1, len(self.topology_inputs) or 1),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._prefetch_workers,
            thread_name_prefix="h3-weight-prefetch",
        )
        use_pinned = (
            getattr(runner, "provider", "CPUExecutionProvider") == "CUDAExecutionProvider"
            and os.environ.get("H3_PINNED_WEIGHTS", "1") != "0"
        )
        self._host_pool = HostWeightPool(use_pinned=use_pinned)
        device_setting = os.environ.get("H3_DEVICE_WEIGHT_PREFETCH", "auto").strip().lower()
        if device_setting not in {"auto", "0", "1"}:
            raise ValueError("H3_DEVICE_WEIGHT_PREFETCH must be auto, 0, or 1")
        memory = self._host_pool.device_memory_info()
        self._device_total_bytes = memory[1] if memory is not None else 0
        self._device_min_total_bytes = max(
            0,
            int(os.environ.get("H3_DEVICE_WEIGHT_MIN_VRAM_GIB", "6")) * 1024**3,
        )
        self._device_prefetch_enabled = bool(
            use_pinned
            and (
                device_setting == "1"
                or (
                    device_setting == "auto"
                    and self._device_total_bytes >= self._device_min_total_bytes
                )
            )
        )
        self._device_reserve_bytes = max(
            0,
            int(os.environ.get("H3_DEVICE_WEIGHT_RESERVE_MIB", "384")) * 1024**2,
        )
        self._device_lock = threading.Lock()
        self._device_inflight_bytes = 0
        self._device_active_slots = 0
        self._device_inflight_slots = 0
        self._device_slot_limit = max(
            1,
            int(os.environ.get("H3_DEVICE_WEIGHT_PREFETCH_SLOTS", "1")),
        )
        self._device_metrics = {
            "persistent_device_prefetch_attempts": 0,
            "persistent_device_prefetch_hits": 0,
            "persistent_device_prefetch_skips": 0,
            "persistent_device_prefetch_failures": 0,
            "persistent_device_prefetch_bytes": 0,
            "persistent_device_prefetch_seconds": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.topology_inputs)

    @property
    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    @property
    def prefetch_workers(self) -> int:
        return self._prefetch_workers

    @property
    def pinned_enabled(self) -> bool:
        return self._host_pool.pinned_enabled

    @property
    def pinned_fallback_reason(self) -> str | None:
        return self._host_pool.fallback_reason

    @property
    def host_pool_allocations(self) -> int:
        return self._host_pool.allocations

    @property
    def device_metrics(self) -> dict[str, int | float | bool]:
        with self._device_lock:
            return {
                **self._device_metrics,
                "persistent_device_prefetch_enabled": self._device_prefetch_enabled,
                "persistent_device_total_bytes": self._device_total_bytes,
                "persistent_device_prefetch_min_total_bytes": self._device_min_total_bytes,
                "persistent_device_prefetch_reserve_bytes": self._device_reserve_bytes,
                "persistent_device_prefetch_slot_limit": self._device_slot_limit,
            }

    def _load_weights(self, graph: str, kind: str) -> LoadedWeights:
        loaded = LoadedWeights(
            self.graph_paths[graph],
            self.topology_inputs[kind],
            self._host_pool,
            kind,
        )
        if not self._device_prefetch_enabled:
            return loaded
        required = loaded.total_bytes
        with self._device_lock:
            self._device_metrics["persistent_device_prefetch_attempts"] += 1
            memory = self._host_pool.device_memory_info()
            slots_full = (
                self._device_active_slots + self._device_inflight_slots
                >= self._device_slot_limit
            )
            if (
                slots_full
                or memory is None
                or not device_prefetch_admitted(
                    memory[0],
                    required,
                    self._device_reserve_bytes,
                    self._device_inflight_bytes,
                )
            ):
                self._device_metrics["persistent_device_prefetch_skips"] += 1
                return loaded
            self._device_inflight_bytes += required
            self._device_inflight_slots += 1
        try:
            cuda_runtime = self._host_pool.cuda_runtime
            if cuda_runtime is None:
                raise RuntimeError("CUDA runtime is unavailable")
            loaded.promote_to_device(cuda_runtime)
        except Exception:
            with self._device_lock:
                self._device_metrics["persistent_device_prefetch_failures"] += 1
            return loaded
        finally:
            with self._device_lock:
                self._device_inflight_bytes -= required
                self._device_inflight_slots -= 1
        with self._device_lock:
            self._device_active_slots += 1
            self._device_metrics["persistent_device_prefetch_hits"] += 1
            self._device_metrics["persistent_device_prefetch_bytes"] += loaded.device_bytes
            self._device_metrics["persistent_device_prefetch_seconds"] += loaded.upload_seconds
        return loaded

    def supports(self, graph: str) -> bool:
        kind = _graph_kind(graph)
        return kind in self.topology_inputs and graph in self.graph_paths

    def session(self, graph: str) -> tuple[Any, bool, float]:
        kind = _graph_kind(graph)
        if kind not in self.topology_inputs:
            raise KeyError(graph)
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
        if kind is None:
            return
        session = self._sessions.pop(kind, None)
        if session is not None:
            del session

    def prefetch(self, graph: str) -> bool:
        if not self.supports(graph) or self._prefetch_depth == 0:
            return False
        if graph in self._weights or graph in self._pending:
            return False
        if len(self._weights) + len(self._pending) >= self._prefetch_depth:
            return False
        kind = _graph_kind(graph)
        if kind not in self.topology_inputs:
            return False
        self._pending[graph] = self._executor.submit(
            self._load_weights,
            graph,
            kind,
        )
        return True

    def weights(
        self, graph: str
    ) -> tuple[dict[str, Any], int, int, int, int, bool, float, float, float, bool, bool]:
        cached = self._weights.get(graph)
        if cached is not None:
            return (
                cached.feeds(),
                cached.total_bytes,
                cached.host_bytes,
                cached.pinned_bytes,
                cached.device_bytes,
                False,
                0.0,
                0.0,
                0.0,
                False,
                cached.device_resident,
            )
        kind = _graph_kind(graph)
        if kind not in self.topology_inputs:
            raise KeyError(graph)
        pending = self._pending.pop(graph, None)
        prefetched = pending is not None
        wait_started = time.perf_counter()
        loaded = (
            pending.result()
            if pending is not None
            else self._load_weights(graph, kind)
        )
        wait_seconds = time.perf_counter() - wait_started
        self._weights[graph] = loaded
        return (
            loaded.feeds(),
            loaded.total_bytes,
            loaded.host_bytes,
            loaded.pinned_bytes,
            loaded.device_bytes,
            True,
            loaded.load_seconds,
            wait_seconds,
            loaded.upload_seconds,
            prefetched,
            loaded.device_resident,
        )

    def release(self, graph: str) -> None:
        loaded = self._weights.pop(graph, None)
        if loaded is not None:
            device_resident = loaded.device_resident
            loaded.close()
            if device_resident:
                with self._device_lock:
                    self._device_active_slots -= 1

    def close(self) -> None:
        self._sessions.clear()
        pending = list(self._pending.values())
        self._pending.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for future in pending:
            if not future.cancelled() and future.exception() is None:
                future.result().close()
        for weights in self._weights.values():
            weights.close()
        self._weights.clear()
        self._host_pool.close()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def publish_persistent_topologies(model_dir: Path) -> dict[str, Any]:
    from h3_workbench.shard_planner import file_sha256

    model_dir = model_dir.resolve()
    schedule = json.loads((model_dir / "schedule.json").read_text(encoding="utf-8"))
    qkv_shard = next(
        shard
        for shard in schedule["shards"]
        if "main_block_00_attention_qkv" in shard["graphs"]
    )
    sources = {
        "attention_qkv": model_dir / qkv_shard["file"],
        "mlp": model_dir / "scaled_mlp" / "block_00" / "scaled_fp16.onnx",
        "attention_output": model_dir / "scaled_attention" / "block_00" / "scaled_fp16.onnx",
    }
    published: dict[str, Any] = {}
    for kind, source in sources.items():
        output = model_dir / PERSISTENT_TOPOLOGIES[kind]
        weights = build_persistent_topology(
            source,
            output,
            canonical_outputs=kind == "attention_qkv",
            all_initializers=kind == "attention_qkv",
        )
        published[kind] = {
            "file": output.name,
            "weight_inputs": len(weights),
            "weight_bytes_per_block": sum(item.length for item in weights),
        }
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    for details in published.values():
        path = model_dir / details["file"]
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest["persistent_weight_topologies"] = published
    manifest.pop("persistent_weight_validation", None)
    if str(manifest.get("hybrid_chain_validation", "")).startswith(
        "four_step_persistent"
    ):
        manifest["hybrid_chain_validation"] = "four_step_scaled_attention_passed"
    _atomic_json(manifest_path, manifest)
    return published
