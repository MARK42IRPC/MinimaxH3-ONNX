"""Correctness-first interpreter for h3-schedule-v2 shard models."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from h3_workbench.profiles import GenerationProfile, PROFILE_360P_17F
from h3_workbench.device_profile import torch_cuda_architecture_supported
from h3_workbench.shard_planner import SCHEDULE_FORMAT, validate_runtime_schedule


_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


class ScheduleMainRuntime:
    """Execute preamble and denoise phases directly from schedule.json."""

    def __init__(
        self,
        directory: Path,
        runner: Any,
        profile: GenerationProfile = PROFILE_360P_17F,
        attention_query_chunk: int = 512,
        activity_callback: Callable[[dict[str, object]], None] | None = None,
        l1_prefetch_shards: int = 2,
        lora_adapter: Any | None = None,
    ) -> None:
        self.directory = directory.resolve()
        self.runner = runner
        self.profile = profile
        self.attention_query_chunk = attention_query_chunk
        self.l1_prefetch_shards = l1_prefetch_shards
        self.activity_callback = activity_callback
        self._lora_adapter = lora_adapter
        self.sampling_step = 0
        self.sampling_steps = 0
        self.audio_fallback_reason: str | None = None
        self._validate_graph_outputs = os.environ.get("H3_VALIDATE_GRAPH_OUTPUTS", "0") == "1"
        self._capture_graph = os.environ.get("H3_CAPTURE_GRAPH")
        capture_path = os.environ.get("H3_CAPTURE_GRAPH_PATH")
        self._capture_graph_path = Path(capture_path).resolve() if capture_path else None
        requested_sdpa_backend = os.environ.get("H3_SDPA_BACKEND", "auto").strip().lower()
        if requested_sdpa_backend not in {"auto", "torch", "ort"}:
            raise ValueError("H3_SDPA_BACKEND must be one of: auto, torch, ort")
        self._device_hidden = (
            getattr(runner, "provider", "CPUExecutionProvider") == "CUDAExecutionProvider"
            and not getattr(runner, "low_vram_mode", False)
            and os.environ.get("H3_DEVICE_RESIDENT_HIDDEN", "1") != "0"
        )
        self._metrics = {
            "session_builds": 0,
            "session_build_seconds": 0.0,
            "graph_runs": 0,
            "graph_seconds": 0.0,
            "device_graph_runs": 0,
            "persistent_session_builds": 0,
            "persistent_session_build_seconds": 0.0,
            "persistent_weight_runs": 0,
            "persistent_weight_bytes": 0,
            "persistent_weight_maps": 0,
            "persistent_weight_map_seconds": 0.0,
            "persistent_host_bytes": 0,
            "persistent_pinned_bytes": 0,
            "persistent_device_bytes": 0,
            "persistent_device_weight_runs": 0,
            "persistent_device_upload_seconds": 0.0,
            "persistent_weight_load_seconds": 0.0,
            "persistent_weight_wait_seconds": 0.0,
            "persistent_prefetches": 0,
            "persistent_prefetch_uses": 0,
            "sdpa_runs": 0,
            "sdpa_seconds": 0.0,
            "device_sdpa_runs": 0,
            "attention_chunk_downgrades": 0,
        }
        self._attention_chunks: dict[int, int] = {}
        self._attention_chunk_free_bytes = 0
        schedule_path = self.directory / "schedule.json"
        if not schedule_path.is_file():
            raise RuntimeError(f"Main model requires {SCHEDULE_FORMAT}: missing {schedule_path}")
        self.schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if self.schedule.get("format") != SCHEDULE_FORMAT:
            raise RuntimeError(
                f"Main model requires {SCHEDULE_FORMAT}, got {self.schedule.get('format')!r}"
            )
        validate_runtime_schedule(self.schedule, self.directory)
        self.shards = {shard["id"]: shard for shard in self.schedule["shards"]}
        self._persistent_weights: Any | None = None
        virtual_weights = None
        try:
            from h3_workbench.ref2va_virtual_slicer import (
                Ref2VASourceWeights,
                ref2va_virtual_ready,
            )

            virtual_ready = ref2va_virtual_ready(self.directory)
        except (ImportError, OSError, ValueError, TypeError):
            virtual_ready = False
        if virtual_ready and self._lora_adapter is not None and getattr(
            self._lora_adapter, "base_component", None
        ) != "ref2va_transformer":
            raise RuntimeError(
                "The selected acceleration LoRA is not compatible with the Ref2VA virtual base"
            )
        if not virtual_ready and getattr(self._lora_adapter, "base_component", None) == "ref2va_transformer":
            raise RuntimeError("Ref2VA acceleration LoRA requires a validated Ref2VA virtual base")
        if virtual_ready:
            graph_paths = {
                graph: self.directory / shard["file"]
                for shard in self.schedule["shards"]
                for graph in shard["graphs"]
            }
            virtual_kwargs = (
                {"topology_paths": self._lora_adapter.topology_paths}
                if self._lora_adapter is not None
                else {}
            )
            virtual_weights = Ref2VASourceWeights(
                self.directory,
                runner,
                graph_paths,
                **virtual_kwargs,
            )
            self._persistent_weights = virtual_weights
            self._ram_cache_prime = virtual_weights.prime_ram_cache()
        elif self._lora_adapter is not None or (
            getattr(runner, "provider", "CPUExecutionProvider") == "CUDAExecutionProvider"
            and os.environ.get("H3_PERSISTENT_WEIGHTS", "1") != "0"
        ):
            from h3_workbench.persistent_weights import PersistentWeightRuntime

            graph_paths = {
                graph: self.directory / shard["file"]
                for shard in self.schedule["shards"]
                for graph in shard["graphs"]
            }
            prefetch_depth = max(
                0,
                min(12, int(os.environ.get("H3_WEIGHT_PREFETCH_DEPTH", "8"))),
            )
            prefetch_workers = max(
                1,
                min(4, int(os.environ.get("H3_WEIGHT_PREFETCH_WORKERS", "3"))),
            )
            persistent = PersistentWeightRuntime(
                self.directory,
                runner,
                graph_paths,
                prefetch_depth=prefetch_depth,
                prefetch_workers=prefetch_workers,
                topology_paths=(
                    self._lora_adapter.topology_paths
                    if self._lora_adapter is not None
                    else None
                ),
            )
            if persistent.enabled:
                self._persistent_weights = persistent
                # Start host-RAM working-set loading as soon as the runtime is
                # created. On the normal path this overlaps prompt encoding or
                # the preceding segment instead of delaying the first graph.
                prime = getattr(persistent, "prime_ram_cache", None)
                self._ram_cache_prime = (
                    prime()
                    if callable(prime)
                    else {
                        "ram_cache_enabled": False,
                        "ram_cache_budget_bytes": 0,
                        "ram_cache_scheduled": 0,
                    }
                )
            elif self._lora_adapter is not None:
                persistent.close()
                raise RuntimeError("Acceleration LoRA adapter has no usable persistent topologies")
        if not hasattr(self, "_ram_cache_prime"):
            self._ram_cache_prime = {
                "ram_cache_enabled": False,
                "ram_cache_budget_bytes": 0,
                "ram_cache_scheduled": 0,
            }
        self._supports_ref2va = bool(virtual_ready)
        self.steps_by_phase = {
            phase: [step for step in self.schedule["steps"] if step["phase"] == phase]
            for phase in ("preamble", "denoise")
        }
        self._sessions: OrderedDict[str, ort.InferenceSession] = OrderedDict()
        total_bytes = 0
        try:
            from h3_workbench.memory_planner import probe_gpu_memory

            total_bytes = probe_gpu_memory().total_bytes
        except Exception:
            total_bytes = 0
        hints = self.schedule["resources"]["session_slot_hints"]
        self._session_limit = (
            len(self.shards)
            if getattr(runner, "provider", "CPUExecutionProvider") == "CPUExecutionProvider"
            else int(hints["vram_24gib"] if total_bytes >= 12 * (1 << 30) else hints["vram_4gib"])
        )
        self._ort_streamed_attention: Any | None = None
        cuda_provider = (
            getattr(runner, "provider", "CPUExecutionProvider") == "CUDAExecutionProvider"
        )
        self._sdpa_backend = "numpy"
        self._sdpa_fallback_reason: str | None = None
        if cuda_provider:
            import torch

            torch_cuda_ready = torch.cuda.is_available()
            if requested_sdpa_backend == "torch" and not torch_cuda_ready:
                raise ValueError("H3_SDPA_BACKEND=torch requires a CUDA-enabled PyTorch")
            device_index = int(getattr(runner, "device_index", 0))
            torch_architecture_ready = (
                torch_cuda_ready and torch_cuda_architecture_supported(device_index)
            )
            if requested_sdpa_backend == "torch" and not torch_architecture_ready:
                try:
                    properties = torch.cuda.get_device_properties(device_index)
                    architecture = f"sm_{properties.major}{properties.minor}"
                except (RuntimeError, AttributeError, IndexError):
                    architecture = "the selected CUDA architecture"
                raise ValueError(
                    "H3_SDPA_BACKEND=torch is unavailable: the installed PyTorch build "
                    f"does not contain kernels for {architecture}. Install a PyTorch build "
                    "supporting this GPU or set H3_SDPA_BACKEND=ort."
                )
            use_torch = torch_architecture_ready and (
                requested_sdpa_backend == "torch"
                or (
                    requested_sdpa_backend == "auto"
                    and not getattr(runner, "low_vram_mode", False)
                )
            )
            if use_torch:
                self._sdpa_backend = "torch"
            else:
                if requested_sdpa_backend == "auto" and torch_cuda_ready and not torch_architecture_ready:
                    try:
                        properties = torch.cuda.get_device_properties(device_index)
                        architecture = f"sm_{properties.major}{properties.minor}"
                    except (RuntimeError, AttributeError, IndexError):
                        architecture = "the selected CUDA architecture"
                    self._sdpa_fallback_reason = (
                        f"torch build lacks {architecture}; selected ORT SDPA"
                    )
                from h3_workbench.inference_runtime import ORTStreamingAttention

                self._ort_streamed_attention = ORTStreamingAttention(self.directory, runner)
                self._sdpa_backend = "ort"
        self._device_sdpa = bool(
            self._ort_streamed_attention is not None
            and os.environ.get("H3_DEVICE_SDPA", "0") == "1"
        )
        recycle_setting = os.environ.get("H3_RECYCLE_PERSISTENT_SESSIONS", "auto").strip().lower()
        if recycle_setting not in {"auto", "0", "1"}:
            raise ValueError("H3_RECYCLE_PERSISTENT_SESSIONS must be auto, 0, or 1")
        recycle_sequence_threshold = max(
            1,
            int(os.environ.get("H3_RECYCLE_SESSION_SEQUENCE_THRESHOLD", "2048")),
        )
        self._recycle_persistent_sessions = bool(
            getattr(runner, "provider", "CPUExecutionProvider") == "CUDAExecutionProvider"
            and (
                recycle_setting == "1"
                or (
                    recycle_setting == "auto"
                    and getattr(runner, "low_vram_mode", False)
                    and not self._device_hidden
                    and profile.sequence_tokens > recycle_sequence_threshold
                )
            )
        )

    @property
    def _main_module_label(self) -> str:
        return "Ref2VA" if self._supports_ref2va else "FL2VA"

    @property
    def _loader_module_label(self) -> str:
        return f"{self._main_module_label} Loader"

    @property
    def _diagnostics_module_label(self) -> str:
        return f"{self._main_module_label} Diagnostics"

    def report_activity(self, module: str, operation: str, **details: object) -> None:
        if self.activity_callback is None:
            return
        payload: dict[str, object] = {"module": module, "operation": operation, **details}
        if self.sampling_step:
            payload["sampling_step"] = self.sampling_step
            payload["sampling_steps"] = self.sampling_steps
        self.activity_callback(payload)

    def _session(self, shard_id: str) -> ort.InferenceSession:
        cached = self._sessions.pop(shard_id, None)
        if cached is not None:
            self._sessions[shard_id] = cached
            return cached
        shard = self.shards[shard_id]
        self.report_activity(
            self._loader_module_label,
            "Building shard session",
            shard=int(shard_id.rsplit("_", 1)[1]) + 1,
            shards=len(self.shards),
        )
        started = time.perf_counter()
        session_paths = getattr(self._lora_adapter, "session_paths", {})
        path = session_paths.get(shard_id, self.directory / shard["file"])
        session = self.runner.session(path)
        elapsed = time.perf_counter() - started
        self._metrics["session_builds"] += 1
        self._metrics["session_build_seconds"] += elapsed
        self.report_activity(
            self._loader_module_label,
            "Shard session ready",
            shard=int(shard_id.rsplit("_", 1)[1]) + 1,
            shards=len(self.shards),
            elapsed_seconds=round(elapsed, 3),
        )
        self._sessions[shard_id] = session
        while len(self._sessions) > max(1, self._session_limit):
            _, evicted = self._sessions.popitem(last=False)
            del evicted
        return session

    def close(self) -> None:
        persistent = self._persistent_weights
        self._persistent_weights = None
        if persistent is not None:
            persistent.close()
        self._sessions.clear()
        if self._ort_streamed_attention is not None:
            self._ort_streamed_attention.close()
        self._ort_streamed_attention = None
        adapter = self._lora_adapter
        self._lora_adapter = None
        if adapter is not None:
            adapter.close()

    def metrics(self) -> dict[str, object]:
        return {
            **self._metrics,
            **(
                self._persistent_weights.device_metrics
                if self._persistent_weights is not None
                else {}
            ),
            "session_build_seconds": round(float(self._metrics["session_build_seconds"]), 3),
            "graph_seconds": round(float(self._metrics["graph_seconds"]), 3),
            "persistent_session_build_seconds": round(
                float(self._metrics["persistent_session_build_seconds"]), 3
            ),
            "persistent_weight_map_seconds": round(
                float(self._metrics["persistent_weight_map_seconds"]), 3
            ),
            "persistent_weight_load_seconds": round(
                float(self._metrics["persistent_weight_load_seconds"]), 3
            ),
            "persistent_weight_wait_seconds": round(
                float(self._metrics["persistent_weight_wait_seconds"]), 3
            ),
            "persistent_device_upload_seconds": round(
                float(self._metrics["persistent_device_upload_seconds"]), 3
            ),
            "resident_sessions": len(self._sessions),
            "session_limit": self._session_limit,
            "device_resident_hidden": self._device_hidden,
            "persistent_session_recycle": self._recycle_persistent_sessions,
            "persistent_pinned_weights": bool(
                self._persistent_weights is not None
                and self._persistent_weights.pinned_enabled
            ),
            "persistent_pinned_fallback_reason": (
                self._persistent_weights.pinned_fallback_reason
                if self._persistent_weights is not None
                else None
            ),
            "persistent_host_pool_allocations": (
                self._persistent_weights.host_pool_allocations
                if self._persistent_weights is not None
                else 0
            ),
            "persistent_prefetch_workers": (
                self._persistent_weights.prefetch_workers
                if self._persistent_weights is not None
                else 0
            ),
            "sdpa_seconds": round(float(self._metrics["sdpa_seconds"]), 3),
            "sdpa_backend": self._sdpa_backend,
            "sdpa_fallback_reason": self._sdpa_fallback_reason,
            "device_sdpa": self._device_sdpa,
            "attention_query_chunk_max": self.attention_query_chunk,
            "attention_query_chunk_counts": dict(sorted(self._attention_chunks.items())),
            "attention_chunk_downgrades": self._metrics["attention_chunk_downgrades"],
            "attention_chunk_last_free_bytes": self._attention_chunk_free_bytes,
            **(
                {
                    "lora_adapter_active": True,
                    "lora_adapter": self._lora_adapter.metrics(),
                }
                if self._lora_adapter is not None
                else {}
            ),
        }

    def warm_fixed_sessions(self) -> dict[str, object]:
        started = time.perf_counter()
        resident = [shard for shard in self.schedule["shards"] if shard.get("resident")]
        warmed = 0
        regular_warmed = 0
        for shard in resident:
            supported_graph = next(
                (
                    graph
                    for graph in shard["graphs"]
                    if self._persistent_weights is not None
                    and self._persistent_weights.supports(graph)
                ),
                None,
            )
            if supported_graph is not None:
                _, built, build_seconds = self._persistent_weights.session(supported_graph)
                if built:
                    self._metrics["session_builds"] += 1
                    self._metrics["session_build_seconds"] += build_seconds
                    self._metrics["persistent_session_builds"] += 1
                    self._metrics["persistent_session_build_seconds"] += build_seconds
                    warmed += 1
                continue
            if regular_warmed >= max(1, self._session_limit):
                continue
            self._session(shard["id"])
            regular_warmed += 1
            warmed += 1
        return {
            **self._ram_cache_prime,
            "warmed_sessions": warmed,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    @staticmethod
    def _read(binding: dict[str, str], external: dict, constants: dict, buffers: dict) -> Any:
        stores = {"external": external, "const": constants, "buffer": buffers}
        return stores[binding["source"]][binding["name"]]

    @staticmethod
    def _write(binding: dict[str, str], value: Any, external: dict, constants: dict, buffers: dict) -> None:
        target = binding["target"]
        if target == "discard":
            return
        stores = {"external": external, "const": constants, "buffer": buffers}
        stores[target][binding["name"]] = value

    @staticmethod
    def _shape_context(arguments: list[Any], values: dict[str, Any], existing: dict[str, int]) -> dict[str, int]:
        context = dict(existing)
        by_name = {argument.name: argument for argument in arguments}
        for name, value in values.items():
            argument = by_name.get(name)
            if argument is None:
                continue
            shape = value.shape() if isinstance(value, ort.OrtValue) else value.shape
            for dimension, actual in zip(argument.shape, shape, strict=False):
                if isinstance(dimension, str):
                    context[dimension] = int(actual)
        if "sequence" not in context and {"text_sequence", "audio_sequence", "video_sequence"} <= context.keys():
            context["sequence"] = (
                context["text_sequence"] + context["audio_sequence"] + context["video_sequence"]
            )
        return context

    def _placeholder(self, argument: Any, context: dict[str, int]) -> np.ndarray:
        dtype = _ORT_DTYPES.get(argument.type)
        if dtype is None:
            raise TypeError(f"Unsupported placeholder dtype {argument.type} for {argument.name}")
        defaults = {
            "sequence": self.profile.sequence_tokens,
            "text_sequence": self.profile.text_tokens,
            "audio_sequence": self.profile.audio_tokens,
            "video_sequence": self.profile.video_tokens,
            "timestep_count": 2,
        }
        shape = tuple(
            int(dimension)
            if isinstance(dimension, int) and dimension > 0
            else int(context.get(str(dimension), defaults.get(str(dimension), 1)))
            for dimension in argument.shape
        )
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return value.numpy() if isinstance(value, ort.OrtValue) else np.asarray(value)

    def _diagnose_outputs(self, operation: str, values: dict[str, Any]) -> None:
        if not self._validate_graph_outputs:
            return
        statistics: dict[str, dict[str, float | int | bool]] = {}
        first_invalid: tuple[str, int] | None = None
        for port, value in values.items():
            array = self._numpy(value)
            finite_mask = np.isfinite(array)
            invalid = int(array.size - np.count_nonzero(finite_mask))
            finite_values = array[finite_mask]
            statistics[port] = {
                "finite": invalid == 0,
                "invalid": invalid,
                "max_abs_finite": (
                    float(np.max(np.abs(finite_values))) if finite_values.size else float("nan")
                ),
            }
            if invalid and first_invalid is None:
                first_invalid = (port, invalid)
        self.report_activity(
            self._diagnostics_module_label,
            operation,
            event="output_stats",
            outputs=statistics,
        )
        if first_invalid is not None:
            port, invalid = first_invalid
            raise FloatingPointError(f"Non-finite output at {operation}/{port}: {invalid} invalid values")

    def _device_output(self, step: dict) -> bool:
        graph = str(step.get("graph", ""))
        return (
            self._device_hidden
            and graph.startswith("main_block_")
            and (
                graph.endswith(("_attention_output", "_mlp"))
                or (self._device_sdpa and graph.endswith("_attention_qkv"))
            )
        )

    def _run_with_iobinding(
        self,
        session: ort.InferenceSession,
        arguments: list[Any],
        feeds: dict[str, Any],
        output_names: list[str],
        device_output: bool,
    ) -> list[Any]:
        binding = session.io_binding()
        for argument in arguments:
            value = feeds[argument.name]
            expected_dtype = _ORT_DTYPES.get(argument.type)
            actual_dtype = None
            if isinstance(value, ort.OrtValue):
                actual_dtype = _ORT_DTYPES.get(value.data_type())
            elif hasattr(value, "dtype"):
                actual_dtype = np.dtype(value.dtype)
            if expected_dtype is not None and actual_dtype is not None and actual_dtype != expected_dtype:
                raise TypeError(
                    f"I/O binding dtype mismatch for {argument.name!r}: "
                    f"expected {np.dtype(expected_dtype)}, got {actual_dtype}"
                )
            if hasattr(value, "bind_input"):
                value.bind_input(binding, argument.name)
            elif isinstance(value, ort.OrtValue):
                binding.bind_ortvalue_input(argument.name, value)
            else:
                array = np.asarray(value)
                binding.bind_cpu_input(
                    argument.name,
                    array if array.flags.c_contiguous else np.ascontiguousarray(array),
                )
        for name in output_names:
            binding.bind_output(name, "cuda" if device_output else "cpu")
        try:
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            outputs = binding.get_outputs()
            if device_output:
                return list(outputs)
            return [value.numpy() for value in outputs]
        finally:
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()

    def _run_graph(
        self,
        step: dict,
        external: dict,
        constants: dict,
        buffers: dict,
        dimension_context: dict[str, int],
    ) -> dict[str, int]:
        persistent = (
            self._persistent_weights
            if self._persistent_weights is not None and self._persistent_weights.supports(step["graph"])
            else None
        )
        persistent_feeds: dict[str, Any] = {}
        persistent_weight_bytes = 0
        persistent_host_bytes = 0
        persistent_pinned_bytes = 0
        persistent_device_bytes = 0
        persistent_load_seconds = 0.0
        persistent_wait_seconds = 0.0
        persistent_upload_seconds = 0.0
        persistent_prefetched = False
        persistent_device_resident = False
        persistent_prefetch_metrics: dict[str, int | float] = {}
        if persistent is None:
            session = self._session(step["shard"])
        else:
            session, built, build_seconds = persistent.session(step["graph"])
            if built:
                self._metrics["session_builds"] += 1
                self._metrics["session_build_seconds"] += build_seconds
                self._metrics["persistent_session_builds"] += 1
                self._metrics["persistent_session_build_seconds"] += build_seconds
                self.report_activity(
                    self._loader_module_label,
                    "Persistent topology session ready",
                    topology=step["graph"].rsplit("_", 1)[-1],
                    elapsed_seconds=round(build_seconds, 3),
                )
            (
                persistent_feeds,
                persistent_weight_bytes,
                persistent_host_bytes,
                persistent_pinned_bytes,
                persistent_device_bytes,
                mapped,
                persistent_load_seconds,
                persistent_wait_seconds,
                persistent_upload_seconds,
                persistent_prefetched,
                persistent_device_resident,
            ) = persistent.weights(step["graph"])
            if mapped:
                self._metrics["persistent_weight_maps"] += 1
                self._metrics["persistent_weight_map_seconds"] += persistent_load_seconds
                self._metrics["persistent_weight_load_seconds"] += persistent_load_seconds
                self._metrics["persistent_weight_wait_seconds"] += persistent_wait_seconds
                self._metrics["persistent_host_bytes"] += persistent_host_bytes
                self._metrics["persistent_pinned_bytes"] += persistent_pinned_bytes
                self._metrics["persistent_device_bytes"] += persistent_device_bytes
                self._metrics["persistent_device_upload_seconds"] += persistent_upload_seconds
            if persistent_prefetched:
                self._metrics["persistent_prefetch_uses"] += 1
            if persistent_device_resident:
                self._metrics["persistent_device_weight_runs"] += 1
            prefetch_metrics = getattr(persistent, "prefetch_metrics", None)
            if callable(prefetch_metrics):
                persistent_prefetch_metrics = prefetch_metrics()
        selected = {
            port: self._read(binding, external, constants, buffers)
            for port, binding in step["inputs"].items()
        }
        adapter_feeds = (
            self._lora_adapter.graph_feeds(step["graph"])
            if self._lora_adapter is not None
            else {}
        )
        embedded_weights = bool(
            self._lora_adapter is not None
            and getattr(self._lora_adapter, "uses_embedded_weights", lambda _graph: False)(step["graph"])
        )
        if adapter_feeds and persistent is None and not embedded_weights:
            raise RuntimeError(
                f"Acceleration LoRA graph {step['graph']} has no persistent topology"
            )
        if step["graph"] == self._capture_graph:
            if self._capture_graph_path is None:
                raise RuntimeError("H3_CAPTURE_GRAPH_PATH is required when H3_CAPTURE_GRAPH is set")
            self._capture_graph_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._capture_graph_path.with_name(f"{self._capture_graph_path.stem}.tmp.npz")
            np.savez(temporary, **{name: self._numpy(value) for name, value in selected.items()})
            os.replace(temporary, self._capture_graph_path)
            self.report_activity(
                self._diagnostics_module_label,
                step["graph"],
                event="inputs_captured",
                path=str(self._capture_graph_path),
            )
        arguments = session.get_inputs()
        context = self._shape_context(arguments, selected, dimension_context)
        feed_names = set(selected) | set(persistent_feeds)
        collisions = feed_names.intersection(adapter_feeds)
        if collisions:
            raise RuntimeError(
                f"Acceleration LoRA feed collision in {step['graph']}: {sorted(collisions)}"
            )
        feeds: dict[str, Any] = {**selected, **persistent_feeds, **adapter_feeds}
        shard_graphs = self.shards[step["shard"]]["graphs"]
        selected_index = shard_graphs.index(step["graph"])
        for argument in arguments:
            if argument.name.startswith("selector_"):
                feeds[argument.name] = np.asarray(selected_index, dtype=np.int64)
            elif argument.name.startswith("run_"):
                index = int(argument.name.rsplit("_", 1)[1])
                feeds[argument.name] = np.asarray(index == selected_index, dtype=np.bool_)
            elif argument.name in constants and "silu_timestep_embedding" in argument.name:
                feeds[argument.name] = constants[argument.name]
            elif (
                self._lora_adapter is not None
                and argument.name not in feeds
                and (
                    argument.name.startswith("lora_")
                )
            ):
                raise RuntimeError(
                    f"Acceleration LoRA graph {step['graph']} is missing feed {argument.name!r}"
                )
            elif argument.name not in feeds:
                feeds[argument.name] = self._placeholder(argument, context)
        output_ports = list(step["outputs"])
        available_outputs = {output.name for output in session.get_outputs()}
        output_names = []
        for port in output_ports:
            prefixed = f"{step['graph']}/{port}"
            if prefixed in available_outputs:
                output_names.append(prefixed)
            elif port in available_outputs:
                output_names.append(port)
            else:
                raise RuntimeError(
                    f"Graph {step['graph']} has no output for port {port!r}; "
                    f"available={sorted(available_outputs)}"
                )
        self.report_activity(
            self._main_module_label,
            step["graph"],
            shard=int(step["shard"].rsplit("_", 1)[1]) + 1,
            shards=len(self.shards),
        )
        device_output = self._device_output(step)
        use_iobinding = device_output or any(
            isinstance(value, ort.OrtValue) or hasattr(value, "bind_input")
            for value in feeds.values()
        )
        started = time.perf_counter()
        try:
            values = (
                self._run_with_iobinding(session, arguments, feeds, output_names, device_output)
                if use_iobinding
                else session.run(output_names, feeds)
            )
        finally:
            if persistent is not None:
                persistent.release(step["graph"])
        self._diagnose_outputs(
            step["graph"],
            dict(zip(output_ports, values, strict=True)),
        )
        elapsed = time.perf_counter() - started
        self._metrics["graph_runs"] += 1
        self._metrics["graph_seconds"] += elapsed
        if use_iobinding:
            self._metrics["device_graph_runs"] += 1
        if persistent is not None:
            self._metrics["persistent_weight_runs"] += 1
            self._metrics["persistent_weight_bytes"] += persistent_weight_bytes
        for port, value in zip(output_ports, values, strict=True):
            self._write(step["outputs"][port], value, external, constants, buffers)
        if persistent is not None and self._recycle_persistent_sessions:
            persistent.release_session(step["graph"])
        self.report_activity(
            self._main_module_label,
            step["graph"],
            event="graph_complete",
            shard=int(step["shard"].rsplit("_", 1)[1]) + 1,
            shards=len(self.shards),
            elapsed_seconds=round(elapsed, 3),
            io_binding=use_iobinding,
            device_output=device_output,
            persistent_weights=persistent is not None,
            weight_bytes=persistent_weight_bytes if persistent is not None else None,
            host_bytes=persistent_host_bytes if persistent is not None else None,
            pinned_bytes=persistent_pinned_bytes if persistent is not None else None,
            device_weight_bytes=persistent_device_bytes if persistent is not None else None,
            weight_load_seconds=(
                round(persistent_load_seconds, 3) if persistent is not None else None
            ),
            weight_wait_seconds=(
                round(persistent_wait_seconds, 3) if persistent is not None else None
            ),
            weight_prefetched=persistent_prefetched if persistent is not None else None,
            weight_device_resident=(
                persistent_device_resident if persistent is not None else None
            ),
            weight_upload_seconds=(
                round(persistent_upload_seconds, 3) if persistent is not None else None
            ),
            **persistent_prefetch_metrics,
        )
        return self._shape_context(arguments, feeds, context)

    def _run_op(self, step: dict, external: dict, constants: dict, buffers: dict) -> None:
        values = {
            port: self._read(binding, external, constants, buffers)
            for port, binding in step["inputs"].items()
        }
        op = step["op"]
        if op == "preamble_inputs":
            raw = np.asarray(values["raw_text_states"])
            outputs = {
                "video_patches": np.zeros((1, 96), dtype=np.float32),
                "audio_patches": np.zeros((1, 32), dtype=np.float32),
                "text_states": raw.astype(np.float16, copy=False),
            }
        elif op == "denoise_inputs":
            from h3_workbench.inference_runtime import modulation_ids, pack_audio, packed_position_ids, patchify_video

            text_count = int(np.asarray(values["text_states"]).shape[0])
            sigma_video = float(values["sigma_video"])
            from h3_workbench.inference_runtime import time_shift_sigma

            sigma_audio = time_shift_sigma(sigma_video, 12.0, 3.0)
            ref2va_layout = external.get("ref2va_layout")
            if ref2va_layout is None:
                times, modulation = modulation_ids(
                    self.profile,
                    text_count,
                    sigma_video,
                    external.get("conditioned_video_indices", ()),
                )
                outputs = {
                    "video_patches": patchify_video(np.asarray(values["video_latent"], dtype=np.float32)),
                    "audio_patches": pack_audio(np.asarray(values["audio_latent"], dtype=np.float32)),
                    "embedding_text_padding": np.zeros((1, 5120), dtype=np.float16),
                    "timesteps": times,
                    "position_ids": packed_position_ids(self.profile, text_count),
                    "modulation_ids": modulation,
                    "sigma_audio": np.asarray(sigma_audio, dtype=np.float32),
                }
            else:
                from h3_workbench.reference import build_row_timesteps

                if int(ref2va_layout.text_indices.size) != text_count:
                    raise ValueError(
                        "Ref2VA text rows do not match the refined text sequence: "
                        f"{ref2va_layout.text_indices.size} != {text_count}"
                    )
                video_latents = (
                    *external.get("reference_video_latents", ()),
                    np.asarray(values["video_latent"], dtype=np.float32),
                )
                audio_latents = (
                    *external.get("reference_audio_latents", ()),
                    np.asarray(values["audio_latent"], dtype=np.float32),
                )
                video_patches = np.concatenate(
                    [patchify_video(np.asarray(latent, dtype=np.float32)) for latent in video_latents],
                    axis=0,
                )
                audio_patches = np.concatenate(
                    [pack_audio(np.asarray(latent, dtype=np.float32)) for latent in audio_latents],
                    axis=0,
                )
                if video_patches.shape[0] != ref2va_layout.video_indices.size:
                    raise ValueError(
                        "Ref2VA visual latent rows do not match the packed layout: "
                        f"{video_patches.shape[0]} != {ref2va_layout.video_indices.size}"
                    )
                if audio_patches.shape[0] != ref2va_layout.audio_indices.size:
                    raise ValueError(
                        "Ref2VA audio latent rows do not match the packed layout: "
                        f"{audio_patches.shape[0]} != {ref2va_layout.audio_indices.size}"
                    )
                times, timestep_indices = build_row_timesteps(
                    ref2va_layout,
                    video_timestep=1.0 - sigma_video,
                    audio_timestep=1.0 - sigma_audio,
                )
                modulation = timestep_indices.astype(np.int64, copy=False) * 3 + np.asarray(
                    ref2va_layout.token_tags,
                    dtype=np.int64,
                )
                outputs = {
                    "video_patches": video_patches,
                    "audio_patches": audio_patches,
                    "embedding_text_padding": np.zeros((1, 5120), dtype=np.float16),
                    "timesteps": times,
                    "position_ids": np.asarray(ref2va_layout.position_ids, dtype=np.float32),
                    "modulation_ids": modulation,
                    "sigma_audio": np.asarray(sigma_audio, dtype=np.float32),
                }
        elif op == "concat_hidden":
            ref2va_layout = external.get("ref2va_layout")
            if ref2va_layout is None:
                hidden = np.concatenate(
                    (values["text_states"], values["audio_embeddings"], values["video_embeddings"]),
                    axis=0,
                ).astype(np.float32, copy=False)
            else:
                hidden = np.empty((ref2va_layout.token_tags.size, 5376), dtype=np.float32)
                hidden[ref2va_layout.text_indices] = np.asarray(values["text_states"], dtype=np.float32)
                hidden[ref2va_layout.audio_indices] = np.asarray(values["audio_embeddings"], dtype=np.float32)
                hidden[ref2va_layout.video_indices] = np.asarray(values["video_embeddings"], dtype=np.float32)
            outputs = {"hidden": hidden}
        elif op == "sdpa":
            from h3_workbench.inference_runtime import select_attention_query_chunk, streamed_attention
            from h3_workbench.memory_planner import probe_gpu_memory

            started = time.perf_counter()
            packed = values["qkv_packed"]
            packed_shape = packed.shape() if isinstance(packed, ort.OrtValue) else np.asarray(packed).shape
            snapshot = probe_gpu_memory()
            query_chunk = select_attention_query_chunk(
                self.attention_query_chunk,
                int(packed_shape[0]),
                snapshot.free_bytes,
            )
            self._attention_chunks[query_chunk] = self._attention_chunks.get(query_chunk, 0) + 1
            self._attention_chunk_free_bytes = snapshot.free_bytes
            if query_chunk < self.attention_query_chunk:
                self._metrics["attention_chunk_downgrades"] += 1
            outputs = {
                "attended": (
                    self._ort_streamed_attention(
                        packed if isinstance(packed, ort.OrtValue) else np.asarray(packed),
                        query_chunk,
                        output_dtype=np.float32,
                    )
                    if self._ort_streamed_attention is not None
                    else streamed_attention(
                        np.asarray(packed),
                        getattr(self.runner, "provider", "CPUExecutionProvider")
                        == "CUDAExecutionProvider",
                        query_chunk,
                        int(getattr(self.runner, "device_index", 0)),
                    )
                )
            }
            self._metrics["sdpa_runs"] += 1
            self._metrics["sdpa_seconds"] += time.perf_counter() - started
            if isinstance(outputs["attended"], ort.OrtValue):
                self._metrics["device_sdpa_runs"] += 1
        elif op == "split_hidden":
            hidden = self._numpy(values["hidden"])
            ref2va_layout = external.get("ref2va_layout")
            if ref2va_layout is None:
                text_count = int(np.asarray(values["text_states"]).shape[0])
                audio_start = text_count
                video_start = audio_start + self.profile.audio_tokens
                outputs = {
                    "audio_hidden": hidden[audio_start:video_start],
                    "video_hidden": hidden[video_start:],
                }
            else:
                outputs = {
                    "audio_hidden": hidden[ref2va_layout.target_audio_indices],
                    "video_hidden": hidden[ref2va_layout.target_video_indices],
                }
        elif op in {"select_head_timestep", "select_head_timestep_turbo"}:
            times = np.asarray(values["timesteps"])
            sigma_video = float(values["sigma_video"])
            sigma_audio = float(values["sigma_audio"])
            video_index = int(np.argmin(np.abs(times - (1.0 - sigma_video))))
            audio_index = int(np.argmin(np.abs(times - (1.0 - sigma_audio))))
            outputs = {
                "video_timestep_embedding": values["timestep_embedding"][video_index : video_index + 1],
                "audio_timestep_embedding": values["timestep_embedding"][audio_index : audio_index + 1],
            }
            if op.endswith("_turbo"):
                silu_timestep_embedding = values.get(
                    "silu_timestep_embedding",
                    constants.get("silu_timestep_embedding"),
                )
                if silu_timestep_embedding is None:
                    raise RuntimeError("Accelerated head selection requires SiLU timestep embeddings")
                constants.update(
                    {
                        "video_silu_timestep_embedding": silu_timestep_embedding[
                            video_index : video_index + 1
                        ],
                        "audio_silu_timestep_embedding": silu_timestep_embedding[
                            audio_index : audio_index + 1
                        ],
                    }
                )
                outputs.update(
                    {
                        "video_silu_timestep_embedding": constants["video_silu_timestep_embedding"],
                        "audio_silu_timestep_embedding": constants["audio_silu_timestep_embedding"],
                    }
                )
        elif op == "unpack_velocity":
            from h3_workbench.inference_runtime import unpack_audio, unpatchify_video

            outputs = {
                "video_velocity": unpatchify_video(
                    np.asarray(values["video_patches"]),
                    self.profile.video_latent_frames,
                    self.profile.video_latent_height,
                    self.profile.video_latent_width,
                ),
                "audio_velocity": unpack_audio(np.asarray(values["audio_patches"])),
            }
        else:
            raise ValueError(f"Unsupported schedule op: {op}")
        self._diagnose_outputs(op, outputs)
        for port, value in outputs.items():
            self._write(step["outputs"][port], value, external, constants, buffers)

    def _execute_phase(self, phase: str, external: dict, constants: dict | None = None) -> tuple[dict, dict]:
        constants = {} if constants is None else constants
        buffers: dict[str, Any] = {}
        dimension_context: dict[str, int] = {}
        steps = self.steps_by_phase[phase]
        for step_index, step in enumerate(steps):
            if (
                self._persistent_weights is not None
                and self._persistent_weights.prefetch_depth > 0
            ):
                candidates: list[str] = []
                for future_step in steps[step_index:]:
                    if (
                        future_step["kind"] == "graph"
                        and self._persistent_weights.supports(future_step["graph"])
                    ):
                        candidates.append(future_step["graph"])
                        if len(candidates) == self._persistent_weights.prefetch_depth:
                            break
                prefetch_many = getattr(self._persistent_weights, "prefetch_many", None)
                if callable(prefetch_many):
                    self._metrics["persistent_prefetches"] += int(
                        prefetch_many(candidates)
                    )
                else:
                    for graph in candidates:
                        if self._persistent_weights.prefetch(graph):
                            self._metrics["persistent_prefetches"] += 1
            if step["kind"] == "graph":
                dimension_context = self._run_graph(
                    step,
                    external,
                    constants,
                    buffers,
                    dimension_context,
                )
            else:
                self._run_op(step, external, constants, buffers)
            for name in step["release"]:
                buffers.pop(name, None)
        return external, constants

    def prepare_text(self, text_states: np.ndarray) -> np.ndarray:
        _, constants = self._execute_phase("preamble", {"raw_text_states": text_states})
        return np.asarray(constants["text_states"])

    def denoise_step(
        self,
        video_latent: np.ndarray,
        audio_latent: np.ndarray,
        text_states: np.ndarray,
        sigma_video: float,
        text_is_refined: bool = False,
        conditioned_video_indices: tuple[int, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        refined = text_states if text_is_refined else self.prepare_text(text_states)
        external = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "sigma_video": float(sigma_video),
            "conditioned_video_indices": conditioned_video_indices,
        }
        external, _ = self._execute_phase("denoise", external, {"text_states": refined})
        return np.asarray(external["video_velocity"]), np.asarray(external["audio_velocity"])

    def denoise_ref2va_step(
        self,
        video_latent: np.ndarray,
        audio_latent: np.ndarray,
        text_states: np.ndarray,
        sigma_video: float,
        reference_video_latents: tuple[np.ndarray, ...],
        reference_audio_latents: tuple[np.ndarray, ...],
        layout: Any,
        text_is_refined: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one dynamic Ref2VA packed-sequence denoise pass.

        Reference rows are embedded and attended on every step but only target
        rows are passed through the output head.  The caller therefore keeps
        reference latents immutable across the Euler loop.
        """
        if not self._supports_ref2va:
            raise RuntimeError("Ref2VA denoising requires a validated Ref2VA virtual model")
        refined = text_states if text_is_refined else self.prepare_text(text_states)
        external = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "sigma_video": float(sigma_video),
            "reference_video_latents": tuple(reference_video_latents),
            "reference_audio_latents": tuple(reference_audio_latents),
            "ref2va_layout": layout,
        }
        external, _ = self._execute_phase("denoise", external, {"text_states": refined})
        return np.asarray(external["video_velocity"]), np.asarray(external["audio_velocity"])
