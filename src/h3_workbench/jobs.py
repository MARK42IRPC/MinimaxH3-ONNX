from __future__ import annotations

import gc
import json
import math
import os
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from h3_workbench.exporter import export_checkpoint
from h3_workbench.acceleration import shifted_flow_sigmas
from h3_workbench.direct_download import download_file
from h3_workbench.inference_runtime import H3MainRuntime, ORTGraphRunner, QwenTextRuntime, initial_latents, sample_latents
from h3_workbench.qwen_persistent import resolve_qwen_directory
from h3_workbench.media_output import (
    decode_audio_latents,
    decode_audio_latents_onnx,
    decode_video_latents,
    decode_video_latents_onnx,
    write_mp4,
)
from h3_workbench.performance_monitor import PerformanceMonitor
from h3_workbench.profiles import PROFILE_360P_17F, GenerationProfile
from h3_workbench.tokenizer import encode_prompt
from h3_workbench.tokenizer import tokenizer_files_ready
from h3_workbench.model_registry import inspect_checkpoint
from h3_workbench.memory_planner import probe_gpu_memory
from h3_workbench.source_catalog import ExportPreset, SourceAsset, export_preset
from h3_workbench.turbo_lora import (
    TURBO_ADAPTER_VARIANT,
    TurboLoraAdapter,
    publish_turbo_adapter,
    validate_turbo_adapter,
)
from h3_workbench.vram_reservation import (
    acquire_reservation,
    configure_reservations,
    refresh_reservation,
    release_reservation,
)


def _complete_main_model(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    schedule = directory / str(manifest.get("schedule", ""))
    return (
        manifest.get("validation_passed") is True
        and manifest.get("build_complete") is True
        and manifest.get("schedule_format") == "h3-schedule-v2"
        and len(manifest.get("blocks", [])) == 50
        and schedule.is_file()
    )


def resolve_main_model_directory(
    workspace: Path,
    output_root: Path,
    accelerated: bool,
) -> Path | None:
    """Resolve a validated main product by capability instead of a legacy folder name."""
    candidates: list[Path] = []
    for root in (
        output_root.resolve(),
        (workspace / "exported").resolve(),
        workspace.resolve(),
    ):
        if not root.is_dir():
            continue
        candidates.extend(path for path in root.iterdir() if path.is_dir())
    ready: list[Path] = []
    for directory in dict.fromkeys(candidates):
        if not _complete_main_model(directory):
            continue
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if bool(manifest.get("acceleration")) == accelerated:
            ready.append(directory)
    if not ready:
        return None
    return max(ready, key=lambda path: (path / "manifest.json").stat().st_mtime_ns)


def _manifest_identity(directory: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"directory": str(directory), "manifest_available": False}
    acceleration = manifest.get("acceleration")
    return {
        "directory": str(directory),
        "component": manifest.get("component"),
        "source": manifest.get("source"),
        "source_quantization": manifest.get("source_quantization"),
        "conversion": manifest.get("conversion"),
        "validation_passed": manifest.get("validation_passed"),
        "acceleration": acceleration if isinstance(acceleration, dict) else None,
    }


def _close_qwen_runtime_weights(runtime: QwenTextRuntime | None) -> None:
    if runtime is None:
        return
    for weights in (runtime.int8_virtual, runtime.persistent):
        if weights is not None:
            weights.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    id: str
    model_id: str
    kind: str = "export"
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    activity: dict[str, Any] = field(default_factory=lambda: {"phase": "queued", "module": "Queue", "operation": "Waiting"})
    prefetch: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    performance_log: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    output_dir: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    def __init__(self, workspace: Path, output_root: Path):
        self.workspace = workspace.resolve()
        self.output_root = output_root.resolve()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-export")
        configure_reservations(self.workspace / ".h3-workbench" / "vram_reservations")

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def create_export(self, model_id: str, video_blocks: str = "all") -> Job:
        source = (self.workspace / model_id).resolve()
        if self.workspace not in source.parents or not source.is_file():
            raise ValueError("Model path is outside the workspace or does not exist")
        record = inspect_checkpoint(source, self.workspace)
        job = Job(id=uuid.uuid4().hex[:12], model_id=model_id)
        component_dir = {
            "audio_vae": "audio_vae",
            "video_vae": "video_vae",
            "text_encoder": "qwen3vl_32b_minimax_h3_int8_virtual",
            "fl2va_transformer": "minimax_h3_fl2va_pruned_fp8_scaled_streaming",
        }.get(record.component, source.stem.replace(".safetensors", ""))
        destination = self.output_root / component_dir
        job.output_dir = str(destination)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run_export, job.id, source, destination, video_blocks)
        return job

    def create_preset_export(self, preset_id: str) -> Job:
        preset = export_preset(preset_id)
        free_bytes = shutil.disk_usage(self.workspace).free
        assets = list(preset.sources)
        missing_download_bytes = sum(
            item.size_bytes for item in assets if self._existing_source_asset(item) is None
        )
        required = missing_download_bytes + preset.output_size_bytes
        if free_bytes < required:
            raise ValueError(f"Insufficient disk space: need about {required / 1024**3:.1f} GiB, available {free_bytes / 1024**3:.1f} GiB")
        job = Job(id=uuid.uuid4().hex[:12], model_id=preset.label, kind="download_export")
        job.output_dir = str(self.workspace / preset.output_dir)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run_preset_export, job.id, preset)
        return job

    def _source_destination(self, asset: SourceAsset) -> Path:
        return self.workspace / ".h3-workbench" / "sources" / asset.repo_id.replace("/", "--") / asset.path

    def _existing_source_asset(self, asset: SourceAsset) -> Path | None:
        candidates = (
            self._source_destination(asset),
            self.workspace / Path(asset.path).name,
            self.workspace / "qwen_tokenizer" / Path(asset.path).name,
        )
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size == asset.size_bytes:
                return candidate.resolve()
        return None

    def _download_source_asset(self, job_id: str, asset: SourceAsset, label: str) -> Path:
        existing = self._existing_source_asset(asset)
        if existing is not None:
            return existing
        destination = self._source_destination(asset)
        self._update(
            job_id,
            message=f"Downloading {label}",
            activity={"phase": "download", "module": label, "operation": asset.path, "url": asset.url},
        )

        def report(downloaded: int, total: int) -> None:
            fraction = downloaded / max(1, total)
            self._update(
                job_id,
                progress=0.01 + 0.10 * fraction,
                message=f"Downloading {label}: {fraction * 100:.1f}%",
                activity={
                    "phase": "download",
                    "module": label,
                    "operation": asset.path,
                    "bytes_downloaded": downloaded,
                    "bytes_total": total,
                    "url": asset.url,
                },
            )

        return download_file(asset.url, destination, asset.size_bytes, report)

    def _run_preset_export(self, job_id: str, preset: ExportPreset) -> None:
        try:
            self._update(job_id, status="running", started_at=_now(), progress=0.01, message="Preparing verified export preset", activity={"phase": "download", "module": preset.label, "operation": "Checking source files"})
            source_label = "Turbo v4 LoRA" if preset.component == "acceleration_lora" else "Original checkpoint"
            source = self._download_source_asset(job_id, preset.source, source_label)
            if preset.component == "tokenizer":
                downloaded = [source]
                for asset in preset.extra_sources:
                    downloaded.append(self._download_source_asset(job_id, asset, asset.path))
                destination = self.workspace / preset.output_dir
                destination.mkdir(parents=True, exist_ok=True)
                for source_path in downloaded:
                    target = destination / source_path.name
                    if source_path.resolve() == target.resolve():
                        continue
                    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
                    shutil.copy2(source_path, temporary)
                    os.replace(temporary, target)
                if not tokenizer_files_ready(destination):
                    raise RuntimeError("Tokenizer files failed the local readiness check")
                self._update(
                    job_id,
                    status="completed",
                    progress=1.0,
                    message="Tokenizer download completed",
                    activity={"phase": "completed", "module": preset.label, "operation": "Files verified"},
                    finished_at=_now(),
                    result={"preset": preset.id, "output": str(destination), "files": [path.name for path in downloaded]},
                )
                return
            if preset.component == "acceleration_lora":
                if preset.support is None:
                    raise RuntimeError("Turbo runtime adapter requires the SiLU timestep grid")
                grid_path = self._download_source_asset(
                    job_id,
                    preset.support,
                    "Turbo timestep grid",
                )
                base_main_dir = resolve_main_model_directory(
                    self.workspace,
                    self.output_root,
                    accelerated=False,
                )
                if base_main_dir is None:
                    raise RuntimeError(
                        "Turbo v4 dynamic LoRA requires the validated FL2VA streaming Base product"
                    )
                destination = self.workspace / preset.output_dir
                self._update(
                    job_id,
                    progress=0.20,
                    message="Publishing graph-only Turbo adapter",
                    activity={
                        "phase": "export",
                        "module": preset.label,
                        "operation": "Building runtime overlay topologies",
                    },
                )
                manifest = publish_turbo_adapter(
                    base_main_dir,
                    destination,
                    source,
                    grid_path,
                )
                self._update(
                    job_id,
                    status="completed",
                    progress=1.0,
                    message="Turbo v4 runtime adapter completed",
                    activity={
                        "phase": "completed",
                        "module": preset.label,
                        "operation": "Dynamic LoRA adapter verified",
                    },
                    finished_at=_now(),
                    result={
                        "preset": preset.id,
                        "source": str(source),
                        "support": str(grid_path),
                        "base": str(base_main_dir),
                        "output": str(destination),
                        "adapter": manifest,
                    },
                )
                return
            lora_path = self._download_source_asset(job_id, preset.lora, "Turbo v4 LoRA") if preset.lora else None
            if preset.support:
                support_path = self._download_source_asset(job_id, preset.support, "Turbo timestep grid")
                if lora_path is None:
                    raise RuntimeError("Turbo support file requires a LoRA checkpoint")
                colocated = lora_path.with_name("h3_silu_temb_grid.safetensors")
                if not colocated.is_file() or colocated.stat().st_size != support_path.stat().st_size:
                    shutil.copy2(support_path, colocated)
            record = inspect_checkpoint(source, source.parent)
            if record.component != preset.component:
                raise RuntimeError(f"Downloaded file is classified as {record.component}, not {preset.component}")
            destination = self.workspace / preset.output_dir
            self._update(job_id, progress=0.12, message="Source files ready; exporting ONNX shards", activity={"phase": "export", "module": preset.label, "operation": "Starting sharded export"})
            def report(progress: float, message: str) -> None:
                self._update(job_id, progress=0.12 + progress * 0.88, message=message, activity={"phase": "export", "module": preset.label, "operation": message})
            exported = export_checkpoint(source, destination, preset.blocks, report, lora_path, 1.0)
            self._update(job_id, status="completed", progress=1.0, message="Verified preset export completed", activity={"phase": "completed", "module": preset.label, "operation": "ONNX validation passed"}, finished_at=_now(), result={"preset": preset.id, "source": str(source), "output": str(destination), "export": exported})
        except Exception as exc:  # noqa: BLE001 - preserve pipeline failure in the job record
            self._update(job_id, status="failed", message=str(exc), activity={"phase": "failed", "module": preset.label, "operation": str(exc)}, finished_at=_now(), error="".join(traceback.format_exception(exc)))

    def create_inference(
        self,
        token_ids: list[int] | None,
        prompt: str | None,
        steps: int,
        seed: int,
        width: int,
        height: int,
        duration_seconds: float,
        temporal_mode: str = "segmented",
        attention_query_chunk: int = 512,
        l1_prefetch_shards: int = 2,
        use_acceleration_lora: bool = False,
    ) -> Job:
        if not 1 <= steps <= 50:
            raise ValueError("Inference steps must be from 1 to 50")
        if use_acceleration_lora and not 4 <= steps <= 8:
            raise ValueError("Turbo v4 acceleration LoRA supports 4-8 sampling steps")
        if not token_ids and not prompt:
            raise ValueError("Provide either prompt or token IDs")
        if not 128 <= width <= 1024 or not 128 <= height <= 1024:
            raise ValueError("Output dimensions must be between 128 and 1024 pixels")
        if not 0 < duration_seconds <= 15:
            raise ValueError("Duration must be greater than zero and at most 15 seconds")
        if temporal_mode not in {"native", "segmented"}:
            raise ValueError("Temporal mode must be 'native' or 'segmented'")
        if attention_query_chunk not in {32, 64, 128, 256, 512}:
            raise ValueError("Attention query chunk must be one of 32, 64, 128, 256, or 512")
        if not 0 <= l1_prefetch_shards <= 4:
            raise ValueError("L1 prefetch shards must be from 0 to 4")
        # Keep the model's validated temporal/FPS geometry as an internal template;
        # output dimensions and duration are always supplied explicitly by the caller.
        base_profile = PROFILE_360P_17F.resized(width, height)
        target_frames = max(1, round(duration_seconds * base_profile.fps))
        segment_count = math.ceil(target_frames / base_profile.frames) if temporal_mode == "segmented" else 1
        profile = base_profile if temporal_mode == "segmented" else base_profile.with_frame_count(target_frames)
        job = Job(id=uuid.uuid4().hex[:12], model_id="manual", kind="inference")
        destination = self.workspace / ".h3-workbench" / "outputs" / f"h3-{job.id}.mp4"
        performance_log = self.workspace / ".h3-workbench" / "performance" / f"h3-{job.id}.jsonl"
        job.output_dir = str(destination.parent)
        job.performance_log = str(performance_log)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(
            self._run_inference,
            job.id,
            token_ids,
            prompt,
            profile,
            steps,
            seed,
            target_frames,
            segment_count,
            temporal_mode,
            attention_query_chunk,
            l1_prefetch_shards,
            destination,
            use_acceleration_lora,
        )
        return job

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                if key == "progress":
                    value = max(job.progress, float(value))
                setattr(job, key, value)

    def performance_log_path(self, job_id: str) -> Path | None:
        job = self.get(job_id)
        if job is None or job.performance_log is None:
            return None
        path = Path(job.performance_log).resolve()
        performance_root = (self.workspace / ".h3-workbench" / "performance").resolve()
        if path.parent != performance_root:
            return None
        return path

    def _performance_state(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            return {
                "job_id": job.id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "activity": dict(job.activity),
                "prefetch": dict(job.prefetch),
            }

    def _run_export(self, job_id: str, source: Path, destination: Path, video_blocks: str) -> None:
        self._update(
            job_id,
            status="running",
            started_at=_now(),
            message="Loading checkpoint",
            activity={"phase": "export", "module": source.name, "operation": "Loading checkpoint"},
        )

        def report(progress: float, message: str) -> None:
            self._update(
                job_id,
                progress=progress,
                message=message,
                activity={"phase": "export", "module": source.name, "operation": message},
            )

        try:
            result = export_checkpoint(source, destination, video_blocks, report)
            self._update(
                job_id,
                status="completed",
                progress=1.0,
                message="Export and validation completed",
                activity={"phase": "completed", "module": source.name, "operation": "Export completed"},
                finished_at=_now(),
                result=result,
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                message=str(exc),
                activity={"phase": "failed", "module": source.name, "operation": str(exc)},
                finished_at=_now(),
                error="".join(traceback.format_exception(exc)),
            )

    def _run_inference(
        self,
        job_id: str,
        token_ids: list[int] | None,
        prompt: str | None,
        profile: GenerationProfile,
        steps: int,
        seed: int,
        target_frames: int,
        segment_count: int,
        temporal_mode: str,
        attention_query_chunk: int,
        l1_prefetch_shards: int,
        destination: Path,
        use_acceleration_lora: bool,
    ) -> None:
        self._update(
            job_id,
            status="running",
            started_at=_now(),
            message="Encoding prompt" if prompt else "Encoding text tokens",
            progress=0.01,
            activity={"phase": "text", "module": "Tokenizer", "operation": "Encoding prompt" if prompt else "Using token IDs"},
        )
        runner: ORTGraphRunner | None = None
        qwen: QwenTextRuntime | None = None
        monitor: PerformanceMonitor | None = None
        warmup_executor: ThreadPoolExecutor | None = None
        warmup_future: Any | None = None
        warm_runtime: H3MainRuntime | None = None
        task_runtime: H3MainRuntime | None = None
        active_runtime: H3MainRuntime | None = None
        reservation_token = f"{os.getpid()}-{job_id}"
        heartbeat_stop = threading.Event()
        reservation_active = False
        try:
            monitor = PerformanceMonitor(
                self.performance_log_path(job_id)
                or self.workspace / ".h3-workbench" / "performance" / f"h3-{job_id}.jsonl",
                lambda: self._performance_state(job_id),
                lambda record: self._update(job_id, performance=record),
            )
            monitor.start()
        except Exception as exc:
            monitor = None
            self._update(
                job_id,
                performance={"timestamp": _now(), "monitor_error": f"Performance monitor unavailable: {exc}"},
            )
        try:
            if prompt:
                token_ids = encode_prompt(prompt, self.workspace / "qwen_tokenizer").tolist()
            assert token_ids
            qwen_dir = resolve_qwen_directory(self.output_root)
            base_main_dir = resolve_main_model_directory(self.workspace, self.output_root, accelerated=False)
            base_ready = base_main_dir is not None
            use_acceleration = use_acceleration_lora
            if not base_ready:
                raise RuntimeError(
                    "No validated FL2VA streaming Base model is installed. "
                    "The dynamic Turbo adapter also depends on this Base product."
                )
            main_dir = base_main_dir
            assert main_dir is not None
            turbo_manifest: dict[str, Any] | None = None
            turbo_adapter_dir: Path | None = None
            turbo_lora_path: Path | None = None
            turbo_grid_path: Path | None = None
            if use_acceleration:
                turbo_preset = export_preset("fl2va_turbo_v4")
                turbo_adapter_dir = (self.workspace / turbo_preset.output_dir).resolve()
                turbo_lora_path = self._existing_source_asset(turbo_preset.source)
                turbo_grid_path = (
                    self._existing_source_asset(turbo_preset.support)
                    if turbo_preset.support is not None
                    else None
                )
                if turbo_lora_path is None or turbo_grid_path is None:
                    raise RuntimeError(
                        "Turbo v4 dynamic LoRA assets are missing; install the runtime adapter before using 4-8 steps"
                    )
                try:
                    turbo_manifest = validate_turbo_adapter(
                        turbo_adapter_dir,
                        base_model_dir=main_dir,
                    )
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        "Turbo v4 dynamic LoRA is not ready for the installed Base model; "
                        "rebuild the runtime adapter from the Models page"
                    ) from exc

            def new_turbo_adapter() -> TurboLoraAdapter | None:
                if not use_acceleration:
                    return None
                assert turbo_adapter_dir is not None
                assert turbo_lora_path is not None
                assert turbo_grid_path is not None
                return TurboLoraAdapter(
                    turbo_adapter_dir,
                    turbo_lora_path,
                    turbo_grid_path,
                    strength=1.0,
                    base_model_dir=main_dir,
                )

            model_label = "Turbo v4 dynamic LoRA" if use_acceleration else "FL2VA Base"
            self._update(
                job_id,
                message=f"Selected {model_label}",
                activity={
                    "phase": "model",
                    "module": model_label,
                    "operation": f"Selected for {steps}-step sampling",
                    "acceleration_active": use_acceleration,
                },
            )
            runner = ORTGraphRunner(prefer_cuda=True)
            provider_name = runner.provider
            if qwen_dir.name.endswith("_int8_virtual") and provider_name != "CUDAExecutionProvider":
                raise RuntimeError("The validated Qwen INT8 virtual runtime requires CUDA execution")
            qwen = QwenTextRuntime(qwen_dir, runner, l1_prefetch_shards)

            # Claim as much VRAM as this job may want so a concurrent process
            # plans against the remainder instead of the full free pool.
            if provider_name == "CUDAExecutionProvider":
                snapshot = probe_gpu_memory()
                requested = max(0, snapshot.free_bytes - 768 * 1024**2 - profile.attention_workspace_bytes)
                if acquire_reservation(reservation_token, requested, device=snapshot.device_key):
                    reservation_active = True

                    def reservation_heartbeat() -> None:
                        while not heartbeat_stop.wait(30.0):
                            refresh_reservation(reservation_token)

                    threading.Thread(
                        target=reservation_heartbeat,
                        name=f"h3-vram-{job_id}",
                        daemon=True,
                    ).start()

            # The Qwen encode leaves the GPU nearly idle for minutes; build the
            # step-invariant FL2VA sessions on a background thread so the first
            # sampling step (and every later step) hits the session cache.
            if (
                qwen.persistent is not None or qwen.int8_virtual is not None
            ) and provider_name == "CUDAExecutionProvider":
                warm_runtime = H3MainRuntime(
                    main_dir,
                    runner,
                    profile,
                    attention_query_chunk,
                    None,
                    l1_prefetch_shards,
                    turbo_adapter=new_turbo_adapter(),
                )
                warmup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-warmup")
                warmup_future = warmup_executor.submit(warm_runtime.warm_fixed_sessions)

            def report_activity(activity: dict[str, object], **context: object) -> None:
                details = {"phase": "inference", **activity, **context}
                module = str(details.get("module", "Runtime"))
                operation = str(details.get("operation", "Running"))
                self._update(job_id, message=f"{module}: {operation}", activity=details)

            text_states = qwen.encode_token_ids(
                np.asarray(token_ids, dtype=np.int64),
                lambda operation, current, total: self._update(
                    job_id,
                    progress=0.02 + 0.14 * current / total,
                    message=f"Qwen: {operation}",
                    activity={
                        "phase": "text",
                        "module": "Qwen",
                        "operation": operation,
                        "current": current,
                        "total": total,
                    },
                ),
                lambda activity: self._update(
                    job_id,
                    prefetch={"phase": "text", "provider": provider_name, **activity},
                ),
            )
            if warmup_future is not None:
                try:
                    warm_result = warmup_future.result()
                    self._update(
                        job_id,
                        prefetch={"phase": "text", "provider": provider_name, "operation": "FL2VA warmup", **warm_result},
                    )
                except Exception as exc:  # noqa: BLE001 - warmup must never fail the job
                    self._update(
                        job_id,
                        prefetch={"phase": "text", "provider": provider_name, "operation": "FL2VA warmup failed", "error": str(exc)},
                    )
                    if warm_runtime is not None:
                        warm_runtime.close()
                        warm_runtime = None
                finally:
                    if warmup_executor is not None:
                        warmup_executor.shutdown(wait=True, cancel_futures=True)
                        warmup_executor = None
                    warmup_future = None
            _close_qwen_runtime_weights(qwen)
            qwen = None
            video_onnx = self.output_root / "video_vae"
            video_manifest = video_onnx / "manifest.json"
            pixel_segments: list[np.ndarray] = []
            audio_segments: list[np.ndarray] = []
            audio_warnings: list[str] = []
            main_runtime_metrics: list[dict[str, object]] = []
            for segment in range(segment_count):
                self._update(
                    job_id,
                    progress=0.18 + 0.64 * segment / segment_count,
                    message=f"Sampling segment {segment + 1}/{segment_count} with {provider_name}",
                    activity={
                        "phase": "sampling",
                        "module": "FL2VA",
                        "operation": "Preparing segment",
                        "segment": segment + 1,
                        "segments": segment_count,
                        "provider": provider_name,
                    },
                )
                video, audio = initial_latents(profile, seed + segment)

                def report_main(activity: dict[str, object], segment: int = segment) -> None:
                    details = {
                        "phase": "sampling",
                        **activity,
                        "segment": segment + 1,
                        "segments": segment_count,
                        "provider": provider_name,
                    }
                    sampling_step = int(details.get("sampling_step", 1))
                    shard = int(details.get("shard", 0))
                    shards = max(1, int(details.get("shards", 1)))
                    within_step = shard / shards if shard else 0.0
                    step_fraction = (sampling_step - 1 + within_step) / steps
                    progress = 0.18 + 0.64 * (segment + 0.85 * step_fraction) / segment_count
                    module = str(details.get("module", "FL2VA"))
                    operation = str(details.get("operation", "Running"))
                    if module.endswith("Loader"):
                        self._update(job_id, prefetch=details)
                        return
                    self._update(job_id, progress=progress, message=f"{module}: {operation}", activity=details)

                if task_runtime is None:
                    if warm_runtime is not None:
                        task_runtime = warm_runtime
                        warm_runtime = None
                    else:
                        task_runtime = H3MainRuntime(
                            main_dir,
                            runner,
                            profile,
                            attention_query_chunk,
                            report_main,
                            l1_prefetch_shards,
                            turbo_adapter=new_turbo_adapter(),
                        )
                # Keep the persistent host-RAM working set alive across
                # segmented clips while changing only the progress callback.
                task_runtime.activity_callback = report_main
                active_runtime = task_runtime
                try:
                    video, audio = sample_latents(
                        active_runtime,
                        video,
                        audio,
                        text_states,
                        steps,
                        lambda current, total, segment=segment: self._update(
                            job_id,
                            progress=0.18 + 0.64 * (segment + 0.85 * current / total) / segment_count,
                            message=(
                                f"Segment {segment + 1}/{segment_count}, "
                                f"sampling step {current}/{total}"
                            ),
                            activity={
                                "phase": "sampling",
                                "module": "FL2VA",
                                "operation": "Sampling step completed",
                                "sampling_step": current,
                                "sampling_steps": total,
                                "segment": segment + 1,
                                "segments": segment_count,
                            },
                        ),
                    )
                    if active_runtime.audio_fallback_reason is not None:
                        audio_warnings.append(active_runtime.audio_fallback_reason)
                    segment_metrics = active_runtime.metrics()
                    segment_metrics["segment"] = segment + 1
                    main_runtime_metrics.append(segment_metrics)
                finally:
                    active_runtime = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._update(
                    job_id,
                    progress=0.18 + 0.64 * (segment + 0.9) / segment_count,
                    message=f"Decoding video segment {segment + 1}/{segment_count}",
                    activity={
                        "phase": "video_decode",
                        "module": "Video VAE",
                        "operation": "Preparing decoder",
                        "segment": segment + 1,
                        "segments": segment_count,
                    },
                )
                def video_callback(activity: dict[str, object], segment: int = segment) -> None:
                    report_activity(
                        activity,
                        phase="video_decode",
                        segment=segment + 1,
                        segments=segment_count,
                    )
                if video_manifest.is_file():
                    segment_pixels = decode_video_latents_onnx(
                        video_onnx,
                        video,
                        profile.output_height,
                        profile.output_width,
                        callback=video_callback,
                    )
                else:
                    segment_pixels = decode_video_latents(
                        self.workspace / "minimax_h3_video_vae_fp16.safetensors",
                        video,
                        profile.output_height,
                        profile.output_width,
                        callback=video_callback,
                    )
                pixel_segments.append(segment_pixels)
                audio_segments.append(audio)
                del video, audio, segment_pixels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            runner.close()
            runner = None
            del text_states
            gc.collect()
            pixels = np.concatenate(pixel_segments, axis=2)[:, :, :target_frames]
            target_audio_frames = math.ceil(target_frames * profile.audio_latents_per_second / profile.fps)
            audio = np.concatenate(audio_segments, axis=3)[:, :, :, :target_audio_frames]
            del pixel_segments, audio_segments
            self._update(
                job_id,
                progress=0.92,
                message="Decoding audio",
                activity={"phase": "audio_decode", "module": "Audio VAE", "operation": "Preparing decoder"},
            )
            audio_onnx = self.output_root / "audio_vae"
            def audio_callback(activity: dict[str, object]) -> None:
                report_activity(activity, phase="audio_decode")

            audio_status = "generated"
            audio_warning = "; ".join(dict.fromkeys(audio_warnings)) if audio_warnings else None
            if audio_warning is not None or not np.isfinite(audio).all():
                audio_status = "silent_fallback"
                audio_warning = audio_warning or "Non-finite FL2VA audio latent"
                waveform = np.zeros((1, 2, target_audio_frames * 800), dtype=np.float32)
                self._update(
                    job_id,
                    message="Audio unstable; using a silent track",
                    activity={
                        "phase": "audio_decode",
                        "module": "Audio VAE",
                        "operation": "Using silent fallback",
                        "reason": audio_warning,
                    },
                )
            else:
                try:
                    if (audio_onnx / "manifest.json").is_file():
                        waveform = decode_audio_latents_onnx(audio_onnx, audio, callback=audio_callback)
                    else:
                        waveform = decode_audio_latents(
                            self.workspace / "minimax_h3_audio_vae_fp32.safetensors",
                            audio,
                            audio_callback,
                        )
                except FloatingPointError as exc:
                    audio_status = "silent_fallback"
                    audio_warning = str(exc)
                    waveform = np.zeros((1, 2, target_audio_frames * 800), dtype=np.float32)
                    self._update(
                        job_id,
                        message="Audio decoder unstable; using a silent track",
                        activity={
                            "phase": "audio_decode",
                            "module": "Audio VAE",
                            "operation": "Using silent fallback",
                            "reason": audio_warning,
                        },
                    )
            self._update(
                job_id,
                progress=0.97,
                message="Writing MP4",
                activity={"phase": "mux", "module": "FFmpeg", "operation": "Encoding H.264/AAC"},
            )
            completed_at = _now()
            metadata = {
                "schema": "h3-workbench-output-v1",
                "job": {
                    "id": job_id,
                    "created_at": self.get(job_id).created_at if self.get(job_id) is not None else None,
                    "started_at": self.get(job_id).started_at if self.get(job_id) is not None else None,
                    "completed_at": completed_at,
                    "performance_log": str(self.performance_log_path(job_id) or ""),
                },
                "input": {
                    "prompt": prompt,
                    "token_ids": token_ids,
                },
                "sampling": {
                    "seed": seed,
                    "steps": steps,
                    "scheduler": "simple_shifted_flow",
                    "video_shift": 12.0,
                    "audio_shift": 3.0,
                    "video_sigmas": shifted_flow_sigmas(steps, 12.0),
                    "audio_schedule": "dual_clock_time_shift_from_video_12_to_audio_3",
                },
                "output": {
                    "path": str(destination),
                    "width": profile.output_width,
                    "height": profile.output_height,
                    "frames": target_frames,
                    "fps": profile.fps,
                    "duration_seconds": target_frames / profile.fps,
                    "audio_sample_rate": 32000,
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                },
                "temporal": {
                    "mode": temporal_mode,
                    "segments": segment_count,
                    "profile": profile.to_dict(),
                },
                "runtime": {
                    "provider": provider_name,
                    "main_variant": TURBO_ADAPTER_VARIANT if use_acceleration else "base",
                    "use_acceleration_lora": use_acceleration,
                    "attention_query_chunk": attention_query_chunk,
                    "l1_prefetch_shards": l1_prefetch_shards,
                    "l2_cache_gib": os.environ.get("H3_L2_CACHE_GIB"),
                    "prefetch_shards": os.environ.get("H3_PREFETCH_SHARDS"),
                    "weight_ram_cache": os.environ.get("H3_WEIGHT_RAM_CACHE", "auto"),
                    "weight_ram_cache_gib": os.environ.get("H3_WEIGHT_RAM_CACHE_GIB"),
                    "segments": main_runtime_metrics,
                },
                "models": {
                    "main": _manifest_identity(main_dir),
                    "turbo_adapter": (
                        {
                            "directory": str(turbo_adapter_dir),
                            "variant": TURBO_ADAPTER_VARIANT,
                            "strength": 1.0,
                            "factor_pairs": turbo_manifest.get("factor_pairs"),
                            "factor_storage_bytes": turbo_manifest.get("factor_storage_bytes"),
                            "assets": turbo_manifest.get("assets"),
                            "application": "runtime_low_rank_overlay",
                        }
                        if turbo_manifest is not None
                        else None
                    ),
                    "text_encoder": _manifest_identity(qwen_dir),
                    "video_vae": _manifest_identity(video_onnx),
                    "audio_vae": _manifest_identity(audio_onnx),
                },
            }
            metadata_path = write_mp4(destination, pixels, waveform, profile.fps, metadata)
            self._update(
                job_id,
                status="completed",
                progress=1.0,
                message=(
                    "Video generation completed with silent audio"
                    if audio_status == "silent_fallback"
                    else "Video generation completed"
                ),
                activity={
                    "phase": "completed",
                    "module": "Output",
                    "operation": "MP4 completed",
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                },
                finished_at=completed_at,
                result={
                    "output": str(destination),
                    "metadata": str(metadata_path) if metadata_path is not None else None,
                    "prompt": prompt,
                    "token_ids": token_ids,
                    "main_variant": TURBO_ADAPTER_VARIANT if use_acceleration else "base",
                    "use_acceleration_lora": use_acceleration,
                    "profile": profile.to_dict(),
                    "steps": steps,
                    "seed": seed,
                    "frames": target_frames,
                    "duration_seconds": target_frames / profile.fps,
                    "segments": segment_count,
                    "temporal_mode": temporal_mode,
                    "attention_query_chunk": attention_query_chunk,
                    "l1_prefetch_shards": l1_prefetch_shards,
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                },
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                message=str(exc),
                activity={"phase": "failed", "module": "Runtime", "operation": str(exc)},
                finished_at=_now(),
                error="".join(traceback.format_exception(exc)),
            )
        finally:
            heartbeat_stop.set()
            if warmup_future is not None:
                warmup_future.cancel()
            if warmup_executor is not None:
                warmup_executor.shutdown(wait=True, cancel_futures=True)
            if active_runtime is not None:
                active_runtime.close()
                active_runtime = None
            if task_runtime is not None:
                task_runtime.close()
                task_runtime = None
            if warm_runtime is not None:
                warm_runtime.close()
            _close_qwen_runtime_weights(qwen)
            if reservation_active:
                release_reservation(reservation_token)
            if runner is not None:
                runner.close()
            if monitor is not None:
                monitor.stop()
