from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from h3_workbench.exporter import export_checkpoint
from h3_workbench.acceleration import shifted_flow_sigmas
from h3_workbench.direct_download import download_file
from h3_workbench.inference_runtime import (
    H3MainRuntime,
    ORTGraphRunner,
    QwenTextRuntime,
    VideoLatentCondition,
    initial_latents,
    initial_ref2va_latents,
    sample_latents,
    sample_ref2va_latents,
)
from h3_workbench.qwen_persistent import resolve_qwen_directory
from h3_workbench.media_output import (
    decode_audio_latents,
    decode_audio_latents_onnx,
    decode_video_latents,
    decode_video_latents_onnx,
    encode_image_frame,
    encode_video_frames,
    audio_vae_encoder_ready,
    load_video_vae_onnx_encoder,
    select_video_vae_encoder_backend,
    video_vae_temporal_encoder_ready,
    write_mp4,
    write_mp4_with_audio_source,
)
from h3_workbench.media_input import (
    InterpolationMode,
    prepare_frame_condition,
    prepare_super_resolution_segment,
    probe_image,
    probe_video,
    read_image,
    read_video_frames,
    resolve_image_path,
    resolve_video_path,
)
from h3_workbench.performance_monitor import PerformanceMonitor
from h3_workbench.profiles import (
    PROFILE_360P_17F,
    GenerationProfile,
    video_latent_index_for_output_frame,
    video_latent_frames_for_output,
    video_vae_output_frames,
)
from h3_workbench.tokenizer import encode_prompt, load_tokenizer
from h3_workbench.tokenizer import tokenizer_files_ready
from h3_workbench.reference import (
    H3_FPS,
    ReferenceSpec,
    align_num_frames,
    resolve_reference_specs,
)
from h3_workbench.reference_conditioning import (
    build_reference_layout,
    encode_reference_latents,
    encode_reference_vision,
    normalize_reference_media,
    resolve_reference_image_short_edge,
)
from h3_workbench.model_registry import inspect_checkpoint
from h3_workbench.memory_planner import probe_gpu_memory
from h3_workbench.source_catalog import ExportPreset, SourceAsset, export_preset
from h3_workbench.ref2va_lora import (
    REF2VA_ADAPTER_VARIANT,
    Ref2VALoraAdapter,
    publish_ref2va_adapter,
    validate_ref2va_adapter,
)
from h3_workbench.vram_reservation import (
    acquire_reservation,
    configure_reservations,
    refresh_reservation,
    release_reservation,
)


logger = logging.getLogger(__name__)


def _complete_main_model(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("component") == "ref2va_transformer":
        try:
            from h3_workbench.ref2va_virtual_slicer import validated_ref2va_virtual_ready

            if not validated_ref2va_virtual_ready(directory):
                return False
        except (ImportError, OSError, TypeError, ValueError):
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
    component: str | None = None,
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
        if component is not None:
            manifest_component = manifest.get("component")
            legacy_component_match = (
                manifest_component is None
                and component == "fl2va_transformer"
                and "fl2va" in directory.name.lower()
            )
            if manifest_component != component and not legacy_component_match:
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


def _resolve_qwen_visual_checkpoint(directory: Path) -> Path:
    """Resolve the visual tower from the same source as the Qwen virtual runtime."""
    candidates = (directory / "runtime_int8_manifest.json", directory / "manifest.json")
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw_source = manifest.get("source_checkpoint") or manifest.get("source")
        if not raw_source:
            continue
        source = Path(str(raw_source))
        if not source.is_absolute():
            source = directory / source
        source = source.resolve()
        if source.is_file():
            return source
    raise RuntimeError(
        "The Qwen virtual runtime does not expose a readable source checkpoint for the visual tower"
    )


def _release_video_encoder(encoder: Any | None) -> None:
    if encoder is None:
        return
    close = getattr(encoder, "close", None)
    if callable(close):
        close()
        return
    try:
        if next(encoder.parameters()).device.type == "cuda":
            encoder.to("cpu", memory_format=torch.contiguous_format)
            torch.cuda.empty_cache()
    except (AttributeError, StopIteration):
        pass


def _segment_video_condition(
    profile: GenerationProfile,
    target_frames: int,
    segment: int,
    segment_count: int,
    temporal_mode: str,
    frame_anchors: dict[str, np.ndarray],
) -> VideoLatentCondition | None:
    anchors: dict[int, np.ndarray] = {}
    if "start" in frame_anchors and segment == 0:
        anchors[0] = frame_anchors["start"]

    if "end" in frame_anchors and segment == segment_count - 1:
        retained_frames = target_frames
        if temporal_mode == "segmented":
            retained_frames = target_frames - segment * profile.frames
        retained_frames = max(1, min(profile.frames, retained_frames))
        latent_index = video_latent_index_for_output_frame(
            retained_frames,
            profile.video_latent_frames,
        )
        if latent_index in anchors:
            raise ValueError("First and last frame conditions map to the same latent token")
        anchors[latent_index] = frame_anchors["end"]

    if not anchors:
        return None
    indices = tuple(sorted(anchors))
    clean = np.ascontiguousarray(np.concatenate([anchors[index] for index in indices], axis=2), dtype=np.float32)
    return VideoLatentCondition(indices=indices, clean=clean)


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
            "ref2va_transformer": "minimax_h3_ref2va_pruned_bf16_virtual",
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

    def _ref2va_acceleration_assets(
        self, main_dir: Path
    ) -> tuple[Path, Path, dict[str, Any]]:
        preset = export_preset("ref2va_turbo_v0_1")
        adapter_dir = (self.workspace / preset.output_dir).resolve()
        lora_path = self._existing_source_asset(preset.source)
        if lora_path is None:
            raise RuntimeError(
                "Ref2VA Turbo LoRA is missing; install the Ref2VA acceleration adapter from the Models page"
            )
        try:
            manifest = validate_ref2va_adapter(adapter_dir, base_model_dir=main_dir)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Ref2VA Turbo LoRA is not ready for the installed Ref2VA base; "
                "rebuild the runtime adapter from the Models page"
            ) from exc
        return adapter_dir, lora_path, manifest

    def _run_preset_export(self, job_id: str, preset: ExportPreset) -> None:
        try:
            self._update(job_id, status="running", started_at=_now(), progress=0.01, message="Preparing verified export preset", activity={"phase": "download", "module": preset.label, "operation": "Checking source files"})
            source_label = "Ref2VA Turbo LoRA" if preset.component == "acceleration_lora" else "Original checkpoint"
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
                base_main_dir = resolve_main_model_directory(
                    self.workspace,
                    self.output_root,
                    accelerated=False,
                    component="ref2va_transformer",
                )
                if base_main_dir is None:
                    raise RuntimeError(
                        "Ref2VA Turbo LoRA requires the validated Ref2VA virtual base product"
                    )
                destination = self.workspace / preset.output_dir
                self._update(
                    job_id,
                    progress=0.20,
                    message="Publishing graph-only Ref2VA adapter",
                    activity={
                        "phase": "export",
                        "module": preset.label,
                        "operation": "Building runtime overlay topologies",
                    },
                )
                manifest = publish_ref2va_adapter(
                    base_main_dir,
                    destination,
                    source,
                )
                self._update(
                    job_id,
                    status="completed",
                    progress=1.0,
                    message="Ref2VA Turbo runtime adapter completed",
                    activity={
                        "phase": "completed",
                        "module": preset.label,
                        "operation": "Dynamic LoRA adapter verified",
                    },
                    finished_at=_now(),
                    result={
                        "preset": preset.id,
                        "source": str(source),
                        "base": str(base_main_dir),
                        "output": str(destination),
                        "adapter": manifest,
                    },
                )
                return
            lora_path = self._download_source_asset(job_id, preset.lora, "Acceleration LoRA") if preset.lora else None
            if preset.support:
                support_path = self._download_source_asset(job_id, preset.support, "Acceleration support asset")
                if lora_path is None:
                    raise RuntimeError("Acceleration support asset requires a LoRA checkpoint")
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
        conditioning_mode: str = "text",
        start_image_path: str | None = None,
        end_image_path: str | None = None,
        references: list[dict[str, object]] | None = None,
    ) -> Job:
        if not 1 <= steps <= 50:
            raise ValueError("Inference steps must be from 1 to 50")
        if use_acceleration_lora and steps != 4:
            raise ValueError("Ref2VA Turbo acceleration LoRA supports exactly 4 sampling steps")
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
        if conditioning_mode not in {"text", "first", "last", "first_last"}:
            raise ValueError("Unknown frame conditioning mode")
        expected = {
            "text": (False, False),
            "first": (True, False),
            "last": (False, True),
            "first_last": (True, True),
        }[conditioning_mode]
        supplied = (bool(start_image_path), bool(end_image_path))
        if supplied != expected:
            raise ValueError(f"Frame conditioning inputs do not match mode {conditioning_mode!r}")
        reference_specs: tuple[ReferenceSpec, ...] = ()
        if references:
            if not prompt or not prompt.strip():
                raise ValueError("Ref2VA references require a prompt")
            if token_ids:
                raise ValueError("Ref2VA references require prompt text, not token IDs")
            if temporal_mode != "native":
                raise ValueError("Ref2VA references require native temporal mode")
            if conditioning_mode != "text" or start_image_path or end_image_path:
                raise ValueError("Ref2VA references are incompatible with first/last frame conditioning")
            if not 5.0 <= duration_seconds <= 15.0:
                raise ValueError("Ref2VA generation duration must be from 5 to 15 seconds")
            reference_specs = resolve_reference_specs(self.workspace, references)
        # Keep the model's validated temporal/FPS geometry as an internal template;
        # output dimensions and duration are always supplied explicitly by the caller.
        base_profile = PROFILE_360P_17F.resized(width, height)
        target_frames = max(1, round(duration_seconds * base_profile.fps))
        if reference_specs:
            target_frames = align_num_frames(target_frames)
        if conditioning_mode == "first_last" and target_frames < 2:
            raise ValueError("First/last conditioning requires an output of at least two frames")
        segment_count = math.ceil(target_frames / base_profile.frames) if temporal_mode == "segmented" else 1
        profile = base_profile if temporal_mode == "segmented" else base_profile.with_frame_count(target_frames)
        frame_conditioning: dict[str, str] | None = None
        if conditioning_mode != "text":
            video_product = (self.workspace / export_preset("video_vae").output_dir).resolve()
            if not video_vae_temporal_encoder_ready(video_product):
                raise ValueError(
                    "Frame conditioning requires the staged Video VAE encoder; "
                    "re-export the Video VAE encoder product"
                )
            frame_conditioning = {"mode": conditioning_mode}
            if start_image_path:
                frame_conditioning["start"] = str(resolve_image_path(self.workspace, start_image_path))
            if end_image_path:
                frame_conditioning["end"] = str(resolve_image_path(self.workspace, end_image_path))
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
            frame_conditioning=frame_conditioning,
            references=reference_specs or None,
        )
        return job

    def create_super_resolution(
        self,
        source_path: str,
        prompt: str | None,
        scale: float,
        interpolation: InterpolationMode,
        noise_strength: float,
        processing_mode: str,
        steps: int,
        seed: int,
        attention_query_chunk: int = 512,
        l1_prefetch_shards: int = 2,
        use_acceleration_lora: bool = False,
    ) -> Job:
        source = resolve_video_path(self.workspace, source_path)
        info = probe_video(source)
        if not 1.0 <= scale <= 4.0:
            raise ValueError("Super-resolution scale must be from 1.0 to 4.0")
        if interpolation not in {"nearest", "bilinear", "bicubic", "trilinear"}:
            raise ValueError("Unsupported interpolation mode")
        if processing_mode not in {"segmented", "direct"}:
            raise ValueError("Super-resolution processing mode must be 'segmented' or 'direct'")
        if not 0.0 <= noise_strength <= 1.0:
            raise ValueError("Noise strength must be from 0.0 to 1.0")
        if not 1 <= steps <= 50:
            raise ValueError("Inference steps must be from 1 to 50")
        if use_acceleration_lora and steps != 4:
            raise ValueError("Ref2VA Turbo acceleration LoRA supports exactly 4 sampling steps")
        if attention_query_chunk not in {32, 64, 128, 256, 512}:
            raise ValueError("Attention query chunk must be one of 32, 64, 128, 256, or 512")
        if not 0 <= l1_prefetch_shards <= 4:
            raise ValueError("L1 prefetch shards must be from 0 to 4")
        output_width = max(32, round(info.width * scale))
        output_height = max(32, round(info.height * scale))
        if output_width > 2048 or output_height > 2048 or output_width * output_height > 4_194_304:
            raise ValueError("Super-resolution output is limited to 2048 px per side and 4 MP")
        manual_prompt = prompt.strip() if prompt and prompt.strip() else None
        final_prompt = manual_prompt or info.prompt
        if not final_prompt:
            raise ValueError("No prompt found in video metadata; provide a manual prompt")
        video_preset = export_preset("video_vae")
        video_product = (self.workspace / video_preset.output_dir).resolve()
        if not video_vae_temporal_encoder_ready(video_product):
            raise ValueError(
                "Super-resolution requires the staged Video VAE encoder; "
                "re-export the Video VAE encoder product"
            )

        profile = PROFILE_360P_17F.resized(output_width, output_height)
        if processing_mode == "direct":
            if info.frames > 360:
                raise ValueError("Direct super-resolution is limited to 360 input frames; use video segmentation")
            profile = profile.with_frame_count(info.frames)
            segment_frames = info.frames
            segment_stride = info.frames
            temporal_overlap = 0
            segment_count = 1
            temporal_mode = "native"
        else:
            temporal_overlap = 5
            segment_stride = PROFILE_360P_17F.frames - temporal_overlap
            segment_frames = PROFILE_360P_17F.frames
            segment_count = max(1, math.ceil(max(1, info.frames - temporal_overlap) / segment_stride))
            temporal_mode = "segmented"
        job = Job(id=uuid.uuid4().hex[:12], model_id=source.name, kind="super_resolution")
        destination = self.workspace / ".h3-workbench" / "outputs" / f"h3-sr-{job.id}.mp4"
        performance_log = self.workspace / ".h3-workbench" / "performance" / f"h3-{job.id}.jsonl"
        job.output_dir = str(destination.parent)
        job.performance_log = str(performance_log)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(
            self._run_inference,
            job.id,
            None,
            final_prompt,
            profile,
            steps,
            seed,
            info.frames,
            segment_count,
            temporal_mode,
            attention_query_chunk,
            l1_prefetch_shards,
            destination,
            use_acceleration_lora,
            output_fps=max(1.0, info.fps),
            super_resolution={
                "source": str(source),
                "source_info": info.to_dict(),
                "scale": scale,
                "interpolation": interpolation,
                "noise_strength": noise_strength,
                "segment_frames": segment_frames,
                "segment_stride": segment_stride,
                "temporal_overlap": temporal_overlap,
                "processing_mode": processing_mode,
                "prompt_source": "manual" if manual_prompt is not None else info.prompt_source,
            },
        )
        return job

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                if key == "progress":
                    value = max(job.progress, float(value))
                setattr(job, key, value)

    def _update_performance(self, job_id: str, record: dict[str, Any]) -> None:
        """Publish performance samples and periodically mirror a heartbeat to the server log."""
        self._update(job_id, performance=record)
        if "performance" not in record:
            return
        sequence = int(record.get("sequence", 0))
        if sequence % 15 != 0:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                return
            activity = dict(job.activity)
            metrics = record.get("performance")
            metrics = metrics if isinstance(metrics, dict) else {}
            gpu = metrics.get("gpu")
            gpu = gpu if isinstance(gpu, dict) else {}
            process = metrics.get("process")
            process = process if isinstance(process, dict) else {}
            position = ""
            if activity.get("current") is not None and activity.get("total") is not None:
                position = f" · {activity['current']}/{activity['total']}"
            elapsed = record.get("elapsed_seconds", "-")
            try:
                elapsed_text = f"{float(elapsed):.0f} 秒"
            except (TypeError, ValueError):
                elapsed_text = str(elapsed)
            resource_parts = []
            if gpu.get("utilization_percent") is not None:
                resource_parts.append(f"GPU {gpu['utilization_percent']:.0f}%")
            if process.get("cpu_percent") is not None:
                resource_parts.append(f"CPU {process['cpu_percent']:.0f}%")
            job.message = (
                f"仍在运行 · {activity.get('module', 'Runtime')} / "
                f"{activity.get('operation', 'Working')}{position} · 已耗时 {elapsed_text}"
                f"{' · ' + ' · '.join(resource_parts) if resource_parts else ''}"
            )
            logger.info(
                "job=%s heartbeat status=%s progress=%.1f%% phase=%s module=%s operation=%s position=%s/%s gpu=%s%% vram=%sMiB cpu=%s%% elapsed=%ss",
                job.id,
                job.status,
                job.progress * 100,
                activity.get("phase", ""),
                activity.get("module", ""),
                activity.get("operation", ""),
                activity.get("current", "-"),
                activity.get("total", "-"),
                gpu.get("utilization_percent", "-"),
                gpu.get("memory_used_mib", "-"),
                process.get("cpu_percent", "-"),
                record.get("elapsed_seconds", "-"),
            )

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

    def _run_ref2va_inference(
        self,
        job_id: str,
        token_ids: list[int] | None,
        prompt: str | None,
        profile: GenerationProfile,
        steps: int,
        seed: int,
        target_frames: int,
        attention_query_chunk: int,
        l1_prefetch_shards: int,
        destination: Path,
        references: tuple[ReferenceSpec, ...],
        use_acceleration_lora: bool,
    ) -> None:
        """Run the ordered multimodal Ref2VA path against the virtual base."""
        del token_ids
        runner: ORTGraphRunner | None = None
        qwen: QwenTextRuntime | None = None
        runtime: H3MainRuntime | None = None
        video_encoder: Any | None = None
        monitor: PerformanceMonitor | None = None
        reservation_token = f"{os.getpid()}-{job_id}"
        reservation_active = False
        heartbeat_stop = threading.Event()
        runtime_profile = profile
        provider_name = "unknown"
        main_dir: Path | None = None
        ref2va_adapter_dir: Path | None = None
        ref2va_lora_path: Path | None = None
        ref2va_adapter_manifest: dict[str, Any] | None = None
        qwen_dir: Path | None = None
        video_onnx = self.output_root / "video_vae"
        audio_onnx = self.output_root / "audio_vae"
        normalized = None
        vision = None
        vision_metadata: dict[str, object] = {}
        condition_video: tuple[np.ndarray, ...] = ()
        condition_audio: tuple[np.ndarray, ...] = ()
        noised_video: tuple[np.ndarray, ...] = ()
        video: np.ndarray | None = None
        audio: np.ndarray | None = None
        text_states: np.ndarray | None = None
        main_runtime_metrics: dict[str, object] = {}
        audio_warning: str | None = None
        audio_status = "generated"

        def update(progress: float, message: str, activity: dict[str, object]) -> None:
            self._update(job_id, progress=progress, message=message, activity=activity)

        def report_activity(activity: dict[str, object], **context: object) -> None:
            details = {"phase": "ref2va", **activity, **context}
            module = str(details.get("module", "Ref2VA"))
            operation = str(details.get("operation", "Running"))
            values: dict[str, object] = {
                "message": f"{module}: {operation}",
                "activity": details,
            }
            stage_progress = details.get("stage_progress")
            if (
                details.get("phase") == "reference_vision"
                and isinstance(stage_progress, (int, float))
                and math.isfinite(float(stage_progress))
            ):
                # Reserve the visible 8%-22% window for visual loading and
                # encoding, while the later VAE stage starts at 22%.
                values["progress"] = 0.08 + 0.12 * min(1.0, max(0.0, float(stage_progress)))
            self._update(job_id, **values)

        try:
            try:
                monitor = PerformanceMonitor(
                    self.performance_log_path(job_id)
                    or self.workspace / ".h3-workbench" / "performance" / f"h3-{job_id}.jsonl",
                    lambda: self._performance_state(job_id),
                    lambda record: self._update_performance(job_id, record),
                )
                monitor.start()
            except Exception as exc:
                monitor = None
                self._update(
                    job_id,
                    performance={"timestamp": _now(), "monitor_error": f"Performance monitor unavailable: {exc}"},
                )
            self._update(
                job_id,
                status="running",
                started_at=_now(),
                progress=0.01,
                message="Preparing Ref2VA references",
                activity={
                    "phase": "ref2va_prepare",
                    "module": "Ref2VA",
                    "operation": "Checking ordered reference assets",
                    "references": len(references),
                },
            )
            if not prompt or not prompt.strip():
                raise ValueError("Ref2VA references require a non-empty prompt")
            if target_frames < 1 or profile.frames != target_frames:
                raise ValueError("Ref2VA profile frame geometry is not aligned with the request")
            main_dir = resolve_main_model_directory(
                self.workspace,
                self.output_root,
                accelerated=False,
                component="ref2va_transformer",
            )
            if main_dir is None:
                raise RuntimeError("No validated Ref2VA virtual model is installed")
            if use_acceleration_lora:
                (
                    ref2va_adapter_dir,
                    ref2va_lora_path,
                    ref2va_adapter_manifest,
                ) = self._ref2va_acceleration_assets(main_dir)
            qwen_dir = resolve_qwen_directory(self.output_root)
            if not tokenizer_files_ready(self.workspace / "qwen_tokenizer"):
                raise RuntimeError("The H3 tokenizer is not installed")
            if not video_vae_temporal_encoder_ready(video_onnx):
                raise RuntimeError(
                    "Ref2VA references require the staged Video VAE encoder; "
                    "re-export the Video VAE product with encoder support"
                )
            if not audio_vae_encoder_ready(audio_onnx):
                raise RuntimeError(
                    "Ref2VA references require the complete Audio VAE encoder and decoder product"
                )

            # The multimodal path has no useful CPU fallback: the visual
            # tower and the split Qwen text runtime are both CUDA-bound on the
            # supported product.  Check the provider before loading the large
            # visual tower so a misconfigured runtime fails without allocating
            # it first.
            runner = ORTGraphRunner(prefer_cuda=True)
            provider_name = runner.provider
            if provider_name != "CUDAExecutionProvider":
                raise RuntimeError(
                    "Ref2VA multimodal conditioning requires CUDAExecutionProvider for the Qwen INT8 virtual runtime"
                )
            if not qwen_dir.name.endswith("_int8_virtual"):
                raise RuntimeError("Ref2VA requires the validated Qwen INT8 virtual runtime")

            tokenizer = load_tokenizer(self.workspace / "qwen_tokenizer")
            visual_checkpoint = _resolve_qwen_visual_checkpoint(qwen_dir)
            reference_image_short_edge = resolve_reference_image_short_edge(
                probe_gpu_memory().total_bytes
            )
            normalized = normalize_reference_media(
                references,
                target_frames,
                callback=lambda details: report_activity(details, phase="reference_normalize"),
                image_short_edge=reference_image_short_edge,
            )
            reference_media_metadata = [
                {
                    "kind": item.kind,
                    "source_width": item.spec.width,
                    "source_height": item.spec.height,
                    "source_duration_seconds": item.spec.duration_seconds,
                    "normalized_shape": list(item.pixels.shape) if item.pixels is not None else None,
                    "normalized_short_edge": (
                        min(int(item.pixels.shape[0]), int(item.pixels.shape[1]))
                        if item.pixels is not None and item.kind == "image"
                        else None
                    ),
                }
                for item in normalized
            ]
            update(
                0.08,
                "Encoding reference vision",
                {
                    "phase": "reference_vision",
                    "module": "Qwen Vision",
                    "operation": "Loading visual tower",
                    "checkpoint": str(visual_checkpoint),
                    "reference_image_short_edge": reference_image_short_edge,
                },
            )
            vision = encode_reference_vision(
                visual_checkpoint,
                tokenizer,
                prompt,
                normalized,
                prefer_cuda=True,
                callback=lambda details: report_activity(details, phase="reference_vision"),
            )
            update(
                0.22,
                "Encoding reference latents",
                {
                    "phase": "reference_vae",
                    "module": "Video/Audio VAE",
                    "operation": "Loading staged encoders",
                    "text_tokens": int(vision.presentation.token_ids.size),
                    "image_tokens": int(vision.visual_condition["image_mask"].sum()),
                    "video_tokens": int(vision.visual_condition["video_mask"].sum()),
                },
            )
            video_encoder = load_video_vae_onnx_encoder(video_onnx)
            condition_video, condition_audio = encode_reference_latents(
                video_encoder,
                audio_onnx,
                normalized,
                prefer_cuda=True,
                posterior_seed=42,
                callback=lambda details: report_activity(details, phase="reference_vae"),
            )
            layout = build_reference_layout(
                normalized,
                vision.presentation,
                condition_video,
                condition_audio,
                (
                    profile.video_latent_frames,
                    profile.video_latent_height,
                    profile.video_latent_width,
                ),
                profile.audio_latent_frames,
            )
            runtime_profile = replace(
                profile,
                text_tokens=int(vision.presentation.token_ids.size),
            )
            vision_metadata = {
                "qwen_position_shape": list(vision.position_ids.shape),
                "qwen_image_grid_thw": vision.image_grid_thw.tolist(),
                "qwen_video_grid_thw": vision.video_grid_thw.tolist(),
                "video_timestamps": [list(value) for value in vision.video_timestamps],
            }
            _release_video_encoder(video_encoder)
            video_encoder = None
            del normalized
            normalized = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            qwen = QwenTextRuntime(qwen_dir, runner, l1_prefetch_shards)
            if qwen.int8_virtual is None:
                raise RuntimeError("The Qwen INT8 virtual runtime is not ready for multimodal conditioning")
            if not isinstance(qwen.int8_virtual.attention_split, dict):
                raise RuntimeError(
                    "The Qwen INT8 virtual runtime is missing split attention graphs required by Ref2VA"
                )

            snapshot = probe_gpu_memory()
            requested = max(
                0,
                snapshot.free_bytes
                - 768 * 1024**2
                - runtime_profile.attention_workspace_bytes,
            )
            if acquire_reservation(reservation_token, requested, device=snapshot.device_key):
                reservation_active = True

                def reservation_heartbeat() -> None:
                    while not heartbeat_stop.wait(30.0):
                        refresh_reservation(reservation_token)

                threading.Thread(
                    target=reservation_heartbeat,
                    name=f"h3-ref2va-vram-{job_id}",
                    daemon=True,
                ).start()

            update(
                0.30,
                "Encoding multimodal prompt",
                {
                    "phase": "text",
                    "module": "Qwen",
                    "operation": "Injecting visual features and running text tower",
                    "tokens": int(vision.presentation.token_ids.size),
                    "image_tokens": int(vision.visual_condition["image_mask"].sum()),
                    "video_tokens": int(vision.visual_condition["video_mask"].sum()),
                },
            )
            assert qwen is not None
            assert vision is not None
            text_states = qwen.encode_token_ids(
                vision.presentation.token_ids,
                lambda operation, current, total: self._update(
                    job_id,
                    progress=0.30 + 0.14 * current / max(1, total),
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
                position_ids=vision.position_ids,
                visual_condition=vision.visual_condition,
            )
            _close_qwen_runtime_weights(qwen)
            qwen = None
            vision = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            noised_video, video, audio = initial_ref2va_latents(
                runtime_profile,
                seed,
                condition_video,
                condition_timestep=0.999,
            )
            update(
                0.46,
                "Sampling Ref2VA sequence",
                {
                    "phase": "sampling",
                    "module": "Ref2VA",
                    "operation": "Preparing packed reference rows",
                    "sampler": "res_multistep" if use_acceleration_lora else "euler",
                    "scheduler": "simple" if use_acceleration_lora else "minimax_h3",
                    "condition_video_rows": layout.num_condition_video_rows,
                    "condition_audio_rows": layout.num_condition_audio_rows,
                    "sequence_tokens": int(layout.token_tags.size),
                },
            )

            def report_main(activity: dict[str, object]) -> None:
                details = {
                    "phase": "sampling",
                    "sampler": "res_multistep" if use_acceleration_lora else "euler",
                    "scheduler": "simple" if use_acceleration_lora else "minimax_h3",
                    **activity,
                    "provider": provider_name,
                }
                module = str(details.get("module", "Ref2VA"))
                # Loader notifications are auxiliary progress. Keep the last
                # actual graph operation visible so a long weight wait does
                # not look like a stalled or failed inference.
                if module.endswith("Loader"):
                    self._update(job_id, prefetch=details)
                    return
                sampling_step = int(details.get("sampling_step", 1))
                shard = int(details.get("shard", 0))
                shards = max(1, int(details.get("shards", 1)))
                fraction = (sampling_step - 1 + (shard / shards if shard else 0.0)) / steps
                self._update(
                    job_id,
                    progress=0.46 + 0.34 * fraction,
                    message=f"{module}: {details.get('operation', 'Running')}",
                    activity=details,
                )

            assert main_dir is not None
            assert runner is not None
            assert video is not None and audio is not None and text_states is not None

            def new_ref2va_adapter() -> Ref2VALoraAdapter | None:
                if not use_acceleration_lora:
                    return None
                assert ref2va_adapter_dir is not None
                assert ref2va_lora_path is not None
                return Ref2VALoraAdapter(
                    ref2va_adapter_dir,
                    ref2va_lora_path,
                    strength=1.0,
                    base_model_dir=main_dir,
                )

            runtime = H3MainRuntime(
                main_dir,
                runner,
                runtime_profile,
                attention_query_chunk,
                report_main,
                l1_prefetch_shards,
                lora_adapter=new_ref2va_adapter(),
            )
            video, audio = sample_ref2va_latents(
                runtime,
                video,
                audio,
                text_states,
                noised_video,
                condition_audio,
                layout,
                steps,
                lambda current, total: self._update(
                    job_id,
                    progress=0.46 + 0.34 * current / max(1, total),
                    message=f"Ref2VA sampling step {current}/{total}",
                    activity={
                        "phase": "sampling",
                        "module": "Ref2VA",
                        "operation": "Sampling step completed",
                        "sampling_step": current,
                        "sampling_steps": total,
                    },
                ),
                sampler="res_multistep" if use_acceleration_lora else "euler",
            )
            main_runtime_metrics = runtime.metrics()
            runtime.close()
            runtime = None
            runner.close()
            runner = None
            del text_states
            text_states = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            update(
                0.82,
                "Decoding generated video",
                {"phase": "video_decode", "module": "Video VAE", "operation": "Decoding target rows"},
            )
            assert video is not None and audio is not None
            if (video_onnx / "manifest.json").is_file():
                pixels = decode_video_latents_onnx(
                    video_onnx,
                    video,
                    profile.output_height,
                    profile.output_width,
                    callback=lambda details: report_activity(details, phase="video_decode"),
                )
            else:
                pixels = decode_video_latents(
                    self.workspace / "minimax_h3_video_vae_fp16.safetensors",
                    video,
                    profile.output_height,
                    profile.output_width,
                    callback=lambda details: report_activity(details, phase="video_decode"),
                )
            pixels = np.ascontiguousarray(pixels[:, :, :target_frames], dtype=np.float32)
            target_audio_frames = runtime_profile.audio_latent_frames
            audio = np.ascontiguousarray(audio[:, :, :, :target_audio_frames], dtype=np.float32)
            update(
                0.91,
                "Decoding generated audio",
                {"phase": "audio_decode", "module": "Audio VAE", "operation": "Decoding target rows"},
            )
            if not np.isfinite(audio).all():
                invalid = int((~np.isfinite(audio)).sum())
                audio_status = "silent_fallback"
                audio_warning = f"Non-finite Ref2VA audio latent: {invalid} invalid values"
                waveform = np.zeros((1, 2, target_audio_frames * 800), dtype=np.float32)
            else:
                try:
                    waveform = decode_audio_latents_onnx(
                        audio_onnx,
                        audio,
                        callback=lambda details: report_activity(details, phase="audio_decode"),
                    )
                except FloatingPointError as exc:
                    audio_status = "silent_fallback"
                    audio_warning = str(exc)
                    waveform = np.zeros((1, 2, target_audio_frames * 800), dtype=np.float32)

            update(
                0.97,
                "Writing Ref2VA MP4",
                {"phase": "mux", "module": "FFmpeg", "operation": "Encoding H.264/AAC"},
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
                    "references": [reference.to_dict() for reference in references],
                    "reference_order": [list(reference.labels) for reference in references],
                },
                "sampling": {
                    "seed": seed,
                    "steps": steps,
                    "sampler": "res_multistep" if use_acceleration_lora else "euler",
                    "scheduler": "simple" if use_acceleration_lora else "minimax_h3",
                    "video_shift": 12.0,
                    "audio_shift": 3.0,
                    "reference_condition_timestep": 0.999,
                    "reference_audio_timestep": 1.0,
                    "condition_noise_order": "visual_references_then_target_video_then_target_audio",
                    "video_sigmas": shifted_flow_sigmas(steps, 12.0, 1.0),
                    "audio_schedule": "dual_clock_time_shift_from_video_12_to_audio_3",
                },
                "reference_conditioning": {
                    "reference_image_short_edge_budget": reference_image_short_edge,
                    "media": reference_media_metadata,
                    "layout_sequence_tokens": int(layout.token_tags.size),
                    "condition_video_rows": layout.num_condition_video_rows,
                    "condition_audio_rows": layout.num_condition_audio_rows,
                    "visual_latent_shapes": [list(value.shape) for value in condition_video],
                    "audio_latent_shapes": [list(value.shape) for value in condition_audio],
                    **vision_metadata,
                },
                "output": {
                    "path": str(destination),
                    "width": profile.output_width,
                    "height": profile.output_height,
                    "frames": target_frames,
                    "fps": H3_FPS,
                    "duration_seconds": target_frames / H3_FPS,
                    "audio_sample_rate": 32000,
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                },
                "temporal": {"mode": "native", "segments": 1, "profile": runtime_profile.to_dict()},
                "runtime": {
                    "provider": provider_name,
                    "main_variant": REF2VA_ADAPTER_VARIANT if use_acceleration_lora else "base",
                    "use_acceleration_lora": use_acceleration_lora,
                    "attention_query_chunk": attention_query_chunk,
                    "l1_prefetch_shards": l1_prefetch_shards,
                    "segments": [main_runtime_metrics],
                },
                "models": {
                    "main": _manifest_identity(main_dir),
                    "ref2va_adapter": (
                        {
                            "directory": str(ref2va_adapter_dir),
                            "variant": REF2VA_ADAPTER_VARIANT,
                            "strength": 1.0,
                            "factor_pairs": ref2va_adapter_manifest.get("factor_pairs"),
                            "assets": ref2va_adapter_manifest.get("assets"),
                            "application": "runtime_low_rank_overlay",
                        }
                        if ref2va_adapter_manifest is not None
                        else None
                    ),
                    "text_encoder": _manifest_identity(qwen_dir),
                    "visual_source": str(visual_checkpoint),
                    "video_vae": _manifest_identity(video_onnx),
                    "audio_vae": _manifest_identity(audio_onnx),
                },
            }
            metadata_path = write_mp4(destination, pixels, waveform, H3_FPS, metadata)
            self._update(
                job_id,
                status="completed",
                progress=1.0,
                message=(
                    "Ref2VA video generation completed with silent audio"
                    if audio_status == "silent_fallback"
                    else "Ref2VA video generation completed"
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
                    "main_variant": REF2VA_ADAPTER_VARIANT if use_acceleration_lora else "base",
                    "use_acceleration_lora": use_acceleration_lora,
                    "profile": runtime_profile.to_dict(),
                    "steps": steps,
                    "seed": seed,
                    "frames": target_frames,
                    "duration_seconds": target_frames / H3_FPS,
                    "segments": 1,
                    "temporal_mode": "native",
                    "attention_query_chunk": attention_query_chunk,
                    "l1_prefetch_shards": l1_prefetch_shards,
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                    "references": [reference.to_dict() for reference in references],
                },
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                message=str(exc),
                activity={"phase": "failed", "module": "Ref2VA", "operation": str(exc)},
                finished_at=_now(),
                error="".join(traceback.format_exception(exc)),
            )
        finally:
            heartbeat_stop.set()
            if runtime is not None:
                runtime.close()
            _close_qwen_runtime_weights(qwen)
            if video_encoder is not None:
                _release_video_encoder(video_encoder)
            if runner is not None:
                runner.close()
            if reservation_active:
                release_reservation(reservation_token)
            if monitor is not None:
                monitor.stop()
            del normalized, vision, condition_video, condition_audio, noised_video, video, audio, text_states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        output_fps: float | None = None,
        super_resolution: dict[str, Any] | None = None,
        frame_conditioning: dict[str, str] | None = None,
        references: tuple[ReferenceSpec, ...] | None = None,
    ) -> None:
        if references:
            self._run_ref2va_inference(
                job_id,
                token_ids,
                prompt,
                profile,
                steps,
                seed,
                target_frames,
                attention_query_chunk,
                l1_prefetch_shards,
                destination,
                references,
                use_acceleration_lora,
            )
            return
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
        super_source: Path | None = None
        super_frames: np.ndarray | None = None
        super_encoder: Any | None = None
        super_encoder_backend: str | None = None
        frame_encoder_backend: str | None = None
        frame_encode_seconds = 0.0
        frame_anchors: dict[str, np.ndarray] = {}
        reservation_token = f"{os.getpid()}-{job_id}"
        heartbeat_stop = threading.Event()
        reservation_active = False
        try:
            monitor = PerformanceMonitor(
                self.performance_log_path(job_id)
                or self.workspace / ".h3-workbench" / "performance" / f"h3-{job_id}.jsonl",
                lambda: self._performance_state(job_id),
                lambda record: self._update_performance(job_id, record),
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
            base_main_dir = resolve_main_model_directory(
                self.workspace,
                self.output_root,
                accelerated=False,
                component="ref2va_transformer" if use_acceleration_lora else None,
            )
            base_ready = base_main_dir is not None
            use_acceleration = use_acceleration_lora
            if not base_ready:
                if use_acceleration:
                    raise RuntimeError(
                        "Ref2VA Turbo LoRA requires a validated Ref2VA virtual base model"
                    )
                raise RuntimeError("No validated Ref2VA or FL2VA main model is installed.")
            main_dir = base_main_dir
            assert main_dir is not None
            ref2va_manifest: dict[str, Any] | None = None
            ref2va_adapter_dir: Path | None = None
            ref2va_lora_path: Path | None = None
            if use_acceleration:
                ref2va_adapter_dir, ref2va_lora_path, ref2va_manifest = self._ref2va_acceleration_assets(main_dir)

            def new_ref2va_adapter() -> Ref2VALoraAdapter | None:
                if not use_acceleration:
                    return None
                assert ref2va_adapter_dir is not None
                assert ref2va_lora_path is not None
                return Ref2VALoraAdapter(
                    ref2va_adapter_dir,
                    ref2va_lora_path,
                    strength=1.0,
                    base_model_dir=main_dir,
                )

            main_component = str(_manifest_identity(main_dir).get("component") or "")
            if not main_component and "fl2va" in main_dir.name.lower():
                main_component = "fl2va_transformer"
            base_label = {
                "ref2va_transformer": "Ref2VA Base",
                "fl2va_transformer": "FL2VA Base",
            }.get(main_component, "Main Base")
            model_label = "Ref2VA Turbo 4-step LoRA" if use_acceleration else base_label
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
                    lora_adapter=new_ref2va_adapter(),
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
            if (super_resolution is not None or frame_conditioning is not None) and warm_runtime is not None:
                # Video VAE encoding is compute-heavy on low-VRAM GPUs. Do not
                # keep the FL2VA warmup sessions resident while it runs; the
                # task runtime is recreated after conditioning is prepared.
                warm_runtime.close()
                warm_runtime = None
            video_onnx = self.output_root / "video_vae"
            video_manifest = video_onnx / "manifest.json"
            pixel_segments: list[np.ndarray] = []
            audio_segments: list[np.ndarray] = []
            audio_warnings: list[str] = []
            main_runtime_metrics: list[dict[str, object]] = []
            super_noise_strength = 0.0
            sampling_start_sigma = 1.0
            super_interpolation: str | None = None
            super_segment_frames = profile.frames
            super_segment_stride = profile.frames
            super_temporal_overlap = 0
            super_encode_seconds = 0.0
            # Conditioning is deliberately prepared in one contiguous phase.
            # The staged ONNX sessions stay resident across segments, then are
            # released before FL2VA sampling claims the device.
            super_conditioning: list[np.ndarray] = []
            if frame_conditioning is not None:
                onnx_encoder_ready = video_vae_temporal_encoder_ready(video_onnx)
                frame_encoder_backend, encoder_selection_reason = select_video_vae_encoder_backend(
                    checkpoint_available=False,
                    onnx_available=onnx_encoder_ready,
                    low_vram_cuda=False,
                )
                self._update(
                    job_id,
                    progress=0.16,
                    message="Encoding first/last frame conditions",
                    activity={
                        "phase": "frame_conditioning_encode",
                        "module": "Video VAE",
                        "operation": "Loading persistent staged ONNX temporal encoder",
                        "mode": frame_conditioning["mode"],
                        "backend": frame_encoder_backend,
                        "selection_reason": encoder_selection_reason,
                    },
                )
                super_encoder = load_video_vae_onnx_encoder(video_onnx)
                encode_started = time.perf_counter()
                encoded_paths: dict[str, np.ndarray] = {}
                roles = [role for role in ("start", "end") if role in frame_conditioning]
                for index, role in enumerate(roles, 1):
                    source = resolve_image_path(self.workspace, frame_conditioning[role])
                    cache_key = str(source)
                    anchor = encoded_paths.get(cache_key)
                    source_size: list[int] | None = None
                    if anchor is None:
                        info = probe_image(source)
                        source_size = [info.width, info.height]
                        image = read_image(source, info)
                        prepared = prepare_frame_condition(
                            image,
                            profile.output_height,
                            profile.output_width,
                            profile.padded_height,
                            profile.padded_width,
                        )
                        anchor = encode_image_frame(
                            super_encoder,
                            np.ascontiguousarray(prepared, dtype=np.float16),
                            callback=lambda activity, role=role: report_activity(
                                activity,
                                phase="frame_conditioning_encode",
                                role=role,
                            ),
                            offload_after=False,
                        )
                        encoded_paths[cache_key] = anchor
                        del image, prepared
                    frame_anchors[role] = anchor.copy()
                    self._update(
                        job_id,
                        progress=0.16 + 0.02 * index / len(roles),
                        activity={
                            "phase": "frame_conditioning_encode",
                            "module": "Video VAE",
                            "operation": "Frame anchor latent prepared",
                            "role": role,
                            "source": str(source),
                            "source_size": source_size,
                            "latent_shape": list(frame_anchors[role].shape),
                            "encoder_protocol": "single_image_causal_token_zero",
                        },
                    )
                frame_encode_seconds = time.perf_counter() - encode_started
                _release_video_encoder(super_encoder)
                super_encoder = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if super_resolution is not None:
                super_source = resolve_video_path(self.workspace, str(super_resolution["source"]))
                source_info = probe_video(super_source)
                self._update(
                    job_id,
                    progress=0.16,
                    message="Reading source video into host RAM",
                    activity={
                        "phase": "super_resolution_input",
                        "module": "Input Video",
                        "operation": "Reading RGB frames and metadata",
                        "frames": source_info.frames,
                        "width": source_info.width,
                        "height": source_info.height,
                    },
                )
                super_frames = read_video_frames(super_source, source_info)
                onnx_encoder_ready = video_vae_temporal_encoder_ready(video_onnx)
                super_encoder_backend, encoder_selection_reason = select_video_vae_encoder_backend(
                    checkpoint_available=False,
                    onnx_available=onnx_encoder_ready,
                    low_vram_cuda=False,
                )
                self._update(
                    job_id,
                    progress=0.17,
                    message="Loading Video VAE encoder weights",
                    activity={
                        "phase": "super_resolution_input",
                        "module": "Video VAE",
                        "operation": "Loading persistent staged ONNX temporal encoder",
                        "graphs": [
                            str(video_onnx / "video_encoder_prelude.onnx"),
                            str(video_onnx / "video_encoder_late.onnx"),
                            str(video_onnx / "video_encoder_tail.onnx"),
                            str(video_onnx / "video_encoder_head.onnx"),
                        ],
                        "backend": super_encoder_backend,
                        "selection_reason": encoder_selection_reason,
                    },
                )
                super_encoder = load_video_vae_onnx_encoder(video_onnx)
                super_noise_strength = float(super_resolution["noise_strength"])
                sampling_start_sigma = super_noise_strength
                super_interpolation = str(super_resolution["interpolation"])
                super_segment_frames = int(super_resolution["segment_frames"])
                super_segment_stride = int(super_resolution.get("segment_stride", super_segment_frames))
                super_temporal_overlap = int(super_resolution.get("temporal_overlap", 0))
                assert super_frames is not None
                assert super_encoder is not None
                assert super_interpolation is not None
                encode_started = time.perf_counter()
                combined_vae_frames = video_vae_output_frames(
                    video_latent_frames_for_output(target_frames)
                )
                combined_prepared_bytes = (
                    3
                    * combined_vae_frames
                    * profile.padded_height
                    * profile.padded_width
                    * np.dtype(np.float16).itemsize
                )
                combine_segment_encoding = bool(
                    segment_count > 1
                    and super_resolution.get("processing_mode") == "segmented"
                    and combined_prepared_bytes <= 1536 * 1024**2
                )
                if combine_segment_encoding:
                    prepared, _ = prepare_super_resolution_segment(
                        super_frames,
                        0,
                        int(super_frames.shape[2]),
                        combined_vae_frames,
                        combined_vae_frames,
                        profile.output_height,
                        profile.output_width,
                        profile.padded_height,
                        profile.padded_width,
                        super_interpolation,  # type: ignore[arg-type]
                    )
                    combined_conditioning = encode_video_frames(
                        super_encoder,
                        np.ascontiguousarray(prepared, dtype=np.float16),
                        callback=lambda activity: report_activity(
                            activity,
                            phase="super_resolution_encode",
                            segment=1,
                            segments=segment_count,
                        ),
                        offload_after=False,
                    )
                    del prepared
                    for segment in range(segment_count):
                        latent_start = segment * 5
                        latent_stop = latent_start + profile.video_latent_frames
                        if latent_stop > combined_conditioning.shape[2]:
                            raise RuntimeError(
                                "Combined Video VAE conditioning is shorter than the segment plan"
                            )
                        conditioning = combined_conditioning[:, :, latent_start:latent_stop].copy()
                        super_conditioning.append(conditioning)
                        segment_start = segment * super_segment_stride
                        actual_source_frames = max(
                            0,
                            min(super_segment_frames, target_frames - segment_start),
                        )
                        self._update(
                            job_id,
                            progress=0.17 + 0.01 * (segment + 1) / segment_count,
                            activity={
                                "phase": "super_resolution_encode",
                                "module": "Video VAE",
                                "operation": "Conditioning latent window prepared",
                                "segment": segment + 1,
                                "segments": segment_count,
                                "source_frames": actual_source_frames,
                                "latent_shape": list(conditioning.shape),
                                "noise_strength": super_noise_strength,
                                "interpolation": super_interpolation,
                                "combined_encode": True,
                            },
                        )
                    del combined_conditioning
                else:
                    for segment in range(segment_count):
                        segment_start = segment * super_segment_stride
                        segment_stop = segment_start + super_segment_frames
                        prepared, actual_source_frames = prepare_super_resolution_segment(
                            super_frames,
                            segment_start,
                            segment_stop,
                            super_segment_frames,
                            profile.frames,
                            profile.output_height,
                            profile.output_width,
                            profile.padded_height,
                            profile.padded_width,
                            super_interpolation,  # type: ignore[arg-type]
                        )
                        conditioning = encode_video_frames(
                            super_encoder,
                            prepared,
                            callback=lambda activity, segment=segment: report_activity(
                                activity,
                                phase="super_resolution_encode",
                                segment=segment + 1,
                                segments=segment_count,
                            ),
                            offload_after=False,
                        )
                        super_conditioning.append(conditioning)
                        self._update(
                            job_id,
                            progress=0.17 + 0.01 * (segment + 1) / segment_count,
                            activity={
                                "phase": "super_resolution_encode",
                                "module": "Video VAE",
                                "operation": "Conditioning latent prepared",
                                "segment": segment + 1,
                                "segments": segment_count,
                                "source_frames": actual_source_frames,
                                "latent_shape": list(conditioning.shape),
                                "noise_strength": super_noise_strength,
                                "interpolation": super_interpolation,
                                "combined_encode": False,
                            },
                        )
                        del prepared
                super_encode_seconds = time.perf_counter() - encode_started
                self._update(
                    job_id,
                    activity={
                        "phase": "super_resolution_encode",
                        "module": "Video VAE",
                        "operation": "Conditioning encode pass completed",
                        "elapsed_seconds": round(super_encode_seconds, 3),
                        "segments": segment_count,
                        "backend": super_encoder_backend,
                    },
                )

                # Sampling and decoding need essentially all available VRAM.
                # Release the encoder session/weights before creating the
                # first FL2VA runtime, while retaining only tiny host latents.
                _release_video_encoder(super_encoder)
                del super_encoder
                super_encoder = None
                del super_frames
                super_frames = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for segment in range(segment_count):
                self._update(
                    job_id,
                    progress=0.18 + 0.64 * segment / segment_count,
                    message=f"Sampling segment {segment + 1}/{segment_count} with {provider_name}",
                    activity={
                        "phase": "sampling",
                        "module": "FL2VA",
                        "operation": "Preparing segment",
                        "sampler": "res_multistep" if use_acceleration else "euler",
                        "scheduler": "simple" if use_acceleration else "minimax_h3",
                        "segment": segment + 1,
                        "segments": segment_count,
                        "provider": provider_name,
                    },
                )
                video, audio = initial_latents(profile, seed + segment)
                video_condition = (
                    _segment_video_condition(
                        profile,
                        target_frames,
                        segment,
                        segment_count,
                        temporal_mode,
                        frame_anchors,
                    )
                    if frame_anchors
                    else None
                )
                if video_condition is not None:
                    self._update(
                        job_id,
                        activity={
                            "phase": "sampling",
                            "module": "Video VAE",
                            "operation": "Applying clean frame anchor mask",
                            "segment": segment + 1,
                            "segments": segment_count,
                            "latent_indices": list(video_condition.indices),
                        },
                    )
                if super_resolution is not None:
                    conditioning = super_conditioning[segment]
                    noise = np.random.default_rng(seed + segment).standard_normal(
                        conditioning.shape,
                        dtype=np.float32,
                    )
                    video = conditioning * (1.0 - super_noise_strength) + noise * super_noise_strength
                    self._update(
                        job_id,
                        activity={
                            "phase": "sampling",
                            "module": "FL2VA",
                            "operation": "Applying conditioning latent",
                            "segment": segment + 1,
                            "segments": segment_count,
                            "latent_shape": list(conditioning.shape),
                            "noise_strength": super_noise_strength,
                        },
                    )
                    del conditioning, noise

                def report_main(activity: dict[str, object], segment: int = segment) -> None:
                    details = {
                        "phase": "sampling",
                        "sampler": "res_multistep" if use_acceleration else "euler",
                        "scheduler": "simple" if use_acceleration else "minimax_h3",
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
                            lora_adapter=new_ref2va_adapter(),
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
                        start_sigma=sampling_start_sigma,
                        video_condition=video_condition,
                        sampler="res_multistep" if use_acceleration else "euler",
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
                if super_resolution is not None and super_temporal_overlap > 0 and segment > 0:
                    overlap = min(super_temporal_overlap, segment_pixels.shape[2])
                    previous = pixel_segments[-1]
                    positions = np.arange(overlap, dtype=np.float32) / overlap
                    left_weight = (1.0 - positions).reshape(1, 1, overlap, 1, 1)
                    right_weight = positions.reshape(1, 1, overlap, 1, 1)
                    blended = (
                        previous[:, :, -overlap:] * left_weight
                        + segment_pixels[:, :, :overlap] * right_weight
                    )
                    pixel_segments[-1] = np.concatenate(
                        (previous[:, :, :-overlap], blended),
                        axis=2,
                    )
                    segment_pixels = segment_pixels[:, :, overlap:]
                pixel_segments.append(segment_pixels)
                if super_resolution is None:
                    audio_segments.append(audio)
                del video, audio, segment_pixels, video_condition
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            runner.close()
            runner = None
            del text_states
            gc.collect()
            pixels = np.concatenate(pixel_segments, axis=2)[:, :, :target_frames]
            del pixel_segments
            audio_onnx = self.output_root / "audio_vae"
            output_fps_value = output_fps or profile.fps
            audio_warning = "; ".join(dict.fromkeys(audio_warnings)) if audio_warnings else None
            if super_resolution is not None:
                waveform = None
                audio_status = "source_preserved" if super_source is not None and probe_video(super_source).has_audio else "no_audio"
                self._update(
                    job_id,
                    progress=0.92,
                    message="Preparing source audio",
                    activity={
                        "phase": "audio_mux",
                        "module": "Input Audio",
                        "operation": "Keeping original audio stream",
                        "audio_status": audio_status,
                    },
                )
            else:
                target_audio_frames = math.ceil(target_frames * profile.audio_latents_per_second / profile.fps)
                audio = np.concatenate(audio_segments, axis=3)[:, :, :, :target_audio_frames]
                del audio_segments
                self._update(
                    job_id,
                    progress=0.92,
                    message="Decoding audio",
                    activity={"phase": "audio_decode", "module": "Audio VAE", "operation": "Preparing decoder"},
                )

                def audio_callback(activity: dict[str, object]) -> None:
                    report_activity(activity, phase="audio_decode")

                audio_status = "generated"
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
                    **(
                        {
                            "video": super_resolution["source_info"],
                            "source_path": str(super_source) if super_source is not None else None,
                            "prompt_source": super_resolution["prompt_source"],
                        }
                        if super_resolution is not None
                        else {}
                    ),
                    **(
                        {
                            "start_image_path": frame_conditioning.get("start"),
                            "end_image_path": frame_conditioning.get("end"),
                        }
                        if frame_conditioning is not None
                        else {}
                    ),
                },
                "sampling": {
                    "seed": seed,
                    "steps": steps,
                    "sampler": "res_multistep" if use_acceleration else "euler",
                    "scheduler": "simple" if use_acceleration else "minimax_h3",
                    "video_shift": 12.0,
                    "audio_shift": 3.0,
                    "start_sigma": sampling_start_sigma,
                    "video_sigmas": shifted_flow_sigmas(steps, 12.0, sampling_start_sigma),
                    "audio_schedule": "dual_clock_time_shift_from_video_12_to_audio_3",
                },
                "output": {
                    "path": str(destination),
                    "width": profile.output_width,
                    "height": profile.output_height,
                    "frames": target_frames,
                    "fps": output_fps_value,
                    "duration_seconds": target_frames / output_fps_value,
                    "audio_sample_rate": 32000,
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                },
                **(
                    {
                        "super_resolution": {
                            "scale": super_resolution["scale"],
                            "interpolation": super_resolution["interpolation"],
                            "noise_strength": super_resolution["noise_strength"],
                            "start_sigma": sampling_start_sigma,
                            "processing_mode": super_resolution["processing_mode"],
                            "segment_frames": super_resolution["segment_frames"],
                            "segment_stride": super_resolution["segment_stride"],
                            "temporal_overlap": super_resolution["temporal_overlap"],
                            "noise_formula": "conditioning * (1 - strength) + standard_normal * strength",
                            "latent_conditioning": "Video VAE temporal/spatial encode",
                            "encoder_backend": super_encoder_backend,
                            "encoder_seconds": round(super_encode_seconds, 3),
                            "audio_policy": "reuse_input_audio",
                        }
                    }
                    if super_resolution is not None
                    else {}
                ),
                **(
                    {
                        "frame_conditioning": {
                            "mode": frame_conditioning["mode"],
                            "policy": "clean_masked_latent_slots",
                            "anchor_tokens_per_boundary": 1,
                            "encoder_protocol": "single_image_causal_token_zero",
                            "condition_timestep": 1.0,
                            "condition_update": "excluded_from_euler_update",
                            "end_alignment": "last_retained_output_frame",
                            "encoder_backend": frame_encoder_backend,
                            "encoder_seconds": round(frame_encode_seconds, 3),
                        }
                    }
                    if frame_conditioning is not None
                    else {}
                ),
                "temporal": {
                    "mode": temporal_mode,
                    "segments": segment_count,
                    "profile": profile.to_dict(),
                },
                "runtime": {
                    "provider": provider_name,
                    "main_variant": REF2VA_ADAPTER_VARIANT if use_acceleration else "base",
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
                    "ref2va_adapter": (
                        {
                            "directory": str(ref2va_adapter_dir),
                            "variant": REF2VA_ADAPTER_VARIANT,
                            "strength": 1.0,
                            "factor_pairs": ref2va_manifest.get("factor_pairs"),
                            "assets": ref2va_manifest.get("assets"),
                            "application": "runtime_low_rank_overlay",
                        }
                        if ref2va_manifest is not None
                        else None
                    ),
                    "text_encoder": _manifest_identity(qwen_dir),
                    "video_vae": _manifest_identity(video_onnx),
                    "audio_vae": _manifest_identity(audio_onnx),
                },
            }
            if super_resolution is not None:
                assert super_source is not None
                metadata_path = write_mp4_with_audio_source(
                    destination,
                    pixels,
                    super_source,
                    output_fps_value,
                    metadata,
                )
            else:
                assert waveform is not None
                metadata_path = write_mp4(destination, pixels, waveform, output_fps_value, metadata)
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
                    "main_variant": REF2VA_ADAPTER_VARIANT if use_acceleration else "base",
                    "use_acceleration_lora": use_acceleration,
                    "profile": profile.to_dict(),
                    "steps": steps,
                    "seed": seed,
                    "frames": target_frames,
                    "duration_seconds": target_frames / output_fps_value,
                    "segments": segment_count,
                    "temporal_mode": temporal_mode,
                    "attention_query_chunk": attention_query_chunk,
                    "l1_prefetch_shards": l1_prefetch_shards,
                    "audio_status": audio_status,
                    "audio_warning": audio_warning,
                    "conditioning_mode": frame_conditioning["mode"] if frame_conditioning is not None else "text",
                    "start_image_path": frame_conditioning.get("start") if frame_conditioning is not None else None,
                    "end_image_path": frame_conditioning.get("end") if frame_conditioning is not None else None,
                    **(
                        {
                            "source": str(super_source) if super_source is not None else None,
                            "scale": super_resolution["scale"],
                            "interpolation": super_resolution["interpolation"],
                            "noise_strength": super_resolution["noise_strength"],
                            "processing_mode": super_resolution["processing_mode"],
                        }
                        if super_resolution is not None
                        else {}
                    ),
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
            if super_encoder is not None:
                _release_video_encoder(super_encoder)
                del super_encoder
            if super_frames is not None:
                del super_frames
            gc.collect()
            if reservation_active:
                release_reservation(reservation_token)
            if runner is not None:
                runner.close()
            if monitor is not None:
                monitor.stop()
