from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
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
from h3_workbench.inference_runtime import H3MainRuntime, ORTGraphRunner, QwenTextRuntime, initial_latents, sample_latents
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
from h3_workbench.model_catalog import MODELSCOPE_REPO, component_by_id


def _complete_main_model(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return manifest.get("validation_passed") is True and len(manifest.get("blocks", [])) == 50


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
        job = Job(id=uuid.uuid4().hex[:12], model_id=model_id)
        component_dir = source.stem.replace(".safetensors", "")
        destination = self.output_root / component_dir
        job.output_dir = str(destination)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run_export, job.id, source, destination, video_blocks)
        return job

    def create_download(self, component_ids: list[str]) -> Job:
        if not component_ids:
            raise ValueError("Select at least one model component")
        components = [component_by_id(item) for item in dict.fromkeys(component_ids)]
        job = Job(id=uuid.uuid4().hex[:12], model_id=MODELSCOPE_REPO, kind="download")
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run_download, job.id, components)
        return job

    def _run_download(self, job_id: str, components: list[Any]) -> None:
        try:
            total = len(components)
            for index, component in enumerate(components, start=1):
                destination = self.workspace / component.relative_path
                self._update(
                    job_id,
                    status="running",
                    started_at=self.get(job_id).started_at or _now(),
                    message=f"下载 {component.label} ({index}/{total})",
                    activity={"phase": "download", "module": component.label, "operation": "ModelScope snapshot"},
                    progress=(index - 1) / total,
                )
                destination.mkdir(parents=True, exist_ok=True)
                local_root = self.workspace if component.id == "tokenizer" else self.output_root
                modelscope_executable = Path(sys.executable).with_name("modelscope.exe" if os.name == "nt" else "modelscope")
                command = [
                    str(modelscope_executable), "download", MODELSCOPE_REPO,
                    "--repo-type", "model", "--local-dir", str(local_root),
                    "--max-workers", "2",
                ]
                command.extend(["--include", *component.include])
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                result = subprocess.run(command, cwd=self.workspace, capture_output=True, text=True, creationflags=flags)
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or "ModelScope download failed").strip()[-2000:]
                    raise RuntimeError(f"{component.label}: {detail}")
                self._update(job_id, progress=index / total, message=f"已下载 {component.label}")
            self._update(job_id, status="completed", progress=1.0, message="模型组件下载完成", finished_at=_now(), result={"components": [item.id for item in components]})
        except Exception as exc:  # noqa: BLE001 - preserve failure in the WebUI job record
            self._update(job_id, status="failed", message="模型下载失败", error=str(exc), finished_at=_now())

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
        attention_query_chunk: int = 256,
        l1_prefetch_shards: int = 2,
    ) -> Job:
        if not 1 <= steps <= 50:
            raise ValueError("Inference steps must be from 1 to 50")
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
        monitor: PerformanceMonitor | None = None
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
            qwen_dir = self.output_root / "qwen3vl_32b_minimax_h3_nvfp4_awq"
            base_main_dir = self.output_root / "minimax_h3_fl2va_pruned_fp8_scaled_streaming"
            accelerated_main_dir = self.output_root / "minimax_h3_fl2va_pruned_fp8_scaled_accelerated"
            accelerated_ready = _complete_main_model(accelerated_main_dir)
            base_ready = _complete_main_model(base_main_dir)
            use_acceleration = 4 <= steps <= 8 and accelerated_ready
            if not use_acceleration and not base_ready:
                supported = "4-8" if accelerated_ready else "none"
                raise RuntimeError(
                    f"No compatible main model is installed for {steps} steps. "
                    f"Turbo v4 supports {supported} steps; install the base streaming model for other step counts."
                )
            main_dir = accelerated_main_dir if use_acceleration else base_main_dir
            model_label = "FL2VA Turbo v4" if use_acceleration else "FL2VA Base"
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
            qwen = QwenTextRuntime(qwen_dir, runner, l1_prefetch_shards)

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
            video_onnx = self.output_root / "video_vae"
            video_manifest = video_onnx / "manifest.json"
            pixel_segments: list[np.ndarray] = []
            audio_segments: list[np.ndarray] = []
            audio_warnings: list[str] = []
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

                runtime = H3MainRuntime(
                    main_dir,
                    runner,
                    profile,
                    attention_query_chunk,
                    report_main,
                    l1_prefetch_shards,
                )
                video, audio = sample_latents(
                    runtime,
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
                if runtime.audio_fallback_reason is not None:
                    audio_warnings.append(runtime.audio_fallback_reason)
                del runtime
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
            del qwen, text_states
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
                    "main_variant": "turbo_v4" if use_acceleration else "base",
                    "attention_query_chunk": attention_query_chunk,
                    "l1_prefetch_shards": l1_prefetch_shards,
                    "l2_cache_gib": os.environ.get("H3_L2_CACHE_GIB"),
                    "prefetch_shards": os.environ.get("H3_PREFETCH_SHARDS"),
                },
                "models": {
                    "main": _manifest_identity(main_dir),
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
                    "main_variant": "turbo_v4" if use_acceleration else "base",
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
            if runner is not None:
                runner.close()
            if monitor is not None:
                monitor.stop()
