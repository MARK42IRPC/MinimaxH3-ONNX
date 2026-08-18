from __future__ import annotations

import asyncio
import json
import platform
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import unquote_to_bytes

import onnxruntime as ort
import psutil
import torch
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from h3_workbench.config import Settings
from h3_workbench.device_profile import probe_device_profiles, select_device_profile
from h3_workbench.jobs import JobManager, resolve_main_model_directory
from h3_workbench.memory_planner import main_model_shards, plan_shard_batches, probe_gpu_memory
from h3_workbench.media_input import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    probe_audio,
    probe_image,
    probe_video,
    resolve_video_path,
)
from h3_workbench.model_registry import scan_models
from h3_workbench.performance_monitor import LivePerformanceMonitor
from h3_workbench.profiles import GENERATION_PROFILES
from h3_workbench.qwen_persistent import resolve_qwen_directory, validated_int8_virtual_qwen_ready
from h3_workbench.shard_cache import default_l2_cache_bytes, default_prefetch_depth
from h3_workbench.tokenizer import tokenizer_files_ready
from h3_workbench.source_catalog import EXPORT_PRESETS, ExportPreset, SourceAsset
from h3_workbench.turbo_lora import validate_turbo_adapter
from h3_workbench.video_vae_persistent import persistent_video_vae_ready, video_vae_block_indices

settings = Settings.from_env()
manager = JobManager(settings.workspace, settings.output_dir)
web_dir = Path(__file__).parent / "web"
live_performance_monitor = LivePerformanceMonitor()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        live_performance_monitor.stop()


app = FastAPI(title="MiniMax H3 Edge Workbench", version="0.1.0", lifespan=_lifespan)
app.mount("/assets", StaticFiles(directory=web_dir), name="assets")


def _manifest_has_blocks(path: Path, count: int) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return manifest.get("validation_passed") is True and len(manifest.get("blocks", [])) == count
    except (OSError, ValueError, TypeError):
        return False


def _complete_main_model(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
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


def _runtime_adapter_ready(directory: Path, base_model_directory: Path | None = None) -> bool:
    try:
        validate_turbo_adapter(directory, base_model_dir=base_model_directory)
    except Exception:  # noqa: BLE001 - corrupt optional artifacts are reported as not ready
        return False
    return True


def _gpu_info() -> dict[str, object]:
    try:
        profiles = probe_device_profiles(refresh=True)
        selected = select_device_profile(profiles)
        return {
            "available": selected.provider == "cuda",
            "name": selected.name,
            "memory_bytes": selected.total_bytes,
            "free_bytes": selected.free_bytes,
            "driver": selected.driver,
            "cuda": (
                selected.cuda_runtime or str(torch.version.cuda)
                if torch.cuda.is_available()
                else selected.cuda_runtime
            ),
            "selected": selected.to_dict(),
            "devices": [item.to_dict() for item in profiles],
        }
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "devices": []}


class ExportRequest(BaseModel):
    model_id: str
    video_blocks: str = Field(default="all", pattern=r"^(all|\d+(,\d+)*)$")


class InferenceRequest(BaseModel):
    token_ids: list[int] | None = None
    prompt: str | None = None
    steps: int = Field(default=4, ge=1, le=50)
    use_acceleration_lora: bool = False
    seed: int = 1
    width: int = Field(default=640, ge=128, le=1024)
    height: int = Field(default=360, ge=128, le=1024)
    duration_seconds: float = Field(default=17 / 24, gt=0, le=15)
    temporal_mode: Literal["native", "segmented"] = "segmented"
    conditioning_mode: Literal["text", "first", "last", "first_last"] = "text"
    start_image_path: str | None = Field(default=None, max_length=2048)
    end_image_path: str | None = Field(default=None, max_length=2048)
    references: list[dict[str, object]] | None = None
    attention_query_chunk: int = Field(default=512, ge=32, le=512)
    l1_prefetch_shards: int = Field(default=2, ge=0, le=4)

    @model_validator(mode="after")
    def require_prompt_or_tokens(self) -> "InferenceRequest":
        self.start_image_path = self.start_image_path.strip() if self.start_image_path else None
        self.end_image_path = self.end_image_path.strip() if self.end_image_path else None
        if not self.prompt and not self.token_ids:
            raise ValueError("Provide either prompt or token_ids")
        if self.token_ids is not None and len(self.token_ids) > 192:
            raise ValueError("token_ids must contain at most 192 items")
        if self.prompt is not None and len(self.prompt) > 4000:
            raise ValueError("prompt must contain at most 4000 characters")
        if self.use_acceleration_lora and not 4 <= self.steps <= 8:
            raise ValueError("Turbo v4 acceleration LoRA supports 4-8 sampling steps")
        if self.attention_query_chunk not in {32, 64, 128, 256, 512}:
            raise ValueError("attention_query_chunk must be one of 32, 64, 128, 256, or 512")
        expected = {
            "text": (False, False),
            "first": (True, False),
            "last": (False, True),
            "first_last": (True, True),
        }[self.conditioning_mode]
        actual = (self.start_image_path is not None, self.end_image_path is not None)
        if actual != expected:
            raise ValueError(
                f"conditioning_mode={self.conditioning_mode!r} requires "
                f"start_image_path={expected[0]} and end_image_path={expected[1]}"
            )
        if self.references is not None:
            if not self.references:
                self.references = None
            else:
                if not self.prompt or not self.prompt.strip():
                    raise ValueError("Ref2VA references require prompt text")
                if self.token_ids:
                    raise ValueError("Ref2VA references require prompt text, not token_ids")
                if self.use_acceleration_lora:
                    raise ValueError("Ref2VA references are incompatible with Turbo v4 acceleration")
                if self.temporal_mode != "native":
                    raise ValueError("Ref2VA references require native temporal mode")
                if self.conditioning_mode != "text":
                    raise ValueError("Ref2VA references are incompatible with frame conditioning")
                if not 5.0 <= self.duration_seconds <= 15.0:
                    raise ValueError("Ref2VA generation duration must be from 5 to 15 seconds")
                if len(self.references) > 12:
                    raise ValueError("MiniMax-H3 accepts at most 12 references in total")
                kinds: list[str] = []
                for reference in self.references:
                    if not isinstance(reference, dict):
                        raise ValueError("Each reference must be an object")
                    kind = str(reference.get("kind") or reference.get("type") or "").strip().lower()
                    path = str(reference.get("path") or reference.get("uri") or "").strip()
                    if kind not in {"image", "video", "audio"}:
                        raise ValueError(f"Unknown reference kind: {kind or '(empty)'}")
                    if not path or len(path) > 2048:
                        raise ValueError("Reference path must contain from 1 to 2048 characters")
                    kinds.append(kind)
                for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
                    if kinds.count(kind) > limit:
                        raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references")
                if set(kinds) == {"audio"}:
                    raise ValueError("An audio reference must be paired with an image or video reference")
        return self


class VideoProbeRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


class SuperResolutionRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=2048)
    prompt: str | None = Field(default=None, max_length=4000)
    scale: float = Field(default=2.0, ge=1.0, le=4.0)
    interpolation: Literal["nearest", "bilinear", "bicubic", "trilinear"] = "bicubic"
    noise_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    processing_mode: Literal["segmented", "direct"] = "segmented"
    steps: int = Field(default=4, ge=1, le=50)
    use_acceleration_lora: bool = False
    seed: int = 1
    attention_query_chunk: int = Field(default=512, ge=32, le=512)
    l1_prefetch_shards: int = Field(default=2, ge=0, le=4)

    @model_validator(mode="after")
    def validate_runtime_options(self) -> "SuperResolutionRequest":
        if self.prompt is not None and len(self.prompt) > 4000:
            raise ValueError("prompt must contain at most 4000 characters")
        if self.use_acceleration_lora and not 4 <= self.steps <= 8:
            raise ValueError("Turbo v4 acceleration LoRA supports 4-8 sampling steps")
        if self.attention_query_chunk not in {32, 64, 128, 256, 512}:
            raise ValueError("attention_query_chunk must be one of 32, 64, 128, 256, or 512")
        return self


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/api/system")
def system_info() -> dict[str, object]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu = _gpu_info()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "pagefile_total_bytes": swap.total,
        "pagefile_used_bytes": swap.used,
        "l2_cache_bytes": default_l2_cache_bytes(),
        "prefetch_shards": default_prefetch_depth(),
        "workspace": str(settings.workspace),
        "gpu": gpu,
    }


@app.get("/api/hardware/snapshot")
def hardware_snapshot() -> dict[str, object]:
    try:
        return live_performance_monitor.snapshot()
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/telemetry", include_in_schema=False)
def telemetry_snapshot() -> dict[str, object]:
    """Compatibility alias used by the polling WebUI client."""
    return hardware_snapshot()


@app.get("/api/hardware/stream")
async def hardware_stream(
    request: Request,
    samples: int | None = Query(default=None, ge=1, le=3600),
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        after_sequence = -1
        sent = 0
        timeout = max(3.0, live_performance_monitor.interval_seconds * 2)
        while samples is None or sent < samples:
            if await request.is_disconnected():
                break
            sample = await asyncio.to_thread(
                live_performance_monitor.wait_for_sample,
                after_sequence,
                timeout,
            )
            if sample is None:
                yield ": keepalive\n\n"
                continue
            after_sequence = int(sample["sequence"])
            payload = json.dumps(sample, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            yield f"id: {after_sequence}\nevent: hardware\ndata: {payload}\n\n"
            sent += 1

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/api/hardware/ws")
async def hardware_websocket(websocket: WebSocket) -> None:
    raw_limit = websocket.query_params.get("samples")
    try:
        limit = None if raw_limit is None else int(raw_limit)
        if limit is not None and not 1 <= limit <= 3600:
            raise ValueError
    except ValueError:
        await websocket.close(code=1008, reason="samples must be from 1 to 3600")
        return

    await websocket.accept()
    after_sequence = -1
    sent = 0
    timeout = max(3.0, live_performance_monitor.interval_seconds * 2)
    try:
        while limit is None or sent < limit:
            sample = await asyncio.to_thread(
                live_performance_monitor.wait_for_sample,
                after_sequence,
                timeout,
            )
            if sample is None:
                continue
            after_sequence = int(sample["sequence"])
            await websocket.send_json(sample)
            sent += 1
    except WebSocketDisconnect:
        return
    if limit is not None:
        await websocket.close(code=1000)


@app.get("/api/models")
def models() -> list[dict[str, object]]:
    return [record.to_dict() for record in scan_models(settings.workspace)]


@app.post("/api/media/probe")
def probe_media(request: VideoProbeRequest) -> dict[str, object]:
    try:
        path = resolve_video_path(settings.workspace, request.path)
        return probe_video(path).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _decode_upload_filename(request: Request, allowed_suffixes: set[str], media_kind: str) -> str:
    encoded = request.headers.get("x-filename", "")
    try:
        decoded = unquote_to_bytes(encoded).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="x-filename must be URI-encoded UTF-8") from exc
    filename = Path(decoded).name
    if not filename or Path(filename).suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"x-filename must be a supported {media_kind} filename")
    return filename


@app.post("/api/media/upload", status_code=201)
async def upload_media(request: Request) -> dict[str, object]:
    filename = _decode_upload_filename(request, VIDEO_SUFFIXES, "video")
    max_upload_bytes = 1024**3
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_upload_bytes:
                raise HTTPException(status_code=413, detail="Uploaded video exceeds the 1 GiB limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    input_root = (settings.workspace / ".h3-workbench" / "inputs").resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    destination = (input_root / f"{uuid.uuid4().hex[:12]}-{filename}").resolve()
    if input_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    written = 0
    try:
        with destination.open("wb") as output:
            async for chunk in request.stream():
                written += len(chunk)
                if written > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds the 1 GiB limit")
                output.write(chunk)
        info = probe_video(destination)
        return {"path": str(destination), "size_bytes": written, "video": info.to_dict()}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/images/upload", status_code=201)
async def upload_image(request: Request) -> dict[str, object]:
    filename = _decode_upload_filename(request, IMAGE_SUFFIXES, "image")
    max_upload_bytes = 32 * 1024**2
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_upload_bytes:
                raise HTTPException(status_code=413, detail="Uploaded image exceeds the 32 MiB limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    input_root = (settings.workspace / ".h3-workbench" / "inputs").resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    destination = (input_root / f"{uuid.uuid4().hex[:12]}-{filename}").resolve()
    if input_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    written = 0
    try:
        with destination.open("wb") as output:
            async for chunk in request.stream():
                written += len(chunk)
                if written > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded image exceeds the 32 MiB limit")
                output.write(chunk)
        info = probe_image(destination)
        return {"path": str(destination), "size_bytes": written, "image": info.to_dict()}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/audio/upload", status_code=201)
async def upload_audio(request: Request) -> dict[str, object]:
    filename = _decode_upload_filename(request, AUDIO_SUFFIXES, "audio")
    max_upload_bytes = 256 * 1024**2
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_upload_bytes:
                raise HTTPException(status_code=413, detail="Uploaded audio exceeds the 256 MiB limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    input_root = (settings.workspace / ".h3-workbench" / "inputs").resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    destination = (input_root / f"{uuid.uuid4().hex[:12]}-{filename}").resolve()
    if input_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    written = 0
    try:
        with destination.open("wb") as output:
            async for chunk in request.stream():
                written += len(chunk)
                if written > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded audio exceeds the 256 MiB limit")
                output.write(chunk)
        info = probe_audio(destination)
        return {"path": str(destination), "size_bytes": written, "audio": info.to_dict()}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _source_asset_status(asset: SourceAsset) -> dict[str, object]:
    cached = settings.state_dir / "sources" / asset.repo_id.replace("/", "--") / asset.path
    local = settings.workspace / Path(asset.path).name
    tokenizer_local = settings.workspace / "qwen_tokenizer" / Path(asset.path).name
    path = next(
        (
            candidate
            for candidate in (cached, local, tokenizer_local)
            if candidate.is_file() and candidate.stat().st_size == asset.size_bytes
        ),
        None,
    )
    return {
        **asdict(asset),
        "ready": path is not None,
        "local_path": str(path.resolve()) if path is not None else None,
    }


def _directory_size(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _preset_product_ready(preset: ExportPreset, directory: Path) -> bool:
    if preset.id == "tokenizer":
        return tokenizer_files_ready(directory)
    if preset.id == "qwen":
        return validated_int8_virtual_qwen_ready(directory)
    if preset.id == "video_vae":
        try:
            blocks = video_vae_block_indices(directory)
        except (OSError, ValueError):
            return False
        long_graphs = (
            directory / "video_decoder_prelude_t7.onnx",
            directory / "video_decoder_head_t7.onnx",
        )
        return (
            len(blocks) == 36
            and all(path.is_file() for path in long_graphs)
            and persistent_video_vae_ready(directory, blocks)
        )
    if preset.product_type == "runtime_adapter":
        return _runtime_adapter_ready(directory)
    if preset.component == "ref2va_transformer":
        try:
            from h3_workbench.ref2va_virtual_slicer import validated_ref2va_virtual_ready

            return validated_ref2va_virtual_ready(directory)
        except (ImportError, OSError, TypeError, ValueError):
            return False
    if preset.component == "fl2va_transformer":
        return _complete_main_model(directory)
    return _manifest_has_blocks(directory / "manifest.json", 0)


def _preset_product_directory(preset: ExportPreset) -> Path:
    product = settings.workspace / preset.output_dir
    if preset.component in {"fl2va_transformer", "ref2va_transformer"}:
        resolved = resolve_main_model_directory(
            settings.workspace,
            settings.output_dir,
            accelerated=False,
            component=preset.component,
        )
        if resolved is not None:
            product = resolved
    return product


def _preset_payload(preset: ExportPreset) -> dict[str, object]:
    assets = list(preset.sources)
    source_status = [_source_asset_status(item) for item in assets]
    product = _preset_product_directory(preset)
    product_ready = _preset_product_ready(preset, product)
    presets_by_id = {item.id: item for item in EXPORT_PRESETS}
    dependencies = []
    for dependency_id in preset.depends_on:
        dependency = presets_by_id[dependency_id]
        dependency_product = _preset_product_directory(dependency)
        dependencies.append(
            {
                "id": dependency.id,
                "label": dependency.label,
                "ready": _preset_product_ready(dependency, dependency_product),
                "path": str(dependency_product.resolve()),
            }
        )
    dependencies_ready = all(bool(item["ready"]) for item in dependencies)
    if product_ready and dependencies_ready:
        status = "ready"
    elif not dependencies_ready:
        status = "dependency_required"
    elif all(bool(item["ready"]) for item in source_status):
        status = "source_ready"
    else:
        status = "download_required"
    return {
        **preset.to_dict(),
        "status": status,
        "sources": source_status,
        "dependencies": dependencies,
        "dependencies_ready": dependencies_ready,
        "product": {
            "ready": product_ready,
            "usable": product_ready and dependencies_ready,
            "path": str(product.resolve()),
            "size_bytes": _directory_size(product),
        },
    }


@app.get("/api/export-presets")
def export_presets() -> dict[str, object]:
    disk = psutil.disk_usage(str(settings.workspace))
    return {
        "download_mode": "direct_http_range",
        "workspace": str(settings.workspace),
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "presets": [_preset_payload(item) for item in EXPORT_PRESETS],
    }


class PresetExportRequest(BaseModel):
    preset_id: str = Field(min_length=1, max_length=64)


@app.post("/api/jobs/download-export", status_code=202)
def create_preset_export(request: PresetExportRequest) -> dict[str, object]:
    try:
        return manager.create_preset_export(request.preset_id).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs")
def jobs() -> list[dict[str, object]]:
    return [job.to_dict() for job in manager.list()]


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, object]:
    item = manager.get(job_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return item.to_dict()


@app.get("/api/jobs/{job_id}/output")
def job_output(job_id: str) -> FileResponse:
    item = manager.get(job_id)
    output = (item.result or {}).get("output") if item is not None else None
    if not isinstance(output, str):
        raise HTTPException(status_code=404, detail="Job output is not available")
    path = Path(output).resolve()
    if settings.workspace not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Job output is not available")
    return FileResponse(path, filename=path.name, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/performance")
def job_performance(job_id: str) -> FileResponse:
    path = manager.performance_log_path(job_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Performance log is not available")
    return FileResponse(path, filename=path.name, media_type="application/x-ndjson")


@app.get("/api/jobs/{job_id}/metadata")
def job_metadata(job_id: str) -> FileResponse:
    item = manager.get(job_id)
    metadata = (item.result or {}).get("metadata") if item is not None else None
    if not isinstance(metadata, str):
        raise HTTPException(status_code=404, detail="Job metadata is not available")
    path = Path(metadata).resolve()
    output_root = (settings.workspace / ".h3-workbench" / "outputs").resolve()
    if path.parent != output_root or not path.is_file():
        raise HTTPException(status_code=404, detail="Job metadata is not available")
    return FileResponse(path, filename=path.name, media_type="application/json")


@app.post("/api/jobs/export", status_code=202)
def create_export(request: ExportRequest) -> dict[str, object]:
    try:
        return manager.create_export(request.model_id, request.video_blocks).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/inference", status_code=202)
def create_inference(request: InferenceRequest) -> dict[str, object]:
    try:
        return manager.create_inference(
            request.token_ids,
            request.prompt,
            request.steps,
            request.seed,
            request.width,
            request.height,
            request.duration_seconds,
            request.temporal_mode,
            request.attention_query_chunk,
            request.l1_prefetch_shards,
            use_acceleration_lora=request.use_acceleration_lora,
            conditioning_mode=request.conditioning_mode,
            start_image_path=request.start_image_path,
            end_image_path=request.end_image_path,
            references=request.references,
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/super-resolution", status_code=202)
def create_super_resolution(request: SuperResolutionRequest) -> dict[str, object]:
    try:
        return manager.create_super_resolution(
            request.source_path,
            request.prompt,
            request.scale,
            request.interpolation,
            request.noise_strength,
            request.processing_mode,
            request.steps,
            request.seed,
            request.attention_query_chunk,
            request.l1_prefetch_shards,
            use_acceleration_lora=request.use_acceleration_lora,
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/profiles")
def generation_profiles() -> list[dict[str, object]]:
    snapshot = probe_gpu_memory()
    device_profiles = probe_device_profiles()
    providers = ort.get_available_providers()
    main_directory = resolve_main_model_directory(settings.workspace, settings.output_dir, accelerated=False)
    ref2va_directory = resolve_main_model_directory(
        settings.workspace,
        settings.output_dir,
        accelerated=False,
        component="ref2va_transformer",
    )
    fl2va_directory = resolve_main_model_directory(
        settings.workspace,
        settings.output_dir,
        accelerated=False,
        component="fl2va_transformer",
    )
    turbo_preset = next(item for item in EXPORT_PRESETS if item.id == "fl2va_turbo_v4")
    turbo_directory = _preset_product_directory(turbo_preset)
    qwen_directory = resolve_qwen_directory(settings.output_dir)
    qwen_manifest = qwen_directory / "manifest.json"
    qwen_ready = (
        validated_int8_virtual_qwen_ready(qwen_directory)
        if qwen_directory.name.endswith("_int8_virtual")
        else _manifest_has_blocks(qwen_manifest, 50)
    )
    acceleration_ready = fl2va_directory is not None and _runtime_adapter_ready(turbo_directory, fl2va_directory)
    main_component: str | None = None
    main_label = "主模型"
    main_capabilities: list[str] = []
    if main_directory is not None:
        try:
            main_manifest = json.loads((main_directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            main_manifest = {}
        main_component = str(main_manifest.get("component") or "") or None
        if main_component is None and "fl2va" in main_directory.name.lower():
            main_component = "fl2va_transformer"
        raw_capabilities = main_manifest.get("capabilities", [])
        main_capabilities = [str(item) for item in raw_capabilities if isinstance(item, str)] if isinstance(raw_capabilities, list) else []
        if main_component == "ref2va_transformer":
            main_label = "Ref2VA 虚拟切片基座"
            if not main_capabilities:
                main_capabilities = ["t2va", "fl2va", "ref2va"]
        elif main_component == "fl2va_transformer":
            main_label = "FL2VA 流式基座"
            if not main_capabilities:
                main_capabilities = ["t2va", "fl2va"]
    tokenizer_ready = tokenizer_files_ready(settings.workspace / "qwen_tokenizer")
    video_directory = settings.output_dir / "video_vae"
    try:
        video_blocks = video_vae_block_indices(video_directory)
    except (OSError, ValueError):
        video_blocks = []
    video_vae_ready = (
        len(video_blocks) == 36
        and (video_directory / "video_decoder_prelude_t7.onnx").is_file()
        and (video_directory / "video_decoder_head_t7.onnx").is_file()
        and persistent_video_vae_ready(video_directory, video_blocks)
    )
    audio_vae_ready = _manifest_has_blocks(settings.output_dir / "audio_vae" / "manifest.json", 0)
    ref2va_ready = ref2va_directory is not None
    shards = main_model_shards(main_directory) if main_directory is not None else []
    main_ready = main_directory is not None
    result: list[dict[str, object]] = []
    for profile in GENERATION_PROFILES.values():
        batches = plan_shard_batches(shards, profile, snapshot.free_bytes) if snapshot.free_bytes and shards else []
        result.append(
            {
                **profile.to_dict(),
                "memory": snapshot.to_dict(),
                "cuda_provider_available": "CUDAExecutionProvider" in providers,
                "device": snapshot.to_dict(),
                "device_profiles": [item.to_dict() for item in device_profiles],
                "qwen_ready": qwen_ready,
                "main_ready": main_ready,
                "ref2va_ready": ref2va_ready,
                "ref2va_label": "Ref2VA 虚拟切片基座",
                "main_component": main_component,
                "main_label": main_label,
                "main_capabilities": main_capabilities,
                "turbo_base_ready": fl2va_directory is not None,
                "turbo_base_label": "FL2VA 流式基座（Turbo 专用）",
                "acceleration_ready": acceleration_ready,
                "acceleration_active": False,
                "tokenizer_ready": tokenizer_ready,
                "video_vae_ready": video_vae_ready,
                "audio_vae_ready": audio_vae_ready,
                "generation_ready": (
                    qwen_ready
                    and main_ready
                    and tokenizer_ready
                    and video_vae_ready
                    and audio_vae_ready
                ),
                "main_shard_batches": [batch.to_dict() for batch in batches],
            }
        )
    return result
