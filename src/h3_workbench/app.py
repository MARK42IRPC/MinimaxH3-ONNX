from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Literal

import onnxruntime as ort
import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from h3_workbench.config import Settings
from h3_workbench.jobs import JobManager
from h3_workbench.memory_planner import main_model_shards, plan_shard_batches, probe_gpu_memory
from h3_workbench.model_registry import scan_models
from h3_workbench.profiles import GENERATION_PROFILES
from h3_workbench.shard_cache import default_l2_cache_bytes, default_prefetch_depth
from h3_workbench.tokenizer import tokenizer_files_ready
from h3_workbench.model_catalog import MODELSCOPE_REPO, GITHUB_REPO, all_component_status
from h3_workbench.source_catalog import EXPORT_PRESETS

settings = Settings.from_env()
manager = JobManager(settings.workspace, settings.output_dir)
web_dir = Path(__file__).parent / "web"

app = FastAPI(title="MiniMax H3 Edge Workbench", version="0.1.0")
app.mount("/assets", StaticFiles(directory=web_dir), name="assets")


def _manifest_has_blocks(path: Path, count: int) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return manifest.get("validation_passed") is True and len(manifest.get("blocks", [])) == count
    except (OSError, ValueError, TypeError):
        return False


def _gpu_info() -> dict[str, object]:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "memory_bytes": props.total_memory,
            "cuda": torch.version.cuda,
        }
    try:
        flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            creationflags=flags,
            text=True,
            timeout=3,
        )
        name, memory_mb, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": name,
            "memory_bytes": int(memory_mb) * 1024 * 1024,
            "driver": driver,
            "cuda": None,
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {"available": False}


class ExportRequest(BaseModel):
    model_id: str
    video_blocks: str = Field(default="all", pattern=r"^(all|\d+(,\d+)*)$")


class InferenceRequest(BaseModel):
    token_ids: list[int] | None = None
    prompt: str | None = None
    steps: int = Field(default=4, ge=1, le=50)
    seed: int = 1
    width: int = Field(default=640, ge=128, le=1024)
    height: int = Field(default=360, ge=128, le=1024)
    duration_seconds: float = Field(default=17 / 24, gt=0, le=15)
    temporal_mode: Literal["native", "segmented"] = "segmented"
    attention_query_chunk: int = Field(default=256, ge=32, le=512)
    l1_prefetch_shards: int = Field(default=2, ge=0, le=4)

    @model_validator(mode="after")
    def require_prompt_or_tokens(self) -> "InferenceRequest":
        if not self.prompt and not self.token_ids:
            raise ValueError("Provide either prompt or token_ids")
        if self.token_ids is not None and len(self.token_ids) > 192:
            raise ValueError("token_ids must contain at most 192 items")
        if self.prompt is not None and len(self.prompt) > 4000:
            raise ValueError("prompt must contain at most 4000 characters")
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


@app.get("/api/models")
def models() -> list[dict[str, object]]:
    return [record.to_dict() for record in scan_models(settings.workspace)]


@app.get("/api/model-components")
def model_components() -> dict[str, object]:
    return {"repo": MODELSCOPE_REPO, "project": GITHUB_REPO, "components": all_component_status(settings.workspace)}


class ModelDownloadRequest(BaseModel):
    components: list[str] = Field(min_length=1, max_length=6)


@app.post("/api/jobs/download", status_code=202)
def create_download(request: ModelDownloadRequest) -> dict[str, object]:
    try:
        return manager.create_download(request.components).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/export-presets")
def export_presets() -> dict[str, object]:
    return {"official_repo": "Comfy-Org/MiniMax-H3", "turbo_repo": "larryvrh/MiniMax-H3-Turbo-Lora", "presets": [item.to_dict() for item in EXPORT_PRESETS]}


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
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/profiles")
def generation_profiles() -> list[dict[str, object]]:
    snapshot = probe_gpu_memory()
    providers = ort.get_available_providers()
    main_directory = settings.output_dir / "minimax_h3_fl2va_pruned_fp8_scaled_streaming"
    qwen_manifest = settings.output_dir / "qwen3vl_32b_minimax_h3_nvfp4_awq" / "manifest.json"
    accelerated_manifest = settings.output_dir / "minimax_h3_fl2va_pruned_fp8_scaled_accelerated" / "manifest.json"
    qwen_ready = _manifest_has_blocks(qwen_manifest, 50)
    acceleration_ready = _manifest_has_blocks(accelerated_manifest, 50)
    tokenizer_ready = tokenizer_files_ready(settings.workspace / "qwen_tokenizer")
    shards = main_model_shards(main_directory) if (main_directory / "manifest.json").is_file() else []
    main_ready = bool(shards)
    result: list[dict[str, object]] = []
    for profile in GENERATION_PROFILES.values():
        batches = plan_shard_batches(shards, profile, snapshot.free_bytes) if snapshot.free_bytes and shards else []
        result.append(
            {
                **profile.to_dict(),
                "memory": snapshot.to_dict(),
                "cuda_provider_available": "CUDAExecutionProvider" in providers,
                "qwen_ready": qwen_ready,
                "main_ready": main_ready,
                "acceleration_ready": acceleration_ready,
                "acceleration_active": False,
                "tokenizer_ready": tokenizer_ready,
                # A complete Turbo export is independently runnable for its
                # supported 4-8 step range, even when the optional base model
                # has been removed to reclaim disk space.
                "generation_ready": qwen_ready and (main_ready or acceleration_ready),
                "main_shard_batches": [batch.to_dict() for batch in batches],
            }
        )
    return result
