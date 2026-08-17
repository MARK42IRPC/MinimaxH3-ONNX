import json
from urllib.parse import quote

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
from dataclasses import replace

from h3_workbench.app import InferenceRequest, SuperResolutionRequest, app


def test_health() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_export_presets_expose_direct_source_and_local_product_state() -> None:
    response = TestClient(app).get("/api/export-presets")

    assert response.status_code == 200
    payload = response.json()
    assert payload["download_mode"] == "direct_http_range"
    qwen = next(item for item in payload["presets"] if item["id"] == "qwen")
    assert qwen["source"]["url"].startswith("https://")
    assert qwen["source"]["repo_id"] == "Comfy-Org/MiniMax-H3"
    assert qwen["source"]["path"] == "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert qwen["output_dir"].endswith("_int8_virtual")
    assert qwen["output_size_bytes"] < 1024**2
    assert qwen["status"] in {"download_required", "source_ready", "ready"}


def test_old_sliced_product_download_endpoint_is_removed() -> None:
    response = TestClient(app).post("/api/jobs/download", json={"components": ["not-a-model"]})

    assert response.status_code == 405


def test_export_presets_only_expose_validated_variants() -> None:
    response = TestClient(app).get("/api/export-presets")

    assert response.status_code == 200
    payload = response.json()
    assert {item["id"] for item in payload["presets"]} == {
        "tokenizer", "audio_vae", "video_vae", "qwen", "fl2va_streaming", "fl2va_turbo_v4"
    }
    turbo = next(item for item in payload["presets"] if item["id"] == "fl2va_turbo_v4")
    assert turbo["label"] == "Turbo v4 动态 LoRA"
    assert turbo["component"] == "acceleration_lora"
    assert turbo["product_type"] == "runtime_adapter"
    assert turbo["depends_on"] == ["fl2va_streaming"]
    assert turbo["source"]["path"] == "minimax_h3_turbo_v4_step600_ema.safetensors"
    assert turbo["source"]["role"] == "lora"
    assert turbo["lora"] is None
    assert turbo["support"]["path"] == "export_support/h3_silu_temb_grid.safetensors"
    assert turbo["support"]["role"] == "silu_timestep_grid"
    assert turbo["source"]["url"].startswith("https://huggingface.co/")
    assert turbo["download_size_bytes"] == 779849816 + 5510600
    assert turbo["output_size_bytes"] < 16 * 1024**2
    assert turbo["required_space_bytes"] < 1024**3
    assert "40GB" in turbo["description"]


def test_turbo_adapter_preset_exposes_base_dependency(monkeypatch, tmp_path) -> None:
    from h3_workbench import app as app_module

    models = tmp_path / "onnx_models"
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, workspace=tmp_path, output_dir=models, state_dir=tmp_path / ".h3-workbench"),
    )

    turbo = next(
        item for item in TestClient(app_module.app).get("/api/export-presets").json()["presets"]
        if item["id"] == "fl2va_turbo_v4"
    )

    assert turbo["status"] == "dependency_required"
    assert turbo["dependencies_ready"] is False
    assert turbo["dependencies"] == [
        {
            "id": "fl2va_streaming",
            "label": "FL2VA 流式基座",
            "ready": False,
            "path": str((models / "minimax_h3_fl2va_pruned_fp8_scaled_streaming").resolve()),
        }
    ]


def test_preset_export_rejects_unknown_variant() -> None:
    response = TestClient(app).post("/api/jobs/download-export", json={"preset_id": "untested"})

    assert response.status_code == 400


def test_system_info_reports_pagefile_capacity() -> None:
    response = TestClient(app).get("/api/system")

    assert response.status_code == 200
    assert response.json()["pagefile_total_bytes"] >= 0


def test_webui_contains_live_job_elapsed_timer() -> None:
    response = TestClient(app).get("/assets/app.js")

    assert response.status_code == 200
    assert "data-job-elapsed" in response.text
    assert "setInterval(updateJobTimers, 1000)" in response.text
    assert "实时性能" in response.text
    assert "缓存与预取" in response.text
    assert "执行参数" in response.text
    assert "job-progress-value" in response.text
    assert "/performance" in response.text
    assert "Turbo v4 动态 LoRA" in response.text
    assert "手动开启" in response.text
    assert "use_acceleration_lora" in response.text
    assert "Turbo v4 尚未就绪，将使用流式基座模型" not in response.text


def test_webui_uri_encodes_upload_filenames() -> None:
    response = TestClient(app).get("/assets/app.js")

    assert response.status_code == 200
    assert '"x-filename": encodeURIComponent(file.name)' in response.text


def test_image_upload_decodes_utf8_filename(monkeypatch, tmp_path) -> None:
    from h3_workbench import app as app_module
    from h3_workbench.media_input import ImageInfo

    monkeypatch.setattr(
        app_module,
        "settings",
        replace(
            app_module.settings,
            workspace=tmp_path,
            output_dir=tmp_path / "onnx_models",
            state_dir=tmp_path / ".h3-workbench",
        ),
    )
    monkeypatch.setattr(
        app_module,
        "probe_image",
        lambda path: ImageInfo(path=str(path), width=320, height=180),
    )

    response = TestClient(app_module.app).post(
        "/api/images/upload",
        headers={"x-filename": quote("首帧测试.png", safe=""), "content-type": "image/png"},
        content=b"image-data",
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["path"].endswith("-首帧测试.png")
    assert payload["image"]["width"] == 320
    assert payload["image"]["height"] == 180


def test_performance_log_endpoint(monkeypatch, tmp_path) -> None:
    from h3_workbench import app as app_module

    log_path = tmp_path / "h3-job.jsonl"
    log_path.write_text('{"sequence":0}\n', encoding="utf-8")
    monkeypatch.setattr(app_module.manager, "performance_log_path", lambda job_id: log_path if job_id == "job" else None)

    response = TestClient(app).get("/api/jobs/job/performance")
    missing = TestClient(app).get("/api/jobs/missing/performance")

    assert response.status_code == 200
    assert response.json() == {"sequence": 0}
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert missing.status_code == 404


def test_hardware_snapshot_sse_and_websocket_share_structured_contract(monkeypatch) -> None:
    from h3_workbench import app as app_module

    class FakeLivePerformanceMonitor:
        interval_seconds = 0.01

        def __init__(self) -> None:
            self.stopped = False

        @staticmethod
        def _sample(sequence: int) -> dict[str, object]:
            return {
                "schema": "h3-hardware-sample-v1",
                "timestamp": "2026-08-15T00:00:00+00:00",
                "sequence": sequence,
                "cpu": {"system_percent": 25.0},
                "memory": {"available_bytes": 1024},
                "gpu": {"available": False, "devices": [], "reason": "not_found"},
                "disk": {"read_bytes_per_second": 128.0, "write_bytes_per_second": 64.0},
                "process": {"pid": 7},
            }

        def snapshot(self) -> dict[str, object]:
            return self._sample(7)

        def wait_for_sample(self, after_sequence: int, timeout: float) -> dict[str, object]:
            assert timeout >= self.interval_seconds
            return self._sample(after_sequence + 1)

        def stop(self) -> None:
            self.stopped = True

    monitor = FakeLivePerformanceMonitor()
    monkeypatch.setattr(app_module, "live_performance_monitor", monitor)

    with TestClient(app_module.app) as client:
        snapshot = client.get("/api/hardware/snapshot")
        stream = client.get("/api/hardware/stream?samples=1")
        with client.websocket_connect("/api/hardware/ws?samples=1") as websocket:
            websocket_sample = websocket.receive_json()

    assert snapshot.status_code == 200
    assert snapshot.json()["sequence"] == 7
    assert snapshot.json()["gpu"]["available"] is False
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert stream.headers["cache-control"] == "no-cache"
    assert "event: hardware" in stream.text
    data_line = next(line for line in stream.text.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: "))["schema"] == "h3-hardware-sample-v1"
    assert websocket_sample["schema"] == "h3-hardware-sample-v1"
    assert websocket_sample["disk"]["write_bytes_per_second"] == 64.0
    assert monitor.stopped is True


def test_hardware_snapshot_reports_monitor_timeout(monkeypatch) -> None:
    from h3_workbench import app as app_module

    class TimedOutMonitor:
        interval_seconds = 1.0

        @staticmethod
        def snapshot():
            raise TimeoutError("sample unavailable")

        @staticmethod
        def stop() -> None:
            pass

    monkeypatch.setattr(app_module, "live_performance_monitor", TimedOutMonitor())

    with TestClient(app_module.app) as client:
        response = client.get("/api/hardware/snapshot")

    assert response.status_code == 503
    assert response.json()["detail"] == "sample unavailable"


def test_job_progress_never_moves_backwards(tmp_path) -> None:
    from h3_workbench.jobs import Job, JobManager

    manager = JobManager(tmp_path, tmp_path / "onnx")
    manager._jobs["job"] = Job(id="job", model_id="model", progress=0.5)
    manager._update("job", progress=0.4)

    assert manager.get("job").progress == 0.5


def test_generation_profiles() -> None:
    response = TestClient(app).get("/api/profiles")

    assert response.status_code == 200
    profile = response.json()[0]
    assert profile["id"] == "360p-17f"
    assert profile["video_tokens"] == 1680
    assert profile["acceleration_active"] is False


def test_models_endpoint_exposes_validated_sliced_product(monkeypatch, tmp_path) -> None:
    from h3_workbench import app as app_module

    product = tmp_path / "exported" / "sliced_main"
    product.mkdir(parents=True)
    (product / "schedule.json").write_text("{}", encoding="utf-8")
    (product / "manifest.json").write_text(
        json.dumps(
            {
                "format": "h3-workbench-onnx-v2",
                "component": "fl2va_transformer",
                "validation_passed": True,
                "build_complete": True,
                "schedule_format": "h3-schedule-v2",
                "schedule": "schedule.json",
                "blocks": list(range(50)),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(
            app_module.settings,
            workspace=tmp_path,
            output_dir=tmp_path / "onnx_models",
            state_dir=tmp_path / ".h3-workbench",
        ),
    )

    response = TestClient(app_module.app).get("/api/models")

    assert response.status_code == 200
    product_record = next(item for item in response.json() if item["id"] == "exported/sliced_main")
    assert product_record["record_type"] == "product"
    assert product_record["ready"] is True
    assert product_record["export_supported"] is False


def test_main_model_resolver_accepts_workspace_root_product(tmp_path) -> None:
    from h3_workbench.jobs import resolve_main_model_directory

    product = tmp_path / "sliced_main"
    product.mkdir()
    (product / "schedule.json").write_text("{}", encoding="utf-8")
    (product / "manifest.json").write_text(
        json.dumps(
            {
                "component": "fl2va_transformer",
                "validation_passed": True,
                "build_complete": True,
                "schedule_format": "h3-schedule-v2",
                "schedule": "schedule.json",
                "blocks": list(range(50)),
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_main_model_directory(tmp_path, tmp_path / "onnx_models", accelerated=False)

    assert resolved == product.resolve()


def test_generation_profile_detects_turbo_capability(monkeypatch, tmp_path) -> None:
    from h3_workbench import app as app_module

    models = tmp_path / "onnx_models"
    base = models / "minimax_h3_fl2va_pruned_fp8_scaled_streaming"
    turbo = tmp_path / ".h3-workbench" / "accelerators" / "turbo_v4"
    qwen = models / "qwen3vl_32b_minimax_h3_nvfp4_awq"
    base.mkdir(parents=True)
    turbo.mkdir(parents=True)
    qwen.mkdir(parents=True)
    blocks = list(range(50))
    (base / "manifest.json").write_text(
        json.dumps(
            {
                "validation_passed": True,
                "build_complete": True,
                "schedule_format": "h3-schedule-v2",
                "schedule": "schedule.json",
                "blocks": blocks,
                "graphs": [],
            }
        ),
        encoding="utf-8",
    )
    (base / "schedule.json").write_text("{}", encoding="utf-8")
    (qwen / "manifest.json").write_text(
        json.dumps({"validation_passed": True, "blocks": blocks}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, workspace=tmp_path, output_dir=models, state_dir=tmp_path / ".h3-workbench"),
    )
    monkeypatch.setattr(app_module, "tokenizer_files_ready", lambda _: True)
    monkeypatch.setattr(app_module, "validate_turbo_adapter", lambda *_, **__: {})

    profile = TestClient(app_module.app).get("/api/profiles").json()[0]

    assert profile["main_ready"] is True
    assert profile["acceleration_ready"] is True
    assert profile["generation_ready"] is False


def test_prompt_inference_request_accepts_dynamic_resolution() -> None:
    request = InferenceRequest(
        prompt="snowfall",
        token_ids=None,
        width=512,
        height=288,
        duration_seconds=15,
        steps=1,
        temporal_mode="native",
        attention_query_chunk=64,
        l1_prefetch_shards=3,
    )

    assert request.token_ids is None
    assert request.width == 512
    assert request.height == 288
    assert request.duration_seconds == 15
    assert request.steps == 1
    assert request.use_acceleration_lora is False
    assert request.temporal_mode == "native"
    assert request.attention_query_chunk == 64
    assert request.l1_prefetch_shards == 3


def test_inference_request_accepts_explicit_acceleration_lora() -> None:
    request = InferenceRequest(prompt="snowfall", steps=4, use_acceleration_lora=True)

    assert request.use_acceleration_lora is True


@pytest.mark.parametrize(
    ("mode", "start", "end"),
    (
        ("text", None, None),
        ("first", "start.png", None),
        ("last", None, "end.png"),
        ("first_last", "start.png", "end.png"),
    ),
)
def test_inference_request_accepts_four_conditioning_modes(
    mode: str,
    start: str | None,
    end: str | None,
) -> None:
    request = InferenceRequest(
        prompt="snowfall",
        conditioning_mode=mode,
        start_image_path=start,
        end_image_path=end,
    )

    assert request.conditioning_mode == mode
    assert request.start_image_path == start
    assert request.end_image_path == end


def test_inference_request_rejects_missing_frame_condition_input() -> None:
    with pytest.raises(ValidationError, match="requires"):
        InferenceRequest(prompt="snowfall", conditioning_mode="first_last", start_image_path="start.png")


def test_inference_request_rejects_acceleration_lora_outside_supported_steps() -> None:
    with pytest.raises(ValidationError, match="supports 4-8"):
        InferenceRequest(prompt="snowfall", steps=3, use_acceleration_lora=True)


def test_super_resolution_request_exposes_manual_processing_mode() -> None:
    assert SuperResolutionRequest(source_path="clip.mp4", prompt="snowfall").processing_mode == "segmented"
    assert (
        SuperResolutionRequest(source_path="clip.mp4", prompt="snowfall", processing_mode="direct").processing_mode
        == "direct"
    )


def test_super_resolution_request_rejects_unknown_processing_mode() -> None:
    with pytest.raises(ValidationError):
        SuperResolutionRequest(source_path="clip.mp4", prompt="snowfall", processing_mode="auto")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duration_seconds", 15.1),
        ("duration_seconds", 0),
        ("steps", 0),
        ("steps", 51),
        ("attention_query_chunk", 48),
        ("l1_prefetch_shards", 5),
    ),
)
def test_inference_request_rejects_out_of_range_duration_and_steps(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="snowfall", **{field: value})
