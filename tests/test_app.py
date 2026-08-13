from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
from dataclasses import replace

from h3_workbench.app import InferenceRequest, app


def test_health() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_components_exposes_download_catalog() -> None:
    response = TestClient(app).get("/api/model-components")

    assert response.status_code == 200
    payload = response.json()
    assert payload["repo"] == "Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS"
    assert payload["project"].endswith("MARK42IRPC/MinimaxH3-ONNX")
    assert {item["id"] for item in payload["components"]} == {"qwen", "turbo", "streaming", "video_vae", "audio_vae", "tokenizer"}


def test_download_rejects_unknown_component() -> None:
    response = TestClient(app).post("/api/jobs/download", json={"components": ["not-a-model"]})

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


def test_generation_profile_is_ready_with_turbo_only(monkeypatch, tmp_path) -> None:
    from h3_workbench import app as app_module

    models = tmp_path / "onnx_models"
    turbo = models / "minimax_h3_fl2va_pruned_fp8_scaled_accelerated"
    qwen = models / "qwen3vl_32b_minimax_h3_nvfp4_awq"
    turbo.mkdir(parents=True)
    qwen.mkdir(parents=True)
    manifest = '{"validation_passed": true, "blocks": [' + ",".join(str(item) for item in range(50)) + "]}"
    (turbo / "manifest.json").write_text(manifest, encoding="utf-8")
    (qwen / "manifest.json").write_text(manifest, encoding="utf-8")
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, output_dir=models))
    monkeypatch.setattr(app_module, "tokenizer_files_ready", lambda _: True)

    profile = TestClient(app_module.app).get("/api/profiles").json()[0]

    assert profile["main_ready"] is False
    assert profile["acceleration_ready"] is True
    assert profile["generation_ready"] is True


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
    assert request.temporal_mode == "native"
    assert request.attention_query_chunk == 64
    assert request.l1_prefetch_shards == 3


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
