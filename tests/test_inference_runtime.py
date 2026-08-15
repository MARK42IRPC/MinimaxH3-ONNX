import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import time
from pathlib import Path
from onnx import TensorProto, helper

from h3_workbench.inference_runtime import (
    ORTGraphRunner,
    _ensure_device_streamed_sdpa_graph,
    _ensure_streamed_sdpa_graph,
    host_prefetch_budget_bytes,
    modulation_ids,
    packed_position_ids,
    pack_audio,
    patchify_video,
    time_shift_sigma,
    streamed_attention,
    select_fl2va_chunk_sizes,
    sample_latents,
    unpack_audio,
    unpatchify_video,
)
from h3_workbench.media_output import (
    _assemble_video_vae_tiles,
    _pad_video_latents_to_tile,
    _release_io_binding,
    _split_tiles,
    _video_vae_temporal_windows,
    decode_video_latents_onnx,
)
from h3_workbench.media_output import decode_audio_latents_onnx
from h3_workbench.profiles import PROFILE_360P_17F, video_latent_frames_for_output, video_vae_output_frames
from h3_workbench.shard_cache import ShardPrefetchCache
from h3_workbench.vendor.video_vae import MiniMaxH3VideoVAE
from h3_workbench.torch_compat import apply_rope_split_half


def test_long_video_vae_temporal_window_plan_matches_reference() -> None:
    windows, frames = _video_vae_temporal_windows(37)
    assert windows[:2] == [(0, 7), (5, 12)]
    assert windows[-1] == (30, 37)
    assert len(windows) == 7
    assert frames == 124


def test_video_vae_io_binding_cleanup_releases_inputs_and_outputs() -> None:
    class Binding:
        inputs_cleared = False
        outputs_cleared = False

        def clear_binding_inputs(self) -> None:
            self.inputs_cleared = True

        def clear_binding_outputs(self) -> None:
            self.outputs_cleared = True

    binding = Binding()
    _release_io_binding(binding)

    assert binding.inputs_cleared
    assert binding.outputs_cleared


def test_video_vae_onnx_runner_closes_when_decode_fails(monkeypatch, tmp_path: Path) -> None:
    state = {"closed": False}

    class Runner:
        def __init__(self, prefer_cuda: bool):
            assert prefer_cuda

        def close(self) -> None:
            state["closed"] = True

    def fail(*_: object) -> np.ndarray:
        raise RuntimeError("decode failed")

    monkeypatch.setattr("h3_workbench.inference_runtime.ORTGraphRunner", Runner)
    monkeypatch.setattr("h3_workbench.media_output._decode_video_latents_onnx_with_runner", fail)

    with pytest.raises(RuntimeError, match="decode failed"):
        decode_video_latents_onnx(
            tmp_path,
            np.zeros((1, 24, 2, 1, 1), dtype=np.float16),
            16,
        )

    assert state["closed"]


def test_video_vae_two_latent_tokens_decode_to_five_frames() -> None:
    with torch.device("meta"):
        model = MiniMaxH3VideoVAE()

    assert model._decode_temporal_chunks(2) == (5, 1)
    assert model.decode_output_shape((1, 24, 2, 24, 40)) == (1, 3, 5, 384, 640)
    assert video_vae_output_frames(2) == 5
    assert video_latent_frames_for_output(1) == 2
    assert video_latent_frames_for_output(5) == 2


def test_small_video_latent_canvas_is_center_padded_to_one_vae_tile() -> None:
    latents = np.ones((1, 24, 2, 8, 10), dtype=np.float16)

    padded = _pad_video_latents_to_tile(latents)

    assert padded.shape == (1, 24, 2, 16, 16)
    np.testing.assert_array_equal(padded[:, :, :, 4:12, 3:13], latents)
    assert np.count_nonzero(padded[:, :, :, :4]) == 0


def test_requested_frames_snap_to_native_h3_temporal_grid() -> None:
    assert video_latent_frames_for_output(17) == 7
    assert video_vae_output_frames(7) == 22
    assert video_latent_frames_for_output(120) == 37
    assert video_vae_output_frames(37) == 124
    assert video_latent_frames_for_output(360) == 107
    assert video_vae_output_frames(107) == 362


def test_dynamic_resolution_profile_and_video_tiles() -> None:
    profile = PROFILE_360P_17F.resized(512, 300)

    assert (profile.padded_width, profile.padded_height) == (512, 320)
    assert profile.video_latent_width == 32
    assert profile.video_latent_height == 20
    assert _split_tiles(384) == ([0, 128], [256, 256], [128])
    assert _split_tiles(640) == ([0, 192, 384], [256, 256, 256], [64, 64])


def test_fl2va_chunk_sizes_follow_available_vram() -> None:
    gib = 1024**3
    assert select_fl2va_chunk_sizes(3 * gib) == {"qkv": 1024, "attention_output": 2048, "mlp": 512}
    assert select_fl2va_chunk_sizes(int(2.5 * gib)) == {"qkv": 1024, "attention_output": 1024, "mlp": 256}
    assert select_fl2va_chunk_sizes(2 * gib) == {"qkv": 512, "attention_output": 512, "mlp": 256}
    assert select_fl2va_chunk_sizes(3 * gib, dynamic=False) == {"qkv": 256, "attention_output": 256, "mlp": 256}


def test_host_prefetch_budget_preserves_commit_reserve(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3_workbench.inference_runtime.probe_host_commit_memory",
        lambda: (28 * 1024**3, 36 * 1024**3),
    )

    assert host_prefetch_budget_bytes() == 4 * 1024**3


def test_host_prefetch_budget_reports_zero_below_reserve(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3_workbench.inference_runtime.probe_host_commit_memory",
        lambda: (35 * 1024**3, 36 * 1024**3),
    )

    assert host_prefetch_budget_bytes() == 0


def test_cuda_runner_defaults_to_non_spinning_single_cpu_thread(monkeypatch) -> None:
    monkeypatch.delenv("H3_ORT_CPU_THREADS", raising=False)
    monkeypatch.delenv("H3_ORT_ALLOW_SPINNING", raising=False)
    monkeypatch.setattr(
        "h3_workbench.inference_runtime.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr("h3_workbench.inference_runtime._preload_cuda_dlls", lambda: None)

    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)

    assert runner.ort_cpu_threads == 1
    assert runner.ort_allow_spinning is False
    runner.close()


def test_cuda_runner_enables_low_vram_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "h3_workbench.inference_runtime.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr("h3_workbench.inference_runtime._preload_cuda_dlls", lambda: None)
    monkeypatch.setattr(
        "h3_workbench.inference_runtime.probe_gpu_memory",
        lambda: type("Memory", (), {"total_bytes": 4 * 1024**3})(),
    )

    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)

    assert runner.low_vram_mode is True
    runner.close()


@pytest.mark.parametrize("frames", (17, 20, 28))
def test_video_vae_tile_assembly_preserves_temporal_length(frames: int) -> None:
    tiles = {
        (0, 0): np.zeros((1, 3, frames, 256, 256), dtype=np.float32),
        (0, 1): np.ones((1, 3, frames, 256, 256), dtype=np.float32),
    }

    canvas = _assemble_video_vae_tiles(
        tiles,
        y_starts=[0],
        y_overlaps=[],
        x_starts=[0, 192],
        x_overlaps=[64],
        padded_height=256,
        padded_width=448,
    )

    assert canvas.shape == (1, 3, frames, 256, 448)


def test_360p_pack_roundtrip() -> None:
    profile = PROFILE_360P_17F
    video = np.arange(1 * 24 * 2 * 24 * 40, dtype=np.float32).reshape(1, 24, 2, 24, 40)
    audio = np.arange(1 * 32 * 2 * 29, dtype=np.float32).reshape(1, 32, 2, 29)
    np.testing.assert_array_equal(unpatchify_video(patchify_video(video), 2, 24, 40), video)
    np.testing.assert_array_equal(unpack_audio(pack_audio(audio)), audio)
    assert packed_position_ids(profile, 192).shape == (profile.sequence_tokens, 3)


def test_video_position_times_follow_official_five_token_cycle() -> None:
    profile = PROFILE_360P_17F
    positions = packed_position_ids(profile, 192)
    video = positions[192 + profile.audio_tokens :].reshape(profile.video_latent_frames, -1, 3)

    expected = 192.0 + np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(np.asarray([1, 4, 4, 4, 4, 1], dtype=np.float32) * (5.0 / 3.0)))
    )
    np.testing.assert_allclose(video[:, 0, 0], expected, rtol=0.0, atol=2e-5)


def test_video_vae_rope_pairs_split_halves() -> None:
    values = torch.tensor([[[[1.0, 2.0, 10.0, 20.0]]]])
    # 90-degree rotation: (a, b) -> (-b, a).
    table = torch.tensor(
        [[[0.0, -1.0], [1.0, 0.0]], [[0.0, -1.0], [1.0, 0.0]]]
    ).reshape(1, 1, 1, 2, 2, 2)

    query, key = apply_rope_split_half(values, values, table)

    expected = torch.tensor([[[[-10.0, -20.0, 1.0, 2.0]]]])
    torch.testing.assert_close(query, expected)
    torch.testing.assert_close(key, expected)


def test_modulation_rows_cover_packed_sequence() -> None:
    profile = PROFILE_360P_17F
    times, ids = modulation_ids(profile, 192, 0.5)
    assert times.shape == (2,)
    assert ids.shape == (profile.sequence_tokens,)
    assert time_shift_sigma(0.5, 12.0, 3.0) < 0.5


def test_streamed_attention_matches_full_attention_on_cpu() -> None:
    random = np.random.default_rng(7)
    sequence = 11
    packed = random.standard_normal((sequence, 56, 384), dtype=np.float32)
    query, key, value = np.split(packed, 3, axis=2)
    query = query.transpose(1, 0, 2)[None]
    key = key.transpose(1, 0, 2)[None]
    value = value.transpose(1, 0, 2)[None]
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(128)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.matmul(probabilities, value).transpose(0, 2, 1, 3).reshape(sequence, -1)

    actual = streamed_attention(packed, use_cuda=False, query_chunk_tokens=3)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_device_streamed_attention_graph_matches_chunked_reference(tmp_path: Path) -> None:
    random = np.random.default_rng(13)
    packed = random.standard_normal((7, 56, 384), dtype=np.float32)
    graph = _ensure_device_streamed_sdpa_graph(
        tmp_path / "device_sdpa.onnx",
        sequence_tokens=7,
        query_chunk_tokens=3,
    )
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])

    actual = session.run(None, {"packed": packed})[0]
    expected = streamed_attention(packed, use_cuda=False, query_chunk_tokens=3)

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_normalized_ort_attention_stays_finite_for_large_qkv(tmp_path) -> None:
    graph = _ensure_streamed_sdpa_graph(tmp_path / "normalized_sdpa.onnx")
    random = np.random.default_rng(17)
    query = random.normal(0, 120, (1, 56, 3, 128)).astype(np.float32)
    key = random.normal(0, 110, (1, 56, 7, 128)).astype(np.float32)
    value = random.normal(0, 90, (1, 56, 7, 128)).astype(np.float32)

    def normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
        scale = max(1.0, float(np.max(np.abs(values))) / 8.0)
        return (values / scale).astype(np.float16), scale

    query_fp16, query_scale = normalize(query)
    key_fp16, key_scale = normalize(key)
    value_fp16, value_scale = normalize(value)
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            "query": query_fp16,
            "key": key_fp16,
            "value": value_fp16,
            "score_scale": np.asarray(query_scale * key_scale / np.sqrt(128.0), dtype=np.float32),
            "value_scale": np.asarray(value_scale, dtype=np.float32),
        },
    )[0]

    assert np.isfinite(actual).all()
    scores = np.matmul(
        query_fp16.astype(np.float32),
        key_fp16.astype(np.float32).transpose(0, 1, 3, 2),
    )
    scores *= query_scale * key_scale / np.sqrt(128.0)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.matmul(
        probabilities.astype(np.float16).astype(np.float32),
        value_fp16.astype(np.float32),
    )
    expected *= value_scale
    expected = expected.transpose(0, 2, 1, 3).reshape(actual.shape)
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=0.5)


def test_sampling_isolates_nonfinite_audio_velocity() -> None:
    class Runtime:
        sampling_step = 0
        sampling_steps = 0
        audio_fallback_reason = None

        @staticmethod
        def prepare_text(text_states):
            return text_states

        @staticmethod
        def denoise_step(video, audio, text_states, sigma, text_is_refined=False):
            return np.zeros_like(video), np.full_like(audio, np.nan)

    runtime = Runtime()
    video = np.ones((1, 1), dtype=np.float32)
    audio = np.ones((1, 1), dtype=np.float32)

    sampled_video, sampled_audio = sample_latents(runtime, video, audio, np.ones((1, 1), dtype=np.float32), steps=1)

    np.testing.assert_array_equal(sampled_video, video)
    np.testing.assert_array_equal(sampled_audio, audio)
    assert runtime.audio_fallback_reason is not None


def test_sample_latents_emits_step_checkpoints() -> None:
    class Runtime:
        audio_fallback_reason = None
        sampling_step = 0
        sampling_steps = 0

        @staticmethod
        def prepare_text(text_states):
            return text_states

        @staticmethod
        def denoise_step(video, audio, text_states, sigma, text_is_refined=False):
            return np.ones_like(video), np.ones_like(audio)

    checkpoints = []
    runtime = Runtime()
    sample_latents(
        runtime,
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        steps=2,
        checkpoint_callback=lambda current, total, video, audio: checkpoints.append(
            (current, total, video.copy(), audio.copy())
        ),
    )

    assert [(current, total) for current, total, _, _ in checkpoints] == [(1, 2), (2, 2)]
    assert not np.array_equal(checkpoints[0][2], checkpoints[1][2])


def test_session_batch_does_not_retry_a_consumer_error(tmp_path) -> None:
    graph = tmp_path / "graph.onnx"
    graph.write_bytes(b"stub")
    runner = ORTGraphRunner.__new__(ORTGraphRunner)
    runner.provider = "CPUExecutionProvider"
    runner.shard_cache = ShardPrefetchCache(budget_bytes=0)
    runner.session = lambda path: object()
    runner._session_cache = {}
    runner._session_cache_bytes = 0
    runner._session_cache_budget = 0
    runner._session_cache_hits = 0
    runner._session_cache_misses = 0
    batches_seen = 0

    with pytest.raises(RuntimeError, match="compute failed"):
        for _ in runner.adaptive_session_batches([[graph]]):
            batches_seen += 1
            raise RuntimeError("compute failed")

    runner.close()
    assert batches_seen == 1


def test_single_graph_audio_decoder_accepts_dynamic_frames(tmp_path) -> None:
    input_info = helper.make_tensor_value_info("latents", TensorProto.FLOAT, [1, 32, 2, "latent_frames"])
    output_info = helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, 32, 2, "latent_frames"])
    graph = helper.make_graph([helper.make_node("Identity", ["latents"], ["waveform"])], "audio", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    onnx.save(model, tmp_path / "audio_decoder.onnx")
    latents = np.random.default_rng(5).standard_normal((1, 32, 2, 7), dtype=np.float32).clip(-1, 1)

    waveform = decode_audio_latents_onnx(tmp_path, latents, prefer_cuda=False)

    np.testing.assert_array_equal(waveform, latents)


def test_l1_session_prefetch_overlaps_following_shard(tmp_path) -> None:
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "prefetch",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
    onnx.save(model, first)
    onnx.save(model, second)
    runner = ORTGraphRunner.__new__(ORTGraphRunner)
    runner.provider = "CUDAExecutionProvider"
    runner.shard_cache = ShardPrefetchCache(budget_bytes=0)
    runner._l1_prefetch_hits = 0
    runner._l1_prefetch_waits = 0
    runner._l1_prefetch_wait_seconds = 0.0
    runner._session_cache = {}
    runner._session_cache_bytes = 0
    runner._session_cache_budget = 0
    runner._session_cache_hits = 0
    runner._session_cache_misses = 0

    def fake_session(path=None, serialized_model=None):
        time.sleep(0.02)
        return object()

    runner.session = fake_session
    operations: list[str] = []

    def loading(details: dict[str, object]) -> None:
        operations.append(str(details["operation"]))

    batches = runner.adaptive_session_batches(
        [[first], [second]], loading_callback=loading, session_prefetch_depth=1, session_prefetch_budget_bytes=100
    )
    first_sessions = next(batches)
    time.sleep(0.06)
    del first_sessions
    second_sessions = next(batches)
    del second_sessions
    with pytest.raises(StopIteration):
        next(batches)
    runner.close()

    assert "L1 prefetch queued" in operations
    assert runner._l1_prefetch_hits >= 1


def test_l1_prefetch_stages_one_stable_rolling_window(tmp_path) -> None:
    graph_paths = []
    for index in range(3):
        graph = tmp_path / f"graph_{index}.onnx"
        graph.write_bytes(b"graph")
        graph_paths.append(graph)

    runner = ORTGraphRunner(prefer_cuda=False)
    runner.provider = "CUDAExecutionProvider"
    staged: list[list[Path]] = []
    runner.shard_cache.stage = lambda paths: staged.append(list(paths))  # type: ignore[method-assign]
    runner.shard_cache.wait = lambda path: None  # type: ignore[method-assign]
    runner.session = lambda path=None, serialized_model=None: object()  # type: ignore[method-assign]

    batches = runner.adaptive_session_batches(
        [[path] for path in graph_paths],
        session_prefetch_depth=2,
        session_prefetch_budget_bytes=1024,
    )
    assert len(list(batches)) == 3
    assert staged[0] == graph_paths


def test_l1_prefetch_barrier_runs_before_building_barrier_batch(tmp_path) -> None:
    qkv = tmp_path / "main_block_00_attention_qkv.onnx"
    output = tmp_path / "main_block_00_attention_output.onnx"
    mlp = tmp_path / "main_block_00_mlp.onnx"
    qkv.write_bytes(b"qkv")
    output.write_bytes(b"output")
    mlp.write_bytes(b"mlp")
    runner = ORTGraphRunner(prefer_cuda=False)
    runner.provider = "CUDAExecutionProvider"
    runner.shard_cache.wait = lambda path: None  # type: ignore[method-assign]
    events: list[str] = []
    runner.session = lambda path=None, serialized_model=None: events.append(f"build:{Path(path).name}") or object()  # type: ignore[method-assign]

    batches = runner.adaptive_session_batches(
        [[qkv], [output], [mlp]],
        before_batch=lambda paths: events.append(f"barrier:{paths[0].name}"),
        session_prefetch_depth=2,
        session_prefetch_budget_bytes=1024,
        prefetch_barrier=lambda paths: paths[0] == output,
    )
    for batch in batches:
        if batch and batch[0][0] == output:
            events.append("consume:main_block_00_attention_output.onnx")
        batch.clear()

    runner.close()
    assert events.index("barrier:main_block_00_attention_output.onnx") < events.index(
        "build:main_block_00_attention_output.onnx"
    )
    assert events.index("consume:main_block_00_attention_output.onnx") < events.index("build:main_block_00_mlp.onnx")


def test_run_reuses_cached_session_across_calls(tmp_path) -> None:
    graph = tmp_path / "identity.onnx"
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
    model = helper.make_model(
        helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "identity", [input_info], [output_info]),
        opset_imports=[helper.make_opsetid("", 20)],
        ir_version=9,
    )
    onnx.save(model, graph)
    runner = ORTGraphRunner(prefer_cuda=False)
    runner.set_session_cache_budget(1 << 20)
    built = 0
    original_session = runner.session

    def counting_session(path):
        nonlocal built
        built += 1
        return original_session(path)

    runner.session = counting_session
    first = runner.run(graph, {"input": np.zeros(2, dtype=np.float32)})
    second = runner.run(graph, {"input": np.ones(2, dtype=np.float32)})

    np.testing.assert_array_equal(first[0], np.zeros(2))
    np.testing.assert_array_equal(second[0], np.ones(2))
    assert built == 1
    assert runner._session_cache_hits == 1
    assert len(runner._session_cache) == 1
    runner.close()


def test_run_does_not_cache_without_budget(tmp_path) -> None:
    graph = tmp_path / "identity.onnx"
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
    model = helper.make_model(
        helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "identity", [input_info], [output_info]),
        opset_imports=[helper.make_opsetid("", 20)],
        ir_version=9,
    )
    onnx.save(model, graph)
    runner = ORTGraphRunner(prefer_cuda=False)
    built = 0
    original_session = runner.session

    def counting_session(path):
        nonlocal built
        built += 1
        return original_session(path)

    runner.session = counting_session
    runner.run(graph, {"input": np.zeros(2, dtype=np.float32)})
    runner.run(graph, {"input": np.zeros(2, dtype=np.float32)})

    assert built == 2
    assert len(runner._session_cache) == 0
    runner.close()
