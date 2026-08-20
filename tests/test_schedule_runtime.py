import json
from pathlib import Path

import numpy as np
import pytest

from h3_workbench.profiles import PROFILE_360P_17F
from h3_workbench.reference import ReferenceSpec, build_ref2va_packed_layout
from h3_workbench.schedule_runtime import ScheduleMainRuntime


class _Argument:
    def __init__(self, name: str, shape: list, dtype: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = dtype


class _ShardSession:
    def get_inputs(self):
        return [
            _Argument("video_patches", ["video_sequence", 96]),
            _Argument("audio_patches", ["audio_sequence", 32]),
            _Argument("text_states", ["text_sequence", 5120], "tensor(float16)"),
            _Argument("hidden_states", ["sequence", 5376]),
            _Argument("run_0", [], "tensor(bool)"),
            _Argument("run_1", [], "tensor(bool)"),
        ]

    def get_outputs(self):
        return [
            _Argument("emb/video_embeddings", []),
            _Argument("emb/audio_embeddings", []),
            _Argument("emb/text_embeddings", []),
            _Argument("norm/hidden_states_out", []),
        ]

    def run(self, output_names, feeds):
        assert set(feeds) == {argument.name for argument in self.get_inputs()}
        if output_names[0].startswith("emb/"):
            assert bool(feeds["run_0"]) and not bool(feeds["run_1"])
            assert feeds["hidden_states"].shape[1] == 5376
            return [
                np.zeros((1, 5376), dtype=np.float32),
                np.zeros((1, 5376), dtype=np.float32),
                np.pad(feeds["text_states"].astype(np.float32), ((0, 0), (0, 256))),
            ]
        assert not bool(feeds["run_0"]) and bool(feeds["run_1"])
        return [feeds["hidden_states"] * 2]


class _Runner:
    provider = "CPUExecutionProvider"

    def __init__(self) -> None:
        self.built = 0

    def session(self, path):
        self.built += 1
        return _ShardSession()


def _schedule() -> dict:
    return {
        "format": "h3-schedule-v2",
        "model": "test",
        "resources": {
            "activation_peak_slots": 4,
            "activation_slots": {},
            "session_slot_hints": {"vram_4gib": 1, "vram_24gib": 8},
        },
        "shards": [{"id": "shard_000", "file": "shard_000.onnx", "resident": True, "graphs": ["emb", "norm"]}],
        "steps": [
            {
                "id": 0,
                "phase": "preamble",
                "kind": "op",
                "op": "preamble_inputs",
                "inputs": {"raw_text_states": {"source": "external", "name": "raw_text_states"}},
                "outputs": {
                    "video_patches": {"target": "buffer", "name": "video"},
                    "audio_patches": {"target": "buffer", "name": "audio"},
                    "text_states": {"target": "buffer", "name": "raw"},
                },
                "release": [],
                "barrier_after": False,
            },
            {
                "id": 1,
                "phase": "preamble",
                "kind": "graph",
                "shard": "shard_000",
                "graph": "emb",
                "inputs": {
                    "video_patches": {"source": "buffer", "name": "video"},
                    "audio_patches": {"source": "buffer", "name": "audio"},
                    "text_states": {"source": "buffer", "name": "raw"},
                },
                "outputs": {
                    "video_embeddings": {"target": "discard"},
                    "audio_embeddings": {"target": "discard"},
                    "text_embeddings": {"target": "buffer", "name": "hidden"},
                },
                "release": ["video", "audio", "raw"],
                "barrier_after": False,
            },
            {
                "id": 2,
                "phase": "preamble",
                "kind": "graph",
                "shard": "shard_000",
                "graph": "norm",
                "inputs": {"hidden_states": {"source": "buffer", "name": "hidden"}},
                "outputs": {"hidden_states_out": {"target": "const", "name": "text_states"}},
                "release": ["hidden"],
                "barrier_after": False,
            },
        ],
    }


def test_runtime_requires_v2_schedule(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        ScheduleMainRuntime(tmp_path, _Runner())


def test_runtime_activity_labels_follow_loaded_main_product() -> None:
    runtime = object.__new__(ScheduleMainRuntime)

    runtime._supports_ref2va = True
    assert runtime._main_module_label == "Ref2VA"
    assert runtime._loader_module_label == "Ref2VA Loader"
    assert runtime._diagnostics_module_label == "Ref2VA Diagnostics"

    runtime._supports_ref2va = False
    assert runtime._main_module_label == "FL2VA"
    assert runtime._loader_module_label == "FL2VA Loader"
    assert runtime._diagnostics_module_label == "FL2VA Diagnostics"


def test_ref2va_ops_scatter_modalities_and_extract_only_target_rows() -> None:
    references = [
        ReferenceSpec(kind="image", path="image.png"),
        ReferenceSpec(kind="audio", path="voice.wav", has_audio=True),
    ]
    layout = build_ref2va_packed_layout(
        text_token_tags=np.asarray([1, 0], dtype=np.int64),
        references=references,
        condition_video_shapes=[(1, 2, 2)],
        condition_audio_row_counts=[4],
        target_video_shape=(2, 2, 2),
        num_target_audio_latents=2,
    )
    runtime = object.__new__(ScheduleMainRuntime)
    runtime.profile = PROFILE_360P_17F
    runtime._lora_adapter = None
    runtime._validate_graph_outputs = False
    external = {
        "video": np.zeros((1, 24, 2, 2, 2), dtype=np.float32),
        "audio": np.zeros((1, 32, 2, 2), dtype=np.float32),
        "sigma": 0.5,
        "reference_video_latents": (np.ones((1, 24, 1, 2, 2), dtype=np.float32),),
        "reference_audio_latents": (np.ones((1, 32, 2, 2), dtype=np.float32),),
        "ref2va_layout": layout,
    }
    constants = {"text_states": np.zeros((2, 5376), dtype=np.float32)}
    buffers: dict[str, np.ndarray] = {}
    denoise_step = {
        "op": "denoise_inputs",
        "inputs": {
            "text_states": {"source": "const", "name": "text_states"},
            "video_latent": {"source": "external", "name": "video"},
            "audio_latent": {"source": "external", "name": "audio"},
            "sigma_video": {"source": "external", "name": "sigma"},
        },
        "outputs": {
            name: {"target": "buffer", "name": name}
            for name in (
                "video_patches",
                "audio_patches",
                "embedding_text_padding",
                "timesteps",
                "position_ids",
                "modulation_ids",
                "sigma_audio",
            )
        },
    }

    runtime._run_op(denoise_step, external, constants, buffers)

    assert buffers["video_patches"].shape == (3, 96)
    assert buffers["audio_patches"].shape == (8, 32)
    np.testing.assert_array_equal(buffers["position_ids"], layout.position_ids.astype(np.float32))
    np.testing.assert_allclose(buffers["timesteps"][-2:], np.asarray([0.999, 1.0], dtype=np.float32))
    assert buffers["modulation_ids"].shape == layout.token_tags.shape

    # The official MiniMax-H3 head emits data-ward velocity; the unpack op
    # must preserve its sign for the scheduler's positive (sigma - next_sigma)
    # Euler update.
    unpack_step = {
        "op": "unpack_velocity",
        "inputs": {
            "video_patches": {"source": "buffer", "name": "video_patches"},
            "audio_patches": {"source": "buffer", "name": "audio_patches"},
        },
        "outputs": {
            "video_velocity": {"target": "external", "name": "video_velocity"},
            "audio_velocity": {"target": "external", "name": "audio_velocity"},
        },
    }
    buffers["video_patches"] = np.ones((runtime.profile.video_tokens, 96), dtype=np.float32)
    buffers["audio_patches"] = np.ones((runtime.profile.audio_tokens, 32), dtype=np.float32)
    runtime._run_op(unpack_step, external, constants, buffers)
    assert np.all(external["video_velocity"] == 1.0)
    assert np.all(external["audio_velocity"] == 1.0)

    audio_embeddings = np.full((layout.audio_indices.size, 5376), 20.0, dtype=np.float32)
    video_embeddings = np.full((layout.video_indices.size, 5376), 30.0, dtype=np.float32)
    buffers.update(audio_embeddings=audio_embeddings, video_embeddings=video_embeddings)
    concat_step = {
        "op": "concat_hidden",
        "inputs": {
            "text_states": {"source": "const", "name": "text_states"},
            "audio_embeddings": {"source": "buffer", "name": "audio_embeddings"},
            "video_embeddings": {"source": "buffer", "name": "video_embeddings"},
        },
        "outputs": {"hidden": {"target": "buffer", "name": "hidden"}},
    }
    runtime._run_op(concat_step, external, constants, buffers)
    np.testing.assert_array_equal(buffers["hidden"][layout.text_indices], constants["text_states"])
    assert np.all(buffers["hidden"][layout.audio_indices] == 20.0)
    assert np.all(buffers["hidden"][layout.video_indices] == 30.0)

    split_step = {
        "op": "split_hidden",
        "inputs": {
            "hidden": {"source": "buffer", "name": "hidden"},
            "text_states": {"source": "const", "name": "text_states"},
        },
        "outputs": {
            "audio_hidden": {"target": "buffer", "name": "target_audio"},
            "video_hidden": {"target": "buffer", "name": "target_video"},
        },
    }
    runtime._run_op(split_step, external, constants, buffers)
    assert buffers["target_audio"].shape[0] == layout.target_audio_indices.size
    assert buffers["target_video"].shape[0] == layout.target_video_indices.size


def test_preamble_executes_if_one_hot_and_reuses_session(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    runner = _Runner()
    runtime = ScheduleMainRuntime(tmp_path, runner)
    raw = np.ones((3, 5120), dtype=np.float16)
    refined = runtime.prepare_text(raw)
    assert refined.shape == (3, 5376)
    assert np.all(refined[:, :5120] == 2)
    assert np.all(refined[:, 5120:] == 0)
    assert runner.built == 1
    assert runtime.warm_fixed_sessions()["warmed_sessions"] == 1
    assert runner.built == 1


def test_runtime_prefers_ref2va_virtual_source_weights(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )

    class VirtualWeights:
        def __init__(self, directory, runner, graph_paths) -> None:
            self.graph_paths = graph_paths
            self.closed = False

        def prime_ram_cache(self):
            return {"virtual_source": True}

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "h3_workbench.ref2va_virtual_slicer.ref2va_virtual_ready",
        lambda directory: True,
    )
    monkeypatch.setattr(
        "h3_workbench.ref2va_virtual_slicer.Ref2VASourceWeights",
        VirtualWeights,
    )

    runtime = ScheduleMainRuntime(tmp_path, _Runner())

    assert isinstance(runtime._persistent_weights, VirtualWeights)
    assert runtime._ram_cache_prime == {"virtual_source": True}
    runtime.close()
    assert runtime._persistent_weights is None


class _DirectSession:
    def get_inputs(self):
        return [_Argument("hidden_states", ["sequence", 5376])]

    def get_outputs(self):
        return [_Argument("hidden_states_out", ["sequence", 5376])]

    def run(self, output_names, feeds):
        assert output_names == ["hidden_states_out"]
        return [feeds["hidden_states"] * 3]


class _DirectRunner(_Runner):
    def session(self, path):
        self.built += 1
        return _DirectSession()


class _NonfiniteDirectSession(_DirectSession):
    def run(self, output_names, feeds):
        output = feeds["hidden_states"].copy()
        output[0, 0] = np.nan
        return [output]


class _NonfiniteDirectRunner(_Runner):
    def session(self, path):
        self.built += 1
        return _NonfiniteDirectSession()


def test_runtime_accepts_plain_outputs_from_direct_graph(tmp_path: Path, monkeypatch) -> None:
    schedule = _schedule()
    schedule["shards"][0]["graphs"] = ["norm"]
    schedule["steps"] = [
        {
            "id": 0,
            "phase": "preamble",
            "kind": "op",
            "op": "preamble_inputs",
            "inputs": {"raw_text_states": {"source": "external", "name": "raw_text_states"}},
            "outputs": {
                "video_patches": {"target": "discard"},
                "audio_patches": {"target": "discard"},
                "text_states": {"target": "buffer", "name": "hidden"},
            },
            "release": [],
            "barrier_after": False,
        },
        {
            "id": 1,
            "phase": "preamble",
            "kind": "graph",
            "shard": "shard_000",
            "graph": "norm",
            "inputs": {"hidden_states": {"source": "buffer", "name": "hidden"}},
            "outputs": {"hidden_states_out": {"target": "const", "name": "text_states"}},
            "release": ["hidden"],
            "barrier_after": False,
        },
    ]
    (tmp_path / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    runtime = ScheduleMainRuntime(tmp_path, _DirectRunner())
    raw = np.ones((2, 5376), dtype=np.float32)
    assert np.all(runtime.prepare_text(raw) == 3)


def test_runtime_diagnostics_identify_first_nonfinite_graph(tmp_path: Path, monkeypatch) -> None:
    schedule = _schedule()
    schedule["shards"][0]["graphs"] = ["norm"]
    schedule["steps"] = [
        {
            "id": 0,
            "phase": "preamble",
            "kind": "op",
            "op": "preamble_inputs",
            "inputs": {"raw_text_states": {"source": "external", "name": "raw_text_states"}},
            "outputs": {
                "video_patches": {"target": "discard"},
                "audio_patches": {"target": "discard"},
                "text_states": {"target": "buffer", "name": "hidden"},
            },
            "release": [],
            "barrier_after": False,
        },
        {
            "id": 1,
            "phase": "preamble",
            "kind": "graph",
            "shard": "shard_000",
            "graph": "norm",
            "inputs": {"hidden_states": {"source": "buffer", "name": "hidden"}},
            "outputs": {"hidden_states_out": {"target": "const", "name": "text_states"}},
            "release": ["hidden"],
            "barrier_after": False,
        },
    ]
    (tmp_path / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    monkeypatch.setenv("H3_VALIDATE_GRAPH_OUTPUTS", "1")
    runtime = ScheduleMainRuntime(tmp_path, _NonfiniteDirectRunner())

    with pytest.raises(FloatingPointError, match="norm/hidden_states_out: 1"):
        runtime.prepare_text(np.ones((2, 5376), dtype=np.float32))


def test_iobinding_preserves_scalar_input_rank() -> None:
    class _Binding:
        def __init__(self) -> None:
            self.shapes = {}
            self.inputs_cleared = False
            self.outputs_cleared = False
            self.inputs_synchronized = False
            self.outputs_synchronized = False

        def bind_cpu_input(self, name, value):
            self.shapes[name] = value.shape

        def bind_output(self, name, device):
            raise AssertionError("No outputs expected")

        def get_outputs(self):
            return []

        def synchronize_inputs(self):
            self.inputs_synchronized = True

        def synchronize_outputs(self):
            self.outputs_synchronized = True

        def clear_binding_inputs(self):
            self.inputs_cleared = True

        def clear_binding_outputs(self):
            self.outputs_cleared = True

    class _Session:
        def __init__(self) -> None:
            self.binding = _Binding()

        def io_binding(self):
            return self.binding

        def run_with_iobinding(self, binding):
            assert binding is self.binding

    runtime = object.__new__(ScheduleMainRuntime)
    session = _Session()

    runtime._run_with_iobinding(
        session,
        [_Argument("scalar", [], "tensor(int64)")],
        {"scalar": np.asarray(0, dtype=np.int64)},
        [],
        device_output=True,
    )

    assert session.binding.shapes["scalar"] == ()
    assert session.binding.inputs_synchronized
    assert session.binding.outputs_synchronized
    assert session.binding.inputs_cleared
    assert session.binding.outputs_cleared


def test_low_vram_runtime_disables_device_resident_hidden(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        low_vram_mode = True

    runtime = ScheduleMainRuntime(
        tmp_path,
        Runner(),
        profile=PROFILE_360P_17F.with_frame_count(120),
    )

    assert runtime._device_hidden is False
    assert runtime._recycle_persistent_sessions is True
    runtime.close()


@pytest.mark.parametrize(
    ("requested", "torch_ready", "low_vram", "expected"),
    (
        ("auto", True, True, "ort"),
        ("auto", True, False, "torch"),
        ("torch", True, True, "torch"),
        ("auto", False, True, "ort"),
        ("ort", True, True, "ort"),
    ),
)
def test_cuda_runtime_selects_requested_sdpa_backend(
    tmp_path: Path,
    monkeypatch,
    requested: str,
    torch_ready: bool,
    low_vram: bool,
    expected: str,
) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    monkeypatch.setenv("H3_PERSISTENT_WEIGHTS", "0")
    monkeypatch.setenv("H3_SDPA_BACKEND", requested)
    monkeypatch.setattr("torch.cuda.is_available", lambda: torch_ready)

    class StreamingAttention:
        def __init__(self, directory, runner) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "h3_workbench.inference_runtime.ORTStreamingAttention",
        StreamingAttention,
    )

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        low_vram_mode = low_vram

    runtime = ScheduleMainRuntime(tmp_path, Runner())

    assert runtime.metrics()["sdpa_backend"] == expected
    assert (runtime._ort_streamed_attention is not None) is (expected == "ort")
    runtime.close()


def test_cuda_runtime_rejects_unavailable_requested_torch_sdpa(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    monkeypatch.setenv("H3_PERSISTENT_WEIGHTS", "0")
    monkeypatch.setenv("H3_SDPA_BACKEND", "torch")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        low_vram_mode = True

    with pytest.raises(ValueError, match="CUDA-enabled PyTorch"):
        ScheduleMainRuntime(tmp_path, Runner())


def test_auto_sdpa_falls_back_when_torch_lacks_selected_architecture(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    monkeypatch.setenv("H3_PERSISTENT_WEIGHTS", "0")
    monkeypatch.setenv("H3_SDPA_BACKEND", "auto")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("h3_workbench.schedule_runtime.torch_cuda_architecture_supported", lambda index: False)

    class StreamingAttention:
        def __init__(self, directory, runner) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "h3_workbench.inference_runtime.ORTStreamingAttention",
        StreamingAttention,
    )

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        device_index = 0
        low_vram_mode = False

    runtime = ScheduleMainRuntime(tmp_path, Runner())

    assert runtime.metrics()["sdpa_backend"] == "ort"
    assert "torch build lacks" in runtime.metrics()["sdpa_fallback_reason"]
    runtime.close()


def test_explicit_torch_sdpa_rejects_unsupported_architecture(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    monkeypatch.setenv("H3_PERSISTENT_WEIGHTS", "0")
    monkeypatch.setenv("H3_SDPA_BACKEND", "torch")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("h3_workbench.schedule_runtime.torch_cuda_architecture_supported", lambda index: False)

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        device_index = 0

    with pytest.raises(ValueError, match="does not contain kernels"):
        ScheduleMainRuntime(tmp_path, Runner())


def test_runtime_rejects_invalid_sdpa_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("H3_SDPA_BACKEND", "invalid")

    with pytest.raises(ValueError, match="auto, torch, ort"):
        ScheduleMainRuntime(tmp_path, _Runner())


def test_low_vram_runtime_retains_topologies_but_not_activations_for_short_sequences(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        low_vram_mode = True

    runtime = ScheduleMainRuntime(tmp_path, Runner(), profile=PROFILE_360P_17F.resized(352, 360))

    assert runtime._device_hidden is False
    assert runtime._recycle_persistent_sessions is False
    runtime.close()


def test_persistent_session_recycling_can_be_disabled_independently(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "schedule.json").write_text(json.dumps(_schedule()), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )
    monkeypatch.setenv("H3_DEVICE_RESIDENT_HIDDEN", "0")
    monkeypatch.setenv("H3_RECYCLE_PERSISTENT_SESSIONS", "0")

    class Runner(_Runner):
        provider = "CUDAExecutionProvider"
        low_vram_mode = True

    runtime = ScheduleMainRuntime(tmp_path, Runner())

    assert runtime._device_hidden is False
    assert runtime._recycle_persistent_sessions is False
    runtime.close()


def test_ref2va_lora_feeds_are_combined_with_persistent_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    schedule = _schedule()
    schedule["shards"] = [
        {
            "id": "shard_000",
            "file": "shard_000.onnx",
            "resident": False,
            "graphs": ["main_block_00_mlp"],
        }
    ]
    schedule["steps"] = []
    (tmp_path / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    monkeypatch.setattr(
        "h3_workbench.schedule_runtime.validate_runtime_schedule",
        lambda schedule, directory: None,
    )

    state = {"persistent_closed": False, "adapter_closed": False}
    lora = np.full((2, 2), 5, dtype=np.float16)

    class Session:
        def get_inputs(self):
            return [
                _Argument("hidden_states", [2, 2]),
                _Argument("base_weight", [2, 2]),
                _Argument("lora_fc1_A", [2, 2], "tensor(float16)"),
            ]

        def get_outputs(self):
            return [_Argument("hidden_states_out", [2, 2])]

        def run(self, output_names, feeds):
            assert output_names == ["hidden_states_out"]
            np.testing.assert_array_equal(feeds["base_weight"], np.eye(2, dtype=np.float32))
            np.testing.assert_array_equal(feeds["lora_fc1_A"], lora)
            return [feeds["hidden_states"] + 1]

    class Persistent:
        enabled = True
        prefetch_depth = 0
        pinned_enabled = False
        pinned_fallback_reason = None
        host_pool_allocations = 0
        prefetch_workers = 1
        device_metrics = {}

        def __init__(self, *args, topology_paths=None, **kwargs):
            assert topology_paths == {"mlp": tmp_path / "adapter_mlp.onnx"}

        @staticmethod
        def supports(graph):
            return graph == "main_block_00_mlp"

        @staticmethod
        def session(graph):
            return Session(), True, 0.01

        @staticmethod
        def weights(graph):
            return (
                {"base_weight": np.eye(2, dtype=np.float32)},
                16,
                16,
                0,
                0,
                True,
                0.01,
                0.01,
                0.0,
                False,
                False,
            )

        @staticmethod
        def release(graph):
            assert graph == "main_block_00_mlp"

        @staticmethod
        def close():
            state["persistent_closed"] = True

    class Adapter:
        topology_paths = {"mlp": tmp_path / "adapter_mlp.onnx"}

        @staticmethod
        def graph_feeds(graph):
            assert graph == "main_block_00_mlp"
            return {"lora_fc1_A": lora}

        @staticmethod
        def metrics():
            return {"variant": "ref2v_turbo_v0.1_dynamic"}

        @staticmethod
        def close():
            state["adapter_closed"] = True

    monkeypatch.setattr("h3_workbench.persistent_weights.PersistentWeightRuntime", Persistent)
    runtime = ScheduleMainRuntime(tmp_path, _Runner(), lora_adapter=Adapter())
    external = {"hidden": np.ones((2, 2), dtype=np.float32)}
    step = {
        "graph": "main_block_00_mlp",
        "shard": "shard_000",
        "inputs": {"hidden_states": {"source": "external", "name": "hidden"}},
        "outputs": {
            "hidden_states_out": {"target": "external", "name": "result"}
        },
    }

    runtime._run_graph(step, external, {}, {}, {})

    np.testing.assert_array_equal(external["result"], np.full((2, 2), 2, dtype=np.float32))
    assert runtime.metrics()["lora_adapter"] == {"variant": "ref2v_turbo_v0.1_dynamic"}
    runtime.close()
    assert state == {"persistent_closed": True, "adapter_closed": True}
