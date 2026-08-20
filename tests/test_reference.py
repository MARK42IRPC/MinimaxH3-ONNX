from pathlib import Path

import numpy as np
import pytest

from h3_workbench.media_input import AudioInfo, ImageInfo, VideoInfo
from h3_workbench.reference_conditioning import (
    reference_image_size,
    resolve_reference_image_short_edge,
)
from h3_workbench.reference import (
    ReferenceSpec,
    build_ref2va_packed_layout,
    build_reference_presentation,
    build_row_timesteps,
    reference_labels,
    resolve_reference_specs,
    sample_video_condition_frames,
)


class _Tokenizer:
    special = {
        "<|vision_start|>": 101,
        "<|vision_end|>": 102,
        "<|image_pad|>": 103,
        "<|video_pad|>": 104,
    }

    def __call__(self, value, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [1000 + ord(character) for character in value]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]


def test_low_vram_reference_images_use_a_bounded_visual_token_budget(monkeypatch) -> None:
    monkeypatch.delenv("H3_REFERENCE_IMAGE_SHORT_EDGE", raising=False)

    short_edge = resolve_reference_image_short_edge(4 * 1024**3)

    assert short_edge == 768
    assert reference_image_size(2880, 1440, short_edge) == (768, 1536)


def test_reference_image_short_edge_allows_an_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("H3_REFERENCE_IMAGE_SHORT_EDGE", "1280")

    assert resolve_reference_image_short_edge(4 * 1024**3) == 1280


def test_reference_labels_preserve_request_order_and_number_per_modality() -> None:
    references = [
        ReferenceSpec(kind="video", path="v.mp4", has_audio=True),
        ReferenceSpec(kind="image", path="i.png"),
        ReferenceSpec(kind="audio", path="a.wav", has_audio=True),
    ]

    assert reference_labels(references) == (
        ("Audio 1", "Video 1"),
        ("Picture 1",),
        ("Audio 2",),
    )


def test_resolve_reference_specs_enforces_official_duration_budgets(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "subject.png"
    video_a = tmp_path / "motion-a.mp4"
    video_b = tmp_path / "motion-b.mp4"
    audio = tmp_path / "voice.wav"
    for path in (image, video_a, video_b, audio):
        path.write_bytes(b"fixture")

    monkeypatch.setattr("h3_workbench.media_input.probe_image", lambda path: ImageInfo(str(path), 640, 480))
    monkeypatch.setattr(
        "h3_workbench.media_input.probe_video",
        lambda path: VideoInfo(str(path), 640, 480, 24.0, 192, 8.0, True, {}),
    )
    monkeypatch.setattr(
        "h3_workbench.media_input.probe_audio",
        lambda path: AudioInfo(str(path), 48_000, 1, 240_000, 5.0),
    )

    specs = resolve_reference_specs(
        tmp_path,
        [
            {"type": "video", "path": video_a.name},
            {"type": "image", "path": image.name},
            {"type": "audio", "path": audio.name},
        ],
    )

    assert [spec.kind for spec in specs] == ["video", "image", "audio"]
    assert specs[0].labels == ("Audio 1", "Video 1")
    assert specs[2].labels == ("Audio 2",)

    with pytest.raises(ValueError, match="total at most 15"):
        resolve_reference_specs(
            tmp_path,
            [
                {"type": "video", "path": video_a.name},
                {"type": "video", "path": video_b.name},
            ],
        )


def test_audio_reference_cannot_be_used_alone(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="paired with"):
        resolve_reference_specs(tmp_path, [{"type": "audio", "path": audio.name}])


def test_video_condition_sampling_matches_qwen_two_fps_pairing() -> None:
    indices, timestamps = sample_video_condition_frames(49, fps=24.0, sample_fps=2.0, temporal_patch=2)

    np.testing.assert_array_equal(indices, np.asarray([0, 12, 24, 36, 48]))
    assert timestamps == (0.25, 1.25, 2.0)
    assert [f"<{value:.1f} seconds>" for value in timestamps] == [
        "<0.2 seconds>",
        "<1.2 seconds>",
        "<2.0 seconds>",
    ]


def test_reference_presentation_places_video_soundtrack_before_video() -> None:
    references = [
        ReferenceSpec(kind="video", path="v.mp4", has_audio=True),
        ReferenceSpec(kind="image", path="i.png"),
        ReferenceSpec(kind="audio", path="a.wav", has_audio=True),
    ]
    tokenizer = _Tokenizer()

    presentation = build_reference_presentation(
        tokenizer,
        "prompt verbatim",
        references,
        image_token_counts=[2],
        video_block_token_counts=[3],
        video_block_timestamps=[[0.25, 0.75]],
    )

    encoded_audio = tokenizer("<Audio 1>: ", add_special_tokens=False)["input_ids"]
    encoded_video = tokenizer("<Video 1>: ", add_special_tokens=False)["input_ids"]
    assert presentation.token_ids[: len(encoded_audio)].tolist() == encoded_audio
    assert presentation.token_ids[len(encoded_audio) : len(encoded_audio) + len(encoded_video)].tolist() == encoded_video
    vision_start = len(encoded_audio) + len(encoded_video) + len("<0.2 seconds>")
    assert presentation.token_ids[vision_start : vision_start + 5].tolist() == [101, 104, 104, 104, 102]
    assert presentation.token_tags[vision_start : vision_start + 5].tolist() == [0, 0, 0, 0, 0]
    assert presentation.mm_token_type_ids[vision_start : vision_start + 5].tolist() == [0, 2, 2, 2, 0]
    assert presentation.token_ids[-len("prompt verbatim") :].tolist() == tokenizer(
        "prompt verbatim", add_special_tokens=False
    )["input_ids"]


def test_ref2va_layout_packs_reference_blocks_before_target_rows() -> None:
    references = [
        ReferenceSpec(kind="image", path="i.png"),
        ReferenceSpec(kind="audio", path="a.wav", has_audio=True),
    ]
    layout = build_ref2va_packed_layout(
        text_token_tags=np.asarray([1, 0], dtype=np.int64),
        references=references,
        condition_video_shapes=[(1, 2, 2)],
        condition_audio_row_counts=[4],
        target_video_shape=(2, 2, 2),
        num_target_audio_latents=2,
    )

    np.testing.assert_array_equal(layout.text_indices, np.asarray([0, 1]))
    np.testing.assert_array_equal(layout.video_indices, np.asarray([2, 11, 12]))
    np.testing.assert_array_equal(layout.audio_indices, np.arange(3, 11))
    np.testing.assert_array_equal(layout.target_audio_indices, np.arange(7, 11))
    np.testing.assert_array_equal(layout.target_video_indices, np.asarray([11, 12]))
    assert layout.num_condition_video_rows == 1
    assert layout.num_condition_audio_rows == 4
    np.testing.assert_allclose(layout.position_ids[3:7, 0], np.asarray([3.0, 4.0, 3.0, 4.0]))
    np.testing.assert_allclose(layout.position_ids[7:11, 0], np.asarray([5.0, 6.0, 5.0, 6.0]))
    np.testing.assert_allclose(layout.position_ids[11:, 0], np.asarray([5.0, 5.0 + 5.0 / 3.0]))

    unique, inverse = build_row_timesteps(layout, video_timestep=0.5, audio_timestep=0.25)
    np.testing.assert_allclose(unique, np.asarray([0.25, 0.5, 0.999, 1.0], dtype=np.float32))
    assert inverse.shape == (13,)


def test_video_soundtrack_rows_are_immediately_before_video_rows() -> None:
    reference = ReferenceSpec(kind="video", path="v.mp4", has_audio=True)
    layout = build_ref2va_packed_layout(
        text_token_tags=np.asarray([1], dtype=np.int64),
        references=[reference],
        condition_video_shapes=[(2, 2, 2)],
        condition_audio_row_counts=[4],
        target_video_shape=(1, 2, 2),
        num_target_audio_latents=1,
    )

    np.testing.assert_array_equal(layout.audio_indices[:4], np.arange(1, 5))
    np.testing.assert_array_equal(layout.video_indices[:2], np.arange(5, 7))
    assert layout.position_ids[1, 0] == layout.position_ids[5, 0]


def test_job_manager_dispatches_ordered_references_to_ref2va_worker(tmp_path: Path, monkeypatch) -> None:
    from h3_workbench import jobs as jobs_module

    specs = (
        ReferenceSpec(kind="video", path="motion.mp4", has_audio=True),
        ReferenceSpec(kind="image", path="subject.png"),
        ReferenceSpec(kind="audio", path="voice.wav", has_audio=True),
    )
    monkeypatch.setattr(jobs_module, "resolve_reference_specs", lambda workspace, values: specs)

    class Executor:
        call = None

        def submit(self, function, *args, **kwargs):
            self.call = (function, args, kwargs)
            return object()

    manager = jobs_module.JobManager(tmp_path, tmp_path / "onnx_models")
    executor = Executor()
    manager._executor = executor

    job = manager.create_inference(
        token_ids=None,
        prompt="Use every reference in order",
        steps=1,
        seed=7,
        width=640,
        height=360,
        duration_seconds=5.0,
        temporal_mode="native",
        attention_query_chunk=32,
        l1_prefetch_shards=2,
        references=[
            {"kind": "video", "path": "motion.mp4"},
            {"kind": "image", "path": "subject.png"},
            {"kind": "audio", "path": "voice.wav"},
        ],
    )

    assert executor.call is not None
    function, args, kwargs = executor.call
    assert function == manager._run_inference
    assert args[0] == job.id
    assert args[6] == 124
    assert args[8] == "native"
    assert kwargs["frame_conditioning"] is None
    assert kwargs["references"] == specs
