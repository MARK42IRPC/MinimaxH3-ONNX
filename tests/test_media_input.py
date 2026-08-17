import json
from pathlib import Path

import numpy as np
import pytest

from h3_workbench.media_input import (
    center_pad_video,
    prepare_frame_condition,
    prepare_super_resolution_segment,
    read_video_prompt,
    resolve_video_path,
    resize_video_spatiotemporal,
)


def test_prompt_prefers_sidecar_over_mp4_tags(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    video.with_suffix(".metadata.json").write_text(
        json.dumps({"input": {"prompt": "sidecar prompt"}}),
        encoding="utf-8",
    )

    assert read_video_prompt(video, {"comment": "tag prompt"}) == ("sidecar prompt", "sidecar")


def test_prompt_reads_json_mp4_comment() -> None:
    assert read_video_prompt(Path("clip.mp4"), {"comment": '{"input":{"prompt":"embedded prompt"}}'}) == (
        "embedded prompt",
        "mp4.comment",
    )


def test_spatiotemporal_interpolation_and_padding_shapes() -> None:
    frames = np.zeros((1, 3, 2, 3, 5), dtype=np.float32)
    for mode in ("nearest", "bilinear", "bicubic", "trilinear"):
        resized = resize_video_spatiotemporal(frames, 4, 6, 10, mode)
        assert resized.shape == (1, 3, 4, 6, 10)
    padded = center_pad_video(frames, 5, 9)
    assert padded.shape == (1, 3, 2, 5, 9)


def test_prepare_super_resolution_segment_pads_last_segment() -> None:
    frames = np.zeros((1, 3, 3, 4, 6), dtype=np.float32)
    frames[:, :, -1] = 1.0
    prepared, actual = prepare_super_resolution_segment(
        frames,
        start=2,
        stop=5,
        segment_frames=3,
        vae_frames=4,
        output_height=8,
        output_width=12,
        padded_height=8,
        padded_width=16,
        interpolation="nearest",
    )
    assert actual == 1
    assert prepared.shape == (1, 3, 4, 8, 16)
    assert np.all(prepared[:, :, :, :, -1] == 1.0)


def test_prepare_frame_condition_keeps_single_image_and_normalizes() -> None:
    image = np.zeros((1, 3, 1, 2, 4), dtype=np.float32)
    image[:, 0] = 1.0

    prepared = prepare_frame_condition(image, 4, 8, 8, 8)

    assert prepared.shape == (1, 3, 1, 8, 8)
    assert prepared.min() == pytest.approx(-1.0)
    assert prepared.max() == pytest.approx(1.0)


def test_resolve_video_path_rejects_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="inside the workspace"):
        resolve_video_path(tmp_path, str(outside))
