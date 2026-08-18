from pathlib import Path

import numpy as np
import torch

from h3_workbench.qwen_transformer import qwen_mrope_position_ids
from h3_workbench.qwen_vision import Qwen3VLVisionEncoder, qwen_vision_config


def test_qwen_vision_model_uses_memory_efficient_sdpa(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    checkpoint = tmp_path / "qwen-vision.safetensors"
    checkpoint.write_bytes(b"test checkpoint")
    selected: dict[str, str] = {}

    class FakeVisionModel(torch.nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            selected["attention"] = config._attn_implementation

    monkeypatch.setattr(modeling_qwen3_vl, "Qwen3VLVisionModel", FakeVisionModel)
    encoder = Qwen3VLVisionEncoder(
        checkpoint,
        prefer_cuda=False,
        config=qwen_vision_config(),
        loader=lambda model, path, device, dtype: None,
    )

    encoder._load_model()

    assert selected == {"attention": "sdpa"}
    encoder.close()


def test_qwen_mrope_keeps_video_temporal_blocks_separate() -> None:
    token_ids = np.asarray(
        [10, 101, 101, 101, 101, 11, 102, 102, 12, 102, 102, 13],
        dtype=np.int64,
    )
    token_types = np.asarray(
        [0, 1, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0],
        dtype=np.int8,
    )

    positions = qwen_mrope_position_ids(
        token_ids,
        image_grid_thw=np.asarray([[1, 4, 4]], dtype=np.int64),
        video_grid_thw=np.asarray([[2, 2, 4]], dtype=np.int64),
        mm_token_type_ids=token_types,
    )

    np.testing.assert_array_equal(
        positions[:, 1:5],
        np.asarray([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]),
    )
    np.testing.assert_array_equal(positions[:, 6:8], np.asarray([[4, 4], [4, 4], [4, 5]]))
    np.testing.assert_array_equal(positions[:, 9:11], np.asarray([[7, 7], [7, 7], [7, 8]]))
    np.testing.assert_array_equal(positions[:, [0, 5, 8, 11]], np.asarray([[0, 3, 6, 9]] * 3))
