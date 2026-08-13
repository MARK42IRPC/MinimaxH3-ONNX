from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from h3_workbench.main_transformer import CheckpointReader


def test_checkpoint_reader_streams_fp8_and_bfloat16(tmp_path: Path) -> None:
    path = tmp_path / "main.safetensors"
    source = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    save_file(
        {
            "test.weight": source.to(torch.float8_e4m3fn),
            "test.weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        },
        path,
    )

    reader = CheckpointReader(path)

    np.testing.assert_array_equal(
        reader.dequant_weight("test").numpy(),
        (source * 0.25).to(torch.float16).numpy(),
    )
    np.testing.assert_array_equal(
        reader.tensor("norm.weight").numpy(),
        np.asarray([1.0, 2.0], dtype=np.float16),
    )
