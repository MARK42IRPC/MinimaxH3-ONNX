from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from h3_workbench.qwen_transformer import QwenAttentionShard, QwenCheckpointReader, QwenMLPShard


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a small Qwen source-reference output")
    parser.add_argument("kind", choices=("attention", "mlp"))
    parser.add_argument("block", type=int)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    with np.load(args.inputs, allow_pickle=False) as archive:
        feeds = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    reader = QwenCheckpointReader(args.checkpoint)
    if args.kind == "attention":
        module = QwenAttentionShard(reader, args.block).eval()
        ordered = ("hidden_states", "cosine", "sine", "attention_mask")
    else:
        module = QwenMLPShard(reader, args.block).eval()
        ordered = ("hidden_states",)
    with torch.inference_mode():
        output = module(*(feeds[name] for name in ordered))
    np.save(args.expected, output.detach().cpu().numpy())


if __name__ == "__main__":
    main()
