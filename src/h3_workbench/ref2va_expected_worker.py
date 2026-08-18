from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from h3_workbench.main_transformer import (
    CheckpointReader,
    load_dit_attention_output_shard,
    load_dit_attention_qkv_shard,
    load_dit_mlp_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a small Ref2VA source-reference output")
    parser.add_argument("kind", choices=("attention_qkv", "attention_output", "mlp"))
    parser.add_argument("block", type=int)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    with np.load(args.inputs, allow_pickle=False) as archive:
        feeds = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    reader = CheckpointReader(args.checkpoint)
    if args.kind == "attention_qkv":
        module = load_dit_attention_qkv_shard(reader, args.block).eval()
        ordered = ("hidden_states", "timestep_embedding", "modulation_ids", "rotary_table")
    elif args.kind == "attention_output":
        module = load_dit_attention_output_shard(reader, args.block).eval()
        ordered = ("hidden_states", "attended", "timestep_embedding", "modulation_ids")
    else:
        module = load_dit_mlp_shard(reader, args.block).eval()
        ordered = ("hidden_states", "timestep_embedding", "modulation_ids")
    with torch.inference_mode():
        output = module(*(feeds[name] for name in ordered))
    np.save(args.expected, output.detach().cpu().numpy())


if __name__ == "__main__":
    main()

