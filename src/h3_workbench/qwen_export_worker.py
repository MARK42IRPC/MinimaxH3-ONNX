from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from h3_workbench.exporter import _export_graph
from h3_workbench.main_transformer import enable_gpu_native_fp16
from h3_workbench.qwen_transformer import (
    QwenAttentionShard,
    QwenCheckpointReader,
    QwenDownShard,
    QwenEmbedding,
    QwenGateShard,
    QwenMLPShard,
    QwenUpShard,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("embedding", "attention", "gate", "up", "down", "mlp"))
    parser.add_argument("index", type=int)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--gpu-native-fp16", action="store_true")
    args = parser.parse_args()

    with np.load(args.inputs, allow_pickle=False) as archive:
        tensors = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    reader = QwenCheckpointReader(args.checkpoint)

    if args.kind == "embedding":
        module = QwenEmbedding(reader).eval()
        input_names = ["token_ids"]
        output_names = ["hidden_states"]
        dynamic_axes = {"token_ids": {0: "sequence"}, "hidden_states": {0: "sequence"}}
    elif args.kind == "attention":
        module = QwenAttentionShard(reader, args.index).eval()
        input_names = ["hidden_states", "cosine", "sine", "attention_mask"]
        output_names = ["hidden_states_out"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "cosine": {0: "sequence"},
            "sine": {0: "sequence"},
            "attention_mask": {2: "sequence", 3: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }
    elif args.kind == "gate":
        module = QwenGateShard(reader, args.index).eval()
        input_names = ["hidden_states"]
        output_names = ["normalized_states", "gate"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "normalized_states": {0: "sequence"},
            "gate": {0: "sequence"},
        }
    elif args.kind == "up":
        module = QwenUpShard(reader, args.index).eval()
        input_names = ["normalized_states"]
        output_names = ["up"]
        dynamic_axes = {"normalized_states": {0: "sequence"}, "up": {0: "sequence"}}
    elif args.kind == "down":
        module = QwenDownShard(reader, args.index).eval()
        input_names = ["hidden_states", "gate", "up"]
        output_names = ["hidden_states_out"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "gate": {0: "sequence"},
            "up": {0: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }
    else:
        module = QwenMLPShard(reader, args.index).eval()
        input_names = ["hidden_states"]
        output_names = ["hidden_states_out"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }

    inputs = tuple(tensors[name] for name in input_names)
    with torch.inference_mode():
        expected = module(*inputs)
    expected_tuple = expected if isinstance(expected, tuple) else (expected,)
    np.savez(
        args.expected,
        **{f"output_{index}": value.detach().cpu().numpy() for index, value in enumerate(expected_tuple)},
    )
    if args.gpu_native_fp16:
        enable_gpu_native_fp16(module)
    _export_graph(module, inputs, args.model, input_names, output_names, dynamic_axes, True)


if __name__ == "__main__":
    main()
