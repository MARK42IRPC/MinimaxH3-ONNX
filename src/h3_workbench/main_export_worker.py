from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from h3_workbench.exporter import _export_graph
from h3_workbench.main_transformer import (
    CheckpointReader,
    load_dit_attention_shard,
    load_dit_attention_output_shard,
    load_dit_attention_qkv_shard,
    load_dit_mlp_shard,
    load_refiner_attention_shard,
    load_refiner_mlp_shard,
    enable_gpu_native_bf16,
    enable_gpu_native_fp16,
    enable_scaled_gpu_native_fp16,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=("refiner_attention", "refiner_mlp", "dit_attention", "dit_attention_qkv", "dit_attention_output", "dit_mlp"),
    )
    parser.add_argument("index", type=int)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--gpu-native-fp16", action="store_true")
    parser.add_argument("--gpu-native-bf16", action="store_true")
    parser.add_argument("--gpu-scaled-fp16", action="store_true")
    args = parser.parse_args()

    with np.load(args.inputs, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}
    reader = CheckpointReader(args.checkpoint, args.lora, args.lora_strength)

    if args.kind == "refiner_attention":
        module = load_refiner_attention_shard(reader, args.index)
        input_names = ["hidden_states"]
        dynamic_axes = {"hidden_states": {0: "sequence"}, "hidden_states_out": {0: "sequence"}}
    elif args.kind == "refiner_mlp":
        module = load_refiner_mlp_shard(reader, args.index)
        input_names = ["hidden_states"]
        dynamic_axes = {"hidden_states": {0: "sequence"}, "hidden_states_out": {0: "sequence"}}
    elif args.kind == "dit_attention":
        module = load_dit_attention_shard(reader, args.index)
        input_names = ["hidden_states", "timestep_embedding", "modulation_ids", "rotary_table"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "timestep_embedding": {0: "timestep_count"},
            "modulation_ids": {0: "sequence"},
            "rotary_table": {1: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }
    elif args.kind == "dit_attention_qkv":
        module = load_dit_attention_qkv_shard(reader, args.index)
        input_names = ["hidden_states", "timestep_embedding", "modulation_ids", "rotary_table"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "timestep_embedding": {0: "timestep_count"},
            "modulation_ids": {0: "sequence"},
            "rotary_table": {1: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }
    elif args.kind == "dit_attention_output":
        module = load_dit_attention_output_shard(reader, args.index)
        input_names = ["hidden_states", "attended", "timestep_embedding", "modulation_ids"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "attended": {0: "sequence"},
            "timestep_embedding": {0: "timestep_count"},
            "modulation_ids": {0: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }
    else:
        module = load_dit_mlp_shard(reader, args.index)
        input_names = ["hidden_states", "timestep_embedding", "modulation_ids"]
        dynamic_axes = {
            "hidden_states": {0: "sequence"},
            "timestep_embedding": {0: "timestep_count"},
            "modulation_ids": {0: "sequence"},
            "hidden_states_out": {0: "sequence"},
        }

    if args.lora is not None and args.kind.startswith("dit_"):
        input_names.append("silu_timestep_embedding")
        dynamic_axes["silu_timestep_embedding"] = {0: "timestep_count"}
    inputs = tuple(tensors[name] for name in input_names)
    with torch.inference_mode():
        expected = module(*inputs)
    np.savez(args.expected, output_0=expected.detach().cpu().numpy())
    selected_modes = sum((args.gpu_native_fp16, args.gpu_native_bf16, args.gpu_scaled_fp16))
    if selected_modes > 1:
        parser.error("Select only one GPU-native precision mode")
    if args.gpu_native_bf16:
        enable_gpu_native_bf16(module)
    elif args.gpu_scaled_fp16:
        enable_scaled_gpu_native_fp16(module)
    elif args.gpu_native_fp16:
        enable_gpu_native_fp16(module)
    _export_graph(
        module,
        inputs,
        args.model,
        input_names,
        ["hidden_states_out"],
        dynamic_axes,
        True,
    )


if __name__ == "__main__":
    main()
