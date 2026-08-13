from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from h3_workbench.inference_runtime import (
    H3MainRuntime,
    ORTGraphRunner,
    QwenTextRuntime,
    initial_latents,
    sample_latents,
)
from h3_workbench.media_output import (
    decode_audio_latents,
    decode_audio_latents_onnx,
    decode_video_latents,
    decode_video_latents_onnx,
    write_mp4,
)
from h3_workbench.memory_planner import main_model_shards, plan_shard_batches, probe_gpu_memory
from h3_workbench.profiles import PROFILE_360P_17F


def _token_ids(value: str) -> np.ndarray:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one token id is required")
    return np.asarray(values, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3 sharded inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    generate_parser.add_argument("--token-ids", type=_token_ids, required=True)
    generate_parser.add_argument("--steps", type=int, choices=range(1, 51), default=4)
    generate_parser.add_argument("--seed", type=int, default=1)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--cpu", action="store_true")
    generate_parser.add_argument("--l1-prefetch-shards", type=int, choices=range(0, 5), default=2)
    args = parser.parse_args()
    profile = PROFILE_360P_17F

    if args.command == "plan":
        snapshot = probe_gpu_memory()
        directory = Path.cwd() / "onnx_models" / "minimax_h3_fl2va_pruned_fp8_scaled_streaming"
        shards = main_model_shards(directory)
        batches = plan_shard_batches(shards, profile, snapshot.free_bytes) if snapshot.free_bytes else []
        print(
            json.dumps(
                {
                    "profile": profile.to_dict(),
                    "memory": snapshot.to_dict(),
                    "batches": [batch.to_dict() for batch in batches],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    workspace = args.workspace.resolve()
    runner = ORTGraphRunner(prefer_cuda=not args.cpu)
    qwen = QwenTextRuntime(
        workspace / "onnx_models" / "qwen3vl_32b_minimax_h3_nvfp4_awq",
        runner,
        args.l1_prefetch_shards,
    )
    text_states = qwen.encode_token_ids(args.token_ids)
    video, audio = initial_latents(profile, args.seed)
    main_runtime = H3MainRuntime(
        workspace / "onnx_models" / "minimax_h3_fl2va_pruned_fp8_scaled_streaming",
        runner,
        profile,
        l1_prefetch_shards=args.l1_prefetch_shards,
    )
    video, audio = sample_latents(main_runtime, video, audio, text_states, args.steps)
    video_onnx = workspace / "onnx_models" / "video_vae"
    if (video_onnx / "manifest.json").is_file():
        pixels = decode_video_latents_onnx(
            video_onnx,
            video,
            profile.output_height,
            profile.output_width,
        )
    else:
        pixels = decode_video_latents(
            workspace / "minimax_h3_video_vae_fp16.safetensors",
            video,
            profile.output_height,
            profile.output_width,
        )
    audio_onnx = workspace / "onnx_models" / "audio_vae"
    if (audio_onnx / "manifest.json").is_file():
        waveform = decode_audio_latents_onnx(audio_onnx, audio, prefer_cuda=not args.cpu)
    else:
        waveform = decode_audio_latents(workspace / "minimax_h3_audio_vae_fp32.safetensors", audio)
    write_mp4(args.output.resolve(), pixels, waveform, profile.fps)


if __name__ == "__main__":
    main()
