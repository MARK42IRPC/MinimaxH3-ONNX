from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from h3_workbench.acceleration import shifted_flow_sigmas
from h3_workbench.inference_runtime import H3MainRuntime, ORTGraphRunner, initial_latents, sample_latents
from h3_workbench.profiles import PROFILE_360P_17F


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event_type: str, durable: bool = False, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
            "event": event_type,
            **details,
        }
        with self._lock:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
            if durable:
                os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()


def save_checkpoint(path: Path, video: np.ndarray, audio: np.ndarray) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(temporary, video=video, audio=audio)
    os.replace(temporary, path)


def _gpu_sample() -> dict[str, float | int] | None:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,power.draw",
                "--format=csv,noheader,nounits",
            ],
            creationflags=flags,
            text=True,
            timeout=3,
        )
        values = [part.strip() for part in output.strip().split(",")]
        return {
            "utilization_percent": int(values[0]),
            "memory_utilization_percent": int(values[1]),
            "memory_used_mib": int(values[2]),
            "memory_free_mib": int(values[3]),
            "power_watts": float(values[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _monitor(logger: JsonlLogger, stop: threading.Event, peaks: dict[str, float]) -> None:
    process = psutil.Process()
    process.cpu_percent(None)
    while not stop.wait(2.0):
        gpu = _gpu_sample()
        rss = process.memory_info().rss
        cpu = process.cpu_percent(None)
        peaks["rss_bytes"] = max(peaks["rss_bytes"], float(rss))
        if gpu is not None:
            peaks["gpu_mib"] = max(peaks["gpu_mib"], float(gpu["memory_used_mib"]))
            peaks["gpu_utilization"] = max(
                peaks["gpu_utilization"], float(gpu["utilization_percent"])
            )
        logger.write(
            "resource_sample",
            process_cpu_percent=cpu,
            process_rss_bytes=rss,
            process_threads=process.num_threads(),
            gpu=gpu,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable schedule-only main-model benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=352)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--temporal-mode", choices=("native", "segmented"), default="segmented")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-latents", type=Path)
    parser.add_argument("--sigma-index", type=int, default=0)
    parser.add_argument("--text-tokens", type=int, default=192)
    parser.add_argument("--attention-query-chunk", type=int, default=512)
    parser.add_argument("--l1-prefetch-shards", type=int, default=2)
    parser.add_argument(
        "--velocity-only",
        action="store_true",
        help="Run one denoise call at the first sigma and persist raw velocities",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    stop = threading.Event()
    peaks = {"rss_bytes": 0.0, "gpu_mib": 0.0, "gpu_utilization": 0.0}
    monitor = threading.Thread(target=_monitor, args=(logger, stop, peaks), name="h3-main-monitor", daemon=True)
    monitor.start()
    runner: ORTGraphRunner | None = None
    runtime: H3MainRuntime | None = None
    last_activity: dict[str, object] = {}
    try:
        base_profile = PROFILE_360P_17F.resized(args.width, args.height)
        profile = (
            base_profile
            if args.temporal_mode == "segmented"
            else base_profile.with_frame_count(args.frames)
        )
        text_states = np.zeros((args.text_tokens, 5120), dtype=np.float16)
        if args.input_latents is None:
            video, audio = initial_latents(profile, args.seed)
        else:
            with np.load(args.input_latents, allow_pickle=False) as archive:
                video = archive["video"].copy()
                audio = archive["audio"].copy()
        sigmas = shifted_flow_sigmas(args.steps)
        if not 0 <= args.sigma_index < args.steps:
            raise ValueError(f"sigma-index must be in [0, {args.steps - 1}]")
        runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)

        def activity(payload: dict[str, object]) -> None:
            nonlocal last_activity
            last_activity = dict(payload)
            runtime_event = payload.get("event")
            details = {key: value for key, value in payload.items() if key != "event"}
            operation_elapsed = details.pop("elapsed_seconds", None)
            if operation_elapsed is not None:
                details["operation_elapsed_seconds"] = operation_elapsed
            logger.write("runtime_activity", runtime_event=runtime_event, **details)
            operation = str(payload.get("operation", ""))
            event = str(payload.get("event", ""))
            if operation.startswith("main_block_") and operation.endswith("_attention_qkv") and not event:
                block = int(operation.split("_")[2]) + 1
                if block == 1 or block % 5 == 0:
                    print(
                        json.dumps(
                            {
                                "step": payload.get("sampling_step"),
                                "steps": payload.get("sampling_steps"),
                                "block": block,
                                "blocks": 50,
                                "elapsed_seconds": round(time.perf_counter() - logger.started, 1),
                            }
                        ),
                        flush=True,
                    )

        runtime = H3MainRuntime(
            args.model,
            runner,
            profile=profile,
            attention_query_chunk=args.attention_query_chunk,
            activity_callback=activity,
            l1_prefetch_shards=args.l1_prefetch_shards,
        )
        logger.write(
            "started",
            durable=True,
            pid=os.getpid(),
            model=str(args.model.resolve()),
            provider=runner.provider,
            ort_cpu_threads=runner.ort_cpu_threads,
            ort_allow_spinning=runner.ort_allow_spinning,
            width=profile.output_width,
            height=profile.output_height,
            requested_frames=args.frames,
            native_frames=profile.frames,
            temporal_mode=args.temporal_mode,
            text_tokens=args.text_tokens,
            audio_tokens=profile.audio_tokens,
            video_tokens=profile.video_tokens,
            sequence_tokens=args.text_tokens + profile.audio_tokens + profile.video_tokens,
            steps=args.steps,
            seed=args.seed,
            input_latents=str(args.input_latents.resolve()) if args.input_latents else None,
            sigma_index=args.sigma_index,
            attention_query_chunk=args.attention_query_chunk,
            l1_prefetch_shards=args.l1_prefetch_shards,
            velocity_only=args.velocity_only,
        )

        def checkpoint(current: int, total: int, current_video: np.ndarray, current_audio: np.ndarray) -> None:
            checkpoint_path = args.output.with_name(f"{args.output.stem}.step-{current}.npz")
            save_checkpoint(checkpoint_path, current_video, current_audio)
            logger.write(
                "step_checkpoint",
                durable=True,
                step=current,
                steps=total,
                path=str(checkpoint_path.resolve()),
                video_finite=bool(np.isfinite(current_video).all()),
                audio_finite=bool(np.isfinite(current_audio).all()),
                video_min=float(np.min(current_video)),
                video_max=float(np.max(current_video)),
                runtime_metrics=runtime.metrics(),
            )

        if args.velocity_only:
            runtime.sampling_step = args.sigma_index + 1
            runtime.sampling_steps = args.steps
            refined_text = runtime.prepare_text(text_states)
            video, audio = runtime.denoise_step(
                video,
                audio,
                refined_text,
                sigmas[args.sigma_index],
                text_is_refined=True,
            )
            if not np.isfinite(video).all():
                invalid = int((~np.isfinite(video)).sum())
                raise FloatingPointError(f"Non-finite FL2VA video velocity: {invalid} invalid values")
            if not np.isfinite(audio).all():
                invalid = int((~np.isfinite(audio)).sum())
                raise FloatingPointError(f"Non-finite FL2VA audio velocity: {invalid} invalid values")
        else:
            video, audio = sample_latents(
                runtime,
                video,
                audio,
                text_states,
                steps=args.steps,
                checkpoint_callback=checkpoint,
            )
        save_checkpoint(args.output, video, audio)
        logger.write(
            "completed",
            durable=True,
            output=str(args.output.resolve()),
            output_kind="velocity" if args.velocity_only else "latent",
            video_shape=list(video.shape),
            audio_shape=list(audio.shape),
            video_finite=bool(np.isfinite(video).all()),
            audio_finite=bool(np.isfinite(audio).all()),
            audio_fallback_reason=runtime.audio_fallback_reason,
            runtime_metrics=runtime.metrics(),
            peak_rss_gib=round(peaks["rss_bytes"] / 2**30, 3),
            peak_gpu_mib=int(peaks["gpu_mib"]),
            peak_gpu_utilization_percent=int(peaks["gpu_utilization"]),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - benchmark must persist every failure
        logger.write(
            "failed",
            durable=True,
            error=str(exc),
            traceback="".join(traceback.format_exception(exc)),
            last_activity=last_activity,
            runtime_metrics=runtime.metrics() if runtime is not None else None,
            peak_rss_gib=round(peaks["rss_bytes"] / 2**30, 3),
            peak_gpu_mib=int(peaks["gpu_mib"]),
            peak_gpu_utilization_percent=int(peaks["gpu_utilization"]),
        )
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        stop.set()
        monitor.join(timeout=5)
        if runtime is not None:
            runtime.close()
        if runner is not None:
            runner.close()
        logger.close()


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
