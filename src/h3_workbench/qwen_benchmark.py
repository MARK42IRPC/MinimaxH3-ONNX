from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from h3_workbench.inference_runtime import ORTGraphRunner, QwenTextRuntime
from h3_workbench.main_benchmark import JsonlLogger
from h3_workbench.qwen_persistent import build_persistent_qwen_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the sharded Qwen ONNX encoder")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--l1-prefetch-shards", type=int, choices=range(0, 5), default=2)
    parser.add_argument("--build-persistent", action="store_true")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.build_persistent:
        print(build_persistent_qwen_graphs(args.model.resolve()))
        return
    logger = JsonlLogger(args.log) if args.log is not None else None
    runner = ORTGraphRunner(prefer_cuda=True)
    try:
        runtime = QwenTextRuntime(args.model.resolve(), runner, args.l1_prefetch_shards)
        if logger is not None:
            logger.write(
                "started",
                durable=True,
                model=str(args.model.resolve()),
                tokens=args.tokens,
                provider=runner.provider,
            )
        started = time.perf_counter()

        def progress(operation: str, current: int, total: int) -> None:
            if logger is not None:
                logger.write("layer", operation=operation, current=current, total=total)

        output = runtime.encode_token_ids(
            np.arange(1, args.tokens + 1, dtype=np.int64),
            callback=progress,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output, output)
        result = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "shape": list(output.shape),
            "finite": bool(np.isfinite(output).all()),
            "provider": runner.provider,
            "output": str(args.output.resolve()) if args.output is not None else None,
            "cache": runner.cache_stats(),
        }
        if logger is not None:
            logger.write("completed", durable=True, **result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        if logger is not None:
            logger.write(
                "failed",
                durable=True,
                error=str(exc),
                traceback="".join(traceback.format_exception(exc)),
            )
        raise
    finally:
        runner.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
