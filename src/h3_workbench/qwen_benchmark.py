from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from h3_workbench.inference_runtime import ORTGraphRunner, QwenTextRuntime
from h3_workbench.qwen_persistent import build_persistent_qwen_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the sharded Qwen ONNX encoder")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--l1-prefetch-shards", type=int, choices=range(0, 5), default=2)
    parser.add_argument("--build-persistent", action="store_true")
    args = parser.parse_args()

    if args.build_persistent:
        print(build_persistent_qwen_graphs(args.model.resolve()))
        return
    runner = ORTGraphRunner(prefer_cuda=True)
    runtime = QwenTextRuntime(args.model.resolve(), runner, args.l1_prefetch_shards)
    started = time.perf_counter()
    output = runtime.encode_token_ids(np.arange(1, args.tokens + 1, dtype=np.int64))
    result = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "shape": list(output.shape),
        "finite": bool(np.isfinite(output).all()),
        "provider": runner.provider,
        "cache": runner.cache_stats(),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    runner.close()


if __name__ == "__main__":
    main()
