from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from h3_workbench.inference_runtime import H3MainRuntime, ORTGraphRunner


def _baseline(
    session,
    rows: int,
    chunk: int,
    inputs_for_chunk,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    outputs = []
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        outputs.append(session.run(None, inputs_for_chunk(start, stop))[0])
    return np.concatenate(outputs, axis=0), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FL2VA chunk execution")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    args = parser.parse_args()

    runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=1)
    runtime = object.__new__(H3MainRuntime)
    runtime.runner = runner
    runtime.chunk_io_binding = True
    runtime.fp16_attention_output = False
    runtime.chunk_sizes = {"qkv": 1024, "attention_output": 2048, "mlp": 512}
    random = np.random.default_rng(19)
    rows = args.rows
    timestep = np.zeros((2, 8), dtype=np.float32)
    mod_ids = np.zeros(rows, dtype=np.int64)

    qkv_session = runner.session(args.model / "main_block_00_attention_qkv.onnx")
    hidden = random.standard_normal((rows, 5376), dtype=np.float32) * 0.01
    rotary = np.zeros((1, rows, 1, 48, 2, 2), dtype=np.float16)

    def qkv_inputs(start: int, stop: int) -> dict[str, np.ndarray]:
        return {
            "hidden_states": hidden[start:stop],
            "timestep_embedding": timestep,
            "modulation_ids": mod_ids[start:stop],
            "rotary_table": rotary[:, start:stop],
        }

    qkv_old, qkv_old_seconds = _baseline(qkv_session, rows, 256, qkv_inputs)
    started = time.perf_counter()
    qkv_new = runtime._run_chunked_session(
        qkv_session,
        rows,
        (56, 384),
        1024,
        "qkv",
        qkv_inputs,
        output_dtype=np.float16,
    )
    qkv_new_seconds = time.perf_counter() - started
    del qkv_session

    output_session = runner.session(args.model / "main_block_00_attention_output.onnx")
    attended = random.standard_normal((rows, 7168), dtype=np.float32) * 0.01

    def output_inputs(start: int, stop: int) -> dict[str, np.ndarray]:
        return {
            "hidden_states": hidden[start:stop],
            "attended": attended[start:stop],
            "timestep_embedding": timestep,
            "modulation_ids": mod_ids[start:stop],
        }

    output_old, output_old_seconds = _baseline(output_session, rows, 256, output_inputs)
    started = time.perf_counter()
    output_new = runtime._run_chunked_session(
        output_session,
        rows,
        (5376,),
        2048,
        "attention_output",
        output_inputs,
    )
    output_new_seconds = time.perf_counter() - started
    result = {
        "provider": runner.provider,
        "rows": rows,
        "qkv": {
            "baseline_seconds": round(qkv_old_seconds, 3),
            "optimized_seconds": round(qkv_new_seconds, 3),
            "speedup": round(qkv_old_seconds / qkv_new_seconds, 3),
            "max_abs": float(np.max(np.abs(qkv_old - qkv_new))),
            "output_dtype": str(qkv_new.dtype),
        },
        "attention_output": {
            "baseline_seconds": round(output_old_seconds, 3),
            "optimized_seconds": round(output_new_seconds, 3),
            "speedup": round(output_old_seconds / output_new_seconds, 3),
            "max_abs": float(np.max(np.abs(output_old - output_new))),
        },
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    runner.close()


if __name__ == "__main__":
    main()
