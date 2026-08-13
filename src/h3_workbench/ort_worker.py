from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("outputs", type=Path)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)
    np.savez(args.outputs, **{f"output_{index}": value for index, value in enumerate(outputs)})


if __name__ == "__main__":
    main()
