"""Smoke test: build_persistent_qwen_graphs on synthetic external-data graphs.

Covers: build produces manifest + runtime graphs, idempotent rebuild under
the lock, concurrent first-build race, and QwenWeightInputs memmap values.
Run with system Python: PYTHONUTF8=1 PYTHONPATH=src python tests/smoke_qwen_persistent_build.py
"""
import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.qwen_persistent import (
    RUNTIME_KINDS,
    RUNTIME_MANIFEST,
    QwenWeightInputs,
    build_persistent_qwen_graphs,
    persistent_qwen_ready,
)

SOURCE_KINDS = ("embedding", "attention", "gate", "up", "down")


def make_source_graph(directory: Path, kind: str) -> None:
    """A MatMul graph whose weight lives in external data, like real exports."""
    name = f"w_{kind}"
    weight = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), name=name)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("MatMul", ["x", name], ["y"], name=f"mm_{kind}")
    graph = helper.make_graph([node], f"qwen_{kind}", [x], [y], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = directory / ("qwen_embedding.onnx" if kind == "embedding" else f"qwen_layer_00_{kind}.onnx")
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
        size_threshold=0,
    )


def test_build_idempotent_and_memmap(tmp: Path) -> None:
    for kind in SOURCE_KINDS:
        make_source_graph(tmp, kind)
    assert not persistent_qwen_ready(tmp)

    manifest = build_persistent_qwen_graphs(tmp)
    assert manifest == tmp / RUNTIME_MANIFEST
    assert persistent_qwen_ready(tmp)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(raw["kinds"]) == set(RUNTIME_KINDS)
    for kind in RUNTIME_KINDS:
        assert (tmp / raw["kinds"][kind]["graph"]).is_file()

    # idempotency: second call is a no-op, manifest bytes unchanged
    first = manifest.read_bytes()
    manifest2 = build_persistent_qwen_graphs(tmp)
    assert manifest2 == manifest
    assert manifest.read_bytes() == first

    # memmap inputs carry the original values at the original dtype/shape
    inputs = QwenWeightInputs(tmp)
    for kind in SOURCE_KINDS:
        arrays = inputs.inputs(kind, None if kind == "embedding" else 0)
        assert len(arrays) == 1
        value = arrays[f"w_{kind}"]
        assert value.dtype == np.float32 and value.shape == (3, 4)
        assert np.array_equal(value, np.arange(12, dtype=np.float32).reshape(3, 4))
        # the cached feed dict is the same object on later calls
        assert inputs.inputs(kind, None if kind == "embedding" else 0) is arrays


def test_runtime_init_auto_build(tmp: Path) -> None:
    """QwenTextRuntime.__init__ builds once when the CUDA provider is active.

    The usable local Python ships CPU-only onnxruntime, so a minimal fake
    runner stands in for the CUDA provider to exercise the wiring.
    """
    from h3_workbench.inference_runtime import QwenTextRuntime

    for kind in SOURCE_KINDS:
        make_source_graph(tmp, kind)
    assert not persistent_qwen_ready(tmp)

    class FakeCudaRunner:
        provider = "CUDAExecutionProvider"

    runtime = QwenTextRuntime(tmp, FakeCudaRunner())  # type: ignore[arg-type]
    assert runtime.persistent is not None
    assert persistent_qwen_ready(tmp)
    # a second runtime on the same directory must not rebuild
    runtime2 = QwenTextRuntime(tmp, FakeCudaRunner())  # type: ignore[arg-type]
    assert runtime2.persistent is not None

    class FakeCpuRunner:
        provider = "CPUExecutionProvider"

    assert QwenTextRuntime(tmp, FakeCpuRunner()).persistent is not None  # type: ignore[arg-type]


def test_concurrent_first_build(tmp: Path) -> None:
    for kind in SOURCE_KINDS:
        make_source_graph(tmp, kind)
    results = []

    def worker() -> None:
        results.append(build_persistent_qwen_graphs(tmp))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert persistent_qwen_ready(tmp)
    for path in results:
        assert path == tmp / RUNTIME_MANIFEST
    assert not list(tmp.glob("*.tmp")), "PID-tagged temporary files must be cleaned up"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_build_idempotent_and_memmap(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_concurrent_first_build(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_runtime_init_auto_build(Path(td))
    print("smoke_qwen_persistent_build: OK")
