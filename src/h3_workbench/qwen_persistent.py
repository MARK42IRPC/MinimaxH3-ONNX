from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

RUNTIME_MANIFEST = "runtime_persistent_manifest.json"
RUNTIME_KINDS = ("embedding", "attention", "gate", "up", "down")


@dataclass(frozen=True)
class ExternalInput:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExternalInput":
        return cls(
            name=str(value["name"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),  # type: ignore[arg-type]
            offset=int(value["offset"]),
            length=int(value["length"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "length": self.length,
        }


def _source_graph(directory: Path, kind: str) -> Path:
    if kind == "embedding":
        return directory / "qwen_embedding.onnx"
    return directory / f"qwen_layer_00_{kind}.onnx"


def _external_fields(initializer: onnx.TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


def build_persistent_qwen_graphs(directory: Path) -> Path:
    """Promote external initializers to inputs without copying any model weights."""
    directory = directory.resolve()
    manifest: dict[str, object] = {"format": "h3-qwen-persistent-v1", "kinds": {}}
    kinds = manifest["kinds"]
    assert isinstance(kinds, dict)
    for kind in RUNTIME_KINDS:
        source = _source_graph(directory, kind)
        model = onnx.load(str(source), load_external_data=False)
        external: list[ExternalInput] = []
        retained: list[onnx.TensorProto] = []
        for initializer in model.graph.initializer:
            fields = _external_fields(initializer)
            if "location" not in fields:
                retained.append(initializer)
                continue
            dtype = np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type))
            item = ExternalInput(
                initializer.name,
                dtype.str,
                tuple(initializer.dims),
                int(fields.get("offset", "0")),
                int(fields.get("length", str(int(np.prod(initializer.dims)) * dtype.itemsize))),
            )
            external.append(item)
            model.graph.input.append(
                helper.make_tensor_value_info(initializer.name, initializer.data_type, list(initializer.dims))
            )
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
        graph_name = f"runtime_qwen_{kind}.onnx"
        model.graph.name = f"h3_persistent_qwen_{kind}"
        onnx.checker.check_model(model)
        onnx.save(model, directory / graph_name)
        kinds[kind] = {"graph": graph_name, "external_inputs": [item.to_dict() for item in external]}
    path = directory / RUNTIME_MANIFEST
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


class QwenWeightInputs:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        raw = json.loads((self.directory / RUNTIME_MANIFEST).read_text(encoding="utf-8"))
        self.kinds = raw["kinds"]

    def graph(self, kind: str) -> Path:
        return self.directory / str(self.kinds[kind]["graph"])

    def data_path(self, kind: str, layer: int | None = None) -> Path:
        if kind == "embedding":
            return self.directory / "qwen_embedding.onnx.data"
        if layer is None:
            raise ValueError(f"A layer index is required for Qwen {kind} weights")
        return self.directory / f"qwen_layer_{layer:02d}_{kind}.onnx.data"

    def inputs(self, kind: str, layer: int | None = None) -> dict[str, np.ndarray]:
        data_path = self.data_path(kind, layer)
        result: dict[str, np.ndarray] = {}
        for raw in self.kinds[kind]["external_inputs"]:
            spec = ExternalInput.from_dict(raw)
            result[spec.name] = np.memmap(
                data_path,
                dtype=np.dtype(spec.dtype),
                mode="r",
                offset=spec.offset,
                shape=spec.shape,
                order="C",
            )
        return result


def persistent_qwen_ready(directory: Path) -> bool:
    manifest = directory / RUNTIME_MANIFEST
    if not manifest.is_file():
        return False
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        return all((directory / raw["kinds"][kind]["graph"]).is_file() for kind in RUNTIME_KINDS)
    except (OSError, KeyError, TypeError, ValueError):
        return False
