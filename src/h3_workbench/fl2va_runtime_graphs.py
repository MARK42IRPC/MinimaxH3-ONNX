from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto

RUNTIME_MANIFEST = "runtime_attention_output_fp16_manifest.json"
RUNTIME_FORMAT = "h3-fl2va-attention-output-fp16-scaled-v3"
ATTENTION_INPUT_SCALE = 0.25
ATTENTION_OUTPUT_SCALE = 1.0 / ATTENTION_INPUT_SCALE


def _attention_output_graphs(directory: Path) -> list[Path]:
    return sorted(directory.glob("main_block_*_attention_output.onnx"))


def _runtime_graph_name(source_name: str) -> str:
    return f"{Path(source_name).stem}.fp16.onnx"


def is_attention_output_graph(path: Path) -> bool:
    return path.name.endswith("_attention_output.onnx") or path.name.endswith("_attention_output.fp16.onnx")


def source_graph_name(path: Path) -> str:
    if path.name.endswith("_attention_output.fp16.onnx"):
        return path.name.replace("_attention_output.fp16.onnx", "_attention_output.onnx")
    return path.name


def build_fp16_attention_output_graphs(directory: Path) -> Path:
    """Create numerically safe attention output graphs around the FP16 GEMM.

    The original projection executes its GEMM in FP16. Scaling the attended
    FP32 input before its existing cast and restoring the scale after the GEMM
    prevents marginal FP16 overflow without changing the projected value.
    """
    directory = directory.resolve()
    sources = _attention_output_graphs(directory)
    if len(sources) != 50:
        raise ValueError(f"Expected 50 attention output graphs, found {len(sources)}")
    graph_names: list[str] = []
    for source in sources:
        model = onnx.load(str(source), load_external_data=False)
        attended = next((item for item in model.graph.input if item.name == "attended"), None)
        if attended is None:
            raise ValueError(f"Missing attended input in {source.name}")
        scale = onnx.helper.make_tensor(
            "runtime_attended_scale", TensorProto.FLOAT, [], [ATTENTION_INPUT_SCALE]
        )
        inv_scale = onnx.helper.make_tensor(
            "runtime_output_scale", TensorProto.FLOAT, [], [ATTENTION_OUTPUT_SCALE]
        )
        nodes = list(model.graph.node)
        input_cast = next(
            (node for node in nodes if node.op_type == "Cast" and node.input and node.input[0] == "attended"),
            None,
        )
        if input_cast is None:
            raise ValueError(f"Missing attended cast in {source.name}")
        gemm = next(
            (
                node
                for node in nodes
                if node.op_type == "Gemm" and input_cast.output and input_cast.output[0] in node.input
            ),
            None,
        )
        output_cast = (
            next(
                (
                    node
                    for node in nodes
                    if node.op_type == "Cast" and gemm is not None and gemm.output and gemm.output[0] in node.input
                ),
                None,
            )
            if gemm is not None
            else None
        )
        if gemm is not None and output_cast is not None:
            input_cast.input[0] = "attended_scaled"
            scale_input = onnx.helper.make_node(
                "Mul", ["attended", "runtime_attended_scale"], ["attended_scaled"]
            )
            nodes.insert(0, scale_input)
            consumer = next(
                (node for node in nodes if output_cast.output and output_cast.output[0] in node.input),
                None,
            )
            if consumer is None:
                raise ValueError(f"Missing FP32 projection consumer in {source.name}")
            scale_output = onnx.helper.make_node(
                "Mul", [output_cast.output[0], "runtime_output_scale"], ["linear_1_scaled"]
            )
            consumer.input[list(consumer.input).index(output_cast.output[0])] = "linear_1_scaled"
            cast_index = next(index for index, node in enumerate(nodes) if node is output_cast)
            nodes.insert(cast_index + 1, scale_output)
            model.graph.initializer.extend([scale, inv_scale])
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        model.graph.name = f"{model.graph.name}_fp16_attended"
        graph_name = _runtime_graph_name(source.name)
        temporary = directory / f"{graph_name}.tmp"
        onnx.save_model(model, temporary)
        onnx.checker.check_model(str(temporary))
        os.replace(temporary, directory / graph_name)
        graph_names.append(graph_name)
    manifest = {
        "format": RUNTIME_FORMAT,
        "graphs": graph_names,
        "external_weights": "original",
    }
    manifest_path = directory / RUNTIME_MANIFEST
    temporary_manifest = directory / f"{RUNTIME_MANIFEST}.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest_path


def fp16_attention_output_ready(directory: Path) -> bool:
    manifest_path = directory / RUNTIME_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    graphs = manifest.get("graphs")
    return (
        manifest.get("format") == RUNTIME_FORMAT
        and isinstance(graphs, list)
        and len(graphs) == 50
        and all(isinstance(name, str) and (directory / name).is_file() for name in graphs)
    )


def runtime_attention_output_graph(directory: Path, source_name: str) -> Path:
    if fp16_attention_output_ready(directory) and is_attention_output_graph(Path(source_name)):
        return directory / _runtime_graph_name(source_name)
    return directory / source_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FL2VA FP16 attended-input runtime graph headers")
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(build_fp16_attention_output_graphs(args.model))


if __name__ == "__main__":
    main()
