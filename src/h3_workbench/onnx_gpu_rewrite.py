from __future__ import annotations

import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _attribute(node: onnx.NodeProto, name: str, default: int | float) -> int | float:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _prune_dead_nodes(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    producers = {output: index for index, node in enumerate(nodes) for output in node.output}
    needed_nodes: set[int] = set()
    pending = [output.name for output in model.graph.output]
    while pending:
        value = pending.pop()
        index = producers.get(value)
        if index is None or index in needed_nodes:
            continue
        needed_nodes.add(index)
        pending.extend(nodes[index].input)
    kept = [node for index, node in enumerate(nodes) if index in needed_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used_initializers = {name for node in kept for name in node.input}
    initializers = [item for item in model.graph.initializer if item.name in used_initializers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def rewrite_chunked_fp32_gemms(model: onnx.ModelProto) -> int:
    """Collapse Slice->Cast->Gemm chunks into one Tensor Core-friendly FP16 Gemm."""
    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    replacements: dict[int, list[onnx.NodeProto]] = {}

    for index, concat in enumerate(nodes):
        if concat.op_type != "Concat" or len(concat.input) < 2:
            continue
        gemms = [producers.get(name) for name in concat.input]
        if any(node is None or node.op_type != "Gemm" for node in gemms):
            continue
        typed_gemms = [node for node in gemms if node is not None]
        casts = [producers.get(node.input[1]) for node in typed_gemms]
        if any(node is None or node.op_type != "Cast" for node in casts):
            continue
        typed_casts = [node for node in casts if node is not None]
        slices = [producers.get(node.input[0]) for node in typed_casts]
        if any(node is None or node.op_type != "Slice" for node in slices):
            continue
        typed_slices = [node for node in slices if node is not None]
        activation_names = {node.input[0] for node in typed_gemms}
        weight_names = {node.input[0] for node in typed_slices}
        if len(activation_names) != 1 or len(weight_names) != 1 or any(len(node.input) != 2 for node in typed_gemms):
            continue

        activation = activation_names.pop()
        weight = weight_names.pop()
        output = concat.output[0]
        fp16_input = f"{output}__gpu_fp16_input"
        fp16_output = f"{output}__gpu_fp16_output"
        reference = typed_gemms[0]
        replacements[index] = [
            helper.make_node("Cast", [activation], [fp16_input], name=f"{output}__gpu_input_cast", to=TensorProto.FLOAT16),
            helper.make_node(
                "Gemm",
                [fp16_input, weight],
                [fp16_output],
                name=f"{output}__gpu_gemm",
                transA=int(_attribute(reference, "transA", 0)),
                transB=int(_attribute(reference, "transB", 1)),
                alpha=float(_attribute(reference, "alpha", 1.0)),
                beta=float(_attribute(reference, "beta", 1.0)),
            ),
            helper.make_node("Cast", [fp16_output], [output], name=f"{output}__gpu_output_cast", to=TensorProto.FLOAT),
        ]

    if not replacements:
        return 0
    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        rewritten.extend(replacements.get(index, [node]))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    _prune_dead_nodes(model)
    return len(replacements)


def rewrite_graph_file(path: Path, destination: Path | None = None) -> int:
    model = onnx.load(path, load_external_data=False)
    replacements = rewrite_chunked_fp32_gemms(model)
    if replacements == 0:
        return 0
    destination = path if destination is None else destination
    temporary = destination.with_name(f"{destination.name}.gpu-rewrite.tmp")
    onnx.save_model(model, temporary)
    onnx.checker.check_model(str(temporary))
    os.replace(temporary, destination)
    return replacements
