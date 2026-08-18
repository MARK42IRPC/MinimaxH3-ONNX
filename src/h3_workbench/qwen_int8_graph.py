from __future__ import annotations

import os
import re
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from h3_workbench.qwen_transformer import QwenCheckpointReader


MLP_WEIGHT_TO_SOURCE = {
    "gate.linear.weight": "mlp.gate_proj",
    "up.linear.weight": "mlp.up_proj",
    "down.linear.weight": "mlp.down_proj",
}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _invalidate_product_validation(directory: Path, reason: str) -> None:
    """Revoke a product validation before publishing a changed runtime manifest."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_passed"] = False
    manifest.pop("validated_product_fingerprint", None)
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    validation.update({"status": "invalidated", "reason": reason})
    manifest["validation"] = validation
    _atomic_json(manifest_path, manifest)


def _tensor_producers(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }


def _copy_value_info(value: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    result = onnx.ValueInfoProto()
    result.CopyFrom(value)
    return result


def _extract_qwen_subgraph(
    model: onnx.ModelProto,
    outputs: list[str],
    replacements: dict[str, onnx.ValueInfoProto] | None = None,
) -> onnx.ModelProto:
    """Extract a dependency-closed graph while treating replacements as inputs."""
    replacements = replacements or {}
    producers = _tensor_producers(model)
    graph_inputs = {value.name: value for value in model.graph.input}
    initializers = {value.name: value for value in model.graph.initializer}
    needed_nodes: set[int] = set()
    needed_tensors = list(outputs)
    while needed_tensors:
        tensor = needed_tensors.pop()
        if tensor in replacements or tensor in graph_inputs or tensor in initializers:
            continue
        node = producers.get(tensor)
        if node is None:
            raise ValueError(f"Qwen topology has no producer for required tensor {tensor!r}")
        node_index = next(index for index, candidate in enumerate(model.graph.node) if candidate is node)
        if node_index in needed_nodes:
            continue
        needed_nodes.add(node_index)
        needed_tensors.extend(name for name in node.input if name)

    nodes = []
    for index, original in enumerate(model.graph.node):
        if index not in needed_nodes:
            continue
        node = onnx.NodeProto()
        node.CopyFrom(original)
        for input_index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[input_index] = replacements[input_name].name
        nodes.append(node)

    used_names = {name for node in nodes for name in node.input if name}
    inputs = []
    for value in model.graph.input:
        if value.name in used_names:
            inputs.append(_copy_value_info(value))
    for value in replacements.values():
        if value.name not in {item.name for item in inputs}:
            inputs.append(_copy_value_info(value))
    output_infos: list[onnx.ValueInfoProto] = []
    known_values = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    for name in outputs:
        if name in replacements:
            output_infos.append(_copy_value_info(replacements[name]))
        elif name in known_values:
            output_infos.append(_copy_value_info(known_values[name]))
        else:
            raise ValueError(f"Qwen topology has no value info for output {name!r}")

    used_initializers = {name for node in nodes for name in node.input if name in initializers}
    graph = helper.make_graph(
        nodes,
        "qwen_attention_split",
        inputs,
        output_infos,
        initializer=[initializers[name] for name in initializers if name in used_initializers],
    )
    result = onnx.ModelProto()
    result.CopyFrom(model)
    result.graph.CopyFrom(graph)
    result.graph.name = "qwen_attention_split"
    onnx.checker.check_model(result, full_check=False)
    return result


def build_split_qwen_attention_graphs(
    fused_topology: Path,
    qkv_output: Path,
    attention_output: Path,
) -> tuple[Path, Path]:
    """Split the validated fused Qwen attention topology at the SDPA boundary."""
    fused_topology = fused_topology.resolve()
    model = onnx.shape_inference.infer_shapes(onnx.load(str(fused_topology), load_external_data=False))
    available = {value.name for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    required = ("add_154", "add_222", "view_8", "view_9")
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"Unsupported Qwen attention topology; missing split tensors: {missing}")

    qkv_model = _extract_qwen_subgraph(model, ["add_154", "add_222", "view_8"])
    # The fused graph applies the output projection's ConvRot in groups of
    # 256 after reshaping the 64x128 attention result to [S, 32, 256].
    attended_info = helper.make_tensor_value_info(
        "attended", TensorProto.FLOAT, ["sequence", 32, 256]
    )
    output_model = _extract_qwen_subgraph(model, ["hidden_states_out"], {"view_9": attended_info})
    qkv_output = qkv_output.resolve()
    attention_output = attention_output.resolve()
    qkv_output.parent.mkdir(parents=True, exist_ok=True)
    attention_output.parent.mkdir(parents=True, exist_ok=True)
    qkv_temporary = qkv_output.with_name(f"{qkv_output.name}.{os.getpid()}.tmp")
    output_temporary = attention_output.with_name(f"{attention_output.name}.{os.getpid()}.tmp")
    try:
        onnx.save_model(qkv_model, str(qkv_temporary), save_as_external_data=False)
        onnx.save_model(output_model, str(output_temporary), save_as_external_data=False)
        os.replace(qkv_temporary, qkv_output)
        os.replace(output_temporary, attention_output)
    finally:
        qkv_temporary.unlink(missing_ok=True)
        output_temporary.unlink(missing_ok=True)
    return qkv_output, attention_output


def install_split_qwen_attention(directory: Path) -> dict[str, object]:
    """Publish split attention graphs and their source-weight specifications."""
    from h3_workbench.qwen_persistent import (
        RUNTIME_INT8_MANIFEST,
        int8_virtual_product_fingerprint,
        runtime_file_identity,
    )

    directory = directory.resolve()
    manifest_path = directory / RUNTIME_INT8_MANIFEST
    _invalidate_product_validation(directory, "qwen_split_attention_graphs_changed")
    runtime = json.loads(manifest_path.read_text(encoding="utf-8"))
    fused = directory / str(runtime["kinds"]["attention"]["graph"])
    qkv_name = "runtime_qwen_attention_qkv_int8.onnx"
    output_name = "runtime_qwen_attention_output_int8.onnx"
    build_split_qwen_attention_graphs(fused, directory / qkv_name, directory / output_name)
    base_specs = list(runtime["kinds"]["attention"]["inputs"])
    qkv_inputs = {value.name for value in onnx.load(str(directory / qkv_name), load_external_data=False).graph.input}
    output_inputs = {value.name for value in onnx.load(str(directory / output_name), load_external_data=False).graph.input}

    def specs(names: set[str]) -> list[dict[str, object]]:
        return [item for item in base_specs if str(item["name"]) in names]

    runtime["attention_split"] = {
        "format": "qkv_sdpa_output_v1",
        "qkv": {
            "graph": qkv_name,
            "graph_identity": runtime_file_identity(directory / qkv_name),
            "inputs": specs(qkv_inputs),
            "outputs": ["add_154", "add_222", "view_8"],
        },
        "output": {
            "graph": output_name,
            "graph_identity": runtime_file_identity(directory / output_name),
            "inputs": specs(output_inputs),
            "outputs": ["hidden_states_out"],
        },
    }
    _atomic_json(manifest_path, runtime)

    # The split graphs are part of the runtime product identity.  A product
    # validated before they were installed must remain unavailable until the
    # split path itself has been validated.
    product_manifest_path = directory / "manifest.json"
    if product_manifest_path.is_file():
        product_manifest = json.loads(product_manifest_path.read_text(encoding="utf-8"))
        product_manifest["validation_passed"] = False
        product_manifest.pop("validated_product_fingerprint", None)
        product_manifest["validation"] = {
            "status": "invalidated",
            "reason": "qwen_split_attention_graphs_changed",
            "product_fingerprint": int8_virtual_product_fingerprint(directory),
        }
        _atomic_json(product_manifest_path, product_manifest)
    return runtime["attention_split"]


def _external_fields(tensor: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _inline_external_tensor(tensor: TensorProto, model_path: Path) -> None:
    fields = _external_fields(tensor)
    location = fields.get("location")
    if location is None:
        return
    offset = int(fields.get("offset", "0"))
    length = int(fields.get("length", "0"))
    data_path = model_path.with_name(location)
    with data_path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"Truncated ONNX external initializer: {data_path}")
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.DEFAULT
    tensor.raw_data = raw


def _set_external_int8(tensor: TensorProto, location: str, offset: int, length: int) -> None:
    tensor.ClearField("raw_data")
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.extend(
        (
            onnx.StringStringEntryProto(key="location", value=location),
            onnx.StringStringEntryProto(key="offset", value=str(offset)),
            onnx.StringStringEntryProto(key="length", value=str(length)),
        )
    )


def _infer_block(path: Path) -> int:
    match = re.search(r"layer_(\d+)_", path.stem)
    if match is None:
        raise ValueError(f"Cannot infer Qwen layer from graph name: {path.name}")
    return int(match.group(1))


def build_int8_qdq_graph(
    fp16_model: Path,
    checkpoint: Path,
    output_model: Path,
    block: int | None = None,
    weight_to_source: dict[str, str] | None = None,
) -> Path:
    """Repack a fused FP16 MLP graph with INT8 external weights and QDQ nodes.

    The source checkpoint is read one matrix at a time. Small graph constants are
    inlined so the resulting candidate has no dependency on the original FP16
    external-data file.
    """
    fp16_model = fp16_model.resolve()
    checkpoint = checkpoint.resolve()
    output_model = output_model.resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    block = _infer_block(fp16_model) if block is None else block
    weight_to_source = MLP_WEIGHT_TO_SOURCE if weight_to_source is None else weight_to_source
    reader = QwenCheckpointReader(checkpoint)
    if reader.source_quantization not in {"int8_per_channel", "int8_tensorwise_convrot"}:
        raise ValueError("The QDQ candidate requires an INT8 per-channel source checkpoint")

    model = onnx.load(str(fp16_model), load_external_data=False)
    graph_initializers = {item.name: item for item in model.graph.initializer}
    replacements: dict[str, tuple[TensorProto, TensorProto, TensorProto]] = {}
    data_path = output_model.with_name(f"{output_model.name}.data")
    temporary_model = output_model.with_name(f"{output_model.name}.{os.getpid()}.tmp")
    temporary_data = data_path.with_name(f"{data_path.name}.{os.getpid()}.tmp")
    for initializer_name, source_suffix in weight_to_source.items():
        source_prefix = f"model.layers.{block}.{source_suffix}"
        weights = reader.raw_tensor(f"{source_prefix}.weight")
        scales = reader.raw_tensor(f"{source_prefix}.weight_scale")
        if weights.dtype != np.int8:
            raise ValueError(f"Expected INT8 weights for {source_prefix}, got {weights.dtype}")
        if scales.shape not in {(weights.shape[0],), (weights.shape[0], 1)}:
            raise ValueError(f"Invalid scale shape for {source_prefix}: {scales.shape}")
        scale_tensor = numpy_helper.from_array(
            np.asarray(scales, dtype=np.float32).reshape(-1), name=f"{initializer_name}.scale"
        )
        zero_tensor = numpy_helper.from_array(
            np.zeros(weights.shape[0], dtype=np.int8), name=f"{initializer_name}.zero"
        )
        int8_tensor = TensorProto(
            name=f"{initializer_name}.int8",
            data_type=TensorProto.INT8,
            dims=list(weights.shape),
        )
        replacements[initializer_name] = (int8_tensor, scale_tensor, zero_tensor)

    # Rebuild the initializer list, inlining only the small constants retained
    # from the FP16 graph and writing INT8 matrices to one compact data file.
    model.graph.ClearField("initializer")
    offset = 0
    with temporary_data.open("wb") as data_handle:
        for old_name, old_tensor in graph_initializers.items():
            if old_name in replacements:
                int8_tensor, scale_tensor, zero_tensor = replacements[old_name]
                source_suffix = weight_to_source[old_name]
                source_prefix = f"model.layers.{block}.{source_suffix}"
                raw = reader.raw_tensor(f"{source_prefix}.weight")
                data_handle.write(raw.tobytes(order="C"))
                _set_external_int8(int8_tensor, data_path.name, offset, raw.nbytes)
                offset += raw.nbytes
                model.graph.initializer.append(int8_tensor)
                model.graph.initializer.extend((scale_tensor, zero_tensor))
            else:
                _inline_external_tensor(old_tensor, fp16_model)
                model.graph.initializer.append(old_tensor)

    for old_name, (int8_tensor, scale_tensor, zero_tensor) in replacements.items():
        dequant_fp32_name = f"{old_name}.dequant_fp32"
        dequant_name = f"{old_name}.dequant"
        model.graph.node.insert(
            0,
            helper.make_node(
                "Cast",
                [dequant_fp32_name],
                [dequant_name],
                name=f"{old_name}.dequant_to_fp16",
                to=TensorProto.FLOAT16,
            ),
        )
        model.graph.node.insert(
            0,
            helper.make_node(
                "DequantizeLinear",
                [int8_tensor.name, scale_tensor.name, zero_tensor.name],
                [dequant_fp32_name],
                name=f"{old_name}.dequantize",
                axis=0,
            ),
        )
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if node.op_type == "Gemm" and input_name == old_name:
                    node.input[index] = dequant_name

    onnx.save_model(model, str(temporary_model), save_as_external_data=False)
    os.replace(temporary_data, data_path)
    os.replace(temporary_model, output_model)
    onnx.checker.check_model(str(output_model), full_check=False)
    return output_model


def build_weight_input_topology(
    source_model: Path,
    output_model: Path,
    input_names: set[str],
) -> Path:
    """Promote selected initializers to inputs for a reusable ORT topology."""
    source_model = source_model.resolve()
    output_model = output_model.resolve()
    model = onnx.load(str(source_model), load_external_data=False)
    retained: list[TensorProto] = []
    promoted: set[str] = set()
    for initializer in model.graph.initializer:
        if initializer.name not in input_names:
            retained.append(initializer)
            continue
        model.graph.input.append(
            helper.make_tensor_value_info(initializer.name, initializer.data_type, list(initializer.dims))
        )
        promoted.add(initializer.name)
    missing = input_names - promoted
    if missing:
        raise ValueError(f"Cannot promote missing Qwen initializers: {sorted(missing)}")
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained)
    if any(item.data_location == TensorProto.EXTERNAL for item in retained):
        raise ValueError("Reusable Qwen topology still depends on external initializers")
    model.graph.name = f"{model.graph.name}_weight_inputs"
    onnx.checker.check_model(model, full_check=False)
    temporary = output_model.with_name(f"{output_model.name}.{os.getpid()}.tmp")
    onnx.save_model(model, str(temporary), save_as_external_data=False)
    os.replace(temporary, output_model)
    return output_model


def initializer_inputs(source_model: Path, input_names: set[str]) -> dict[str, np.ndarray]:
    """Map promoted initializer data without mapping the complete ONNX model."""
    source_model = source_model.resolve()
    model = onnx.load(str(source_model), load_external_data=False)
    result: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        if initializer.name not in input_names:
            continue
        fields = _external_fields(initializer)
        if "location" in fields:
            dtype = np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type))
            result[initializer.name] = np.memmap(
                source_model.with_name(fields["location"]),
                dtype=dtype,
                mode="r",
                offset=int(fields.get("offset", "0")),
                shape=tuple(initializer.dims),
                order="C",
            )
        else:
            result[initializer.name] = np.ascontiguousarray(numpy_helper.to_array(initializer))
    missing = input_names - result.keys()
    if missing:
        raise ValueError(f"Cannot load missing Qwen initializers: {sorted(missing)}")
    return result
