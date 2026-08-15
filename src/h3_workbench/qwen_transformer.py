from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_workbench.main_transformer import FrozenLinear, _StreamingSafeTensorFile, rms_norm

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 25600
HEAD_DIM = 128
QUERY_HEADS = 64
KV_HEADS = 8
KV_REPEAT = QUERY_HEADS // KV_HEADS
NORM_EPS = 1e-6
CONVROT_FORMAT = "int8_tensorwise"
E2M1_LUT = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _from_blocked(blocked: np.ndarray, rows: int, columns: int) -> np.ndarray:
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded_rows = row_blocks * 128
    padded_columns = column_blocks * 4
    step1 = blocked.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
    step3 = step2.reshape(row_blocks, column_blocks, 4, 32, 4)
    step4 = step3.reshape(row_blocks, column_blocks, 128, 4)
    step5 = step4.transpose(0, 2, 1, 3)
    return step5.reshape(padded_rows, padded_columns)[:rows, :columns]


def dequantize_nvfp4_array(
    packed: np.ndarray,
    per_tensor_scale: float,
    blocked_scales: np.ndarray,
    rows_per_chunk: int = 128,
) -> np.ndarray:
    rows, packed_columns = packed.shape
    logical_columns = packed_columns * 2
    block_columns = logical_columns // 16
    scales = _from_blocked(np.asarray(blocked_scales, dtype=np.float32), rows, block_columns)
    output = np.empty((rows, logical_columns), dtype=np.float16)
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        source = packed[start:stop]
        codes = np.empty((stop - start, logical_columns), dtype=np.uint8)
        codes[:, 0::2] = source >> 4
        codes[:, 1::2] = source & 0x0F
        values = E2M1_LUT[codes]
        total_scale = np.repeat(scales[start:stop], 16, axis=1) * per_tensor_scale
        output[start:stop] = values * total_scale
    return output


class QwenCheckpointReader:
    def __init__(self, path: Path):
        self._file = _StreamingSafeTensorFile(path)
        probe = "model.layers.0.self_attn.q_proj"
        if self._file.has_tensor(f"{probe}.weight_scale_2"):
            self.source_quantization = "nvfp4_awq"
        elif self._file.has_tensor(f"{probe}.weight_scale"):
            config = self._quantization_config(probe)
            self.source_quantization = (
                "int8_tensorwise_convrot" if config is not None and config.get("convrot") is True
                else "int8_per_channel"
            )
        else:
            raise ValueError(f"Unsupported Qwen checkpoint quantization: {path.name}")

    def tensor(self, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        array = self._file.get_tensor(key)
        if dtype == torch.int8:
            return torch.from_numpy(np.asarray(array, dtype=np.int8).copy()).contiguous()
        target = np.float16 if dtype == torch.float16 else np.float32
        return torch.from_numpy(np.asarray(array, dtype=target).copy()).to(dtype).contiguous()

    def optional_tensor(self, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        if not self._file.has_tensor(key):
            return None
        return self.tensor(key, dtype)

    def raw_tensor(self, key: str) -> np.ndarray:
        """Return one native-dtype tensor for graph repacking without a full map."""
        return self._file.get_tensor(key)

    def _quantization_config(self, prefix: str) -> dict[str, object] | None:
        key = f"{prefix}.comfy_quant"
        if not self._file.has_tensor(key):
            return None
        raw = np.asarray(self._file.get_tensor(key), dtype=np.uint8).tobytes().rstrip(b"\0")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Comfy quantization metadata for {prefix}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Invalid Comfy quantization metadata for {prefix}")
        return value

    def convrot_group_size(self, prefix: str) -> int | None:
        config = self._quantization_config(prefix)
        if self.source_quantization != "int8_tensorwise_convrot":
            return None
        if config is None or config.get("format") != CONVROT_FORMAT or config.get("convrot") is not True:
            raise ValueError(f"Missing INT8 ConvRot metadata for {prefix}")
        group_size = int(config.get("convrot_groupsize", 0))
        if group_size < 4 or group_size & (group_size - 1) or math.log(group_size, 4) % 1:
            raise ValueError(f"Invalid ConvRot group size for {prefix}: {group_size}")
        return group_size

    def dequant_weight(self, prefix: str) -> torch.Tensor:
        if self.source_quantization in {"int8_per_channel", "int8_tensorwise_convrot"}:
            weight = self._file.get_tensor(f"{prefix}.weight")
            scales = self._file.get_tensor(f"{prefix}.weight_scale")
            if weight.dtype != np.int8 or scales.shape not in {(weight.shape[0],), (weight.shape[0], 1)}:
                raise ValueError(f"Invalid per-channel INT8 tensors for {prefix}")
            row_scales = np.asarray(scales, dtype=np.float32).reshape(weight.shape[0], 1)
            converted = np.empty(weight.shape, dtype=np.float16)
            for start in range(0, weight.shape[0], 256):
                stop = min(start + 256, weight.shape[0])
                converted[start:stop] = np.asarray(weight[start:stop], dtype=np.float32) * row_scales[start:stop]
            result = torch.from_numpy(converted).contiguous()
            del weight, scales, row_scales, converted
            gc.collect()
            return result

        packed = self._file.get_tensor(f"{prefix}.weight")
        blocked_scales = self._file.get_tensor(f"{prefix}.weight_scale")
        scale = float(self._file.get_tensor(f"{prefix}.weight_scale_2"))
        converted = dequantize_nvfp4_array(packed, scale, blocked_scales)
        result = torch.from_numpy(converted).contiguous()
        del packed, blocked_scales, converted
        gc.collect()
        return result


class SmoothedLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        pre_quant_scale: torch.Tensor | None,
        convrot_group_size: int | None = None,
    ):
        super().__init__()
        self.linear = FrozenLinear(weight, compute_fp32=True)
        if pre_quant_scale is None:
            self.register_parameter("pre_quant_scale", None)
        else:
            self.pre_quant_scale = nn.Parameter(pre_quant_scale.float(), requires_grad=False)
        if convrot_group_size is None:
            self.register_buffer("convrot_hadamard", None, persistent=False)
            self.convrot_group_size = 0
        else:
            self.register_buffer(
                "convrot_hadamard",
                regular_hadamard(convrot_group_size),
                persistent=False,
            )
            self.convrot_group_size = convrot_group_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_quant_scale is not None:
            inputs = inputs.float() * self.pre_quant_scale
        if self.convrot_hadamard is not None:
            original_shape = inputs.shape
            groups = original_shape[-1] // self.convrot_group_size
            grouped = inputs.reshape(-1, groups, self.convrot_group_size)
            inputs = torch.matmul(grouped, self.convrot_hadamard.to(inputs)).reshape(original_shape)
        return self.linear(inputs)


def regular_hadamard(size: int) -> torch.Tensor:
    """Return the normalized regular Hadamard matrix used by Comfy ConvRot."""
    if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    result = h4
    current = 4
    while current < size:
        result = torch.kron(result, h4)
        current *= 4
    return result / math.sqrt(size)


class QwenEmbedding(nn.Module):
    def __init__(self, reader: QwenCheckpointReader):
        super().__init__()
        scale = reader.optional_tensor("model.embed_tokens.weight_scale")
        weight_dtype = torch.int8 if scale is not None else torch.float16
        self.weight = nn.Parameter(reader.tensor("model.embed_tokens.weight", weight_dtype), requires_grad=False)
        if scale is None:
            self.register_parameter("scale", None)
        else:
            self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        rows = F.embedding(token_ids.long(), self.weight).float()
        if self.scale is None:
            return rows
        scales = F.embedding(token_ids.long(), self.scale).float()
        return rows * scales


def _apply_rope(values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return values * cosine[:, None, :] + rotated * sine[:, None, :]


class QwenAttentionShard(nn.Module):
    def __init__(self, reader: QwenCheckpointReader, index: int):
        super().__init__()
        prefix = f"model.layers.{index}"
        self.input_norm = nn.Parameter(reader.tensor(f"{prefix}.input_layernorm.weight"), requires_grad=False)
        self.q_norm = nn.Parameter(reader.tensor(f"{prefix}.self_attn.q_norm.weight"), requires_grad=False)
        self.k_norm = nn.Parameter(reader.tensor(f"{prefix}.self_attn.k_norm.weight"), requires_grad=False)
        self.query = _load_linear(reader, f"{prefix}.self_attn.q_proj")
        self.key = _load_linear(reader, f"{prefix}.self_attn.k_proj")
        self.value = _load_linear(reader, f"{prefix}.self_attn.v_proj")
        self.output = _load_linear(reader, f"{prefix}.self_attn.o_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence = hidden_states.shape[0]
        normalized = rms_norm(hidden_states.float(), self.input_norm, NORM_EPS)
        query = self.query(normalized).view(sequence, QUERY_HEADS, HEAD_DIM)
        key = self.key(normalized).view(sequence, KV_HEADS, HEAD_DIM)
        value = self.value(normalized).view(sequence, KV_HEADS, HEAD_DIM)
        query = _apply_rope(rms_norm(query, self.q_norm, NORM_EPS), cosine, sine)
        key = _apply_rope(rms_norm(key, self.k_norm, NORM_EPS), cosine, sine)
        key = key.repeat_interleave(KV_REPEAT, dim=1)
        value = value.repeat_interleave(KV_REPEAT, dim=1)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (HEAD_DIM**-0.5)
        probabilities = torch.softmax(scores + attention_mask.float(), dim=-1)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(sequence, QUERY_HEADS * HEAD_DIM)
        return hidden_states.float() + self.output(attended)


class QwenGateShard(nn.Module):
    def __init__(self, reader: QwenCheckpointReader, index: int):
        super().__init__()
        prefix = f"model.layers.{index}"
        self.norm = nn.Parameter(reader.tensor(f"{prefix}.post_attention_layernorm.weight"), requires_grad=False)
        self.gate = _load_linear(reader, f"{prefix}.mlp.gate_proj")

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = rms_norm(hidden_states.float(), self.norm, NORM_EPS)
        return normalized, self.gate(normalized)


class QwenUpShard(nn.Module):
    def __init__(self, reader: QwenCheckpointReader, index: int):
        super().__init__()
        self.up = _load_linear(reader, f"model.layers.{index}.mlp.up_proj")

    def forward(self, normalized_states: torch.Tensor) -> torch.Tensor:
        return self.up(normalized_states)


class QwenDownShard(nn.Module):
    def __init__(self, reader: QwenCheckpointReader, index: int):
        super().__init__()
        self.down = _load_linear(reader, f"model.layers.{index}.mlp.down_proj")

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return hidden_states.float() + self.down(F.silu(gate.float()) * up.float())


class QwenMLPShard(nn.Module):
    """One-layer MLP boundary used to measure launch and transfer reduction."""

    def __init__(self, reader: QwenCheckpointReader, index: int):
        super().__init__()
        prefix = f"model.layers.{index}"
        self.norm = nn.Parameter(reader.tensor(f"{prefix}.post_attention_layernorm.weight"), requires_grad=False)
        self.gate = _load_linear(reader, f"{prefix}.mlp.gate_proj")
        self.up = _load_linear(reader, f"{prefix}.mlp.up_proj")
        self.down = _load_linear(reader, f"{prefix}.mlp.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = rms_norm(hidden_states.float(), self.norm, NORM_EPS)
        gate = self.gate(normalized)
        up = self.up(normalized)
        return hidden_states.float() + self.down(F.silu(gate.float()) * up.float())


def _load_linear(reader: QwenCheckpointReader, prefix: str) -> SmoothedLinear:
    return SmoothedLinear(
        reader.dequant_weight(prefix),
        reader.optional_tensor(f"{prefix}.pre_quant_scale"),
        reader.convrot_group_size(prefix),
    )
