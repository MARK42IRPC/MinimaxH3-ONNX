from __future__ import annotations

import gc
import json
import math
import struct
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_workbench.acceleration import LoRAMerger

HIDDEN_SIZE = 5376
HEADS = 56
HEAD_DIM = 128
ATTENTION_INNER = HEADS * HEAD_DIM
FFN_SIZE = 14336
CURVE_DIM = 8
MODALITIES = 3
ROPE_HALF = 48
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
SAFETENSORS_HEADER_LIMIT = 32 * 1024 * 1024


class _StreamingSafeTensorFile:
    """Read individual tensors without mapping the entire checkpoint on Windows."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("file is too short to be a Safetensors checkpoint")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length > SAFETENSORS_HEADER_LIMIT:
                raise ValueError(f"Safetensors header is unexpectedly large: {header_length} bytes")
            payload = handle.read(header_length)
            if len(payload) != header_length:
                raise ValueError("Safetensors header is truncated")
        header = json.loads(payload)
        self._entries: dict[str, dict[str, Any]] = {
            key: value for key, value in header.items() if key != "__metadata__"
        }
        self._data_start = 8 + header_length
        self._file_size = path.stat().st_size

    @staticmethod
    def _dtype(name: str) -> np.dtype[Any]:
        dtypes: dict[str, Any] = {
            "BOOL": np.bool_,
            "U8": np.uint8,
            "I8": np.int8,
            "I16": np.int16,
            "I32": np.int32,
            "I64": np.int64,
            "F16": np.float16,
            "BF16": ml_dtypes.bfloat16,
            "F32": np.float32,
            "F64": np.float64,
            "F8_E4M3": ml_dtypes.float8_e4m3fn,
            "F8_E5M2": ml_dtypes.float8_e5m2,
        }
        try:
            return np.dtype(dtypes[name])
        except KeyError as exc:
            raise ValueError(f"Unsupported Safetensors dtype: {name}") from exc

    def get_tensor(self, key: str) -> np.ndarray:
        try:
            entry = self._entries[key]
        except KeyError as exc:
            raise KeyError(f"Tensor not found in {self.path.name}: {key}") from exc
        shape = tuple(int(value) for value in entry["shape"])
        start, stop = (int(value) for value in entry["data_offsets"])
        dtype = self._dtype(entry["dtype"])
        expected_bytes = math.prod(shape) * dtype.itemsize
        if start < 0 or stop < start or stop - start != expected_bytes:
            raise ValueError(f"Invalid Safetensors bounds for {key}: {start}:{stop}")
        absolute_start = self._data_start + start
        absolute_stop = self._data_start + stop
        if absolute_stop > self._file_size:
            raise ValueError(f"Safetensors tensor is truncated: {key}")
        with self.path.open("rb") as handle:
            handle.seek(absolute_start)
            payload = handle.read(expected_bytes)
        if len(payload) != expected_bytes:
            raise ValueError(f"Safetensors tensor is truncated: {key}")
        return np.frombuffer(payload, dtype=dtype).reshape(shape)

    def has_tensor(self, key: str) -> bool:
        return key in self._entries

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        try:
            return tuple(int(value) for value in self._entries[key]["shape"])
        except KeyError as exc:
            raise KeyError(f"Tensor not found in {self.path.name}: {key}") from exc

    def memmap_tensor(self, key: str) -> np.ndarray:
        try:
            entry = self._entries[key]
        except KeyError as exc:
            raise KeyError(f"Tensor not found in {self.path.name}: {key}") from exc
        start, stop = (int(value) for value in entry["data_offsets"])
        shape = tuple(int(value) for value in entry["shape"])
        dtype = self._dtype(entry["dtype"])
        expected_bytes = math.prod(shape) * dtype.itemsize
        if stop - start != expected_bytes or self._data_start + stop > self._file_size:
            raise ValueError(f"Invalid Safetensors bounds for {key}: {start}:{stop}")
        return np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=self._data_start + start,
            shape=shape,
            order="C",
        )


class CheckpointReader:
    """Load one tensor at a time without mapping the complete checkpoint.

    Torch's Windows CPU backend crashes while reading native FP8 tensors, while
    the NumPy safetensors backend reserves commit for the complete 21 GB file.
    Offset-based reads keep both virtual memory and resident memory bounded.
    """

    def __init__(self, path: Path, lora_path: Path | None = None, lora_strength: float = 1.0):
        np.float8_e4m3fn = ml_dtypes.float8_e4m3fn  # type: ignore[attr-defined]
        np.bfloat16 = ml_dtypes.bfloat16  # type: ignore[attr-defined]
        self._file = _StreamingSafeTensorFile(path)
        self._lora = LoRAMerger(lora_path, lora_strength) if lora_path is not None else None

    def tensor(self, key: str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        array = self._file.get_tensor(key)
        target = np.float32 if dtype == torch.float32 else np.float16
        converted = np.asarray(array, dtype=target).copy()
        return torch.from_numpy(converted).to(dtype).contiguous()

    def raw_tensor(self, key: str) -> np.ndarray:
        """Native-dtype array (fp8 stays fp8) for ONNX fp8 initializer embedding."""
        return self._file.get_tensor(key)

    def dequant_weight(self, prefix: str) -> torch.Tensor:
        weight = self._file.get_tensor(f"{prefix}.weight")
        # Ref2VA ships the same block layout in BF16 rather than FP8 plus a
        # scalar scale. Keep the FP8 path unchanged and let the virtual slicer
        # use the exported FP16/FP32 topology dtype for these full-precision
        # weights.
        if not self._file.has_tensor(f"{prefix}.weight_scale"):
            del weight
            return self.full_precision_weight(prefix)
        scale = float(self._file.get_tensor(f"{prefix}.weight_scale"))
        # Multiplication in FP32 matches the quantized layout's dequantization;
        # stream rows so the FP8, a full FP32 copy, and FP16 output never coexist.
        converted = np.empty(weight.shape, dtype=np.float16)
        rows_per_chunk = 256
        for start in range(0, weight.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, weight.shape[0])
            converted[start:stop] = np.asarray(weight[start:stop], dtype=np.float32) * scale
        if self._lora is not None:
            self._lora.merge(prefix, converted)
        result = torch.from_numpy(converted).contiguous()
        del weight
        gc.collect()
        return result

    def full_precision_weight(self, prefix: str) -> torch.Tensor:
        source = self._file.get_tensor(f"{prefix}.weight")
        converted = np.asarray(source, dtype=np.float16).copy()
        if self._lora is not None:
            self._lora.merge(prefix, converted)
        return torch.from_numpy(converted).contiguous()

    def lora_factors(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._lora is None:
            return None
        factors = self._lora.factors(prefix)
        if factors is None:
            return None
        a, b = factors
        return torch.from_numpy(a.copy()), torch.from_numpy((b * self._lora.strength).copy())


class FrozenLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        compute_fp32: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.compute_fp32 = compute_fp32
        self.gpu_native_fp16 = False
        self.gpu_native_dtype: torch.dtype | None = None
        self.gpu_input_scale = 1.0
        self.gpu_output_scale = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.compute_fp32 and self.gpu_native_dtype is not None:
            values = inputs.float()
            if self.gpu_input_scale != 1.0:
                values = values / self.gpu_input_scale
            result = F.linear(values.to(self.gpu_native_dtype), self.weight, self.bias).float()
            if self.gpu_output_scale != 1.0:
                result = result * self.gpu_output_scale
            return result
        if self.compute_fp32:
            outputs = []
            rows_per_chunk = 256
            values = inputs.float()
            for start in range(0, self.weight.shape[0], rows_per_chunk):
                stop = min(start + rows_per_chunk, self.weight.shape[0])
                bias = self.bias[start:stop].float() if self.bias is not None else None
                outputs.append(F.linear(values, self.weight[start:stop].float(), bias))
            return torch.cat(outputs, dim=-1)
        return F.linear(inputs, self.weight, self.bias)


def enable_gpu_native_fp16(module: nn.Module) -> None:
    _enable_gpu_native(module, torch.float16)


def enable_gpu_native_bf16(module: nn.Module) -> None:
    _enable_gpu_native(module, torch.bfloat16)


def enable_scaled_gpu_native_fp16(
    module: nn.Module,
    fc1_weight_scale: float = 16.0,
    fc2_input_scale: float = 128.0,
    fc2_weight_scale: float = 16.0,
) -> None:
    for name, child in module.named_modules():
        if not isinstance(child, FrozenLinear) or not child.compute_fp32:
            continue
        input_scale = fc2_input_scale if name.endswith("fc2") else 1.0
        weight_scale = fc2_weight_scale if name.endswith("fc2") else fc1_weight_scale
        _configure_gpu_native(child, torch.float16, input_scale, weight_scale)


def _enable_gpu_native(module: nn.Module, dtype: torch.dtype) -> None:
    for child in module.modules():
        if isinstance(child, FrozenLinear) and child.compute_fp32:
            _configure_gpu_native(child, dtype, 1.0, 1.0)


def _configure_gpu_native(
    linear: FrozenLinear,
    dtype: torch.dtype,
    input_scale: float,
    weight_scale: float,
) -> None:
    if input_scale <= 0.0 or weight_scale <= 0.0:
        raise ValueError("Tensor Core scales must be positive")
    output_scale = input_scale * weight_scale
    linear.weight = nn.Parameter((linear.weight.float() / weight_scale).to(dtype), requires_grad=False)
    if linear.bias is not None:
        linear.bias = nn.Parameter((linear.bias.float() / output_scale).to(dtype), requires_grad=False)
    linear.gpu_native_fp16 = dtype == torch.float16
    linear.gpu_native_dtype = dtype
    linear.gpu_input_scale = float(input_scale)
    linear.gpu_output_scale = float(output_scale)


def rms_norm(inputs: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = inputs.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(inputs.dtype)


class MainAttention(nn.Module):
    def __init__(
        self,
        qkv_weight: torch.Tensor,
        out_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
    ):
        super().__init__()
        self.qkv = FrozenLinear(qkv_weight, compute_fp32=True)
        self.out = FrozenLinear(out_weight, compute_fp32=True)
        self.q_weight = nn.Parameter(q_weight, requires_grad=False)
        self.k_weight = nn.Parameter(k_weight, requires_grad=False)

    @staticmethod
    def _apply_rope(values: torch.Tensor, rotary_table: torch.Tensor) -> torch.Tensor:
        first = values[..., :ROPE_HALF]
        second = values[..., ROPE_HALF : ROPE_HALF * 2]
        pairs = torch.stack((first, second), dim=-1).unsqueeze(0)
        rotated = torch.matmul(rotary_table.to(values.dtype), pairs.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return torch.cat((rotated[..., 0], rotated[..., 1], values[..., ROPE_HALF * 2 :]), dim=-1)

    def forward(self, inputs: torch.Tensor, rotary_table: torch.Tensor | None = None) -> torch.Tensor:
        query, key, value = self.project_qkv(inputs, rotary_table)
        query_heads = query.transpose(0, 1).unsqueeze(0)
        key_heads = key.transpose(0, 1).unsqueeze(0)
        value_heads = value.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(query_heads.float(), key_heads.float().transpose(-2, -1)) * (HEAD_DIM**-0.5)
        probabilities = torch.softmax(scores, dim=-1).to(value_heads.dtype)
        attended = torch.matmul(probabilities, value_heads)
        attended = attended.transpose(1, 2).reshape(1, inputs.shape[0], ATTENTION_INNER).squeeze(0)
        return self.out(attended)

    def project_qkv(
        self, inputs: torch.Tensor, rotary_table: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = inputs.shape[0]
        qkv = self.qkv(inputs)
        query, key, value = qkv.split(ATTENTION_INNER, dim=-1)
        query = rms_norm(query.view(sequence, HEADS, HEAD_DIM), self.q_weight, QK_NORM_EPS)
        key = rms_norm(key.view(sequence, HEADS, HEAD_DIM), self.k_weight, QK_NORM_EPS)
        value = value.view(sequence, HEADS, HEAD_DIM)
        if rotary_table is not None:
            query = self._apply_rope(query, rotary_table)
            key = self._apply_rope(key, rotary_table)

        return query, key, value


class MainMLP(nn.Module):
    def __init__(self, fc1_weight: torch.Tensor, fc2_weight: torch.Tensor):
        super().__init__()
        self.fc1 = FrozenLinear(fc1_weight, compute_fp32=True)
        self.fc2 = FrozenLinear(fc2_weight, compute_fp32=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(inputs).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class MainDiTBlock(nn.Module):
    def __init__(
        self,
        attention: MainAttention,
        mlp: MainMLP,
        norm1_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        adaln_weight: torch.Tensor,
        adaln_bias: torch.Tensor,
    ):
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.norm1_weight = nn.Parameter(norm1_weight, requires_grad=False)
        self.norm2_weight = nn.Parameter(norm2_weight, requires_grad=False)
        # Curve-form AdaLN is evaluated in FP32 by ComfyUI's manual-cast linear.
        self.adaln = FrozenLinear(adaln_weight.float(), adaln_bias.float())

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        modulation_ids: torch.Tensor,
        rotary_table: torch.Tensor,
    ) -> torch.Tensor:
        modulation = self.adaln(timestep_embedding.float())
        modulation = modulation.view(-1, MODALITIES, 6, HIDDEN_SIZE).reshape(-1, 6, HIDDEN_SIZE)
        selected = modulation.index_select(0, modulation_ids)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = selected.unbind(dim=1)

        normalized = rms_norm(hidden_states, self.norm1_weight, NORM_EPS)
        attn_input = normalized * (1.0 + scale_msa.to(normalized.dtype)) + shift_msa.to(normalized.dtype)
        hidden_states = hidden_states + self.attention(attn_input, rotary_table) * gate_msa.to(hidden_states.dtype)
        normalized = rms_norm(hidden_states, self.norm2_weight, NORM_EPS)
        mlp_input = normalized * (1.0 + scale_mlp.to(normalized.dtype)) + shift_mlp.to(normalized.dtype)
        return hidden_states + self.mlp(mlp_input) * gate_mlp.to(hidden_states.dtype)


class CurveModulation(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        lora_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.linear = FrozenLinear(weight.float(), bias.float())
        if lora_factors is None:
            self.lora_a = None
            self.lora_b = None
        else:
            self.lora_a = nn.Parameter(lora_factors[0].float(), requires_grad=False)
            self.lora_b = nn.Parameter(lora_factors[1].float(), requires_grad=False)

    @property
    def uses_turbo_adaln(self) -> bool:
        return self.lora_a is not None and self.lora_b is not None

    def forward(
        self,
        timestep_embedding: torch.Tensor,
        modulation_ids: torch.Tensor,
        silu_timestep_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.linear(timestep_embedding.float())
        if self.uses_turbo_adaln:
            if silu_timestep_embedding is None:
                raise ValueError("Turbo AdaLN requires the full-width SiLU timestep embedding")
            values = values + F.linear(F.linear(silu_timestep_embedding.float(), self.lora_a), self.lora_b)
        values = values.view(-1, MODALITIES, 6, HIDDEN_SIZE).reshape(-1, 6, HIDDEN_SIZE)
        return values.index_select(0, modulation_ids)


class DiTAttentionShard(nn.Module):
    def __init__(
        self,
        attention: MainAttention,
        norm_weight: torch.Tensor,
        modulation: CurveModulation,
    ):
        super().__init__()
        self.attention = attention
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.modulation = modulation

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        modulation_ids: torch.Tensor,
        rotary_table: torch.Tensor,
        silu_timestep_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift, scale, gate = self.modulation(
            timestep_embedding, modulation_ids, silu_timestep_embedding
        )[:, :3].unbind(dim=1)
        normalized = rms_norm(hidden_states, self.norm_weight, NORM_EPS)
        inputs = normalized * (1.0 + scale.to(normalized.dtype)) + shift.to(normalized.dtype)
        return hidden_states + self.attention(inputs, rotary_table) * gate.to(hidden_states.dtype)


class DiTAttentionQKVShard(nn.Module):
    """Projection half of a streaming attention block."""

    def __init__(self, attention: MainAttention, norm_weight: torch.Tensor, modulation: CurveModulation):
        super().__init__()
        self.attention = attention
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.modulation = modulation

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        modulation_ids: torch.Tensor,
        rotary_table: torch.Tensor,
        silu_timestep_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift, scale, _ = self.modulation(
            timestep_embedding, modulation_ids, silu_timestep_embedding
        )[:, :3].unbind(dim=1)
        normalized = rms_norm(hidden_states, self.norm_weight, NORM_EPS)
        inputs = normalized * (1.0 + scale.to(normalized.dtype)) + shift.to(normalized.dtype)
        query, key, value = self.attention.project_qkv(inputs, rotary_table)
        return torch.cat((query, key, value), dim=-1)


class DiTAttentionOutputShard(nn.Module):
    """Output projection and residual half of a streaming attention block."""

    def __init__(self, out_weight: torch.Tensor, norm_weight: torch.Tensor, modulation: CurveModulation):
        super().__init__()
        self.out = FrozenLinear(out_weight, compute_fp32=True)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.modulation = modulation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attended: torch.Tensor,
        timestep_embedding: torch.Tensor,
        modulation_ids: torch.Tensor,
        silu_timestep_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, gate = self.modulation(
            timestep_embedding, modulation_ids, silu_timestep_embedding
        )[:, :3].unbind(dim=1)
        return hidden_states + self.out(attended) * gate.to(hidden_states.dtype)


class DiTMLPShard(nn.Module):
    def __init__(self, mlp: MainMLP, norm_weight: torch.Tensor, modulation: CurveModulation):
        super().__init__()
        self.mlp = mlp
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.modulation = modulation

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        modulation_ids: torch.Tensor,
        silu_timestep_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift, scale, gate = self.modulation(
            timestep_embedding, modulation_ids, silu_timestep_embedding
        )[:, 3:].unbind(dim=1)
        normalized = rms_norm(hidden_states, self.norm_weight, NORM_EPS)
        inputs = normalized * (1.0 + scale.to(normalized.dtype)) + shift.to(normalized.dtype)
        return hidden_states + self.mlp(inputs) * gate.to(hidden_states.dtype)


class RefinerBlock(nn.Module):
    def __init__(
        self,
        attention: MainAttention,
        mlp: MainMLP,
        norm1_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
    ):
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.norm1_weight = nn.Parameter(norm1_weight, requires_grad=False)
        self.norm2_weight = nn.Parameter(norm2_weight, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(rms_norm(hidden_states, self.norm1_weight, NORM_EPS))
        return hidden_states + self.mlp(rms_norm(hidden_states, self.norm2_weight, NORM_EPS))


class RefinerAttentionShard(nn.Module):
    def __init__(self, attention: MainAttention, norm_weight: torch.Tensor):
        super().__init__()
        self.attention = attention
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.attention(rms_norm(hidden_states, self.norm_weight, NORM_EPS))


class RefinerMLPShard(nn.Module):
    def __init__(self, mlp: MainMLP, norm_weight: torch.Tensor):
        super().__init__()
        self.mlp = mlp
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(rms_norm(hidden_states, self.norm_weight, NORM_EPS))


class RefinerNorm(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, NORM_EPS)


class MainEmbeddings(nn.Module):
    def __init__(self, reader: CheckpointReader):
        super().__init__()
        self.video = FrozenLinear(
            reader.tensor("video_patch_proj.weight", torch.float32),
            reader.tensor("video_patch_proj.bias", torch.float32),
        )
        self.audio = FrozenLinear(
            reader.tensor("audio_patch_proj.weight", torch.float32),
            reader.tensor("audio_patch_proj.bias", torch.float32),
        )
        self.condition = FrozenLinear(
            reader.tensor("condition_proj.weight"),
            reader.tensor("condition_proj.bias"),
            compute_fp32=True,
        )

    def forward(
        self,
        video_patches: torch.Tensor,
        audio_patches: torch.Tensor,
        text_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = self.video(video_patches.float()).float()
        audio = self.audio(audio_patches.float()).float()
        text = self.condition(text_states.to(torch.float16)).float()
        return video, audio, text


class MainConditioning(nn.Module):
    def __init__(self, reader: CheckpointReader, silu_temb_grid: torch.Tensor | None = None):
        super().__init__()
        self.adaln_t_table = nn.Parameter(reader.tensor("adaln_t_table", torch.float32), requires_grad=False)
        self.inv_freq = nn.Parameter(reader.tensor("rope.inv_freq", torch.float32), requires_grad=False)
        self.silu_temb_grid = (
            nn.Parameter(silu_temb_grid.float(), requires_grad=False) if silu_temb_grid is not None else None
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        position = timesteps.float().clamp(0.0, 1.0) * (self.adaln_t_table.shape[0] - 1)
        lower = position.floor().long().clamp(max=self.adaln_t_table.shape[0] - 2)
        fraction = (position - lower).unsqueeze(1)
        timestep_embedding = torch.lerp(
            self.adaln_t_table.index_select(0, lower),
            self.adaln_t_table.index_select(0, lower + 1),
            fraction,
        )

        per_axis = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        time_freq, height_freq, width_freq = per_axis.unbind(dim=1)
        angles = torch.cat((time_freq, height_freq, width_freq), dim=-1)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        rotary_table = torch.stack((cosine, -sine, sine, cosine), dim=-1)
        rotary_table = rotary_table.reshape(1, position_ids.shape[0], 1, ROPE_HALF, 2, 2).to(torch.float16)
        if self.silu_temb_grid is None:
            return timestep_embedding, rotary_table
        silu_timestep_embedding = torch.lerp(
            self.silu_temb_grid.index_select(0, lower),
            self.silu_temb_grid.index_select(0, lower + 1),
            fraction,
        )
        return timestep_embedding, rotary_table, silu_timestep_embedding


class MainHead(nn.Module):
    def __init__(self, reader: CheckpointReader):
        super().__init__()
        self.norm_weight = nn.Parameter(reader.tensor("final_layer.norm.weight"), requires_grad=False)
        self.adaln = FrozenLinear(
            reader.tensor("final_layer.adaln_proj.linear.weight", torch.float32),
            reader.tensor("final_layer.adaln_proj.linear.bias", torch.float32),
        )
        factors = reader.lora_factors("final_layer.adaln_proj.linear")
        if factors is None:
            self.lora_a = None
            self.lora_b = None
        else:
            self.lora_a = nn.Parameter(factors[0].float(), requires_grad=False)
            self.lora_b = nn.Parameter(factors[1].float(), requires_grad=False)
        self.video_out = FrozenLinear(
            reader.tensor("final_layer.video_out.weight", torch.float32),
            reader.tensor("final_layer.video_out.bias", torch.float32),
        )
        self.audio_out = FrozenLinear(
            reader.tensor("final_layer.audio_out.weight", torch.float32),
            reader.tensor("final_layer.audio_out.bias", torch.float32),
        )

    def _modulate(
        self,
        hidden: torch.Tensor,
        timestep_embedding: torch.Tensor,
        silu_timestep_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = self.adaln(timestep_embedding.float())
        if self.lora_a is not None and self.lora_b is not None:
            if silu_timestep_embedding is None:
                raise ValueError("Turbo head AdaLN requires the full-width SiLU timestep embedding")
            modulation = modulation + F.linear(F.linear(silu_timestep_embedding.float(), self.lora_a), self.lora_b)
        shift, scale = modulation.chunk(2, dim=-1)
        normalized = rms_norm(hidden, self.norm_weight, NORM_EPS).float()
        return normalized * (1.0 + scale) + shift

    def forward(
        self,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        video_timestep_embedding: torch.Tensor,
        audio_timestep_embedding: torch.Tensor,
        video_silu_timestep_embedding: torch.Tensor | None = None,
        audio_silu_timestep_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = self.video_out(
            self._modulate(video_hidden, video_timestep_embedding, video_silu_timestep_embedding)
        )
        audio = self.audio_out(
            self._modulate(audio_hidden, audio_timestep_embedding, audio_silu_timestep_embedding)
        )
        return video, audio


def load_dit_block(reader: CheckpointReader, index: int) -> MainDiTBlock:
    prefix = f"blocks.{index}"
    attention = MainAttention(
        reader.dequant_weight(f"{prefix}.attn.qkv_proj"),
        reader.dequant_weight(f"{prefix}.attn.out_proj"),
        reader.tensor(f"{prefix}.attn.q_norm.weight"),
        reader.tensor(f"{prefix}.attn.k_norm.weight"),
    )
    mlp = MainMLP(
        reader.dequant_weight(f"{prefix}.mlp.fc1"),
        reader.dequant_weight(f"{prefix}.mlp.fc2"),
    )
    return MainDiTBlock(
        attention,
        mlp,
        reader.tensor(f"{prefix}.norm1.weight"),
        reader.tensor(f"{prefix}.norm2.weight"),
        reader.tensor(f"{prefix}.adaln_proj.linear.weight", torch.float32),
        reader.tensor(f"{prefix}.adaln_proj.linear.bias", torch.float32),
    ).eval()


def load_refiner_block(reader: CheckpointReader, index: int) -> RefinerBlock:
    prefix = f"token_refiner.blocks.{index}"
    attention = MainAttention(
        reader.full_precision_weight(f"{prefix}.attn.qkv_proj"),
        reader.full_precision_weight(f"{prefix}.attn.out_proj"),
        reader.tensor(f"{prefix}.attn.q_norm.weight"),
        reader.tensor(f"{prefix}.attn.k_norm.weight"),
    )
    mlp = MainMLP(
        reader.full_precision_weight(f"{prefix}.mlp.fc1"),
        reader.full_precision_weight(f"{prefix}.mlp.fc2"),
    )
    return RefinerBlock(
        attention,
        mlp,
        reader.tensor(f"{prefix}.norm1.weight"),
        reader.tensor(f"{prefix}.norm2.weight"),
    ).eval()


def load_dit_attention_shard(reader: CheckpointReader, index: int) -> DiTAttentionShard:
    prefix = f"blocks.{index}"
    attention = MainAttention(
        reader.dequant_weight(f"{prefix}.attn.qkv_proj"),
        reader.dequant_weight(f"{prefix}.attn.out_proj"),
        reader.tensor(f"{prefix}.attn.q_norm.weight"),
        reader.tensor(f"{prefix}.attn.k_norm.weight"),
    )
    modulation = CurveModulation(
        reader.tensor(f"{prefix}.adaln_proj.linear.weight", torch.float32),
        reader.tensor(f"{prefix}.adaln_proj.linear.bias", torch.float32),
        reader.lora_factors(f"{prefix}.adaln_proj.linear"),
    )
    return DiTAttentionShard(attention, reader.tensor(f"{prefix}.norm1.weight"), modulation).eval()


def load_dit_attention_qkv_shard(reader: CheckpointReader, index: int) -> DiTAttentionQKVShard:
    prefix = f"blocks.{index}"
    attention = MainAttention(
        reader.dequant_weight(f"{prefix}.attn.qkv_proj"),
        reader.dequant_weight(f"{prefix}.attn.out_proj"),
        reader.tensor(f"{prefix}.attn.q_norm.weight"),
        reader.tensor(f"{prefix}.attn.k_norm.weight"),
    )
    modulation = CurveModulation(
        reader.tensor(f"{prefix}.adaln_proj.linear.weight", torch.float32),
        reader.tensor(f"{prefix}.adaln_proj.linear.bias", torch.float32),
        reader.lora_factors(f"{prefix}.adaln_proj.linear"),
    )
    return DiTAttentionQKVShard(attention, reader.tensor(f"{prefix}.norm1.weight"), modulation).eval()


def load_dit_attention_output_shard(reader: CheckpointReader, index: int) -> DiTAttentionOutputShard:
    prefix = f"blocks.{index}"
    modulation = CurveModulation(
        reader.tensor(f"{prefix}.adaln_proj.linear.weight", torch.float32),
        reader.tensor(f"{prefix}.adaln_proj.linear.bias", torch.float32),
        reader.lora_factors(f"{prefix}.adaln_proj.linear"),
    )
    return DiTAttentionOutputShard(
        reader.dequant_weight(f"{prefix}.attn.out_proj"),
        reader.tensor(f"{prefix}.norm1.weight"),
        modulation,
    ).eval()


def load_dit_mlp_shard(reader: CheckpointReader, index: int) -> DiTMLPShard:
    prefix = f"blocks.{index}"
    mlp = MainMLP(
        reader.dequant_weight(f"{prefix}.mlp.fc1"),
        reader.dequant_weight(f"{prefix}.mlp.fc2"),
    )
    modulation = CurveModulation(
        reader.tensor(f"{prefix}.adaln_proj.linear.weight", torch.float32),
        reader.tensor(f"{prefix}.adaln_proj.linear.bias", torch.float32),
        reader.lora_factors(f"{prefix}.adaln_proj.linear"),
    )
    return DiTMLPShard(mlp, reader.tensor(f"{prefix}.norm2.weight"), modulation).eval()


def load_refiner_attention_shard(reader: CheckpointReader, index: int) -> RefinerAttentionShard:
    prefix = f"token_refiner.blocks.{index}"
    attention = MainAttention(
        reader.full_precision_weight(f"{prefix}.attn.qkv_proj"),
        reader.full_precision_weight(f"{prefix}.attn.out_proj"),
        reader.tensor(f"{prefix}.attn.q_norm.weight"),
        reader.tensor(f"{prefix}.attn.k_norm.weight"),
    )
    return RefinerAttentionShard(attention, reader.tensor(f"{prefix}.norm1.weight")).eval()


def load_refiner_mlp_shard(reader: CheckpointReader, index: int) -> RefinerMLPShard:
    prefix = f"token_refiner.blocks.{index}"
    mlp = MainMLP(
        reader.full_precision_weight(f"{prefix}.mlp.fc1"),
        reader.full_precision_weight(f"{prefix}.mlp.fc2"),
    )
    return RefinerMLPShard(mlp, reader.tensor(f"{prefix}.norm2.weight")).eval()
