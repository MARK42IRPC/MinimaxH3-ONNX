from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NoInitMixin:
    def reset_parameters(self) -> None:
        pass


class Linear(_NoInitMixin, nn.Linear):
    pass


class Conv1d(_NoInitMixin, nn.Conv1d):
    pass


class Conv3d(_NoInitMixin, nn.Conv3d):
    def forward(self, input: torch.Tensor, autopad: str | None = None) -> torch.Tensor:
        weight = self.weight
        if autopad == "causal_zero":
            temporal_frames = getattr(self, "export_causal_frames", input.shape[2])
            weight = weight[:, :, -temporal_frames:, :, :]
        return self._conv_forward(input, weight, self.bias)


class ConvTranspose1d(_NoInitMixin, nn.ConvTranspose1d):
    pass


class GroupNorm(_NoInitMixin, nn.GroupNorm):
    pass


class LayerNorm(_NoInitMixin, nn.LayerNorm):
    pass


class RMSNorm(_NoInitMixin, nn.RMSNorm):
    pass


class disable_weight_init:
    Linear = Linear
    Conv1d = Conv1d
    Conv3d = Conv3d
    ConvTranspose1d = ConvTranspose1d
    GroupNorm = GroupNorm
    LayerNorm = LayerNorm
    RMSNorm = RMSNorm


def cast_to_input(weight: torch.Tensor, input: torch.Tensor, non_blocking: bool = False, copy: bool = False) -> torch.Tensor:
    if weight.device == input.device and weight.dtype == input.dtype and not copy:
        return weight
    return weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking, copy=copy)


def cast_bias_weight(module: nn.Module, input: torch.Tensor, **_: object):
    weight = cast_to_input(module.weight, input)
    bias = None if module.bias is None else cast_to_input(module.bias, input)
    return weight, bias, None


def uncast_bias_weight(*_: object) -> None:
    return None


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: object, **kwargs: object):
    return F.scaled_dot_product_attention(q, k, v, *args, **kwargs)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = (x.float() * torch.rsqrt(variance + eps)).to(dtype=x.dtype)
    if weight is not None:
        normalized = normalized * cast_to_input(weight, x)
    return normalized


def apply_rope_split_half(query: torch.Tensor, key: torch.Tensor, table: torch.Tensor):
    pair_count = table.shape[-3]
    # Split-half RoPE pairs the first half of the rotated channels with the
    # second half: (x[0], x[pairs]), (x[1], x[pairs + 1]), ... .
    q = torch.stack((query[..., :pair_count], query[..., pair_count : pair_count * 2]), dim=-1)
    k = torch.stack((key[..., :pair_count], key[..., pair_count : pair_count * 2]), dim=-1)
    q = torch.matmul(table, q.unsqueeze(-1)).squeeze(-1)
    k = torch.matmul(table, k.unsqueeze(-1)).squeeze(-1)
    q = torch.cat((q[..., 0], q[..., 1]), dim=-1)
    k = torch.cat((k[..., 0], k[..., 1]), dim=-1)
    return q, k


def optimized_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, heads: int, skip_reshape: bool = True):
    del heads, skip_reshape
    output = F.scaled_dot_product_attention(query, key, value)
    return output.transpose(1, 2).flatten(2)


def intermediate_device() -> torch.device:
    return torch.device("cpu")
