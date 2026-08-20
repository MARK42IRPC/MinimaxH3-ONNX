from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class AccelerationConfig:
    name: str
    path: str
    strength: float
    default_steps: int
    minimum_steps: int
    maximum_steps: int
    partial_conversion: bool
    warning: str | None
    sha256: str
    tensor_pairs: int
    backbone_pairs: int
    adaln_pairs: int

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        return asdict(self)


def shifted_flow_sigmas(
    steps: int,
    shift: float = 12.0,
    start_sigma: float = 1.0,
) -> list[float]:
    """Build the MiniMax-H3 sigma grid for ``steps`` model evaluations.

    The official scheduler receives the number of grid points and exposes one
    fewer model timestep because the terminal zero is not evaluated. The
    workbench API exposes the number of model evaluations, so it constructs one
    extra point internally. ``start_sigma`` is expressed in shifted sigma
    space; for a partial-noise request the inverse shift is used to preserve
    the requested starting sigma exactly.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be a positive finite number")
    if not np.isfinite(start_sigma) or not 0.0 <= start_sigma <= 1.0:
        raise ValueError("start_sigma must be between 0.0 and 1.0")
    base_start = start_sigma / (shift - (shift - 1.0) * start_sigma)
    base = np.linspace(base_start, 0.0, steps + 1, dtype=np.float32)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    shifted = np.asarray(shifted, dtype=np.float32)
    shifted[0] = np.float32(start_sigma)
    shifted[-1] = np.float32(0.0)
    # Match MiniMaxH3Scheduler's unique_consecutive behavior if float32
    # rounding collapses adjacent points at an unusually large step count.
    keep = np.concatenate(([True], shifted[1:] != shifted[:-1]))
    values = shifted[keep].tolist()
    # Preserve an explicitly supplied Python start value at the public API
    # boundary; the official torch scheduler rounds it when it materializes
    # the device tensor, but callers use this value to label the first pass.
    values[0] = float(start_sigma)
    values[-1] = 0.0
    return values


def minimax_h3_denoised(
    sample: np.ndarray,
    velocity: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Convert MiniMax-H3's data-ward velocity into a denoised sample."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Denoising sigma must be positive and finite, got {sigma}")
    sample_values = np.asarray(sample)
    velocity_values = np.asarray(velocity)
    if sample_values.shape != velocity_values.shape:
        raise ValueError(
            f"Sample and velocity shapes must match, got {sample_values.shape} and {velocity_values.shape}"
        )
    result = sample_values.astype(np.float32, copy=False) + (
        np.float32(sigma) * velocity_values.astype(np.float32, copy=False)
    )
    return np.asarray(result, dtype=sample_values.dtype)


def minimax_h3_euler_step(
    sample: np.ndarray,
    velocity: np.ndarray,
    sigma: float,
    sigma_next: float,
) -> np.ndarray:
    """Apply MiniMax-H3's data-ward rectified-flow Euler step.

    MiniMax-H3's transformer predicts a data-ward velocity. Its official
    scheduler therefore uses ``x_next = x + (sigma - sigma_next) * velocity``
    (equivalently, it blends ``x`` with ``x0 = x + sigma * velocity``).
    Compute the update in float32, as the reference scheduler does for half
    precision samples.
    """
    if not np.isfinite(sigma) or not np.isfinite(sigma_next):
        raise ValueError("Scheduler sigmas must be finite")
    if sigma <= 0.0 or sigma_next < 0.0 or sigma_next > sigma:
        raise ValueError(
            f"MiniMax-H3 Euler step requires 0 <= sigma_next <= sigma with sigma > 0, "
            f"got sigma={sigma}, sigma_next={sigma_next}"
        )
    sample_values = np.asarray(sample)
    velocity_values = np.asarray(velocity)
    if sample_values.shape != velocity_values.shape:
        raise ValueError(
            f"Sample and velocity shapes must match, got {sample_values.shape} and {velocity_values.shape}"
        )
    result = sample_values.astype(np.float32, copy=False) + (
        np.float32(sigma - sigma_next) * velocity_values.astype(np.float32, copy=False)
    )
    return np.asarray(result, dtype=sample_values.dtype)


def minimax_h3_res_multistep_step(
    sample: np.ndarray,
    velocity: np.ndarray,
    sigma: float,
    sigma_next: float,
    previous_sigma: float | None = None,
    previous_sigma_down: float | None = None,
    previous_denoised: np.ndarray | None = None,
    current_denoised: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ComfyUI's ``res_multistep`` update to MiniMax-H3 output.

    The H3 head returns a data-ward velocity, so the solver's denoised value
    is ``sample + sigma * velocity``. The first step uses Euler because there
    is no previous denoised value, and the terminal step returns that denoised
    value directly instead of taking logarithms of zero.
    """
    if not np.isfinite(sigma) or not np.isfinite(sigma_next):
        raise ValueError("Scheduler sigmas must be finite")
    if sigma <= 0.0 or sigma_next < 0.0 or sigma_next > sigma:
        raise ValueError(
            f"MiniMax-H3 res_multistep requires 0 <= sigma_next <= sigma with sigma > 0, "
            f"got sigma={sigma}, sigma_next={sigma_next}"
        )
    sample_values = np.asarray(sample)
    velocity_values = np.asarray(velocity)
    if sample_values.shape != velocity_values.shape:
        raise ValueError(
            f"Sample and velocity shapes must match, got {sample_values.shape} and {velocity_values.shape}"
        )
    if current_denoised is None:
        current_denoised_values = minimax_h3_denoised(sample_values, velocity_values, sigma).astype(
            np.float32, copy=False
        )
    else:
        current_denoised_values = np.asarray(current_denoised)
        if current_denoised_values.shape != sample_values.shape:
            raise ValueError(
                "Current denoised shape must match sample, "
                f"got {current_denoised_values.shape} and {sample_values.shape}"
            )
        current_denoised_values = current_denoised_values.astype(np.float32, copy=False)

    if sigma_next == 0.0:
        return np.asarray(current_denoised_values, dtype=sample_values.dtype)

    history = (previous_sigma, previous_sigma_down, previous_denoised)
    if all(value is None for value in history):
        return minimax_h3_euler_step(sample_values, velocity_values, sigma, sigma_next)
    if any(value is None for value in history):
        raise ValueError(
            "res_multistep history must provide previous_sigma, previous_sigma_down, and previous_denoised together"
        )
    assert previous_sigma is not None
    assert previous_sigma_down is not None
    assert previous_denoised is not None
    if (
        not np.isfinite(previous_sigma)
        or not np.isfinite(previous_sigma_down)
        or previous_sigma <= 0.0
        or previous_sigma_down <= 0.0
        or previous_sigma_down > previous_sigma
    ):
        raise ValueError(
            "res_multistep previous sigmas must be finite, positive, and satisfy "
            f"previous_sigma_down <= previous_sigma, got {previous_sigma}, {previous_sigma_down}"
        )
    previous_values = np.asarray(previous_denoised)
    if previous_values.shape != sample_values.shape:
        raise ValueError(
            f"Previous denoised shape must match sample, got {previous_values.shape} and {sample_values.shape}"
        )

    # This is the two-step exponential multistep formula used by ComfyUI.
    # With a degenerate history, fall back to the stable H3 Euler update.
    t = np.float32(-np.log(np.float32(sigma)))
    t_old = np.float32(-np.log(np.float32(previous_sigma_down)))
    t_next = np.float32(-np.log(np.float32(sigma_next)))
    t_prev = np.float32(-np.log(np.float32(previous_sigma)))
    h = np.float32(t_next - t)
    c2 = np.float32((t_prev - t_old) / h) if h != 0.0 else np.float32(np.nan)
    if not np.isfinite(h) or h <= 0.0 or not np.isfinite(c2) or abs(float(c2)) < 1e-7:
        return minimax_h3_euler_step(sample_values, velocity_values, sigma, sigma_next)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi1 = np.float32(np.expm1(-h) / (-h))
        phi2 = np.float32((phi1 - np.float32(1.0)) / (-h))
        phi2_over_c2 = np.float32(phi2 / c2)
        b1 = np.float32(phi1 - phi2_over_c2)
        b2 = phi2_over_c2
    b1 = np.float32(np.nan_to_num(b1, nan=float(phi1), posinf=float(phi1), neginf=float(phi1)))
    b2 = np.float32(np.nan_to_num(b2, nan=0.0, posinf=0.0, neginf=0.0))
    if not np.isfinite((b1, b2)).all():
        return minimax_h3_euler_step(sample_values, velocity_values, sigma, sigma_next)
    result = np.exp(-h, dtype=np.float32) * sample_values.astype(np.float32, copy=False) + h * (
        b1 * current_denoised_values + b2 * previous_values.astype(np.float32, copy=False)
    )
    return np.asarray(result, dtype=sample_values.dtype)


def load_silu_temb_grid(lora_path: Path) -> np.ndarray:
    candidates = (
        lora_path.with_name("h3_silu_temb_grid.safetensors"),
        Path.cwd() / "h3_silu_temb_grid.safetensors",
    )
    for path in candidates:
        if not path.is_file():
            continue
        with safe_open(str(path), framework="numpy") as checkpoint:
            grid = np.asarray(checkpoint.get_tensor("silu_t_emb_grid"), dtype=np.float32)
        if grid.shape != (1025, 2688):
            raise ValueError(f"Unexpected Turbo timestep grid shape in {path}: {grid.shape}")
        return grid.copy()
    raise FileNotFoundError(
        "Turbo LoRA on a pruned base requires h3_silu_temb_grid.safetensors next to the LoRA file"
    )


def inspect_acceleration_lora(path: Path, strength: float = 1.0) -> AccelerationConfig:
    with safe_open(str(path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = set(checkpoint.keys())
    prefixes = {
        key.removesuffix(".lora_A.weight")
        for key in keys
        if key.endswith(".lora_A.weight") and f'{key.removesuffix(".lora_A.weight")}.lora_B.weight' in keys
    }
    adaln_pairs = sum("adaln_proj" in prefix for prefix in prefixes)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return AccelerationConfig(
        name=path.name,
        path=str(path.resolve()),
        strength=strength,
        default_steps=4,
        minimum_steps=4,
        maximum_steps=8,
        partial_conversion=metadata.get("partial_conversion", "false").lower() == "true",
        warning=metadata.get("warning"),
        sha256=digest.hexdigest(),
        tensor_pairs=len(prefixes),
        backbone_pairs=len(prefixes) - adaln_pairs,
        adaln_pairs=adaln_pairs,
    )


class LoRAMerger:
    def __init__(self, path: Path, strength: float = 1.0):
        np.bfloat16 = ml_dtypes.bfloat16  # type: ignore[attr-defined]
        self._file = safe_open(str(path), framework="numpy")
        self._keys = set(self._file.keys())
        self.strength = float(strength)
        self.merged: set[str] = set()

    def _prefix(self, base_prefix: str) -> str | None:
        for prefix in (base_prefix, f"diffusion_model.{base_prefix}"):
            if f"{prefix}.lora_A.weight" in self._keys and f"{prefix}.lora_B.weight" in self._keys:
                return prefix
        return None

    def has(self, base_prefix: str) -> bool:
        return self._prefix(base_prefix) is not None

    def factors(self, base_prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
        prefix = self._prefix(base_prefix)
        if prefix is None:
            return None
        a = np.asarray(self._file.get_tensor(f"{prefix}.lora_A.weight"), dtype=np.float32)
        b = np.asarray(self._file.get_tensor(f"{prefix}.lora_B.weight"), dtype=np.float32)
        return a, b

    def merge(self, base_prefix: str, weight: np.ndarray, rows_per_chunk: int = 128) -> None:
        prefix = self._prefix(base_prefix)
        if prefix is None:
            return
        a = np.asarray(self._file.get_tensor(f"{prefix}.lora_A.weight"), dtype=np.float32)
        b = self._file.get_tensor(f"{prefix}.lora_B.weight")
        if a.shape[1] != weight.shape[1] or b.shape[0] != weight.shape[0] or b.shape[1] != a.shape[0]:
            raise ValueError(
                f"LoRA shape mismatch for {base_prefix}: A={a.shape}, B={b.shape}, weight={weight.shape}"
            )
        for start in range(0, weight.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, weight.shape[0])
            delta = np.asarray(b[start:stop], dtype=np.float32) @ a
            merged = np.asarray(weight[start:stop], dtype=np.float32) + self.strength * delta
            weight[start:stop] = merged
        self.merged.add(base_prefix)
