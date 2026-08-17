from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

GIB = 1024**3
_PROFILE_TTL_SECONDS = 2.0
_last_profiles_at = 0.0
_last_profiles: tuple[DeviceProfile, ...] | None = None
_selected_selector: str | None = None
_selected_device_key: str | None = None


@dataclass(frozen=True)
class DeviceProfile:
    """One physical execution device and its conservative runtime policy."""

    index: int
    name: str
    uuid: str | None
    provider: str
    total_bytes: int
    free_bytes: int
    driver: str | None = None
    cuda_runtime: str | None = None
    compute_capability: str | None = None
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_cuda_graph: bool = False
    tier: str = "cpu"
    reason: str | None = None

    @property
    def device_key(self) -> str:
        return self.uuid or f"index:{self.index}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["device_key"] = self.device_key
        return result


def _parse_selector(value: str | None) -> str:
    selector = (value or os.environ.get("H3_CUDA_DEVICE", "0")).strip()
    return selector or "0"


def _capability_flags(compute_capability: str | None) -> dict[str, bool]:
    if not compute_capability:
        return {
            "supports_fp16": False,
            "supports_bf16": False,
            "supports_fp8": False,
            "supports_cuda_graph": False,
        }
    try:
        major, minor = (int(part) for part in compute_capability.split(".", 1))
    except (TypeError, ValueError):
        return {
            "supports_fp16": False,
            "supports_bf16": False,
            "supports_fp8": False,
            "supports_cuda_graph": False,
        }
    # These are hardware capability hints. Operator support and numerical
    # validation remain separate runtime gates.
    return {
        "supports_fp16": (major, minor) >= (7, 0),
        "supports_bf16": (major, minor) >= (8, 0),
        "supports_fp8": (major, minor) >= (8, 9),
        "supports_cuda_graph": (major, minor) >= (7, 0),
    }


def _tier(total_bytes: int, provider: str) -> str:
    if provider != "cuda" or total_bytes <= 0:
        return "cpu"
    if total_bytes <= 6 * GIB:
        return "low_vram"
    if total_bytes < 12 * GIB:
        return "standard"
    if total_bytes < 20 * GIB:
        return "high_vram"
    return "enthusiast"


def _nvidia_smi_rows() -> dict[int, dict[str, str]]:
    flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    query_variants = (
        "index,uuid,name,memory.total,memory.free,driver_version,compute_cap",
        "index,uuid,name,memory.total,memory.free,driver_version",
    )
    for query in query_variants:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                creationflags=flags,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        rows: dict[int, dict[str, str]] = {}
        for raw in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
            if len(raw) not in {6, 7}:
                continue
            try:
                index = int(raw[0].strip())
                int(raw[3].strip())
                int(raw[4].strip())
            except ValueError:
                continue
            rows[index] = {
                "uuid": raw[1].strip(),
                "name": raw[2].strip(),
                "total_mib": raw[3].strip(),
                "free_mib": raw[4].strip(),
                "driver": raw[5].strip(),
                "compute_capability": raw[6].strip() if len(raw) == 7 else "",
            }
        if rows:
            return rows
    return {}


def _torch_rows() -> dict[int, dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        runtime = str(torch.version.cuda) if torch.version.cuda else None
        rows: dict[int, dict[str, Any]] = {}
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes = 0
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
            except (RuntimeError, TypeError):
                pass
            rows[index] = {
                "name": str(properties.name),
                "total_bytes": int(properties.total_memory),
                "free_bytes": int(free_bytes),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "cuda_runtime": runtime,
            }
        return rows
    except (ImportError, RuntimeError, AttributeError):
        return {}


def torch_cuda_architecture_supported(index: int = 0) -> bool:
    """Return whether the installed PyTorch build contains this GPU's CUDA kernels.

    ``torch.cuda.is_available()`` only tells us that CUDA can be initialized. It
    does not mean that the wheel was compiled for the visible GPU. This matters
    for newer cards such as Blackwell, where an older wheel can initialize CUDA
    and then fail on the first convolution or SDPA call with ``no kernel image``.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        properties = torch.cuda.get_device_properties(index)
        architecture = f"sm_{properties.major}{properties.minor}"
        return architecture in set(torch.cuda.get_arch_list())
    except (ImportError, RuntimeError, AttributeError, IndexError):
        return False


def _available_cuda_provider() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except (ImportError, RuntimeError):
        return False


def _build_profiles() -> tuple[DeviceProfile, ...]:
    smi = _nvidia_smi_rows()
    torch_rows = _torch_rows()
    indices = sorted(set(smi) | set(torch_rows))
    if not indices:
        return (
            DeviceProfile(
                index=-1,
                name="CPU",
                uuid=None,
                provider="cpu",
                total_bytes=0,
                free_bytes=0,
                tier="cpu",
                reason="No CUDA device was detected",
            ),
        )
    cuda_provider = _available_cuda_provider()
    profiles: list[DeviceProfile] = []
    for index in indices:
        smi_row = smi.get(index, {})
        torch_row = torch_rows.get(index, {})
        total_bytes = int(torch_row.get("total_bytes", int(smi_row.get("total_mib", 0)) * 1024**2))
        free_bytes = int(torch_row.get("free_bytes", int(smi_row.get("free_mib", 0)) * 1024**2))
        name = str(torch_row.get("name") or smi_row.get("name") or f"CUDA device {index}")
        capability = torch_row.get("compute_capability") or smi_row.get("compute_capability") or None
        flags = _capability_flags(capability)
        provider = "cuda" if cuda_provider else "cpu"
        profiles.append(
            DeviceProfile(
                index=index,
                name=name,
                uuid=smi_row.get("uuid") or None,
                provider=provider,
                total_bytes=total_bytes,
                free_bytes=free_bytes,
                driver=smi_row.get("driver") or None,
                cuda_runtime=torch_row.get("cuda_runtime"),
                compute_capability=capability,
                **flags,
                tier=_tier(total_bytes, provider),
                reason=None if cuda_provider else "CUDAExecutionProvider is unavailable",
            )
        )
    return tuple(profiles)


def probe_device_profiles(*, refresh: bool = False) -> tuple[DeviceProfile, ...]:
    global _last_profiles_at, _last_profiles
    now = time.monotonic()
    if not refresh and _last_profiles is not None and now - _last_profiles_at < _PROFILE_TTL_SECONDS:
        return _last_profiles
    _last_profiles = _build_profiles()
    _last_profiles_at = now
    return _last_profiles


def select_device_profile(
    profiles: tuple[DeviceProfile, ...] | list[DeviceProfile] | None = None,
    selector: str | None = None,
) -> DeviceProfile:
    available = tuple(profiles if profiles is not None else probe_device_profiles())
    if not available:
        return DeviceProfile(-1, "CPU", None, "cpu", 0, 0, tier="cpu", reason="No device profile")
    if len(available) == 1 and available[0].provider == "cpu":
        return available[0]
    requested = _parse_selector(selector)
    if requested.lower() == "auto":
        return max(available, key=lambda item: (item.free_bytes, item.total_bytes, -item.index))
    for item in available:
        if (
            requested == item.uuid
            or requested == item.device_key
            or (item.uuid is not None and requested.lower() == item.uuid.lower())
        ):
            return item
    try:
        index = int(requested, 0)
    except ValueError as exc:
        raise ValueError(
            f"H3_CUDA_DEVICE={requested!r} did not match a visible GPU index or UUID"
        ) from exc
    for item in available:
        if item.index == index:
            return item
    visible = ", ".join(f"{item.index}:{item.name}" for item in available)
    raise ValueError(f"CUDA device index {index} is unavailable; visible devices: {visible}")


def selected_device_profile(*, refresh: bool = False) -> DeviceProfile:
    global _selected_selector, _selected_device_key
    profiles = probe_device_profiles(refresh=refresh)
    requested = _parse_selector(None)
    selector = requested.lower()
    if selector != _selected_selector:
        _selected_selector = selector
        _selected_device_key = None
    if _selected_device_key is not None:
        for profile in profiles:
            if profile.device_key == _selected_device_key:
                return profile
    selected = select_device_profile(profiles, requested)
    _selected_device_key = selected.device_key
    return selected


def selected_device_index(*, refresh: bool = False) -> int:
    profile = selected_device_profile(refresh=refresh)
    return max(0, profile.index)
