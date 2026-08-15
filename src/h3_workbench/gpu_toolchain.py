from __future__ import annotations

import importlib.util
import ctypes
import hashlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from h3_workbench.device_profile import selected_device_index, selected_device_profile


FP8_DEQUANT_KERNEL = r"""
#include <cuda_fp8.h>
#include <cuda_fp16.h>

extern "C" __global__ void fp8_scale_lora(
    const unsigned char* source,
    half* output,
    const half* lora_a,
    const half* lora_b,
    int rows,
    int columns,
    int rank,
    float scale,
    float lora_strength,
    int transpose_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int count = rows * columns;
    if (index >= count) return;
    int row = index / columns;
    int column = index - row * columns;
    const __nv_fp8_e4m3* values = reinterpret_cast<const __nv_fp8_e4m3*>(source);
    float value = static_cast<float>(values[index]) * scale;
    if (rank > 0) {
        float delta = 0.0f;
        for (int item = 0; item < rank; ++item) {
            delta += __half2float(lora_b[row * rank + item])
                * __half2float(lora_a[item * columns + column]);
        }
        value += delta * lora_strength;
    }
    int output_index = transpose_output ? column * rows + row : index;
    output[output_index] = __float2half_rn(value);
}
"""


CUTLASS_GEMM_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

extern "C" __declspec(dllexport) int h3_cutlass_gemm_f16(
    void const* a,
    void const* b,
    void const* c,
    void* d,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    void* stream) {
  using RowMajor = cutlass::layout::RowMajor;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float>;
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,
      RowMajor,
      cutlass::half_t,
      RowMajor,
      cutlass::half_t,
      RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      TargetArch,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      OutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2>;
  typename Gemm::Arguments arguments(
      {m, n, k},
      {static_cast<cutlass::half_t const*>(a), k},
      {static_cast<cutlass::half_t const*>(b), n},
      {static_cast<cutlass::half_t const*>(c), n},
      {static_cast<cutlass::half_t*>(d), n},
      {alpha, beta});
  Gemm operation;
  cutlass::Status status = operation.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 100 + static_cast<int>(status);
  }
  size_t workspace_size = operation.get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t allocation = cudaMalloc(&workspace, workspace_size);
    if (allocation != cudaSuccess) {
      return 1000 + static_cast<int>(allocation);
    }
  }
  status = operation.initialize(arguments, workspace, static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    if (workspace != nullptr) cudaFree(workspace);
    return 200 + static_cast<int>(status);
  }
  status = operation.run(static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    if (workspace != nullptr) cudaFree(workspace);
    return 300 + static_cast<int>(status);
  }
  if (workspace != nullptr) cudaFree(workspace);
  return 0;
}
"""


@dataclass(frozen=True)
class GpuToolchainStatus:
    cuda_path: str | None
    nvcc: str | None
    cupy_available: bool
    cupy_version: str | None
    compute_capability: str | None
    device: str | None
    cutlass_available: bool
    cutlass_include: str | None
    tensorrt_available: bool
    tensorrt_version: str | None
    cuda_graph_supported: bool
    fp8_tensor_core_supported: bool
    fp8_fused_kernel_supported: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ascii_cache_root() -> Path:
    configured = os.environ.get("H3_CUDA_CACHE_DIR")
    candidates = [
        Path(configured) if configured else None,
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "h3-workbench" / "cuda-cache",
        Path(r"C:\h3-cuda-cache"),
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_absolute() or not str(candidate).isascii():
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable ASCII CUDA cache directory is available")


def discover_cuda_path() -> Path | None:
    for name in ("CUDA_PATH", "CUDA_PATH_V12_6"):
        value = os.environ.get(name)
        if value and (Path(value) / "bin" / "nvcc.exe").is_file():
            return Path(value)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.is_dir():
        versions = sorted(toolkit_root.glob("v12.*"), reverse=True)
        for version in versions:
            if (version / "bin" / "nvcc.exe").is_file():
                return version
    return None


def prepare_cuda_environment() -> tuple[Path, Path]:
    cuda_path = discover_cuda_path()
    if cuda_path is None:
        raise RuntimeError("CUDA Toolkit with nvcc was not found")
    cache = _ascii_cache_root()
    temporary = cache / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_PATH"] = str(cuda_path)
    os.environ["CUDA_INSTALL_PATH"] = str(cuda_path)
    os.environ["CUPY_CACHE_DIR"] = str(cache / "cupy")
    # NVRTC on Windows 12.6 cannot open source/header paths containing non-ASCII text.
    if not os.environ.get("TEMP", "").isascii() or not os.environ.get("TMP", "").isascii():
        os.environ["TEMP"] = str(temporary)
        os.environ["TMP"] = str(temporary)
    return cuda_path, cache


def _cutlass_include(cache: Path) -> Path | None:
    spec = importlib.util.find_spec("cutlass_library")
    if spec is None or spec.origin is None:
        return None
    source_root = Path(spec.origin).parent / "source"
    source = source_root / "include"
    if not (source / "cutlass" / "cutlass.h").is_file():
        return None
    if str(source_root).isascii():
        os.environ["CUTLASS_PATH"] = str(source_root)
        return source
    destination_root = cache / "cutlass-source"
    destination = destination_root / "include"
    marker = destination / "cutlass" / "cutlass.h"
    if not marker.is_file():
        shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
    os.environ["CUTLASS_PATH"] = str(destination_root)
    return destination


def probe_gpu_toolchain() -> GpuToolchainStatus:
    try:
        cuda_path, cache = prepare_cuda_environment()
    except RuntimeError as exc:
        return GpuToolchainStatus(
            None, None, False, None, None, None, False, None, False, None, False, False, False, str(exc)
        )
    cupy_available = False
    cupy_version = None
    capability = None
    device = None
    reason = None
    try:
        import cupy as cp

        cupy_available = True
        cupy_version = cp.__version__
        device_index = selected_device_index()
        cp.cuda.Device(device_index).use()
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        capability = f"{properties['major']}.{properties['minor']}"
        raw_name = properties["name"]
        device = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
    except Exception as exc:  # noqa: BLE001 - report optional toolchain failures
        reason = str(exc)
    cutlass_include = _cutlass_include(cache)
    tensorrt_available = False
    tensorrt_version = None
    try:
        import tensorrt as trt

        tensorrt_available = True
        tensorrt_version = trt.__version__
    except Exception:  # noqa: BLE001 - optional extra
        pass
    try:
        major, minor = (int(part) for part in capability.split(".", 1)) if capability else (0, 0)
    except ValueError:
        major, minor = 0, 0
    return GpuToolchainStatus(
        str(cuda_path),
        str(cuda_path / "bin" / "nvcc.exe"),
        cupy_available,
        cupy_version,
        capability,
        device,
        cutlass_include is not None,
        str(cutlass_include) if cutlass_include else None,
        tensorrt_available,
        tensorrt_version,
        cupy_available and major >= 7,
        (major, minor) >= (8, 9),
        cupy_available and major >= 8,
        reason,
    )


def initialize_cutlass() -> Any:
    cuda_path, cache = prepare_cuda_environment()
    cutlass_include = _cutlass_include(cache)
    if cutlass_include is None:
        raise RuntimeError("The nvidia-cutlass GPU extra is not installed")
    import cutlass_cppgen

    # CUTLASS 4.2 evaluates its Linux `which nvcc` fallback eagerly on Windows.
    cutlass_cppgen._CUDA_INSTALL_PATH = str(cuda_path)
    cutlass_cppgen.CUTLASS_PATH = str(cutlass_include.parent)
    cutlass_cppgen.source_path = cutlass_cppgen.CUTLASS_PATH
    return cutlass_cppgen


def _visual_studio_environment() -> dict[str, str]:
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
    ]
    vcvars = next(
        (
            root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            for root in roots
            if (root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat").is_file()
        ),
        None,
    )
    if vcvars is None:
        raise RuntimeError("Visual Studio 2022 x64 C++ build tools were not found")
    short_buffer = ctypes.create_unicode_buffer(32768)
    short_length = ctypes.windll.kernel32.GetShortPathNameW(
        str(vcvars),
        short_buffer,
        len(short_buffer),
    )
    vcvars_command = short_buffer.value if short_length else str(vcvars)
    ascii_working_directory = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    command_environment = os.environ.copy()
    command_environment["TEMP"] = str(ascii_working_directory)
    command_environment["TMP"] = str(ascii_working_directory)
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", f"call {vcvars_command} >nul && set"],
        cwd=ascii_working_directory,
        env=command_environment,
        capture_output=True,
        check=True,
        text=True,
        encoding="mbcs",
        errors="replace",
        timeout=60,
    )
    environment = os.environ.copy()
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            environment[name] = value
    return environment


def build_cutlass_gemm_library() -> Path:
    cuda_path, cache = prepare_cuda_environment()
    cutlass_include = _cutlass_include(cache)
    if cutlass_include is None:
        raise RuntimeError("The nvidia-cutlass GPU extra is not installed")
    profile = selected_device_profile()
    capability = profile.compute_capability or "8.6"
    try:
        major, minor = (int(part) for part in capability.split(".", 1))
    except ValueError:
        major, minor = 8, 6
    target_sm = major * 10 + minor
    digest = hashlib.sha256(f"{target_sm}:{CUTLASS_GEMM_SOURCE}".encode("ascii")).hexdigest()[:12]
    library = cache / f"h3_cutlass_sm{target_sm}_{digest}.dll"
    if library.is_file():
        return library
    source = cache / f"h3_cutlass_sm{target_sm}_{digest}.cu"
    target_arch = "cutlass::arch::Sm75" if target_sm < 80 else "cutlass::arch::Sm80"
    source_code = CUTLASS_GEMM_SOURCE.replace(
        "using RowMajor = cutlass::layout::RowMajor;",
        f"using TargetArch = {target_arch};\n  using RowMajor = cutlass::layout::RowMajor;",
    )
    source.write_text(source_code, encoding="ascii")
    environment = _visual_studio_environment()
    temporary = cache / "tmp"
    environment["TEMP"] = str(temporary)
    environment["TMP"] = str(temporary)
    command = [
        str(cuda_path / "bin" / "nvcc.exe"),
        "--use-local-env",
        "-std=c++17",
        "-O3",
        f"-arch=sm_{target_sm}",
        "--shared",
        "-Xcompiler=/MD",
        f"-I{cutlass_include}",
        source.name,
        "-o",
        library.name,
    ]
    result = subprocess.run(
        command,
        cwd=cache,
        env=environment,
        capture_output=True,
        text=True,
        encoding="mbcs",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0 or not library.is_file():
        details = result.stderr.strip() or result.stdout.strip() or f"nvcc exited {result.returncode}"
        raise RuntimeError(f"CUTLASS build failed: {details}")
    return library


class CutlassFp16Gemm:
    def __init__(self) -> None:
        import cupy as cp

        self.cp = cp
        cp.cuda.Device(selected_device_index()).use()
        self.path = build_cutlass_gemm_library()
        self.library = ctypes.WinDLL(str(self.path))
        self.function = self.library.h3_cutlass_gemm_f16
        self.function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        self.function.restype = ctypes.c_int

    def __call__(self, a: Any, b: Any, *, stream: Any | None = None) -> FusedDeviceTensor:
        cp = self.cp
        if a.dtype != cp.float16 or b.dtype != cp.float16 or a.ndim != 2 or b.ndim != 2:
            raise ValueError("CUTLASS GEMM requires two FP16 matrices")
        if not a.flags.c_contiguous or not b.flags.c_contiguous or a.shape[1] != b.shape[0]:
            raise ValueError("CUTLASS GEMM requires contiguous compatible row-major matrices")
        m, k = (int(value) for value in a.shape)
        n = int(b.shape[1])
        c = cp.asarray(np.zeros((m, n), dtype=np.float16))
        output = cp.empty((m, n), dtype=cp.float16)
        selected_stream = stream or cp.cuda.get_current_stream()
        status = self.function(
            ctypes.c_void_p(a.data.ptr),
            ctypes.c_void_p(b.data.ptr),
            ctypes.c_void_p(c.data.ptr),
            ctypes.c_void_p(output.data.ptr),
            m,
            n,
            k,
            1.0,
            0.0,
            ctypes.c_void_p(selected_stream.ptr),
        )
        if status:
            raise RuntimeError(f"CUTLASS GEMM launch failed with status {status}")
        return FusedDeviceTensor(output, (a, b, c))


class FusedDeviceTensor:
    def __init__(self, array: Any, sources: tuple[Any, ...] = (), device_index: int | None = None) -> None:
        self.array = array
        self.sources = sources
        self.device_index = selected_device_index() if device_index is None else int(device_index)
        self.pointer = int(array.data.ptr)
        self.dtype = np.dtype(array.dtype)
        self.shape = tuple(int(value) for value in array.shape)

    def bind_input(self, binding: Any, name: str) -> None:
        binding.bind_input(name, "cuda", self.device_index, self.dtype, self.shape, self.pointer)

    def numpy(self) -> np.ndarray:
        import cupy as cp

        return cp.asnumpy(self.array)


class FusedFp8Dequantizer:
    """NVRTC kernel for FP8 scale, optional transpose, and dynamic LoRA merge."""

    def __init__(self) -> None:
        cuda_path, _ = prepare_cuda_environment()
        import cupy as cp

        device_index = selected_device_index()
        cp.cuda.Device(device_index).use()
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        architecture = f"sm_{properties['major']}{properties['minor']}"
        include = str(cuda_path / "include").replace("\\", "/")
        self.cp = cp
        self.kernel = cp.RawKernel(
            FP8_DEQUANT_KERNEL,
            "fp8_scale_lora",
            options=(f"-I{include}", f"-arch={architecture}"),
        )

    def __call__(
        self,
        source: np.ndarray,
        scale: float,
        *,
        transpose: bool = False,
        lora_a: np.ndarray | None = None,
        lora_b: np.ndarray | None = None,
        lora_strength: float = 1.0,
        stream: Any | None = None,
    ) -> FusedDeviceTensor:
        if source.ndim != 2 or source.dtype.itemsize != 1:
            raise ValueError("FP8 source must be a two-dimensional one-byte tensor")
        rows, columns = (int(value) for value in source.shape)
        rank = 0
        if (lora_a is None) != (lora_b is None):
            raise ValueError("LoRA A and B must be provided together")
        if lora_a is not None and lora_b is not None:
            if lora_a.ndim != 2 or lora_b.ndim != 2:
                raise ValueError("LoRA factors must be matrices")
            rank = int(lora_a.shape[0])
            if lora_a.shape != (rank, columns) or lora_b.shape != (rows, rank):
                raise ValueError("LoRA factors do not match the base weight")
        cp = self.cp
        device_source = cp.asarray(source.view(np.uint8))
        device_a = cp.asarray(np.asarray(lora_a, dtype=np.float16)) if lora_a is not None else None
        device_b = cp.asarray(np.asarray(lora_b, dtype=np.float16)) if lora_b is not None else None
        output_shape = (columns, rows) if transpose else (rows, columns)
        output = cp.empty(output_shape, dtype=cp.float16)
        count = rows * columns
        args = (
            device_source,
            output,
            device_a if device_a is not None else np.uint64(0),
            device_b if device_b is not None else np.uint64(0),
            np.int32(rows),
            np.int32(columns),
            np.int32(rank),
            np.float32(scale),
            np.float32(lora_strength),
            np.int32(transpose),
        )
        selected_stream = stream or cp.cuda.get_current_stream()
        self.kernel(((count + 255) // 256,), (256,), args, stream=selected_stream)
        return FusedDeviceTensor(output, (device_source, device_a, device_b), device_index=selected_device_index())


def cuda_graph_eligibility(*, stable_device_inputs: bool, fixed_shapes: bool) -> tuple[bool, str | None]:
    status = probe_gpu_toolchain()
    if not status.cuda_graph_supported:
        return False, status.reason or "CUDA Graph requires the CuPy CUDA toolchain"
    if not fixed_shapes:
        return False, "CUDA Graph requires fixed input and output shapes"
    if not stable_device_inputs:
        return False, "CUDA Graph requires stable device addresses for every bound tensor"
    return True, None
