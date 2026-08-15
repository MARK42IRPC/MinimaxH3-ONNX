from dataclasses import replace

import numpy as np

from h3_workbench.gpu_toolchain import (
    FusedDeviceTensor,
    GpuToolchainStatus,
    cuda_graph_eligibility,
)


def _status(**values: object) -> GpuToolchainStatus:
    base = GpuToolchainStatus(
        cuda_path="C:/CUDA",
        nvcc="C:/CUDA/bin/nvcc.exe",
        cupy_available=True,
        cupy_version="14",
        compute_capability="8.6",
        device="RTX 3050",
        cutlass_available=True,
        cutlass_include="C:/cutlass",
        tensorrt_available=False,
        tensorrt_version=None,
        cuda_graph_supported=True,
        fp8_tensor_core_supported=False,
        fp8_fused_kernel_supported=True,
    )
    return replace(base, **values)


def test_cuda_graph_gate_requires_fixed_shapes_and_stable_device_addresses(monkeypatch) -> None:
    monkeypatch.setattr("h3_workbench.gpu_toolchain.probe_gpu_toolchain", _status)

    assert cuda_graph_eligibility(stable_device_inputs=True, fixed_shapes=True) == (True, None)
    assert cuda_graph_eligibility(stable_device_inputs=False, fixed_shapes=True)[0] is False
    assert cuda_graph_eligibility(stable_device_inputs=True, fixed_shapes=False)[0] is False


def test_fused_device_tensor_binds_owned_device_pointer() -> None:
    class Array:
        class Data:
            ptr = 1234

        data = Data()
        dtype = np.dtype(np.float16)
        shape = (2, 3)

    class Binding:
        call = None

        def bind_input(self, *args) -> None:
            self.call = args

    tensor = FusedDeviceTensor(Array(), sources=(object(),))
    binding = Binding()
    tensor.bind_input(binding, "weight")

    assert binding.call == ("weight", "cuda", 0, np.dtype(np.float16), (2, 3), 1234)
    assert len(tensor.sources) == 1
