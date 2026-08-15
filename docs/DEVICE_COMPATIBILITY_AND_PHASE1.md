# Device Compatibility And Phase 1

## Scope

Phase 1 establishes one source of truth for GPU selection, device capability
reporting, VRAM planning, and safe fallback. It does not change the exported
ONNX model format or enable experimental TensorRT, unified-stream, or device
weight-prefetch paths by default.

The current production target is Windows plus NVIDIA CUDA. CPUExecutionProvider
remains a validation and fallback path, but the validated Qwen INT8 virtual
product still requires CUDA. AMD, Intel, and Apple GPU execution are not yet
supported production backends.

The locked Windows environment uses the official PyTorch CUDA 12.6 wheel.
ONNX Runtime CUDA and PyTorch CUDA are independent checks: a machine can have
an NVIDIA driver and a working ORT CUDA provider while still having a CPU-only
PyTorch installation, in which case the runtime falls back from PyTorch SDPA
to the ORT streaming-attention implementation.

## Device policy

The runtime reports every visible NVIDIA device and selects one device for the
whole process. `H3_CUDA_DEVICE` accepts a numeric index, a GPU UUID, or `auto`.
The default remains index `0` for compatibility. `auto` selects the device with
the most free VRAM at startup. A selected device is identified by UUID when
available, otherwise by its stable index key.

Each device receives a tier based on total VRAM:

| Tier | Total VRAM | Default intent |
| --- | ---: | --- |
| `cpu` | no CUDA device | CPU provider and conservative host path |
| `low_vram` | up to 6 GiB | one-session streaming and small attention chunks |
| `standard` | 6-12 GiB | bounded persistent sessions and adaptive chunks |
| `high_vram` | 12-20 GiB | persistent topologies, device-resident activations, warmup |
| `enthusiast` | 20 GiB or more | high-VRAM policy; fixed-shape CUDA Graph is a later gate |

The tier is a scheduling hint, not a correctness claim. Runtime free VRAM,
cross-process reservations, model shape, and host commit headroom remain the
final admission checks.

## Capability rules

Compute capability is reported as the complete `major.minor` value. FP16,
BF16, FP8, and CUDA Graph support are separate capability flags. In particular,
FP8 is never inferred from a major version alone; an operator still needs a
device-specific numerical and performance gate. Custom CUDA libraries must be
compiled for the selected SM or a validated fat binary and cached with the
device capability in the key.

## Phase 1 deliverables

1. Add `DeviceProfile` discovery from `nvidia-smi` and PyTorch, with UUID,
   index, VRAM, driver, CUDA runtime, compute capability, tier, and capability
   flags.
2. Route `probe_gpu_memory`, ONNX Runtime provider options, CuPy helpers, and
   CUDA I/O binding through the selected device index.
3. Scope VRAM reservations by GPU UUID/index so concurrent jobs on separate
   GPUs do not block one another.
4. Expose the complete inventory and selected policy through `/api/system` and
   `/api/profiles`.
5. Make CPU affinity and process priority opt-in. The launcher must not assume
   the current workstation's core topology on another machine.

## Acceptance gates

- CPU-only startup still works and reports a CPU profile.
- A single CUDA device behaves as before, with explicit `device_id` passed to
  ONNX Runtime.
- `H3_CUDA_DEVICE=auto` and UUID selection are deterministic for the lifetime
  of a process.
- Reservations for GPU A are not subtracted from GPU B.
- Invalid device selection fails with an actionable message rather than a
  silent CPU fallback.
- Existing numerical tests and the full CPU test suite remain green.

## Follow-up phases

Phase 2 will add fixed-shape resolution/duration buckets, reusable device
buffers, and a high-VRAM double-buffer upload policy. CUDA Graph will only be
enabled for those fixed buckets after an end-to-end numerical gate.

Phase 3 may evaluate FP8 exports, VAE-local TensorRT engines, and multi-GPU
job scheduling. The current main-model TensorRT result is not suitable for a
default production path: its FP16 candidate produced non-finite output and its
FP32 candidate had a larger numerical error than CUDA.
