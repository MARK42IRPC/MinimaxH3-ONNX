# FL2VA GPU Utilization Optimization Plan

## Baseline

Target host:

- NVIDIA RTX 3050 Laptop GPU, 4 GiB VRAM
- Intel Core i5-12450H, 32 GiB RAM
- Native 360p generation, up to 15 seconds

Observed during a five-step native task on 2026-08-12:

| Module | Average GPU utilization | Peak GPU utilization |
| --- | ---: | ---: |
| MLP | 76% | 100% |
| Streaming SDPA | 46% | 65% |
| Attention QKV | 37% | 55% |
| Attention output | 19% | 27% |

Storage was not the dominant stall: L1 session prefetch waited 2.46 seconds and
L2 waited 15.00 seconds at the sampled point. The main loss is caused by small
256-token ORT calls, repeated CPU/GPU synchronization, CPU output allocation,
and activation round trips at graph boundaries.

## Objectives

- Raise QKV and Attention output utilization without exceeding 3.7 GiB VRAM.
- Reduce ORT calls per 15-second block by at least 50%.
- Preserve exact tensor order and numerical behavior.
- Keep every optimization independently reversible.
- Show actual chunk sizes, cache waits, module progress, and total task elapsed
  time in the WebUI.

## Phase 1: Dynamic Chunking And Bound Output Buffers

1. Choose chunk sizes from the current free VRAM.
2. Prefer 1024 tokens for QKV and 2048 for Attention output when at least
   2.25 GiB is free.
3. Use 512 tokens for MLP only when at least 2.75 GiB is free; otherwise retain
   256 because the MLP already reaches full GPU utilization.
4. Bind every chunk directly into a slice of a preallocated CPU output array.
   This removes per-call output allocation and the final `numpy.concatenate`.
5. On CUDA OOM, halve the failing chunk and retry down to 256 tokens.

Acceptance criteria:

- All tests pass and outputs remain finite.
- No task regression or OOM on the 4 GiB card.
- Attention output average utilization improves over the 19% baseline.
- End-to-end FL2VA step time does not regress by more than 5%.

Initial validation on 2026-08-12:

| Graph | Rows | Baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: | ---: |
| QKV | 4096 | 1.589s | 0.979s | 1.62x |
| Attention output | 4096 | 0.217s | 0.140s | 1.55x |

QKV matched exactly. Attention output had a maximum absolute difference of
`3.12e-4`, consistent with FP16 CUDA GEMM scheduling differences. A real
17-frame CUDA Video VAE decode completed in 49.45 seconds with output shape
`[1, 3, 17, 360, 640]`, finite values, and range `[0, 1]`.

Rollback:

- Set `H3_FL2VA_DYNAMIC_CHUNKS=0` to restore 256-token calls.
- Set `H3_FL2VA_IO_BINDING=0` to restore `session.run()` output allocation.

### Phase 1.1: Native 15-Second Commit-Memory Fix

A native 15-second task failed on 2026-08-12 after Windows reported virtual
memory exhaustion. The Python process had committed 23.8 GB, while the host's
page file was only 4.84 GiB. NumPy then failed to allocate a final 7 MiB SDPA
output chunk.

Implemented safeguards:

- Store the 26,352-token QKV buffer as FP16 instead of FP32, reducing it from
  2.11 GiB to 1.06 GiB.
- Preallocate the FP32 attended output and copy every FP16 CUDA result directly
  into its destination slice. This removes the list of chunk outputs and a
  final 0.70 GiB concatenation copy.
- Treat Attention output as an L1 prefetch barrier and a single-session batch.
  Run SDPA only after the QKV session is released, and resume MLP prefetch only
  after Attention output consumes the large attended tensor.
- Poll WebUI jobs continuously so a fast failure is rendered even after the
  browser's previous state no longer contains a running job.
- Report page-file capacity on the system page and FP16 QKV size in the request
  estimate.

The estimated host-memory peak reduction for 15-second 640x360 native mode is
1.76 GiB. CUDA SDPA validation produced finite `[257, 7168]` FP32 output with a
maximum absolute error of `2.38e-6`. The real 4096-row QKV graph produced FP16
output with a maximum absolute difference of `0.00191`; its runtime was neutral
(0.898s versus 0.887s baseline). Attention output retained a 2.06x speedup.

The code no longer requires a larger page file for this specific peak, but a
system-managed or at least 24 GiB page file remains recommended as protection
against ONNX Runtime session-construction commit spikes.

### Phase 1.2: FP16 Attended Sidecar Graphs

The streamed SDPA result now stays FP16 through the Attention output graph
boundary. Fifty small sidecar graph headers change only the `attended` input
type and reuse the original external `.onnx.data` files. The sidecars total
735 KiB and copy no weights.

On the real block-0 Attention output graph at 4096 rows:

- attended host storage fell from 112 MiB to 56 MiB;
- warm execution time fell by about 10% (`0.899x` of the FP32-input path);
- output matched exactly (`max_abs=0`, cosine `1.0`).

For a 15-second 640x360 native sequence, this removes another approximately
0.35 GiB of resident host memory. Combined with Phase 1.1, the estimated peak
reduction is about 2.11 GiB.

The L1 scheduler also caps prefetch by Windows commit headroom and preserves a
4 GiB reserve. If commit headroom reaches the reserve, concurrent L1 prefetch
is disabled and loading becomes sequential.

### Audio Failure Isolation

A one-step base-model run completed FL2VA and Video VAE but produced non-finite
audio latents. The Audio VAE itself remained finite for 200-frame CUDA tests
with finite latent magnitudes up to 271, confirming that the failure originated
in the one-step FL2VA audio branch rather than the dynamic Audio VAE graph.

Sampling now checks video and audio branches separately. Non-finite video data
remains a hard failure. Non-finite audio velocity or latents are isolated, the
last finite audio state is retained, and the completed video is muxed with an
equal-duration silent track. The job result records `audio_status` and the
fallback reason. The WebUI warns that the base model below five steps may have
unstable audio; acceleration readiness is kept separate from actual activation.

## Phase 2: CUDA-Resident Attention Activations

Keep QKV output, K/V, attended rows, Attention output, and the next MLP input as
CUDA OrtValues. Copy only final block outputs to host when session eviction is
required. This phase is only started after Phase 1 is benchmarked because it
changes ownership and lifetime of several large activation buffers.

Acceptance criteria:

- At least 50% fewer activation H2D/D2H transfers.
- No numerical deviation outside the existing ONNX tolerance.
- Stable VRAM below 3.8 GiB.

## Phase 3: Fused Streaming Attention

Evaluate ONNX Runtime fused attention, FlashAttention, or a custom CUDA kernel
that keeps K/V resident and streams query tiles. Select an implementation only
after measuring kernel support on the RTX 3050 and verifying exact long-context
semantics.

## Phase 4: Persistent FL2VA Topologies

Reuse one QKV, Attention output, and MLP topology with per-layer weights supplied
as inputs. The Qwen persistent topology reduced its four-token benchmark from
129.8 seconds to 67.2 seconds, but FL2VA must be evaluated separately because
its long activation tensors and five repeated sampling steps change the H2D
tradeoff.

## Measurement Protocol

For each accepted change record:

- Total task elapsed time and FL2VA step time.
- GPU average, P50, P95, and peak utilization by module.
- Peak VRAM and host RAM.
- L1/L2 wait counts and seconds.
- Selected QKV, Attention output, MLP, and SDPA chunks.
- Output finite check and generated MP4 validation.

Use the same seed, prompt, resolution, duration, step count, and temporal mode
for A/B runs. Do not compare runs that differ in model cache warmth without
recording that distinction.

## 2026-08-14 Single-Block Tensor Core Gate

The product FP8-storage MLP expands weights to FP32 at run time and contains
134 Gemm, 137 Slice, and 137 Cast nodes. The RTX 3050 Laptop GPU is Ampere and
does not have native FP8 Tensor Cores. Four direct, selector-free MLP exports
were tested for blocks 0, 4, and 49:

- FP32 reference: selector-free but retains 134 chunked Gemm operations.
- Native FP16: three Gemm operations, but blocks 0 and 49 produced non-finite
  output and were rejected.
- BF16: three Gemm operations and finite output, but relative L2 reached
  `4.92e-3` on block 49 and was rejected.
- Scaled FP16: scales FC1 weights by 16, FC2 inputs by 128, and FC2 weights by
  16 before Tensor Core Gemm, then restores the scale in FP32. RMSNorm, AdaLN,
  residuals, and outputs remain FP32.

At 64 rows, scaled FP16 reduced the warm MLP time from approximately 53 ms to
4.9-5.2 ms (10.3-10.7x). At the cf4 shape of 1190 rows, warm time fell from
145.6-146.0 ms to 64.3-68.4 ms (2.13-2.27x). The graph contains 3 Gemm and 4
Cast nodes. Its sampled session VRAM was approximately 691 MiB versus 2207 MiB
for the product FP8-to-FP32 graph.

All four real cf4 sampling timestep/modulation combinations were checked at
1190 rows. Worst scaled-FP16 relative L2 was `2.29e-4`, `3.02e-4`, and
`5.17e-4` for blocks 0, 4, and 49 respectively; every output was finite. Model
storage increases from approximately 224 MiB to 445 MiB per MLP shard.

Scaled FP16 passes the single-block gate but is not yet a product conversion.
The next gate must run a chained 50-block trace with real preceding-block
activations and then the four-step cf4 main-model test. Full publication also
needs a storage/session strategy because converting all 50 MLP shards adds
approximately 10.8 GiB.

## 2026-08-14 Full-Chain Scaled-FP16 Gate

All 50 MLP blocks were exported with a durable, resumable state file and a
90 GiB free-space guard checked before every new block. The 47 new exports took
803.693 seconds of worker time; the complete batch and hybrid build took
852.176 seconds. The candidate set occupies 21.70 GiB. C: free space moved from
151.01 GiB to 130.26 GiB, preserving 40.26 GiB beyond the hard reserve. Blocks
0, 4, and 49 reuse the validated files through hard links.

Every graph built a CUDA session, ran finite, contained 3 Gemm and 4 Cast
operators, and stored its large MLP initializers as FP16. Across 50 independent
64-row checks, relative L2 ranged from `2.873e-4` to `5.342e-4`, below the
`2e-3` single-block limit.

The cf4 full-chain A/B used identical initial latents and raw velocity outputs.
Video relative L2 was `2.497e-3` with cosine `0.9999970`; audio relative L2 was
`1.590e-3` with cosine `0.9999986`. Both outputs were finite and passed the
`5e-3` accumulated 50-block gate. Product internal elapsed time was 122.966
seconds versus 107.091 seconds for scaled FP16 (1.148x). Graph time fell from
42.182 to 18.361 seconds, while session build time rose from 59.755 to 70.053
seconds because FP16 storage is larger than the product FP8 storage.

The four-step scaled-FP16 run saved finite checkpoints at 104.944, 206.841,
and 310.024 seconds. It then failed after the fourth `main_head` with exactly
88704 non-finite video velocity values at 412.782 seconds, matching the product
failure boundary and invalid count. Scaled FP16 therefore does not cause or fix
the known fourth-step instability. The hybrid remains unpublished with
`validation_passed=false`; its one-step numerical and performance results are
valid, but full product publication is blocked on the sampling instability.

## 2026-08-14 ORT CPU Thread Gate

CUDA sessions now default to one ORT CPU operator thread with intra/inter-op
spinning disabled. `H3_ORT_CPU_THREADS` and `H3_ORT_ALLOW_SPINNING` retain
explicit overrides. This does not change the ONNX or CUDA compute path.

On the same scaled-FP16 cf4 velocity test, average process CPU fell from
626.87% to 66.68% and peak CPU fell from 657.4% to 79.2%. Internal elapsed time
improved from 107.091 to 91.271 seconds (1.173x), session build time from 70.053
to 57.934 seconds, and graph time from 18.361 to 17.132 seconds. Video and audio
outputs were elementwise identical to the spinning/multithreaded run. Session
construction still consumes 63.5% of internal elapsed time and is the next
measured optimization target.

## 2026-08-14 Fourth-Step Overflow Fix

Graph-output diagnostics resumed directly from the third cf4 checkpoint and
located the first non-finite value at
`main_block_44_attention_output/hidden_states_out`. SDPA remained finite with a
maximum absolute value of 594.85, while the residual stream had grown above
3.5 million. The attention output projection used an unscaled native FP16 Gemm,
which overflowed before its FP32 residual addition.

On the captured block-44 inputs, the existing native FP16 graph reproduced the
non-finite output. The chunked FP32 reference was finite at 36.5 ms warm median.
Scaled FP16 was finite with relative L2 `1.410e-4` and a 24.1 ms warm median.
All 50 attention-output graphs were then exported with the same scale restore
in FP32. Independent relative L2 ranged from `2.521e-4` to `3.301e-4`; the set
occupies 3.754 GiB and passed CUDA and graph-structure validation.

The repaired fourth-step resume stayed finite through block 49 and `main_head`.
The full four-step cf4 run then completed with finite video and audio at
90.141, 180.501, 268.332, and 356.788 seconds. Total internal elapsed was
356.791 seconds; session build was 224.828 seconds and graph execution was
68.474 seconds. Average GPU utilization was only 19.51% with 2224.6 MiB average
VRAM, confirming that session construction and short-kernel gaps now dominate.

## 2026-08-14 Persistent Weight Runtime Gate

Three reusable runtime topologies now cover QKV, scaled Attention output,
and scaled MLP. Per-block external data is bulk-read sequentially on one worker,
kept in a bounded three-graph host window, and supplied as runtime inputs. This
separates ONNX parsing/session construction from layer weights and removes file
page faults from the synchronous H2D path. Buffers are released after each graph
instead of retaining all 50 blocks in process RSS.

The first mmap-only prototype reduced session builds but regressed the one-step
cf4 test from 91.271 to 92.354 seconds and raised RSS to 23.31 GiB. It was
rejected. Sequential prefetch reduced the final one-step result to 59.540
seconds, a 34.77% reduction, with elementwise-identical video and audio. QKV
reuse required all 23 initializers, including block-specific inline scales and
shape constants; converting only the four external tensors failed the numerical
gate and was not retained.

The complete four-step cf4 validation finished in 230.913 seconds versus
356.791 seconds (35.28% reduction). Session builds fell from 618 to 21 and
session-build time from 224.828 to 3.595 seconds. Graph time remained comparable
at 69.382 versus 68.474 seconds. Final video and audio were elementwise
identical; peak RSS was 2.166 GiB, peak VRAM was 2808 MiB, and average GPU
utilization rose from 19.51% to 22.13%. The remaining measured runtime target is
the 81.156 seconds of foreground wait across 156.24 GB of four-step weight
streaming, not ONNX session construction. The pinned-host gate below addresses
that target.

## 2026-08-14 Pinned Host Weight Gate

The three-graph host window now reuses one `cudaHostAllocPortable` buffer for
each persistent topology. The prefetch worker reads directly into those buffers
and ORT copies from pinned memory; allocations are not repeated per block. CUDA
runtime loading prefers the packaged `nvidia-cuda-runtime-cu12` dependency and
falls back to reusable pageable buffers with a recorded reason if unavailable.

At 1190 rows the isolated MLP median fell from about 115.0 ms to 95.8 ms. The
one-step cf4 test fell from 59.540 to 44.726 seconds with elementwise-identical
video and audio. The full four-step run completed at 45.425, 88.459, 132.120,
and 176.272 seconds, a 50.60% reduction from the original 356.791-second path.
Graph time fell to 61.329 seconds and foreground weight wait to 57.969 seconds.
Only three host allocations were made; peak RSS was 2.418 GiB, peak VRAM was
2936 MiB, and average GPU utilization reached 29.88% with a 95% peak.

## 2026-08-14 Parallel Weight Prefetch Gate

The three graph kinds have independent pinned buffers, so their sequential
file reads can overlap without allocating a second buffer per kind. An isolated
2.18 GiB-per-round read test improved from 0.755 GiB/s with one worker to
1.067 GiB/s with three. Real inference showed that three workers caused enough
CPU and storage contention to offset most of that gain; two workers provided
the best balance and are now the default. `H3_WEIGHT_PREFETCH_WORKERS` can
override the bounded 1-3 worker setting.

Two velocity runs completed in 37.975 and 44.003 seconds versus one-worker
controls at 44.726 and 45.719 seconds; all outputs were elementwise identical.
The final four-step validation completed at 37.859, 73.117, 108.708, and
144.496 seconds, with 144.500 seconds total. This is 18.02% faster than the
176.272-second pinned single-worker path and 59.50% faster than the original
356.791-second runtime. Foreground weight wait fell from 57.969 to 12.620
seconds (78.23%). The tradeoff is higher summed concurrent load time and graph
contention: graph time rose from 61.329 to 68.467 seconds and average process
CPU from 116.05% to 160.44%. Peak RSS remained 2.420 GiB, peak VRAM was
2935 MiB, average GPU utilization was 31.46%, and video/audio remained
elementwise identical.

## 2026-08-14 Device SDPA and Weight Upload Experiments

A device-resident ORT SDPA graph kept QKV and attended rows as CUDA
`OrtValue`s and reduced 50 one-step SDPA calls to 1.52-1.69 seconds. Two
velocity runs completed in 36.142 and 39.149 seconds. However, the four-step
output accumulated 5.04% video and 6.47% audio relative L2 error versus the
accepted path, so `H3_DEVICE_SDPA` defaults to off and the implementation is
retained only for further numerical work.

Adaptive device-weight upload was also tested. Admission checks live CUDA free
memory, a 384 MiB safety reserve, in-flight bytes, and a single device slot;
automatic use additionally requires at least 6 GiB total VRAM. ORT-owned
device allocations retained their arena high-water mark and pushed the 4 GiB
GPU to 3913 MiB, regressing one step to 51.709 seconds. Direct
`cudaMalloc`/`cudaMemcpy` buffers released correctly and limited the peak to
3554 MiB, but only 6 of 150 graphs qualified and wall time still regressed to
40.239 seconds. Automatic upload therefore remains disabled on this 4 GiB
device; `H3_DEVICE_WEIGHT_PREFETCH=1` is an explicit experimental override.

The device-SDPA four-step run also exposed a storage-state limit: 156.24 GB of
weights accumulated 406.95 seconds of concurrent load time and 148.00 seconds
of foreground wait, versus 224.06 and 12.62 seconds in the accepted run. Its
235.87-second wall time is not a valid compute comparison. The fixed
three-buffer pinned window held RSS to 2.223 GiB while the SSD stayed busy,
which is expected and confirms that spare host RAM alone does not remove the
sustained storage bottleneck.

## Operator Fusion Direction

ONNX operator fusion is applicable, but it must be measured separately from
weight/session lifetime. The runtime currently uses `ORT_DISABLE_ALL`, so the
first gate is an offline ORT optimized-model A/B on selector-free direct graphs.
After that, prioritize these CUDA fusions:

- RMSNorm + AdaLN scale/shift before QKV and MLP.
- QKV projection + split + Q/K norm + RoPE.
- SwiGLU (`Split + SiLU + Mul`) between the two MLP Gemm operations.
- Attention output projection + gate + FP32 residual add, retaining the new
  overflow-safe scale restoration.

Offline ORT Basic, Extended, and All candidates were measured on the persistent
topologies. Only Extended/All changed a graph, replacing MLP `Sigmoid+Mul` with
`com.microsoft::QuickGelu` and reducing 36 nodes to 33. In a three-round,
24-sample interleaved test all four modes remained within 1% at 116-117 ms, so
the ORT candidates were rejected and are not part of the published model.

ORT built-in fusion may reduce launch count but will not remove the repeated
session build and weight upload gaps. If offline ORT optimization is
insufficient, the next topology should use persistent selector-free graphs with
layer weights supplied as inputs, followed by TensorRT or custom ORT CUDA ops
only for patterns that remain unfused.

A selector-free QKV topology was then isolated and measured. Its median graph
time improved from 147.41 ms to 130.46 ms, and the four-step graph total fell
from 61.329 to 58.191 seconds with elementwise-identical output. However, the
four-step wall time regressed from 176.272 to 179.363 seconds because the extra
disk wait offset the graph saving. The extraction algorithm remains covered for
future fused or resident-weight work, but the published default retains the
validated `Equal + If` QKV wrapper.

## Workspace Cleanup

The validated model was published at
`exported/minimax_h3_fl2va_scaled_fp16_tensor_core_v3`. The obsolete product,
MLP-only hybrid, all-block candidate directories, precision experiments, and
captured diagnostic tensors were removed after the v3 schedule and manifest
were validated. The attempted D: archive copy was cancelled and its partial
directory deleted at the user's request; the source checkpoint remains at
`D:\ckpt\minimax_h3_fl2va_pruned_fp8_scaled.safetensors`.

C: free space increased from 125.69 GiB to 145.02 GiB, releasing approximately
19.33 GiB while retaining the complete 320-file v3 model and all durable JSONL
benchmark logs.

## 2026-08-14 Qwen INT8 Virtual Slices

The Qwen source was changed to
`qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27,141,342,152 bytes).
Text-tower linear weights are signed INT8 with one FP32 scale per output row;
the embedding and normalization tensors remain BF16. Opening the complete file
through the NumPy safetensors backend exhausted Windows commit with error 1455,
so export and runtime now use header offsets and read or mmap one tensor at a
time. The BF16 embedding is gathered by token row on CPU instead of uploading
the complete vocabulary table.

Gate, Up, and Down were fused into one persistent MLP topology. At four tokens,
the three FP16 graphs took a 5.784 ms warm median and the fused graph took
4.961 ms (1.166x) with identical output. The more important streaming gate
promoted weights to topology inputs. INT8 cut MLP input bytes from 786,452,480
to 393,461,760 and improved the accurate streaming median from 106.164 to
85.023 ms (1.249x). Attention input bytes fell from 188,765,184 to 94,454,784
and its median fell from 27.628 to 19.861 ms (1.391x).

An initially faster QDQ graph converted FP32 scales to FP16 before
dequantization. Its single-layer streaming gates reached approximately
1.59-1.61x, but block 24 exceeded the strict numerical gate and its complete
50-layer output differed from the accurate candidate by 1.5036% relative L2.
That candidate is rejected. The accepted topology performs INT8 plus FP32-scale
dequantization followed by an explicit FP16 cast, matching the source export's
weight conversion order.

The published virtual product contains two topology graphs and manifests only;
it is approximately 167 KiB and maps large matrices directly from the original
safetensors file. No 23 GiB text-weight copy is generated. Representative
blocks 0, 24, and 49 passed a 1e-3 source-reference gate. Worst Attention and
MLP relative L2 were 7.20e-4 and 4.27e-4. A complete 50-layer four-token run
finished in 51.042 seconds with finite `[4, 5120]` FP32 output. The prior
persistent product's historical four-token result was 67.2 seconds, so the
measured reduction is 24.0%; the values are reported separately because the
source quantization and storage locations differ.

An attempted current-product comparison on D: was cancelled at layer 24 after
567.7 seconds when the drive was confirmed to be mechanical. It is not a valid
performance result. Full-chain logs show the accepted C:-resident INT8 path is
still storage-bound: median Attention, MLP, and complete-layer spans were
0.183, 0.660, and 0.838 seconds, versus approximately 0.020 and 0.085 seconds
once their weights were cached. Host prefetch must therefore remain bounded by
live memory headroom; pinning the complete text tower is unsafe on the 32 GiB
host.

## 2026-08-14 Persistent Video VAE Boundary

The original decoder uses one session per transformer block and returns every
tile to the host between all 36 blocks. On a representative 1,285-token tile,
plain `session.run` took 71.886 ms and CUDA I/O Binding took 69.436 ms (3.4%
faster) with elementwise-identical output. A relaxed batch-2 graph took 71.760
ms per tile and was rejected because it was slower than resident batch 1.

All 12 block initializers were then promoted to inputs. Blocks 0, 1, and 35
have the same topology fingerprint; each block supplies 134,287,360 bytes of
FP16 weights and the reusable topology is 27,942 bytes. On a C:-local copy,
the original session built in 163.561 ms and ran at a 65.709 ms warm median.
The persistent path uploaded weights in 28.917 ms and ran at 59.503 ms; output
was elementwise identical. The topology session is built once, activations and
rotary tables remain CUDA OrtValues across blocks, and only final hidden tiles
return to the host.

Loading one embedded-weight ONNX file into host arrays took 209.127 ms. This
cost makes a serial single-tile persistent path unattractive. The production
runtime therefore prefetches the next block on one host worker while the
current block processes all tiles. A typical 360p 2x2 tile set provides about
4 x 59.5 ms of GPU work, enough to cover the measured host load. Only current
weights occupy device memory; the next block remains in host memory. The new
path activates only when its topology manifest passes the ready check and can
be disabled with `H3_VIDEO_VAE_PERSISTENT=0`.

## 2026-08-15 GPU Toolchain And Six-Point Gate

The Windows GPU development path is now reproducible with CUDA Toolkit 12.6,
Visual Studio 2022 Community, CuPy 14.1.1, CUTLASS 4.2, and `cuda-python`
12.9.7. Python packages use the Tsinghua mirror. `cuda-python` remains below
version 13 because CUTLASS 4.2 is incompatible with its changed API. CUDA
source, compiler temporary files, and copied CUTLASS headers use
`C:\ProgramData\h3-workbench\cuda-cache`; NVRTC and nvcc fail on parts of the
toolchain when their temporary path contains the current non-ASCII user name.

TensorRT is a separate optional dependency. ONNX Runtime 1.23.2 requires the
TensorRT 10 ABI, so the extra is pinned to 10.9.0.34. TensorRT packages alone
use the explicit NVIDIA package index through the system proxy; all other
Python packages continue to use the Tsinghua mirror. A missing TensorRT runtime
does not block the main GPU extra or silently turn a TensorRT result into a
CUDA result.

The six optimization points reached these gates:

1. Persistent weights use reusable pinned host buffers and a non-blocking CUDA
   upload stream. `cudaMemcpyAsync` is submitted during prefetch, the pinned
   lease remains alive until the consumer synchronizes, and metrics separate
   total upload span from foreground upload wait. Device admission checks live
   free VRAM against the next shard, in-flight bytes, one device slot, and a
   384 MiB reserve. A real 4 GiB run admitted 66 of 150 graphs but increased
   graph time and exposed unsafe interaction with device-resident activations,
   so automatic device upload retains a 6 GiB minimum. It remains available on
   4 GiB only through the explicit `H3_DEVICE_WEIGHT_PREFETCH=1` experiment.
2. QKV, Attention output, and MLP already reuse one persistent topology session
   per graph kind. A caller-owned non-blocking compute stream was accepted by
   ORT and reported `has_user_compute_stream=1`, but a real multi-session
   schedule exited in native code immediately after its first QKV graph. It is
   therefore opt-in through `H3_UNIFIED_CUDA_STREAM=1`, not a default. Combining
   all three graphs into one topology is
   rejected for now: streaming SDPA is an intentional lifetime boundary, and
   a 13,371-token native sequence cannot keep the complete intermediate state
   inside the 4 GiB device without losing the low-memory guarantee.
3. CUDA Graph passed the fixed-shape, stable-device-address Block 00 gate. Five
   warm runs had a 4.09 ms replay median versus 4.72 ms for ordinary CUDA, a
   13.3% reduction. Capture is therefore available to experiments but is not
   enabled on the dynamic production path; changing weight and activation
   pointers invalidate the graph contract.
4. Streaming SDPA selects 512, 384, 256, 128, 64, or 32 query rows from current
   free VRAM before every block. The user value is a hard upper bound, a
   512 MiB device reserve is retained, and selected sizes plus downgrade counts
   are persisted in task metrics.
5. An NVRTC kernel now fuses FP8 E4M3 conversion, scale application, optional
   LoRA `B @ A`, optional transpose, and FP16 output. Base and LoRA-transposed
   smoke tests matched the NumPy FP16 reference elementwise. Ampere compute
   capability 8.6 has no native FP8 Tensor Core path, so the useful design is
   fused software conversion followed by FP16 Tensor Core GEMM. This kernel is
   an experimental adapter primitive until a source FP8 topology needs it; the
   current published main model already stores its accepted scaled weights as
   FP16.
6. A native CUTLASS FP16 TensorOp DLL is compiled and hash-cached through nvcc
   and Visual Studio. The GA107-safe kernel uses a 64x64x32 tile and two stages
   to remain below the laptop GPU's per-block shared-memory limit. A
   128x256x128 GEMM matched the FP32-accumulated FP16 reference with maximum
   absolute difference `2.4414e-4`. The backend benchmark rejects ORT CPU
   fallback and measures TensorRT FP16 and FP32 engines separately.

For the validated scaled-FP16 Block 00, ordinary CUDA and CUDA Graph both had
relative L2 `2.867e-4` against the stored reference and finite output. The
large maximum absolute error (`104.90625`) is attached to very large activation
magnitudes and is not a CUDA Graph difference; both backends produced the same
error metrics. End-to-end promotion still requires a same-task A/B because the
single-block 13.3% graph replay gain does not include storage and weight upload
stalls.

TensorRT 10.9 was then measured on the same Block 00. The FP16 engine produced
non-finite output and failed the numerical gate. The FP32 engine was finite,
but relative L2 rose to `2.613e-3`, versus `2.867e-4` for CUDA, and its 4.68 ms
warm median only narrowly beat ordinary CUDA while remaining slower than the
4.23 ms CUDA Graph replay. TensorRT session plus first-run engine construction
took about 40 seconds and sampled approximately 1.37 GiB VRAM, around 430 MiB
above the CUDA sessions. TensorRT therefore remains an installed, optional
benchmark backend and is rejected for the production main-model path.

The same cf4 one-step benchmark also rejected two aggressive 4 GiB runtime
combinations. A unified caller stream exited in native code after the first QKV
graph. Retaining all three persistent topology sessions and device activations
then exited at the Attention Output to MLP handoff, with and without device
weight upload. The accepted policy separates session and activation lifetime:
sequences up to 2,048 tokens retain the three topology sessions while returning
block activations through host memory; larger sequences recycle sessions to
preserve the long-context VRAM reserve. The cf4 one-step result was 38.770 s,
with elementwise-identical video/audio, 2,878 MiB peak VRAM, three persistent
session builds, and 18.232 s graph time. This is 11.9% faster than the 44.003 s
historical repeat and within 2.1% of the 37.975 s best run. Device activations,
unified streams, and automatic 4 GiB weight upload remain disabled by default.

A final validation with no experimental environment overrides confirmed the
adaptive policy as the production default. It completed in 41.833 s with
finite, elementwise-identical video and audio output, 2,878 MiB peak VRAM,
three persistent session builds, 18.838 s graph time, 2.235 s foreground
weight wait, and 17.014 s streaming SDPA time. This is 4.9% faster than the
44.003 s historical repeat, 10.2% slower than the 37.975 s historical best,
54.2% faster than the 91.271 s CPU-optimized run, and 60.9% faster than the
107.091 s early scaled-FP16 path. The retained-session experiment above shows
the remaining short-sequence opportunity; the accepted default restores most
of the regression without weakening long-sequence VRAM safety.

## 2026-08-15 CUDA Torch SDPA Gate

The CUDA PyTorch wheel was installed as `torch 2.10.0+cu126` and the schedule
runtime was given an explicit `H3_SDPA_BACKEND=auto|torch|ort` selector. The
historical `36d8d47f53d7` geometry was replayed with the published scaled-FP16
model: 352x360, 17 requested frames, segmented mode, 1,190 sequence tokens,
seed 1, four steps, and query chunk 256. The benchmark used zero text states
and excluded Qwen, Video VAE, audio VAE, and MP4 encoding.

The one-step gate selected the requested backend and kept both outputs finite.
ORT completed in 42.725 s with 17.968 s of SDPA; Torch completed in 44.852 s
with 20.506 s of SDPA. A reverse-order repeat kept Torch at 44.803 s average,
while ORT averaged 47.294 s because one run incurred a storage/runtime stall.
Torch used about 2,607 MiB peak VRAM versus ORT's 2,877 MiB, but its one-step
video/audio relative L2 error against ORT was 2.106e-3 / 1.308e-3.

The complete four-step gate is the production decision: ORT finished in
170.399 s with 82.411 s of SDPA, while Torch finished in 185.743 s with
88.589 s of SDPA. Both remained finite, but final Torch video/audio relative
L2 error reached 4.370e-2 / 2.765e-2, increasing at each checkpoint. On the
4 GiB RTX 3050, `auto` therefore keeps ORT for low-VRAM devices; Torch remains
available as an explicit experimental override and as the automatic candidate
for higher-VRAM devices. The measured peak VRAM was 2,880 MiB for ORT and
2,633 MiB for Torch.

## 2026-08-15 Host RAM Working-Set Gate

The long Video VAE decode trace showed that the decoder's 36 embedded-weight
blocks were being read one block at a time. The complete validated working set
is 4.502 GiB on the current product, while the host reported about 23 GiB of
available memory. The decoder now preloads all blocks into host RAM when the
working set fits with a 6 GiB system reserve, then reuses those arrays for every
CUDA tile. `H3_VIDEO_VAE_RAM_CACHE=0` restores streaming and
`H3_VIDEO_VAE_RAM_CACHE=1` forces the cache for an explicit experiment.

The main FL2VA ONNX product is 37.93 GiB and cannot be made fully resident on
the current 32 GiB host without paging. Its bounded persistent-weight and L2
read-ahead paths therefore remain memory-budgeted; only a full-working-set
cache is promoted automatically on hosts where it genuinely fits.

## 2026-08-15 Task-Level Host Weight Hot Set

The previous L2 path warmed files in the Windows page cache, but persistent
weights were still copied from those files into a temporary host buffer on each
sampling step. The new CUDA path plans a stable host-RAM hot set when the
persistent topology is created. It loads selected QKV, Attention output, and
MLP arrays once in the background, reuses them across denoising steps, and
keeps the same runtime alive across segmented clips in one task. The cache is
host-only on purpose; the existing VRAM admission policy still controls device
weight upload.

Auto mode budgets current available RAM minus a 12 GiB safety reserve and a
2 GiB L2 reserve. It caches complete weight entries in schedule order until
the budget is consumed, so the 32 GiB workstation gets a bounded hot set while
64 GiB-class hosts can approach the full 37.93 GiB working set. `H3_WEIGHT_RAM_CACHE=0`
disables the feature and `H3_WEIGHT_RAM_CACHE_GIB` caps its budget. The runtime
metrics now report candidate/resident bytes, resident entries, cache hits, and
load seconds.

L2 read-ahead now has a resizable budget. When host available memory drops, it
evicts the oldest mmap entries and reports budget adjustments and pressure
evictions. When the weight hot set is active, automatic L2 read-ahead is capped
at 2 GiB so page warming cannot consume the memory reserved for ORT sessions,
activations, and the next uncached graph. This addresses the observed failure
mode where SSD activity became low but available RAM approached zero and GPU
gaps grew instead of shrinking.
