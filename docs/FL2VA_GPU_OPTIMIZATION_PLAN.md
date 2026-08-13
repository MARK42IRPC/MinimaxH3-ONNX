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
