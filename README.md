# MiniMax H3 Edge Workbench

Local model inventory, staged ONNX export, numerical validation, and a lightweight WebUI for MiniMax H3 edge experiments.

Companion ONNX model package: [ModelScope](https://www.modelscope.cn/models/Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS). The WebUI can inspect installed components and enqueue selected ModelScope downloads. This GitHub repository intentionally does not contain the multi-hundred-gigabyte ONNX artifacts.

The current exporter supports the Comfy-Org H3 VAE and FL2VA Transformer checkpoints:

- `minimax_h3_audio_vae_fp32.safetensors`
- `minimax_h3_video_vae_fp16.safetensors`
- `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`

Video VAE export is deliberately sharded into a CNN encoder, decoder prelude, individual Transformer blocks, and a decoder head. This keeps ONNX Runtime from loading the full 4.85 GB decoder at once.

The FL2VA Transformer is split into input projections, two token-refiner Attention/MLP pairs, conditioning, 50 DiT QKV/output/MLP groups, and a final video/audio head. FP8 weights are dequantized to FP16 storage and linear operations are streamed in FP32 to avoid FP16 overflow on later blocks. Main attention is exact full-sequence attention: ONNX handles QKV and output projections, while runtime SDPA processes query chunks without materializing the full attention matrix.

## Start

On Windows, run `install.bat` once. It creates the Python 3.11 environment and installs the locked dependencies, including the official PyTorch CUDA 12.6 wheel. The installer detects a usable NVIDIA device with `nvidia-smi` and adds the optional `gpu` extra (`cuda-python`, CuPy, and CUTLASS) automatically. Set `H3_INSTALL_GPU=0` before running it to skip those optional tools, or `H3_INSTALL_GPU=1` to force them. Installation ends with Torch CUDA and ONNX Runtime provider checks; ONNX Runtime remains the primary sharded graph backend and `CPUExecutionProvider` remains available as a fallback.

The WebUI's `切片` page can reproduce the validated package directly from ModelScope. It exposes only the tested Comfy-Org source variants. The Turbo v4 preset also downloads Larryvrh's `minimax_h3_turbo_v4_step600_ema.safetensors` and the validated `h3_silu_temb_grid.safetensors` support file before exporting the complete 259/259 LoRA path.

```powershell
uv sync --locked --extra dev --no-editable
uv run h3-workbench --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

On Windows, double-click `start_webui.bat` from the repository root to use the existing `.venv` directly.

The workbench scans the current directory by default. Override it with `--workspace PATH`.

## CLI export

```powershell
uv run python -m h3_workbench.exporter inspect .
uv run python -m h3_workbench.exporter export minimax_h3_audio_vae_fp32.safetensors --output onnx_models/audio_vae
uv run python -m h3_workbench.exporter export minimax_h3_video_vae_fp16.safetensors --output onnx_models/video --video-blocks all
uv run python -m h3_workbench.exporter export minimax_h3_fl2va_pruned_fp8_scaled.safetensors --output onnx_models/main --main-blocks all
```

Use `--video-blocks 0` for a quick block-level smoke export before committing several gigabytes of output.

## 360p inference

The edge runtime supports two temporal modes at 24 fps on a 640x384 internal canvas, then center-crops to 640x360. `segmented` reuses the validated 17-frame profile (`[1, 24, 2, 24, 40]`) and concatenates edge-safe clips. `native` builds one long temporal latent profile (15 seconds at 360p is 104 video latents, 26,352 total transformer tokens) and keeps full-sequence attention. The WebUI exposes durations up to 15 seconds, manual 1-50 sampling steps, and the SDPA query chunk (`32/64/128/256/512`).

```powershell
uv run h3-infer plan
uv run h3-infer generate --token-ids 1,42,1000,151935 --steps 6 --output output.mp4
```

The runtime probes free VRAM before every denoising step and streams the validated Base shards within the available budget. Larryvrh Turbo v4 is installed as a runtime adapter over that Base product; 4 steps is the default and 4-8 is its supported quality range. The WebUI exposes a manual `加载 Turbo v4 加速 LoRA` switch, which is off by default; step count no longer silently selects the adapter. When enabled, the request must use 4-8 steps and the validated adapter must be ready. The workbench does not export or retain a second merged 40GB main model.

Device support, GPU selection, VRAM tiers, and the first compatibility phase are documented in [docs/DEVICE_COMPATIBILITY_AND_PHASE1.md](docs/DEVICE_COMPATIBILITY_AND_PHASE1.md). Set `H3_CUDA_DEVICE` to a GPU index, UUID, or `auto` when more than one NVIDIA GPU is visible.

The adapter publishes six graph-only ONNX topologies and keeps all 259 LoRA pairs dynamic. Backbone/refiner factors run as FP16 low-rank GEMMs, while 50 DiT AdaLN pairs and the final head pair use FP32 factors with the author's 1025-row full-width SiLU timestep grid. The adapter manifest locks the Base topology, LoRA, and grid identities by SHA-256; no Base weights are copied into the adapter directory.

The WebUI accepts a natural-language prompt and dynamically pads requested output dimensions to multiples of 32. The current edge profile accepts output widths and heights from 128 to 1024 pixels while retaining the 17-frame temporal profile. Main-model sequence length, position IDs, attention memory planning, Video VAE tile layout, and final center crop are derived from the requested dimensions.

Each completed MP4 embeds a compact JSON record in its container comment and is accompanied by a same-name `.metadata.json` sidecar. The sidecar preserves the complete prompt and token IDs, seed, steps, dual video/audio schedule, output geometry, temporal mode, active model/LoRA manifest identity, runtime cache settings, audio status, and performance-log path for reproducible A/B tests.

Prompt tokenization uses the MiniMax H3 tokenizer files under `qwen_tokenizer/`:

```text
qwen_tokenizer/
├── merges.txt
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

The H3 tokenizer configuration must be used instead of a generic Qwen3-VL tokenizer because it defines H3-specific tokens. Raw Token IDs remain available under the WebUI's advanced input.

Video VAE decoding uses reusable 256-pixel ONNX tiles. The decoder prelude and head remain resident only while needed, and each of the 36 dynamic-sequence Transformer blocks is loaded once per tile batch through ONNX Runtime CUDA. When the complete block working set fits in host memory, all 36 blocks are preloaded into RAM before the first CUDA tile; set `H3_VIDEO_VAE_RAM_CACHE=0` to compare streaming or `H3_VIDEO_VAE_RAM_CACHE=1` to force the cache. Auto mode preserves 6 GiB for the rest of the system. The PyTorch CPU decoder remains as a fallback.

## Design limits

- Video VAE ONNX uses a fixed 256-pixel tile profile with dynamic output resolution assembled by the runtime. H3 durations are computed on the native `17k+5` frame grid: 2 latent tokens decode to 5 frames, 7 decode to 22, and a 5-second request runs 37 latent tokens / 124 native frames before trimming to 120 frames.
- Validation uses CPUExecutionProvider so it also works when ONNX Runtime CUDA is unavailable.
- Qwen3-VL text layers export as INT8 embedding plus per-layer Attention/Gate/Up/Down ONNX shards. The vision tower is not yet part of the 360p text-only path.
- Segmented mode uses independent edge-safe clips and can jump at boundaries because first-frame continuation is not wired yet.
- Native mode uses streaming QKV/SDPA attention. Long temporal Video VAE decoding currently falls back to the PyTorch temporal chunker; use it for staged 1-step memory/time measurements before attempting 5-8 step production renders.
- `H3_SDPA_BACKEND=auto|torch|ort` selects the streaming-attention backend. `auto` keeps ORT on `low_vram` devices (at most 6 GiB) after the RTX 3050 numerical/performance gate; high-VRAM CUDA devices may use Torch SDPA automatically. Force `torch` only after a same-shape numerical gate.
- Main-model shards use a three-level pipeline: ONNX files on SSD, bounded asynchronous mmap read-ahead in system RAM, and dependency-safe dynamic session batches in VRAM. The L1 planner reserves activation and streaming K/V memory before loading 1-3 adjacent graphs; every QKV boundary forces a release before SDPA.
- `H3_L2_CACHE_GIB` overrides the RAM read-ahead budget (default: available RAM minus an 8 GiB reserve, clamped to 1-12 GiB). `H3_PREFETCH_SHARDS` controls the look-ahead depth (default: 16 graphs). Set `H3_L2_CACHE_GIB=0` to disable read-ahead for comparison.
- CUDA jobs also build a task-level host-RAM weight hot set before denoising. Auto mode caches as many persistent QKV, Attention output, and MLP weights as fit after a 12 GiB safety reserve plus a 2 GiB L2 reserve; all task segments reuse that set. Set `H3_WEIGHT_RAM_CACHE=0` to disable it, or use `H3_WEIGHT_RAM_CACHE_GIB` to cap it. `H3_WEIGHT_RAM_WORKERS` controls background loading. The complete 37.93 GiB main product only becomes fully resident on hosts with enough free RAM; a 32 GiB host receives a bounded hot set and streams the remainder.
- The root launcher leaves CPU affinity and process priority on automatic OS defaults. Set `H3_CPU_AFFINITY` to a hexadecimal mask or `H3_CPU_PRIORITY=AboveNormal` only after measuring a specific host. GPUs with at most 6 GiB VRAM load one CUDA session at a time; larger GPUs may retain up to three dependency-safe sessions.
- The vendored VAE architecture is adapted from ComfyUI's MiniMax H3 implementation. See `THIRD_PARTY_NOTICES.md`.
