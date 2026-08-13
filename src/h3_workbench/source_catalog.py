from __future__ import annotations

from dataclasses import asdict, dataclass


OFFICIAL_REPO = "Comfy-Org/MiniMax-H3"
TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
SUPPORT_REPO = "Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS"


@dataclass(frozen=True)
class SourceAsset:
    repo_id: str
    path: str
    size_bytes: int


@dataclass(frozen=True)
class ExportPreset:
    id: str
    label: str
    component: str
    source: SourceAsset
    output_dir: str
    output_size_bytes: int
    description: str
    blocks: str = "all"
    lora: SourceAsset | None = None
    support: SourceAsset | None = None

    @property
    def download_size_bytes(self) -> int:
        return self.source.size_bytes + (self.lora.size_bytes if self.lora else 0) + (self.support.size_bytes if self.support else 0)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["download_size_bytes"] = self.download_size_bytes
        result["required_space_bytes"] = self.download_size_bytes + self.output_size_bytes
        return result


EXPORT_PRESETS: tuple[ExportPreset, ...] = (
    ExportPreset("audio_vae", "Audio VAE", "audio_vae", SourceAsset(OFFICIAL_REPO, "vae/minimax_h3_audio_vae_fp32.safetensors", 605254808), "onnx_models/audio_vae", 261860256, "FP32 Audio VAE 单图解码器"),
    ExportPreset("video_vae", "Video VAE", "video_vae", SourceAsset(OFFICIAL_REPO, "vae/minimax_h3_video_vae_fp16.safetensors", 5207808496), "onnx_models/video_vae", 4982896191, "已修正 split-half RoPE 的 36 Block Video VAE"),
    ExportPreset("qwen", "Qwen 文本编码器", "text_encoder", SourceAsset(OFFICIAL_REPO, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15687142551), "onnx_models/qwen3vl_32b_minimax_h3_nvfp4_awq", 49556923492, "实测 NVFP4 AWQ 文本塔，导出 GPU 原生 FP16 GEMM 图"),
    ExportPreset("fl2va_streaming", "FL2VA 流式基座", "fl2va_transformer", SourceAsset(OFFICIAL_REPO, "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors", 20958205608), "onnx_models/minimax_h3_fl2va_pruned_fp8_scaled_streaming", 40764106323, "实测成功的 pruned FP8 scaled 基座与流式注意力"),
    ExportPreset(
        "fl2va_turbo_v4",
        "FL2VA Turbo v4",
        "fl2va_transformer",
        SourceAsset(OFFICIAL_REPO, "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors", 20958205608),
        "onnx_models/minimax_h3_fl2va_pruned_fp8_scaled_accelerated",
        41732628275,
        "实测成功的 v4 step600 EMA，完整 259/259 LoRA 覆盖，推荐 4-8 步",
        lora=SourceAsset(TURBO_REPO, "minimax_h3_turbo_v4_step600_ema.safetensors", 779849816),
        support=SourceAsset(SUPPORT_REPO, "export_support/h3_silu_temb_grid.safetensors", 5510600),
    ),
)

_BY_ID = {preset.id: preset for preset in EXPORT_PRESETS}


def export_preset(preset_id: str) -> ExportPreset:
    try:
        return _BY_ID[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown export preset: {preset_id}") from exc
