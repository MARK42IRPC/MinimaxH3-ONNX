from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import quote, urlencode


OFFICIAL_REPO = "Comfy-Org/MiniMax-H3"
TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
SUPPORT_REPO = "Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS"
TOKENIZER_REPO = SUPPORT_REPO


def _hugging_face_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{quote(path, safe='/')}?download=true"


def _modelscope_url(repo_id: str, path: str) -> str:
    query = urlencode({"Revision": "master", "FilePath": path})
    return f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo?{query}"


@dataclass(frozen=True)
class SourceAsset:
    repo_id: str
    path: str
    size_bytes: int
    url: str
    role: str = "checkpoint"


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
    extra_sources: tuple[SourceAsset, ...] = ()
    product_type: str = "onnx_model"
    depends_on: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[SourceAsset, ...]:
        return (
            self.source,
            *(item for item in (self.lora, self.support) if item is not None),
            *self.extra_sources,
        )

    @property
    def download_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.sources)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["download_size_bytes"] = self.download_size_bytes
        result["required_space_bytes"] = self.download_size_bytes + self.output_size_bytes
        return result


EXPORT_PRESETS: tuple[ExportPreset, ...] = (
    ExportPreset(
        "tokenizer",
        "Qwen VL Tokenizer",
        "tokenizer",
        SourceAsset(
            TOKENIZER_REPO,
            "qwen_tokenizer/tokenizer.json",
            7032403,
            _modelscope_url(TOKENIZER_REPO, "qwen_tokenizer/tokenizer.json"),
            role="tokenizer",
        ),
        "qwen_tokenizer",
        11492078,
        "提示词编码所需的四个原始 Tokenizer 文件，无模型转换步骤",
        extra_sources=(
            SourceAsset(
                TOKENIZER_REPO,
                "qwen_tokenizer/tokenizer_config.json",
                11003,
                _modelscope_url(TOKENIZER_REPO, "qwen_tokenizer/tokenizer_config.json"),
                role="tokenizer",
            ),
            SourceAsset(
                TOKENIZER_REPO,
                "qwen_tokenizer/vocab.json",
                2776833,
                _modelscope_url(TOKENIZER_REPO, "qwen_tokenizer/vocab.json"),
                role="tokenizer",
            ),
            SourceAsset(
                TOKENIZER_REPO,
                "qwen_tokenizer/merges.txt",
                1671839,
                _modelscope_url(TOKENIZER_REPO, "qwen_tokenizer/merges.txt"),
                role="tokenizer",
            ),
        ),
    ),
    ExportPreset(
        "audio_vae",
        "Audio VAE",
        "audio_vae",
        SourceAsset(
            OFFICIAL_REPO,
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            605254808,
            _modelscope_url(OFFICIAL_REPO, "vae/minimax_h3_audio_vae_fp32.safetensors"),
        ),
        "onnx_models/audio_vae",
        261860256,
        "FP32 Audio VAE 单图解码器",
    ),
    ExportPreset(
        "video_vae",
        "Video VAE",
        "video_vae",
        SourceAsset(
            OFFICIAL_REPO,
            "vae/minimax_h3_video_vae_fp16.safetensors",
            5207808496,
            _modelscope_url(OFFICIAL_REPO, "vae/minimax_h3_video_vae_fp16.safetensors"),
        ),
        "onnx_models/video_vae",
        4982896191,
        "36 Block 切片、全块验证与持久解码拓扑",
    ),
    ExportPreset(
        "qwen",
        "Qwen VL 编码器",
        "text_encoder",
        SourceAsset(
            OFFICIAL_REPO,
            "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            27141342152,
            _modelscope_url(OFFICIAL_REPO, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
        ),
        "onnx_models/qwen3vl_32b_minimax_h3_int8_virtual",
        200000,
        "INT8 原权重零拷贝虚拟切片，代表层与 50 层全链校验",
    ),
    ExportPreset(
        "fl2va_streaming",
        "FL2VA 流式基座",
        "fl2va_transformer",
        SourceAsset(
            OFFICIAL_REPO,
            "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            20958205608,
            _modelscope_url(OFFICIAL_REPO, "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
        ),
        "onnx_models/minimax_h3_fl2va_pruned_fp8_scaled_streaming",
        40764106323,
        "实测成功的 pruned FP8 scaled 基座与流式注意力",
    ),
    ExportPreset(
        "fl2va_turbo_v4",
        "Turbo v4 动态 LoRA",
        "acceleration_lora",
        SourceAsset(
            TURBO_REPO,
            "minimax_h3_turbo_v4_step600_ema.safetensors",
            779849816,
            _hugging_face_url(TURBO_REPO, "minimax_h3_turbo_v4_step600_ema.safetensors"),
            role="lora",
        ),
        ".h3-workbench/accelerators/turbo_v4",
        1048576,
        "运行时叠加 v4 step600 EMA 与 SiLU 时间步网格，完整覆盖 259/259 LoRA，推荐 4-8 步；不会生成合并的 40GB 主模型",
        support=SourceAsset(
            SUPPORT_REPO,
            "export_support/h3_silu_temb_grid.safetensors",
            5510600,
            _modelscope_url(SUPPORT_REPO, "export_support/h3_silu_temb_grid.safetensors"),
            role="silu_timestep_grid",
        ),
        product_type="runtime_adapter",
        depends_on=("fl2va_streaming",),
    ),
)

_BY_ID = {preset.id: preset for preset in EXPORT_PRESETS}


def export_preset(preset_id: str) -> ExportPreset:
    try:
        return _BY_ID[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown export preset: {preset_id}") from exc
