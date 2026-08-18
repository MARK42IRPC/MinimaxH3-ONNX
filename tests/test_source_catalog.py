from h3_workbench.source_catalog import EXPORT_PRESETS, OFFICIAL_REPO, SUPPORT_REPO
from urllib.parse import parse_qs, urlparse


def test_large_checkpoint_sources_use_official_modelscope() -> None:
    large_assets = {
        preset.source
        for preset in EXPORT_PRESETS
        if preset.id in {"audio_vae", "video_vae", "qwen", "ref2va", "fl2va_streaming"}
    }

    assert large_assets
    assert all(asset.repo_id == OFFICIAL_REPO for asset in large_assets)
    for asset in large_assets:
        parsed = urlparse(asset.url)
        assert parsed.path == "/api/v1/models/Comfy-Org/MiniMax-H3/repo"
        assert parse_qs(parsed.query) == {"Revision": ["master"], "FilePath": [asset.path]}


def test_ref2va_is_the_required_main_generation_base() -> None:
    ref2va = next(preset for preset in EXPORT_PRESETS if preset.id == "ref2va")
    fl2va = next(preset for preset in EXPORT_PRESETS if preset.id == "fl2va_streaming")

    assert ref2va.component == "ref2va_transformer"
    assert ref2va.required_for_generation is True
    assert ref2va.source.size_bytes == 40225724176
    assert ref2va.output_dir.endswith("minimax_h3_ref2va_pruned_bf16_virtual")
    assert fl2va.required_for_generation is False


def test_small_support_assets_use_private_modelscope_collection() -> None:
    tokenizer = next(preset for preset in EXPORT_PRESETS if preset.id == "tokenizer")
    assets = tokenizer.sources

    assert len(assets) == 4
    assert all(asset.repo_id == SUPPORT_REPO for asset in assets)
    assert {asset.path for asset in assets} == {
        "qwen_tokenizer/merges.txt",
        "qwen_tokenizer/tokenizer.json",
        "qwen_tokenizer/tokenizer_config.json",
        "qwen_tokenizer/vocab.json",
    }
    for asset in assets:
        parsed = urlparse(asset.url)
        assert parsed.path == "/api/v1/models/Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS/repo"
        assert parse_qs(parsed.query) == {"Revision": ["master"], "FilePath": [asset.path]}

    turbo = next(preset for preset in EXPORT_PRESETS if preset.id == "fl2va_turbo_v4")
    assert turbo.support is not None
    assert turbo.support.repo_id == SUPPORT_REPO
    assert turbo.support.path == "export_support/h3_silu_temb_grid.safetensors"
