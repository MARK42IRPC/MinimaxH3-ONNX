from h3_workbench.source_catalog import EXPORT_PRESETS, OFFICIAL_REPO, SUPPORT_REPO


def test_large_checkpoint_sources_use_official_modelscope() -> None:
    large_assets = {
        preset.source
        for preset in EXPORT_PRESETS
        if preset.id in {"audio_vae", "video_vae", "qwen", "fl2va_streaming"}
    }

    assert large_assets
    assert all(asset.repo_id == OFFICIAL_REPO for asset in large_assets)
    assert all(
        asset.url.startswith("https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master/")
        for asset in large_assets
    )


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
    assert all(asset.url.startswith("https://www.modelscope.cn/models/") for asset in assets)

    turbo = next(preset for preset in EXPORT_PRESETS if preset.id == "fl2va_turbo_v4")
    assert turbo.support is not None
    assert turbo.support.repo_id == SUPPORT_REPO
    assert turbo.support.path == "export_support/h3_silu_temb_grid.safetensors"
