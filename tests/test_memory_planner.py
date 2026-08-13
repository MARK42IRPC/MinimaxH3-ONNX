from h3_workbench.memory_planner import (
    MIB,
    halve_batch,
    plan_shard_batches,
    plan_streaming_shard_batches,
    streaming_kv_bytes,
)
from h3_workbench.profiles import PROFILE_360P_17F


def test_360p_profile_dimensions_and_tokens() -> None:
    profile = PROFILE_360P_17F
    assert (profile.video_latent_frames, profile.video_latent_height, profile.video_latent_width) == (7, 24, 40)
    assert profile.video_tokens == 1680
    assert profile.audio_latent_frames == 37
    assert profile.sequence_tokens == 1946


def test_shards_are_grouped_within_vram_budget() -> None:
    shards = [(f"block_{index}.onnx", 400 * MIB) for index in range(6)]
    batches = plan_shard_batches(
        shards,
        PROFILE_360P_17F,
        free_bytes=3500 * MIB,
        fixed_reserve_bytes=768 * MIB,
        weight_factor=1.2,
    )
    assert [len(batch.shards) for batch in batches] == [2, 2, 2]
    assert [item.graph for item in batches[0].shards] == ["block_0.onnx", "block_1.onnx"]


def test_batch_can_be_halved_after_provider_oom() -> None:
    shards = [(f"block_{index}.onnx", 100 * MIB) for index in range(5)]
    batch = plan_shard_batches(shards, PROFILE_360P_17F, free_bytes=8 * 1024 * MIB)[0]
    left, right = halve_batch(batch)
    assert len(left.shards) == 2
    assert right is not None
    assert len(right.shards) == 3


def test_streaming_batches_end_at_qkv_barriers_and_fit_available_vram() -> None:
    shards = [
        ("main_block_00_attention_qkv.onnx", 220 * MIB),
        ("main_block_00_attention_output.onnx", 80 * MIB),
        ("main_block_00_mlp.onnx", 440 * MIB),
        ("main_block_01_attention_qkv.onnx", 220 * MIB),
        ("main_block_01_attention_output.onnx", 80 * MIB),
        ("main_block_01_mlp.onnx", 440 * MIB),
    ]
    batches = plan_streaming_shard_batches(shards, PROFILE_360P_17F, free_bytes=3500 * MIB)

    assert [len(batch.shards) for batch in batches] == [1, 1, 2, 1, 1]
    assert all(
        len(batch.shards) == 1
        for batch in batches
        if batch.shards[0].graph.endswith("_attention_output.onnx")
    )
    assert streaming_kv_bytes(PROFILE_360P_17F) == PROFILE_360P_17F.sequence_tokens * 56 * 128 * 4


def test_streaming_batches_shrink_when_vram_is_tight() -> None:
    shards = [
        ("main_block_00_attention_qkv.onnx", 220 * MIB),
        ("main_block_00_attention_output.onnx", 80 * MIB),
        ("main_block_00_mlp.onnx", 440 * MIB),
    ]

    batches = plan_streaming_shard_batches(shards, PROFILE_360P_17F, free_bytes=1100 * MIB)

    assert [len(batch.shards) for batch in batches] == [1, 1, 1]
