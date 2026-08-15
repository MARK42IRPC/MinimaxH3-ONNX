from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace


def video_vae_output_frames(latent_frames: int) -> int:
    """Mirror the H3 Video VAE temporal decoder's output-shape planning."""
    if latent_frames < 1:
        raise ValueError("latent_frames must be positive")
    if latent_frames == 1:
        return 1

    token_drop = 3
    tokens_per_chunk = 5
    temporal_ratio = 4
    pseudo_total = latent_frames + token_drop
    pad_tokens = (-pseudo_total) % tokens_per_chunk
    pseudo_total += pad_tokens
    num_chunks = pseudo_total // tokens_per_chunk - 1
    if num_chunks < 1:
        pad_tokens += tokens_per_chunk
        num_chunks += 1

    # Every seven-token decode contributes 17 main frames. The final five
    # overlap frames are emitted once after the last chunk.
    total_frames = num_chunks * 17 + 5
    padded_length = latent_frames + pad_tokens
    before_padding = padded_length - pad_tokens
    for index in range(pad_tokens):
        total_frames -= (
            1 if (before_padding + index) % tokens_per_chunk == 0 else temporal_ratio
        )
    return total_frames


def video_latent_frames_for_output(output_frames: int) -> int:
    """Return the next native H3 latent length (2 + 5k) covering the request."""
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    if output_frames <= 5:
        return 2
    chunks = math.ceil((output_frames - 5) / 17)
    return chunks * 5 + 2


@dataclass(frozen=True)
class GenerationProfile:
    id: str
    label: str
    output_width: int
    output_height: int
    padded_width: int
    padded_height: int
    frames: int
    fps: int
    video_latent_frames: int
    text_tokens: int
    audio_latents_per_second: int = 40
    spatial_vae_ratio: int = 16
    video_patch_size: int = 2
    main_attention_heads: int = 56

    @property
    def video_latent_height(self) -> int:
        return self.padded_height // self.spatial_vae_ratio

    @property
    def video_latent_width(self) -> int:
        return self.padded_width // self.spatial_vae_ratio

    @property
    def audio_latent_frames(self) -> int:
        return math.ceil(self.frames * self.audio_latents_per_second / self.fps)

    @property
    def video_tokens(self) -> int:
        return (
            self.video_latent_frames
            * (self.video_latent_height // self.video_patch_size)
            * (self.video_latent_width // self.video_patch_size)
        )

    @property
    def audio_tokens(self) -> int:
        return self.audio_latent_frames * 2

    @property
    def sequence_tokens(self) -> int:
        return self.text_tokens + self.video_tokens + self.audio_tokens

    @property
    def attention_workspace_bytes(self) -> int:
        # FP32 scores and probabilities. Q/K/V and residual buffers are covered
        # by the planner's fixed activation reserve.
        return self.main_attention_heads * self.sequence_tokens**2 * 4 * 2

    def resized(self, output_width: int, output_height: int) -> "GenerationProfile":
        padded_width = math.ceil(output_width / 32) * 32
        padded_height = math.ceil(output_height / 32) * 32
        return replace(
            self,
            output_width=output_width,
            output_height=output_height,
            padded_width=padded_width,
            padded_height=padded_height,
        )

    def with_frame_count(self, frames: int) -> "GenerationProfile":
        latent_frames = video_latent_frames_for_output(frames)
        native_frames = video_vae_output_frames(latent_frames)
        return replace(self, frames=native_frames, video_latent_frames=latent_frames)

    def to_dict(self) -> dict[str, int | str]:
        result = asdict(self)
        result.update(
            {
                "video_latent_height": self.video_latent_height,
                "video_latent_width": self.video_latent_width,
                "audio_latent_frames": self.audio_latent_frames,
                "video_tokens": self.video_tokens,
                "audio_tokens": self.audio_tokens,
                "sequence_tokens": self.sequence_tokens,
                "attention_workspace_bytes": self.attention_workspace_bytes,
            }
        )
        return result


PROFILE_360P_17F = GenerationProfile(
    id="360p-17f",
    label="360p / 22 native frames",
    output_width=640,
    output_height=360,
    padded_width=640,
    padded_height=384,
    frames=22,
    fps=24,
    video_latent_frames=7,
    text_tokens=192,
)

GENERATION_PROFILES = {PROFILE_360P_17F.id: PROFILE_360P_17F}
