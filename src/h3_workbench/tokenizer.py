from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

_REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def tokenizer_files_ready(directory: Path) -> bool:
    return all((directory / name).is_file() for name in _REQUIRED_FILES)


@lru_cache(maxsize=2)
def _load_tokenizer(directory: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(directory, local_files_only=True)


def load_tokenizer(directory: Path) -> AutoTokenizer:
    """Load the tokenizer object used to build multimodal presentations."""
    if not tokenizer_files_ready(directory):
        raise FileNotFoundError(f"Incomplete H3 tokenizer directory: {directory}")
    return _load_tokenizer(str(directory.resolve()))


def encode_prompt(prompt: str, directory: Path, max_tokens: int = 192) -> np.ndarray:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must not be empty")
    tokenizer = load_tokenizer(directory)
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
    return np.asarray(token_ids, dtype=np.int64)
