from pathlib import Path

import numpy as np

import h3_workbench.tokenizer as tokenizer_module
from h3_workbench.tokenizer import encode_prompt, tokenizer_files_ready


class _FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": [151644, 872, 198, 151645], "attention_mask": [1, 1, 1, 1]}


def test_encode_prompt_accepts_batch_encoding(tmp_path: Path, monkeypatch) -> None:
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (tmp_path / name).touch()
    monkeypatch.setattr(tokenizer_module, "_load_tokenizer", lambda _: _FakeTokenizer())

    result = encode_prompt("snow", tmp_path)

    np.testing.assert_array_equal(result, np.asarray([151644, 872, 198, 151645], dtype=np.int64))
    assert tokenizer_files_ready(tmp_path)
