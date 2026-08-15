from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from h3_workbench import direct_download


class _Response(BytesIO):
    def __init__(self, value: bytes, status: int) -> None:
        super().__init__(value)
        self.status = status

    def getcode(self) -> int:
        return self.status


def test_direct_download_publishes_complete_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = b"verified-model-bytes"
    requests = []

    def open_request(request, timeout):  # noqa: ANN001, ANN202 - compact URL opener stub
        requests.append((request, timeout))
        return _Response(payload, 200)

    monkeypatch.setattr(direct_download, "urlopen", open_request)
    destination = tmp_path / "model.safetensors"

    result = direct_download.download_file("https://models.example/model", destination, len(payload))

    assert result == destination
    assert destination.read_bytes() == payload
    assert not destination.with_name("model.safetensors.part").exists()
    assert requests[0][0].get_header("Range") is None


def test_direct_download_resumes_partial_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = b"abcdefghij"
    destination = tmp_path / "model.safetensors"
    partial = destination.with_name("model.safetensors.part")
    partial.write_bytes(payload[:4])
    ranges: list[str | None] = []

    def open_request(request, timeout):  # noqa: ANN001, ANN202 - compact URL opener stub
        ranges.append(request.get_header("Range"))
        return _Response(payload[4:], 206)

    monkeypatch.setattr(direct_download, "urlopen", open_request)

    direct_download.download_file("https://models.example/model", destination, len(payload))

    assert ranges == ["bytes=4-"]
    assert destination.read_bytes() == payload


def test_direct_download_refuses_wrong_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"wrong")

    with pytest.raises(RuntimeError, match="unexpected size"):
        direct_download.download_file("https://models.example/model", destination, 99)
