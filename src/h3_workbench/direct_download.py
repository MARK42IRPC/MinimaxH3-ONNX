from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DownloadProgress = Callable[[int, int], None]
_CHUNK_BYTES = 8 * 1024 * 1024


def _part_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    callback: DownloadProgress | None = None,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download one immutable model asset with HTTP Range resume support."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            if callback is not None:
                callback(actual_size, expected_size)
            return destination
        raise RuntimeError(
            f"Existing model file has an unexpected size: {destination} "
            f"({actual_size} bytes, expected {expected_size})"
        )

    partial = _part_path(destination)
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        raise RuntimeError(
            f"Partial download is larger than expected: {partial} "
            f"({offset} bytes, expected {expected_size})"
        )
    if offset == expected_size:
        os.replace(partial, destination)
        if callback is not None:
            callback(expected_size, expected_size)
        return destination

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "minimax-h3-edge-workbench/0.1",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout_seconds)  # noqa: S310 - catalog URLs are explicit HTTPS assets
    except HTTPError as exc:
        if exc.code == 416 and partial.is_file() and partial.stat().st_size == expected_size:
            os.replace(partial, destination)
            return destination
        raise RuntimeError(f"Direct model download failed with HTTP {exc.code}: {url}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if offset > 0 and not append:
            offset = 0
        mode = "ab" if append else "wb"
        downloaded = offset
        last_report = 0.0
        with partial.open(mode) as output:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if callback is not None and (now - last_report >= 0.5 or downloaded == expected_size):
                    callback(downloaded, expected_size)
                    last_report = now
            output.flush()
            os.fsync(output.fileno())

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Direct model download is incomplete: {partial} "
            f"({actual_size} bytes, expected {expected_size})"
        )
    os.replace(partial, destination)
    if callback is not None:
        callback(expected_size, expected_size)
    return destination
