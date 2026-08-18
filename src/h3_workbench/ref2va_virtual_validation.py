from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from h3_workbench.exporter import _metrics
from h3_workbench.inference_runtime import ORTGraphRunner
from h3_workbench.ref2va_virtual_slicer import (
    LAYERS,
    VIRTUAL_KINDS,
    Ref2VASourceWeights,
    ref2va_virtual_ready,
    virtual_product_fingerprint,
)


def _inputs(tokens: int, block: int, kind: str) -> dict[str, np.ndarray]:
    random = np.random.default_rng(193 + block * 7 + VIRTUAL_KINDS.index(kind))
    hidden = (random.standard_normal((tokens, 5376), dtype=np.float32) * 0.25).astype(np.float32)
    values: dict[str, np.ndarray] = {
        "hidden_states": hidden,
        "timestep_embedding": random.standard_normal((1, 8), dtype=np.float32),
        "modulation_ids": np.resize(np.arange(3, dtype=np.int64), tokens),
    }
    angles = np.zeros((tokens, 48), dtype=np.float32)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotary = np.empty((1, tokens, 1, 48, 2, 2), dtype=np.float16)
    rotary[..., 0, 0] = cosine[None, :, None, :]
    rotary[..., 0, 1] = -sine[None, :, None, :]
    rotary[..., 1, 0] = sine[None, :, None, :]
    rotary[..., 1, 1] = cosine[None, :, None, :]
    if kind == "attention_qkv":
        values["rotary_table"] = rotary
    elif kind == "attention_output":
        values["attended"] = random.standard_normal((tokens, 7168), dtype=np.float32)
    return values


def _source_expected(
    source: Path,
    kind: str,
    block: int,
    feeds: dict[str, np.ndarray],
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="ref2va-virtual-validation-") as temporary_name:
        temporary = Path(temporary_name)
        inputs = temporary / "inputs.npz"
        expected = temporary / "expected.npy"
        np.savez(inputs, **feeds)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "h3_workbench.ref2va_expected_worker",
                kind,
                str(block),
                str(source),
                str(inputs),
                str(expected),
            ],
            capture_output=True,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"Ref2VA {kind} block {block} source reference failed: {details}")
        return np.load(expected)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate representative Ref2VA virtual block slices")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 24, 49])
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--relative-l2-max", type=float, default=3e-3)
    return parser


def run(args: argparse.Namespace) -> int:
    model = args.model.resolve()
    manifest_path = model / "manifest.json"
    product_fingerprint: str | None = None
    runner: ORTGraphRunner | None = None
    weights: Ref2VASourceWeights | None = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_passed"] = False
        manifest.pop("validated_product_fingerprint", None)
        manifest["validation"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        product_fingerprint = virtual_product_fingerprint(model)
        if not ref2va_virtual_ready(model):
            raise RuntimeError(f"Ref2VA virtual product is stale or incomplete: {model}")
        if any(not 0 <= block < LAYERS for block in args.blocks):
            raise ValueError(f"Ref2VA validation blocks must be in [0, {LAYERS - 1}]")
        runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
        graph_paths = {
            graph: model / graph
            for graph in manifest.get("graphs", [])
            if isinstance(graph, str)
        }
        weights = Ref2VASourceWeights(model, runner, graph_paths)
        sessions = {
            kind: runner.session(weights.graph(kind))
            for kind in VIRTUAL_KINDS
        }
        validation: dict[str, Any] = {
            "status": "running",
            "provider": runner.provider,
            "blocks": {},
            "relative_l2_max": args.relative_l2_max,
        }
        failed: list[str] = []
        for block in args.blocks:
            record: dict[str, Any] = {}
            for kind in VIRTUAL_KINDS:
                feeds = _inputs(args.tokens, block, kind)
                expected = _source_expected(weights.source, kind, block, feeds)
                graph = f"main_block_{block:02d}_{kind}.onnx"
                actual = sessions[kind].run(None, {**feeds, **weights.inputs(graph)})[0]
                weights.release(graph)
                metrics = _metrics(expected, actual)
                passed = bool(np.isfinite(actual).all() and metrics["relative_l2"] <= args.relative_l2_max)
                record[kind] = {**metrics, "passed": passed}
                if not passed:
                    failed.append(f"{block}:{kind}")
            validation["blocks"][str(block)] = record
        if failed:
            raise RuntimeError(f"Ref2VA virtual validation failed: {failed}")
        validation["status"] = "passed"
        validation["validation_passed"] = True
        validation["product_fingerprint"] = product_fingerprint
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation"] = validation
        manifest["validation_passed"] = True
        manifest["validated_product_fingerprint"] = product_fingerprint
        _atomic_json(manifest_path, manifest)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist every validation failure
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["validation_passed"] = False
            manifest.pop("validated_product_fingerprint", None)
            manifest["validation"] = {
                "status": "failed",
                "product_fingerprint": product_fingerprint,
                "error": str(exc),
            }
            _atomic_json(manifest_path, manifest)
        except Exception:  # noqa: BLE001 - preserve the original validation error
            pass
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        if weights is not None:
            weights.close()
        if runner is not None:
            runner.close()


def validate_virtual_ref2va_product(
    model: Path,
    log: Path,
    blocks: tuple[int, ...] = (0, 24, 49),
    tokens: int = 3,
    relative_l2_max: float = 3e-3,
) -> dict[str, Any]:
    args = argparse.Namespace(
        model=model,
        log=log,
        blocks=list(blocks),
        tokens=tokens,
        relative_l2_max=relative_l2_max,
    )
    if run(args) != 0:
        manifest = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
        raise RuntimeError(str(manifest.get("validation", {}).get("error", "Ref2VA validation failed")))
    return json.loads((model / "manifest.json").read_text(encoding="utf-8"))["validation"]


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()

