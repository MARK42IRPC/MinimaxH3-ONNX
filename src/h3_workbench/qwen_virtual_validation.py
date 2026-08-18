from __future__ import annotations

import argparse
import hashlib
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
from h3_workbench.main_benchmark import JsonlLogger
from h3_workbench.qwen_transformer import streamed_qwen_causal_attention
from h3_workbench.qwen_persistent import (
    QwenInt8SourceWeights,
    int8_virtual_product_fingerprint,
    int8_virtual_qwen_ready,
)


def _attention_inputs(tokens: int, block: int) -> dict[str, np.ndarray]:
    random = np.random.default_rng(71 + block)
    hidden = (random.standard_normal((tokens, 5120), dtype=np.float32) * 0.25).astype(np.float32)
    positions = np.arange(tokens, dtype=np.float32)
    frequencies = 1.0 / (500_000.0 ** (np.arange(0, 128, 2, dtype=np.float32) / 128.0))
    angles = np.outer(positions, frequencies)
    angles = np.concatenate((angles, angles), axis=-1)
    return {
        "hidden_states": hidden,
        "cosine": np.cos(angles).astype(np.float32),
        "sine": np.sin(angles).astype(np.float32),
        "attention_mask": np.triu(np.full((1, 1, tokens, tokens), -10_000.0, dtype=np.float32), k=1),
    }


def _source_expected(
    source: Path,
    kind: str,
    block: int,
    feeds: dict[str, np.ndarray],
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="qwen-virtual-validation-") as temporary_name:
        temporary = Path(temporary_name)
        inputs = temporary / "inputs.npz"
        output = temporary / "expected.npy"
        np.savez(inputs, **feeds)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "h3_workbench.qwen_expected_worker",
                kind,
                str(block),
                str(source),
                str(inputs),
                str(output),
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
            raise RuntimeError(f"Qwen {kind} block {block} source reference failed: {details}")
        return np.load(output)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate representative layers of a virtual Qwen INT8 product")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 24, 49])
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--relative-l2-max", type=float, default=1e-3)
    parser.add_argument("--full-chain-output", type=Path)
    parser.add_argument("--run-full-chain", action="store_true")
    return parser


def _run_full_chain(
    weights: QwenInt8SourceWeights,
    attention: Any,
    mlp: Any,
    tokens: int,
    attention_qkv: Any | None = None,
    attention_output: Any | None = None,
    runner: ORTGraphRunner | None = None,
) -> np.ndarray:
    base_ids = np.asarray([1, 42, 1000, 151935], dtype=np.int64)
    token_ids = np.resize(base_ids, tokens)
    hidden = weights.embedding(token_ids)
    positions = np.arange(tokens, dtype=np.float32)
    frequencies = 1.0 / (500_000.0 ** (np.arange(0, 128, 2, dtype=np.float32) / 128.0))
    angles = np.concatenate((np.outer(positions, frequencies),) * 2, axis=-1)
    shared = {
        "cosine": np.cos(angles).astype(np.float32),
        "sine": np.sin(angles).astype(np.float32),
        "attention_mask": np.triu(
            np.full((1, 1, tokens, tokens), -10_000.0, dtype=np.float32),
            k=1,
        ),
    }
    for block in range(50):
        if attention_qkv is None or attention_output is None or runner is None:
            hidden = attention.run(
                None,
                {"hidden_states": hidden, **shared, **weights.inputs("attention", block)},
            )[0]
        else:
            query, key, value = attention_qkv.run(
                None,
                {
                    "hidden_states": hidden,
                    "cosine": shared["cosine"],
                    "sine": shared["sine"],
                    **weights.inputs("attention_qkv", block),
                },
            )
            attended = streamed_qwen_causal_attention(
                query,
                key,
                value,
                query_chunk_tokens=max(1, tokens),
                use_cuda=runner.provider == "CUDAExecutionProvider",
                device_index=int(getattr(runner, "device_index", 0)),
                output_dtype=np.dtype(np.float16),
            )
            hidden = attention_output.run(
                None,
                {
                    "hidden_states": hidden,
                    "sine": shared["sine"],
                    "attended": attended.reshape(-1, 32, 256).astype(np.float32),
                    **weights.inputs("attention_output", block),
                },
            )[0]
        hidden = mlp.run(
            None,
            {"hidden_states": hidden, **weights.inputs("mlp", block)},
        )[0]
    return np.ascontiguousarray(hidden)


def _run_split_attention(
    qkv: Any,
    output: Any,
    runner: ORTGraphRunner,
    weights: QwenInt8SourceWeights,
    block: int,
    feeds: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the split attention path exactly as the multimodal runtime does."""
    query, key, value = qkv.run(
        None,
        {
            "hidden_states": feeds["hidden_states"],
            "cosine": feeds["cosine"],
            "sine": feeds["sine"],
            **weights.inputs("attention_qkv", block),
        },
    )
    attended = streamed_qwen_causal_attention(
        query,
        key,
        value,
        query_chunk_tokens=max(1, int(feeds["hidden_states"].shape[0])),
        use_cuda=runner.provider == "CUDAExecutionProvider",
        device_index=int(getattr(runner, "device_index", 0)),
        output_dtype=np.dtype(np.float16),
    )
    return np.asarray(
        output.run(
            None,
            {
                "hidden_states": feeds["hidden_states"],
                "sine": feeds["sine"],
                "attended": attended.reshape(-1, 32, 256).astype(np.float32),
                **weights.inputs("attention_output", block),
            },
        )[0],
        dtype=np.float32,
    )


def run(args: argparse.Namespace) -> int:
    logger = JsonlLogger(args.log)
    runner: ORTGraphRunner | None = None
    model = args.model.resolve()
    manifest_path = model / "manifest.json"
    product_fingerprint: str | None = None
    weights: QwenInt8SourceWeights | None = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_passed"] = False
        manifest.pop("validated_product_fingerprint", None)
        manifest["validation"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)

        product_fingerprint = int8_virtual_product_fingerprint(model)
        if not int8_virtual_qwen_ready(model):
            raise RuntimeError(f"Qwen virtual product is stale or incomplete: {model}")
        runner = ORTGraphRunner(prefer_cuda=True, prefetch_depth=0)
        weights = QwenInt8SourceWeights(model)
        attention = runner.session(weights.graph("attention"))
        split = weights.attention_split
        if not isinstance(split, dict) or "qkv" not in split or "output" not in split:
            raise RuntimeError("Qwen virtual product is missing split attention graphs")
        attention_qkv = runner.session(weights.graph("attention_qkv"))
        attention_output = runner.session(weights.graph("attention_output"))
        mlp = runner.session(weights.graph("mlp"))
        logger.write(
            "started",
            durable=True,
            model=str(model),
            source=str(weights.source),
            blocks=args.blocks,
            tokens=args.tokens,
            provider=runner.provider,
        )
        validation: dict[str, Any] = {"blocks": {}}
        failed_blocks: list[int] = []
        for block in args.blocks:
            attention_feeds = _attention_inputs(args.tokens, block)
            attention_expected = _source_expected(weights.source, "attention", block, attention_feeds)
            attention_actual = attention.run(None, {**attention_feeds, **weights.inputs("attention", block)})[0]
            attention_split_actual = _run_split_attention(
                attention_qkv,
                attention_output,
                runner,
                weights,
                block,
                attention_feeds,
            )
            attention_metrics = _metrics(attention_expected, attention_actual)
            attention_split_metrics = _metrics(attention_actual, attention_split_actual)
            mlp_feeds = {"hidden_states": attention_expected}
            mlp_expected = _source_expected(weights.source, "mlp", block, mlp_feeds)
            mlp_actual = mlp.run(None, {**mlp_feeds, **weights.inputs("mlp", block)})[0]
            mlp_metrics = _metrics(mlp_expected, mlp_actual)
            passed = (
                np.isfinite(attention_actual).all()
                and np.isfinite(attention_split_actual).all()
                and np.isfinite(mlp_actual).all()
                and attention_metrics["relative_l2"] <= args.relative_l2_max
                and attention_split_metrics["relative_l2"] <= args.relative_l2_max
                and mlp_metrics["relative_l2"] <= args.relative_l2_max
            )
            record = {
                "attention": attention_metrics,
                "attention_split": attention_split_metrics,
                "mlp": mlp_metrics,
                "passed": bool(passed),
            }
            validation["blocks"][str(block)] = record
            logger.write("block_completed", durable=True, block=block, **record)
            if not passed:
                failed_blocks.append(block)

        if failed_blocks:
            raise RuntimeError(
                f"Qwen virtual blocks failed the {args.relative_l2_max:g} relative-L2 gate: {failed_blocks}"
            )

        full_chain: dict[str, Any] | None = None
        if getattr(args, "run_full_chain", False):
            output = _run_full_chain(
                weights,
                attention,
                mlp,
                args.tokens,
                attention_qkv=attention_qkv,
                attention_output=attention_output,
                runner=runner,
            )
            full_chain = {
                "executed": True,
                "shape": list(output.shape),
                "finite": bool(np.isfinite(output).all()),
                "minimum": float(output.min()),
                "maximum": float(output.max()),
                "sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            }
            if output.shape != (args.tokens, 5120) or not full_chain["finite"]:
                raise RuntimeError(f"Qwen full-chain execution failed validation: {full_chain}")
            logger.write("full_chain_completed", durable=True, **full_chain)
        elif args.full_chain_output is not None:
            output = np.load(args.full_chain_output)
            full_chain = {
                "executed": False,
                "path": str(args.full_chain_output.resolve()),
                "shape": list(output.shape),
                "finite": bool(np.isfinite(output).all()),
                "minimum": float(output.min()),
                "maximum": float(output.max()),
            }
            if output.shape != (args.tokens, 5120) or not full_chain["finite"]:
                raise RuntimeError(f"Qwen full-chain output failed validation: {full_chain}")
        validation["full_chain"] = full_chain
        validation["relative_l2_max"] = args.relative_l2_max
        validation["status"] = "passed"
        validation["product_fingerprint"] = product_fingerprint
        validation["validation_passed"] = True
        if not int8_virtual_qwen_ready(model) or int8_virtual_product_fingerprint(model) != product_fingerprint:
            raise RuntimeError("Qwen virtual product changed during validation")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation"] = validation
        manifest["validation_passed"] = True
        manifest["validated_product_fingerprint"] = product_fingerprint
        _atomic_json(manifest_path, manifest)
        logger.write("completed", durable=True, validation=validation)
        print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist all validation failures
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
        except Exception:  # noqa: BLE001 - preserve the original validation failure
            pass
        logger.write("failed", durable=True, error=str(exc), traceback="".join(traceback.format_exception(exc)))
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        if weights is not None:
            weights.close()
        if runner is not None:
            runner.close()
        logger.close()


def validate_virtual_qwen_product(
    model: Path,
    log: Path,
    blocks: tuple[int, ...] = (0, 24, 49),
    tokens: int = 4,
    relative_l2_max: float = 1e-3,
    run_full_chain: bool = True,
) -> dict[str, Any]:
    args = argparse.Namespace(
        model=model,
        log=log,
        blocks=list(blocks),
        tokens=tokens,
        relative_l2_max=relative_l2_max,
        full_chain_output=None,
        run_full_chain=run_full_chain,
    )
    if run(args) != 0:
        manifest = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
        raise RuntimeError(str(manifest.get("validation", {}).get("error", "Qwen validation failed")))
    return json.loads((model / "manifest.json").read_text(encoding="utf-8"))["validation"]


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
