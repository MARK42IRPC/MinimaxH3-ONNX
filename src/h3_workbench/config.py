from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    workspace: Path
    output_dir: Path
    state_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = Path(os.environ.get("H3_WORKSPACE", Path.cwd())).resolve()
        return cls(
            workspace=workspace,
            output_dir=workspace / "onnx_models",
            state_dir=workspace / ".h3-workbench",
        )

