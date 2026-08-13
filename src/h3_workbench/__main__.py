from __future__ import annotations

import argparse
import os
import platform
import psutil
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3 Edge Workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["H3_WORKSPACE"] = str(args.workspace.resolve())
    if platform.system() == "Windows":
        process = psutil.Process()
        affinity = os.environ.get("H3_CPU_AFFINITY")
        if affinity:
            mask = int(affinity, 0)
            process.cpu_affinity([index for index in range(os.cpu_count() or 1) if mask & (1 << index)])
        priority = os.environ.get("H3_CPU_PRIORITY")
        if priority == "AboveNormal":
            process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    uvicorn.run("h3_workbench.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
