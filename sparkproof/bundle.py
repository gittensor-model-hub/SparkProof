"""Assemble a publishable SparkProof bundle (no API keys)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def write_bundle(
    *,
    out_dir: Path,
    trajectories: list[dict[str, Any]],
    manifest: dict[str, Any],
    prompts_path: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "trajectories.jsonl").open("w") as f:
        for record in trajectories:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if prompts_path is not None:
        shutil.copy2(prompts_path, out_dir / "prompts.jsonl")

    return out_dir
