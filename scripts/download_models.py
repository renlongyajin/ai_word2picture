"""Utility script for downloading required diffusion weights."""

from __future__ import annotations

from pathlib import Path


def download_all(target_dir: Path) -> None:
    """Download model checkpoints into the target directory."""
    raise NotImplementedError("Model download logic will be implemented before release.")


if __name__ == "__main__":
    download_all(Path("models"))
