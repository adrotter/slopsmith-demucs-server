"""Prefetch the default BS-RoFormer checkpoint used by the RoFormer backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from roformer_download import download_roformer_files


def _default_cache_dir() -> Path:
    return Path(os.environ.get("SLOPSMITH_DEMUCS_CACHE", Path.home() / ".cache" / "slopsmith-demucs"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BS-Roformer-SW-Fixed model files")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="Server cache directory (default: SLOPSMITH_DEMUCS_CACHE or ~/.cache/slopsmith-demucs)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist")
    args = parser.parse_args()

    checkpoint, config = download_roformer_files(args.cache_dir, force=args.force)
    print(f"checkpoint={checkpoint}")
    print(f"config={config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
