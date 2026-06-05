"""Download the default BS-RoFormer model files used by the server."""

from __future__ import annotations

import urllib.request
from pathlib import Path


ROFORMER_MIRROR_BASE_URL = (
    "https://github.com/adrotter/slopsmith-model-mirror/releases/download/"
    "roformer-bs-rofo-sw-fixed-v1"
)
ROFORMER_CHECKPOINT_FILENAME = "BS-Rofo-SW-Fixed.ckpt"
ROFORMER_CONFIG_FILENAME = "BS-Rofo-SW-Fixed.yaml"
ROFORMER_FILES = (ROFORMER_CHECKPOINT_FILENAME, ROFORMER_CONFIG_FILENAME)
USER_AGENT = "slopsmith-demucs-server/roformer-downloader"


def default_roformer_dir(cache_dir: Path) -> Path:
    return Path(cache_dir).expanduser() / "models" / "BS-ROFO-SW-Fixed"


def default_roformer_paths(cache_dir: Path) -> tuple[Path, Path]:
    model_dir = default_roformer_dir(cache_dir)
    return (
        model_dir / ROFORMER_CHECKPOINT_FILENAME,
        model_dir / ROFORMER_CONFIG_FILENAME,
    )


def download_roformer_files(cache_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Download the default RoFormer checkpoint/config pair if missing."""
    model_dir = default_roformer_dir(cache_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in ROFORMER_FILES:
        destination = model_dir / filename
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            print(f"[roformer] using cached {destination}", flush=True)
            continue
        _download_file(filename, destination, force=force)

    return default_roformer_paths(cache_dir)


def _download_file(filename: str, destination: Path, force: bool = False) -> None:
    print(f"[roformer] downloading {destination.name}", flush=True)
    url = f"{ROFORMER_MIRROR_BASE_URL}/{filename}"
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            output.write(response.read())
        partial.replace(destination)
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc
    print(f"[roformer] ready {destination}", flush=True)
