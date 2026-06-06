"""Download the default BS-RoFormer model files used by the server."""

from __future__ import annotations

import socket
import time
import urllib.error
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
TIMEOUT = 600
CHUNK_SIZE = 8192
MIN_DOWNLOAD_RATE_BYTES_PER_SECOND = 64 * 1024


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
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT) as response,
            partial.open("wb") as output,
        ):
            expected_total = _content_length(response)
            deadline_seconds = _download_deadline_seconds(expected_total)
            started = time.monotonic()
            downloaded = 0
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                elapsed = time.monotonic() - started
                if elapsed > deadline_seconds:
                    raise TimeoutError(
                        f"download exceeded {deadline_seconds:.0f}s deadline after "
                        f"{elapsed:.0f}s ({downloaded}/{expected_total or 'unknown'} bytes)"
                    )
        partial.replace(destination)
    except (socket.timeout, urllib.error.URLError) as exc:
        print(f"[roformer] failed {destination.name}: {exc}", flush=True)
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc
    print(f"[roformer] ready {destination}", flush=True)


def _content_length(response) -> int | None:
    value = response.getheader("Content-Length")
    if not value:
        return None
    try:
        total = int(value)
    except ValueError:
        return None
    return total if total > 0 else None


def _download_deadline_seconds(expected_total: int | None) -> float:
    if expected_total is None:
        return float(TIMEOUT)
    return max(float(TIMEOUT), expected_total / MIN_DOWNLOAD_RATE_BYTES_PER_SECOND)
