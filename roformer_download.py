"""Download the default BS-RoFormer model files used by the server."""

from __future__ import annotations

import hashlib
import socket
import time
import tempfile
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
ROFORMER_SHA256 = {
    ROFORMER_CHECKPOINT_FILENAME: "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
    ROFORMER_CONFIG_FILENAME: "f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b",
}
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
        expected_sha256 = ROFORMER_SHA256[filename]
        if destination.is_file() and not force:
            actual_sha256 = _file_sha256(destination)
            if actual_sha256 == expected_sha256:
                print(f"[roformer] using cached {destination}", flush=True)
                continue
            print(
                f"[roformer] cached {destination.name} checksum mismatch; re-downloading",
                flush=True,
            )
        _download_file(filename, destination, expected_sha256)

    return default_roformer_paths(cache_dir)


def _download_file(filename: str, destination: Path, expected_sha256: str) -> None:
    print(f"[roformer] downloading {destination.name}", flush=True)
    url = f"{ROFORMER_MIRROR_BASE_URL}/{filename}"
    temp_path: Path | None = None

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f"{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        with (
            urllib.request.urlopen(request, timeout=TIMEOUT) as response,
            temp_path.open("wb") as output,
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

        actual_sha256 = _file_sha256(temp_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"checksum mismatch for {filename}: expected {expected_sha256}, got {actual_sha256}"
            )

        temp_path.replace(destination)
        temp_path = None
    except (socket.timeout, urllib.error.URLError) as exc:
        print(f"[roformer] failed {destination.name}: {exc}", flush=True)
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"failed to download {filename} from {url}: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(f"[roformer] ready {destination}", flush=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


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
