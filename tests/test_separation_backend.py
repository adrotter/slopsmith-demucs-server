from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import server
import roformer_download


MODEL_SHA256 = "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"


class FakeProcess:
    returncode = 0
    command = None
    timeout = None

    def __init__(self, command, **kwargs):
        type(self).command = command

    def communicate(self, timeout):
        type(self).timeout = timeout
        return "", ""

    def kill(self):
        pass


class SeparationBackendTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "_model": server._model,
            "_device": server._device,
            "_gpu_available": server._gpu_available,
            "_backend": server._backend,
            "_roformer_checkpoint": server._roformer_checkpoint,
            "_roformer_config": server._roformer_config,
            "_roformer_auto_download": server._roformer_auto_download,
            "CACHE_DIR": server.CACHE_DIR,
            "CACHE_MAX_COMPLETED_JOBS": server.CACHE_MAX_COMPLETED_JOBS,
            "cache_order": list(server.cache_order),
            "warmup_state": dict(server.warmup_state),
        }
        with server.jobs_lock:
            server.jobs.clear()
        with server.cache_order_lock:
            server.cache_order.clear()

    def tearDown(self):
        for name, value in self.originals.items():
            if name == "warmup_state":
                server.warmup_state.clear()
                server.warmup_state.update(value)
            elif name == "cache_order":
                continue
            else:
                setattr(server, name, value)
        with server.jobs_lock:
            server.jobs.clear()
        with server.cache_order_lock:
            server.cache_order.clear()
            server.cache_order.extend(self.originals["cache_order"])

    def test_cache_entries_are_isolated_by_backend_and_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            server.CACHE_DIR = Path(temp_dir)
            cache_path = server.CACHE_DIR / "job"
            cache_path.mkdir()
            (cache_path / "vocals.wav").write_bytes(b"stem")

            server._backend = "demucs"
            (cache_path / ".model").write_text("demucs:htdemucs_ft", encoding="utf-8")
            self.assertEqual(
                server._check_cache("job", ["vocals"], "htdemucs_ft"),
                {"vocals": "/download/job/vocals.wav"},
            )

            server._backend = "roformer"
            server._roformer_checkpoint = "BS-Rofo-SW-Fixed.ckpt"
            self.assertIsNone(server._check_cache("job", ["vocals"], "htdemucs_ft"))
            prepared_path = server._prepare_cache_path("job", "htdemucs_ft")
            self.assertEqual(prepared_path, cache_path)
            self.assertFalse((cache_path / "vocals.wav").exists())

    def test_cache_eviction_ignores_roformer_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            server.CACHE_DIR = Path(temp_dir)
            server.CACHE_MAX_COMPLETED_JOBS = 1
            models_dir = server.CACHE_DIR / "models"
            models_dir.mkdir()
            (models_dir / "BS-Rofo-SW-Fixed.ckpt").write_bytes(b"model")

            old_cache = server.CACHE_DIR / "old"
            old_cache.mkdir()
            (old_cache / ".model").write_text("demucs:htdemucs_ft", encoding="utf-8")
            (old_cache / "vocals.wav").write_bytes(b"stem")

            new_cache = server.CACHE_DIR / "new"
            new_cache.mkdir()
            (new_cache / ".model").write_text("demucs:htdemucs_ft", encoding="utf-8")

            server._initialize_cache_order()
            server._remember_cache_entry("new")

            self.assertTrue(models_dir.exists())
            self.assertFalse(old_cache.exists())
            self.assertTrue(new_cache.exists())

    def test_roformer_worker_invokes_local_runner_and_caches_requested_stems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "vocals.wav").write_bytes(b"stem")
            audio_path = root / "input.wav"
            audio_path.write_bytes(b"audio")

            server.CACHE_DIR = root / "cache"
            server.CACHE_DIR.mkdir()
            server._backend = "roformer"
            server._roformer_checkpoint = str(root / "model.ckpt")
            server._roformer_config = str(root / "model.yaml")
            with server.jobs_lock:
                server.jobs["job"] = {"status": "processing"}

            with (
                patch("server.tempfile.mkdtemp", return_value=str(output_dir)),
                patch("server.subprocess.Popen", FakeProcess),
            ):
                server._run_separation("job", str(audio_path), ["vocals"], "ignored")

            self.assertIn("run_roformer.py", FakeProcess.command[1])
            self.assertIn(server._roformer_checkpoint, FakeProcess.command)
            self.assertIn(server._roformer_config, FakeProcess.command)
            self.assertEqual(FakeProcess.timeout, 600)
            self.assertEqual(
                (server.CACHE_DIR / "job" / ".model").read_text(encoding="utf-8"),
                "roformer:model",
            )
            with server.jobs_lock:
                self.assertEqual(server.jobs["job"]["status"], "complete")
                self.assertEqual(
                    server.jobs["job"]["stems"],
                    {"vocals": "/download/job/vocals.wav"},
                )

    def test_roformer_auto_download_uses_cache_managed_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            server.CACHE_DIR = Path(temp_dir)
            server._roformer_checkpoint = ""
            server._roformer_config = ""

            server._set_default_roformer_paths()

            self.assertEqual(
                server._roformer_checkpoint,
                str(Path(temp_dir) / "models" / "BS-ROFO-SW-Fixed" / "BS-Rofo-SW-Fixed.ckpt"),
            )
            self.assertEqual(
                server._roformer_config,
                str(Path(temp_dir) / "models" / "BS-ROFO-SW-Fixed" / "BS-Rofo-SW-Fixed.yaml"),
            )

    def test_roformer_warmup_downloads_default_model_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            server.CACHE_DIR = Path(temp_dir)
            server._backend = "roformer"
            server._roformer_auto_download = True
            server._roformer_checkpoint = ""
            server._roformer_config = ""

            def fake_download(cache_dir):
                checkpoint, config = server.default_roformer_paths(cache_dir)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"model")
                config.write_text("audio: {}", encoding="utf-8")
                return checkpoint, config

            with patch("server.download_roformer_files", side_effect=fake_download) as download:
                server._warmup_demucs()

            download.assert_called_once_with(server.CACHE_DIR)
            self.assertEqual(server.warmup_state["demucs"], "ready")

    def test_roformer_downloader_uses_release_asset_urls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeResponse:
                def __enter__(self):
                    self.chunks = [b"model", b""]
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def read(self, size=-1):
                    return self.chunks.pop(0)

                def getheader(self, name):
                    return "5" if name == "Content-Length" else None

            with patch("roformer_download.urllib.request.urlopen", return_value=FakeResponse()) as download:
                with patch.dict(
                    roformer_download.ROFORMER_SHA256,
                    {
                        roformer_download.ROFORMER_CHECKPOINT_FILENAME: MODEL_SHA256,
                        roformer_download.ROFORMER_CONFIG_FILENAME: MODEL_SHA256,
                    },
                ):
                    checkpoint, config = roformer_download.download_roformer_files(root)

            self.assertEqual(checkpoint.name, roformer_download.ROFORMER_CHECKPOINT_FILENAME)
            self.assertEqual(config.name, roformer_download.ROFORMER_CONFIG_FILENAME)
            self.assertEqual(download.call_count, 2)
            first_url = download.call_args_list[0].args[0].full_url
            self.assertEqual(
                first_url,
                f"{roformer_download.ROFORMER_MIRROR_BASE_URL}/{roformer_download.ROFORMER_CHECKPOINT_FILENAME}",
            )
            self.assertEqual(download.call_args_list[0].kwargs["timeout"], roformer_download.TIMEOUT)

    def test_roformer_downloader_skips_cache_only_when_checksum_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint, _ = roformer_download.default_roformer_paths(root)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"model")

            with (
                patch(
                    "roformer_download.ROFORMER_FILES",
                    (roformer_download.ROFORMER_CHECKPOINT_FILENAME,),
                ),
                patch.dict(
                    roformer_download.ROFORMER_SHA256,
                    {roformer_download.ROFORMER_CHECKPOINT_FILENAME: MODEL_SHA256},
                ),
                patch("roformer_download.urllib.request.urlopen") as download,
            ):
                roformer_download.download_roformer_files(root)

            download.assert_not_called()

    def test_roformer_downloader_refetches_cache_when_checksum_mismatches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint, _ = roformer_download.default_roformer_paths(root)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"bad")

            class FakeResponse:
                def __enter__(self):
                    self.chunks = [b"model", b""]
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def read(self, size=-1):
                    return self.chunks.pop(0)

                def getheader(self, name):
                    return "5" if name == "Content-Length" else None

            with (
                patch(
                    "roformer_download.ROFORMER_FILES",
                    (roformer_download.ROFORMER_CHECKPOINT_FILENAME,),
                ),
                patch.dict(
                    roformer_download.ROFORMER_SHA256,
                    {roformer_download.ROFORMER_CHECKPOINT_FILENAME: MODEL_SHA256},
                ),
                patch("roformer_download.urllib.request.urlopen", return_value=FakeResponse()) as download,
            ):
                roformer_download.download_roformer_files(root)

            download.assert_called_once()
            self.assertEqual(checkpoint.read_bytes(), b"model")

    def test_roformer_downloader_uses_unique_temp_file_and_cleans_only_its_own(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint, _ = roformer_download.default_roformer_paths(root)
            checkpoint.parent.mkdir(parents=True)
            shared_partial = checkpoint.with_suffix(checkpoint.suffix + ".part")
            shared_partial.write_bytes(b"other process")

            class FakeResponse:
                def __enter__(self):
                    self.chunks = [b"bad", b""]
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def read(self, size=-1):
                    return self.chunks.pop(0)

                def getheader(self, name):
                    return "3" if name == "Content-Length" else None

            with patch("roformer_download.urllib.request.urlopen", return_value=FakeResponse()):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    roformer_download._download_file(
                        roformer_download.ROFORMER_CHECKPOINT_FILENAME,
                        checkpoint,
                        MODEL_SHA256,
                    )

            self.assertTrue(shared_partial.exists())
            self.assertEqual(shared_partial.read_bytes(), b"other process")
            self.assertEqual(list(checkpoint.parent.glob(f"{checkpoint.name}.*.part")), [])

    def test_roformer_downloader_aborts_when_overall_deadline_expires(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeResponse:
                def __enter__(self):
                    self.chunks = [b"model", b""]
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def read(self, size=-1):
                    return self.chunks.pop(0)

                def getheader(self, name):
                    return "1" if name == "Content-Length" else None

            with (
                patch("roformer_download.TIMEOUT", 1),
                patch("roformer_download.time.monotonic", side_effect=[0.0, 2.0]),
                patch("roformer_download.urllib.request.urlopen", return_value=FakeResponse()),
            ):
                with self.assertRaisesRegex(RuntimeError, "download exceeded 1s deadline"):
                    roformer_download.download_roformer_files(root)

    def test_roformer_downloader_logs_url_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with (
                patch("roformer_download.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")),
                patch("builtins.print") as log,
            ):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    roformer_download.download_roformer_files(root)

            self.assertTrue(
                any("failed BS-Rofo-SW-Fixed.ckpt" in str(call) for call in log.call_args_list)
            )

    def test_roformer_warmup_rejects_zero_byte_local_model_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.ckpt"
            config = root / "model.yaml"
            checkpoint.write_bytes(b"")
            config.write_text("audio: {}", encoding="utf-8")
            server._backend = "roformer"
            server._roformer_auto_download = False
            server._roformer_checkpoint = str(checkpoint)
            server._roformer_config = str(config)

            server._warmup_demucs()

            self.assertEqual(
                server.warmup_state["demucs"],
                f"failed: missing local model file: {checkpoint}",
            )

    def test_cli_model_roformer_selects_auto_download_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            server.CACHE_DIR = Path(temp_dir)
            with (
                patch("server.sys.argv", ["server.py", "--model", "roformer", "--skip-warmup"]),
                patch("server._detect_gpu", return_value=False),
                patch("server.uvicorn.run") as run_server,
            ):
                server.main()

            run_server.assert_called_once()
            self.assertEqual(server._backend, "roformer")
            self.assertTrue(server._roformer_auto_download)
            self.assertTrue(server._roformer_uses_default_paths())

    def test_cli_model_roformer_rejects_demucs_backend(self):
        with (
            patch("server.sys.argv", ["server.py", "--model", "roformer", "--backend", "demucs"]),
            patch("server.uvicorn.run"),
        ):
            with self.assertRaises(SystemExit):
                server.main()


if __name__ == "__main__":
    unittest.main()
