from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server
import roformer_download


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
            "warmup_state": dict(server.warmup_state),
        }
        with server.jobs_lock:
            server.jobs.clear()

    def tearDown(self):
        for name, value in self.originals.items():
            if name == "warmup_state":
                server.warmup_state.clear()
                server.warmup_state.update(value)
            else:
                setattr(server, name, value)
        with server.jobs_lock:
            server.jobs.clear()

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
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def read(self):
                    return b"model"

            with patch("roformer_download.urllib.request.urlopen", return_value=FakeResponse()) as download:
                checkpoint, config = roformer_download.download_roformer_files(root)

            self.assertEqual(checkpoint.name, roformer_download.ROFORMER_CHECKPOINT_FILENAME)
            self.assertEqual(config.name, roformer_download.ROFORMER_CONFIG_FILENAME)
            self.assertEqual(download.call_count, 2)
            first_url = download.call_args_list[0].args[0].full_url
            self.assertEqual(
                first_url,
                f"{roformer_download.ROFORMER_MIRROR_BASE_URL}/{roformer_download.ROFORMER_CHECKPOINT_FILENAME}",
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
