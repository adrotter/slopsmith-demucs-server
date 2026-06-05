"""Run a local BS-RoFormer checkpoint through audio-separator.

The checkpoint and YAML config are supplied explicitly so private or custom
UVR models can be used without copying them into audio-separator's model cache.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from audio_separator.separator import Separator


class LocalRoformerSeparator(Separator):
    """Load one local UVR-compatible RoFormer pair without downloading files."""

    def __init__(self, checkpoint: Path, config: Path, device: str, **kwargs):
        self.local_checkpoint = checkpoint.resolve()
        self.local_config = config.resolve()
        self.requested_device = device
        super().__init__(**kwargs)

    def setup_torch_device(self, system_info):
        """Honor the server's explicit device selection."""
        if not self.requested_device:
            return super().setup_torch_device(system_info)

        self.torch_device_cpu = torch.device("cpu")
        if self.requested_device == "cpu":
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]
            return

        if not self.requested_device.startswith("cuda"):
            raise ValueError(f"Unsupported RoFormer device: {self.requested_device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.torch_device = torch.device(self.requested_device)
        self.onnx_execution_provider = ["CUDAExecutionProvider"]

    def download_model_files(self, model_filename):
        """Return the local model pair in audio-separator's expected shape."""
        return (
            self.local_checkpoint.name,
            "MDXC",
            "Local BS-RoFormer",
            str(self.local_checkpoint),
            str(self.local_config),
        )

    def load_model_data_from_yaml(self, yaml_config_filename):
        """UVR metadata identifies this custom filename as a BS-RoFormer."""
        model_data = super().load_model_data_from_yaml(yaml_config_filename)
        model_data["is_roformer"] = True
        return model_data


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def _pad_short_audio(audio_path: Path, config_path: Path):
    """Pad clips shorter than one model window; audio-separator cannot."""
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    sample_rate = int(config["audio"]["sample_rate"])
    hop_length = int(config["model"].get("stft_hop_length", config["audio"]["hop_length"]))
    minimum_samples = hop_length * (int(config["inference"]["dim_t"]) - 1)
    if librosa.get_duration(path=str(audio_path)) >= minimum_samples / sample_rate:
        return audio_path, None, None

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    original_samples = audio.shape[-1]
    audio = np.pad(audio, ((0, 0), (0, minimum_samples - original_samples)))

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.close()
    sf.write(temp_audio.name, audio.T, sample_rate)
    return Path(temp_audio.name), Path(temp_audio.name), original_samples


def _trim_padded_outputs(output_files, output_dir: Path, original_samples: int):
    """Remove padding from stems produced for a short input clip."""
    for output_file in output_files:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = output_dir / output_path
        stem, sample_rate = sf.read(output_path)
        sf.write(output_path, stem[:original_samples], sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local BS-RoFormer model")
    parser.add_argument("--checkpoint", required=True, type=_existing_file)
    parser.add_argument("--config", required=True, type=_existing_file)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="")
    parser.add_argument("audio", type=_existing_file)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    separator = LocalRoformerSeparator(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        log_level=logging.INFO,
        model_file_dir=str(args.checkpoint.parent.resolve()),
        output_dir=str(args.output_dir.resolve()),
        output_format="WAV",
        use_soundfile=True,
        use_autocast=args.device.startswith("cuda"),
    )
    separator.load_model(args.checkpoint.name)
    audio_path, padded_audio_path, original_samples = _pad_short_audio(
        args.audio.resolve(),
        args.config.resolve(),
    )
    try:
        output_files = separator.separate(
            str(audio_path),
            custom_output_names={
                "bass": "bass",
                "drums": "drums",
                "other": "other",
                "vocals": "vocals",
                "guitar": "guitar",
                "piano": "piano",
            },
        )
    finally:
        if padded_audio_path:
            os.unlink(padded_audio_path)

    if not output_files:
        raise RuntimeError("RoFormer separation produced no output files")
    if original_samples is not None:
        _trim_padded_outputs(output_files, args.output_dir.resolve(), original_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
