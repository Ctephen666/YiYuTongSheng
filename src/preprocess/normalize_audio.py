from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input


class AudioNormalizer:
    """Placeholder loudness and sample-rate normalization stage."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with audio paths and sample rate.
        Output:
            AudioNormalizer instance.
        TODO:
            Add sample-rate conversion and loudness target parameters.
        """
        self.config = config

    def run(self) -> dict:
        """Validate separated stems before downstream analysis.

        Input:
            data/stems/vocals.wav and data/stems/accompaniment.wav.
        Output:
            Dict containing status, output paths, and message.
        TODO:
            Normalize with soundfile/librosa/ffmpeg instead of pass-through.
        """
        vocals = path_from_config(self.config, "vocals")
        accompaniment = path_from_config(self.config, "accompaniment")
        vocal_status = require_or_mock_input(vocals, self.config, "vocals for normalization")
        acc_status = require_or_mock_input(accompaniment, self.config, "accompaniment for normalization")
        return {
            "status": "mock" if "mock" in {vocal_status, acc_status} else "success",
            "outputs": {"vocals": str(vocals), "accompaniment": str(accompaniment)},
            "message": "TODO: normalize sample rate, channels, and loudness.",
        }
