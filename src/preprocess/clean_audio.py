from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input


class AudioCleaner:
    """Placeholder cleanup stage for separated vocals."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.vocals.
        Output:
            AudioCleaner instance.
        TODO:
            Add denoise, silence trimming, and artifact removal settings.
        """
        self.config = config

    def run(self) -> dict:
        """Validate separated vocal availability.

        Input:
            data/stems/vocals.wav.
        Output:
            Dict containing status, output path, and message.
        TODO:
            Apply real vocal cleanup with librosa, noisereduce, or DAW exports.
        """
        vocals = path_from_config(self.config, "vocals")
        status = require_or_mock_input(vocals, self.config, "vocals for cleaning")
        return {
            "status": status,
            "outputs": {"vocals": str(vocals)},
            "message": "TODO: add vocal denoising and silence cleanup.",
        }
