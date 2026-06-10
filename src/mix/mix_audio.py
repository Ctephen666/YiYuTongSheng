from __future__ import annotations

from src.common.audio_utils import write_placeholder_audio
from src.common.io_utils import path_from_config, require_or_mock_input


class AudioMixer:
    """Mock final audio mixer."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.rvc_converted, paths.accompaniment, and paths.final_mix.
        Output:
            AudioMixer instance.
        TODO:
            Add gain, pan, limiter, and output format settings.
        """
        self.config = config

    def run(self) -> dict:
        """Create final_mix.wav placeholder.

        Input:
            data/rvc/converted_voice.wav and data/stems/accompaniment.wav.
        Output:
            data/final/final_mix.wav.
        TODO:
            Use pydub overlay or ffmpeg to mix converted vocals and accompaniment.
        """
        converted = path_from_config(self.config, "rvc_converted")
        accompaniment = path_from_config(self.config, "accompaniment")
        converted_status = require_or_mock_input(converted, self.config, "converted vocal for mixing")
        accompaniment_status = require_or_mock_input(accompaniment, self.config, "accompaniment for mixing")
        final_mix = path_from_config(self.config, "final_mix")
        write_placeholder_audio(final_mix, "mock final mixed song", self.config)
        return {
            "status": "mock" if "mock" in {converted_status, accompaniment_status} else "success",
            "outputs": {"final_mix": str(final_mix)},
            "message": "TODO: use pydub/librosa/ffmpeg to mix vocal and accompaniment.",
        }
