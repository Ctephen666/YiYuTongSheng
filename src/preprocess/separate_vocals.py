from __future__ import annotations

from src.common.audio_utils import write_placeholder_audio
from src.common.io_utils import path_from_config, require_or_mock_input


class VocalSeparator:
    """Mock vocal/accompaniment separation stage."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.raw_song, paths.vocals, and paths.accompaniment.
        Output:
            VocalSeparator instance.
        TODO:
            Add Demucs/UVR backend selection and model configuration.
        """
        self.config = config

    def run(self) -> dict:
        """Separate original song into vocals and accompaniment placeholders.

        Input:
            data/raw/original_song.wav from project config.
        Output:
            Dict containing status, output paths, and message.
        TODO:
            Replace mock separation with Demucs/UVR output.
        """
        raw_song = path_from_config(self.config, "raw_song")
        status = require_or_mock_input(raw_song, self.config, "raw song")

        vocals = path_from_config(self.config, "vocals")
        accompaniment = path_from_config(self.config, "accompaniment")
        write_placeholder_audio(vocals, "mock separated vocals.wav", self.config)
        write_placeholder_audio(accompaniment, "mock separated accompaniment.wav", self.config)

        message = "TODO: replace mock separation with Demucs/UVR output."
        return {
            "status": "mock" if status == "mock" else "success",
            "outputs": {"vocals": str(vocals), "accompaniment": str(accompaniment)},
            "message": message,
        }
