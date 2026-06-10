from __future__ import annotations

from src.common.audio_utils import write_placeholder_audio
from src.common.io_utils import path_from_config, require_or_mock_input


class RVCInferencer:
    """Mock RVC/SVC timbre conversion stage."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.svs_vocal and paths.rvc_converted.
        Output:
            RVCInferencer instance.
        TODO:
            Add RVC model, index, f0 method, and CLI command settings.
        """
        self.config = config

    def run(self) -> dict:
        """Create placeholder converted_voice.wav.

        Input:
            data/svs/target_language_vocal.wav.
        Output:
            data/rvc/converted_voice.wav.
        TODO:
            Replace mock with RVC inference command.
        """
        svs_vocal = path_from_config(self.config, "svs_vocal")
        input_status = require_or_mock_input(svs_vocal, self.config, "SVS vocal for RVC")
        converted = path_from_config(self.config, "rvc_converted")
        write_placeholder_audio(converted, "mock RVC converted voice", self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"rvc_converted": str(converted)},
            "message": "TODO: replace mock with RVC inference command.",
        }
