from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input


class VocalAligner:
    """Placeholder vocal alignment stage."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.rvc_converted.
        Output:
            VocalAligner instance.
        TODO:
            Add onset alignment and time-stretch correction.
        """
        self.config = config

    def run(self) -> dict:
        """Validate converted vocal before mixing.

        Input:
            data/rvc/converted_voice.wav.
        Output:
            Dict with converted vocal path.
        TODO:
            Align converted vocal to accompaniment using beat/onset features.
        """
        converted = path_from_config(self.config, "rvc_converted")
        status = require_or_mock_input(converted, self.config, "converted vocal")
        return {
            "status": status,
            "outputs": {"rvc_converted": str(converted)},
            "message": "TODO: align converted vocal timing before final mix.",
        }
