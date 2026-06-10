from __future__ import annotations

from src.common.audio_utils import write_placeholder_audio
from src.common.io_utils import allow_mock, path_from_config, project_root, require_or_mock_input


class SingingRenderer:
    """Mock target-language singing renderer."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.svs_vocal.
        Output:
            SingingRenderer instance.
        TODO:
            Add OpenUtau/DiffSinger executable paths and render presets.
        """
        self.config = config

    def run(self) -> dict:
        """Create or validate the target-language vocal output.

        Input:
            data/svs/openutau_export_plan.json.
        Output:
            data/svs/target_language_vocal.wav placeholder.
        TODO:
            Render target language singing voice using OpenUtau or DiffSinger.
        """
        plan = project_root(self.config) / "data" / "svs" / "openutau_export_plan.json"
        input_status = require_or_mock_input(plan, self.config, "OpenUtau export plan")
        svs_vocal = path_from_config(self.config, "svs_vocal")
        if not svs_vocal.exists() and allow_mock(self.config):
            write_placeholder_audio(svs_vocal, "mock target-language SVS vocal", self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"svs_vocal": str(svs_vocal)},
            "message": "TODO: render target language singing voice using OpenUtau or DiffSinger.",
        }
