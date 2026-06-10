from __future__ import annotations

from src.common.io_utils import path_from_config, project_root, require_or_mock_input
from src.common.json_utils import write_json


class OpenUtauExporter:
    """Create an OpenUtau export plan."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with melody, lyrics, and alignment paths.
        Output:
            OpenUtauExporter instance.
        TODO:
            Add .ustx export options and voicebank metadata.
        """
        self.config = config

    def run(self) -> dict:
        """Generate data/svs/openutau_export_plan.json.

        Input:
            melody_midi, lyrics_singable, and note_lyric_alignment paths.
        Output:
            JSON plan describing what a real OpenUtau exporter should consume.
        TODO:
            Generate a .ustx project with notes, lyrics, phonemes, and timing.
        """
        melody_midi = path_from_config(self.config, "melody_midi")
        lyrics = path_from_config(self.config, "lyrics_singable")
        alignment = path_from_config(self.config, "note_lyric_alignment")
        statuses = {
            require_or_mock_input(melody_midi, self.config, "melody MIDI"),
            require_or_mock_input(lyrics, self.config, "singable lyrics"),
            require_or_mock_input(alignment, self.config, "note lyric alignment"),
        }
        output = project_root(self.config) / "data" / "svs" / "openutau_export_plan.json"
        data = {
            "melody_midi": str(melody_midi),
            "lyrics_singable": str(lyrics),
            "alignment": str(alignment),
            "todo": "generate .ustx project for OpenUtau or export DiffSinger input.",
        }
        write_json(output, data, self.config)
        return {
            "status": "mock" if "mock" in statuses else "success",
            "outputs": {"openutau_export_plan": str(output)},
            "message": "TODO: generate .ustx project from melody, lyrics, and alignment.",
        }
