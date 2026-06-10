from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input, write_placeholder


class ScoreExporter:
    """Export mock melody score files."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.melody_notes and paths.melody_midi.
        Output:
            ScoreExporter instance.
        TODO:
            Add MIDI and MusicXML export settings.
        """
        self.config = config

    def run(self) -> dict:
        """Create placeholder score files from melody notes.

        Input:
            data/score/melody_notes.json.
        Output:
            data/score/melody.mid and data/score/melody.musicxml placeholder files.
        TODO:
            Use pretty_midi, music21, or MuseScore CLI for real score export.
        """
        notes = path_from_config(self.config, "melody_notes")
        input_status = require_or_mock_input(notes, self.config, "melody notes")
        melody_midi = path_from_config(self.config, "melody_midi")
        melody_musicxml = melody_midi.with_suffix(".musicxml")
        write_placeholder(melody_midi, "mock melody MIDI", self.config)
        write_placeholder(melody_musicxml, "mock melody MusicXML", self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"melody_midi": str(melody_midi), "melody_musicxml": str(melody_musicxml)},
            "message": "TODO: export real MIDI/MusicXML and optionally inspect in MuseScore.",
        }
