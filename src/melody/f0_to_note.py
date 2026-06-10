from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import write_json


class F0ToNoteConverter:
    """Convert F0 contours to a note-level mock melody."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.f0_csv and paths.melody_notes.
        Output:
            F0ToNoteConverter instance.
        TODO:
            Add note segmentation thresholds and quantization options.
        """
        self.config = config

    def run(self) -> dict:
        """Generate melody_notes.json from the mock F0 contour.

        Input:
            data/score/f0.csv.
        Output:
            data/score/melody_notes.json.
        TODO:
            Implement real pitch tracking, smoothing, note segmentation, and rests.
        """
        f0_csv = path_from_config(self.config, "f0_csv")
        input_status = require_or_mock_input(f0_csv, self.config, "F0 CSV")
        melody_notes = path_from_config(self.config, "melody_notes")
        data = {
            "phrases": [
                {
                    "id": 1,
                    "start": 0.0,
                    "end": 3.2,
                    "notes": [
                        {"pitch": "C4", "start": 0.0, "duration": 0.5},
                        {"pitch": "D4", "start": 0.5, "duration": 0.5},
                        {"pitch": "E4", "start": 1.0, "duration": 1.0},
                    ],
                }
            ]
        }
        write_json(melody_notes, data, self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"melody_notes": str(melody_notes)},
            "message": "TODO: convert F0 contours into calibrated note events.",
        }
