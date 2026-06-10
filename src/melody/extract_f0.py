from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input, write_text


class F0Extractor:
    """Mock F0 extraction stage."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.vocals and paths.f0_csv.
        Output:
            F0Extractor instance.
        TODO:
            Add RMVPE/librosa.pyin configuration.
        """
        self.config = config

    def run(self) -> dict:
        """Generate a mock F0 CSV file.

        Input:
            data/stems/vocals.wav.
        Output:
            data/score/f0.csv containing time,f0,voiced columns.
        TODO:
            Replace mock F0 with RMVPE or librosa.pyin output.
        """
        vocals = path_from_config(self.config, "vocals")
        input_status = require_or_mock_input(vocals, self.config, "vocals for F0 extraction")
        f0_csv = path_from_config(self.config, "f0_csv")
        csv_content = "time,f0,voiced\n0.00,261.63,1\n0.50,293.66,1\n1.00,329.63,1\n1.50,0,0\n"
        write_text(f0_csv, csv_content, self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"f0_csv": str(f0_csv)},
            "message": "TODO: replace mock F0 with RMVPE/librosa extraction.",
        }
