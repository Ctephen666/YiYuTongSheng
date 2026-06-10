from __future__ import annotations

from src.common.io_utils import path_from_config, project_root, write_text


class ReportGenerator:
    """Generate a markdown report for the mock pipeline."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with all stage output paths.
        Output:
            ReportGenerator instance.
        TODO:
            Add objective metrics, figures, and comparison tables.
        """
        self.config = config

    def _artifact_rows(self) -> list[tuple[str, str, bool, str]]:
        """Collect expected stage outputs.

        Input:
            Pipeline config paths.
        Output:
            Rows containing stage, path, existence, and TODO note.
        TODO:
            Include stage execution metadata from a run manifest.
        """
        paths = self.config["paths"]
        return [
            ("preprocess", paths["vocals"], path_from_config(self.config, "vocals").exists(), "Connect Demucs/UVR separation."),
            ("preprocess", paths["accompaniment"], path_from_config(self.config, "accompaniment").exists(), "Validate stem quality."),
            ("melody", paths["f0_csv"], path_from_config(self.config, "f0_csv").exists(), "Replace mock F0 with RMVPE/librosa.pyin."),
            ("melody", paths["melody_notes"], path_from_config(self.config, "melody_notes").exists(), "Quantize and correct notes."),
            ("lyrics", paths["lyrics_singable"], path_from_config(self.config, "lyrics_singable").exists(), "Use LLM and human review."),
            ("phoneme", paths["phonemes_target"], path_from_config(self.config, "phonemes_target").exists(), "Use real phonemizer backend."),
            ("alignment", paths["note_lyric_alignment"], path_from_config(self.config, "note_lyric_alignment").exists(), "Add dynamic programming alignment."),
            ("svs", paths["svs_vocal"], path_from_config(self.config, "svs_vocal").exists(), "Render with OpenUtau/DiffSinger."),
            ("svc", paths["rvc_converted"], path_from_config(self.config, "rvc_converted").exists(), "Run RVC/SVC inference."),
            ("mix", paths["final_mix"], path_from_config(self.config, "final_mix").exists(), "Mix with pydub/librosa/ffmpeg."),
        ]

    def run(self) -> dict:
        """Write outputs/reports/pipeline_report.md.

        Input:
            Current pipeline artifacts under data/.
        Output:
            Markdown report with outputs, mock status, TODOs, and a text flowchart.
        TODO:
            Add charts for F0 similarity, rhythm alignment, and speaker similarity.
        """
        report_path = project_root(self.config) / "outputs" / "reports" / "pipeline_report.md"
        rows = self._artifact_rows()
        table = "\n".join(
            f"| {stage} | `{path}` | {'yes' if exists else 'no'} | mock | {todo} |"
            for stage, path, exists, todo in rows
        )
        content = f"""# YiyuTongsheng Pipeline Report

## Artifact Summary

| Stage | Output | Exists | Current Mode | TODO |
| --- | --- | --- | --- | --- |
{table}

## Text Flowchart

original song
-> vocal/accompaniment separation
-> F0 extraction and melody reconstruction
-> Chinese phrase mapping
-> target-language lyric translation
-> singable lyric adaptation
-> target-language phonemization
-> syllable-note alignment
-> target-language SVS rendering
-> RVC/SVC timbre transfer
-> vocal/accompaniment mixing
-> final cross-lingual song output

## Notes

All generated media artifacts in the current scaffold are placeholders unless real tools are connected.
"""
        write_text(report_path, content, self.config)
        return {
            "status": "mock",
            "outputs": {"pipeline_report": str(report_path)},
            "message": "TODO: add real objective metrics and figures.",
        }
