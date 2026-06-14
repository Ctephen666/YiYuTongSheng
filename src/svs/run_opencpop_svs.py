from __future__ import annotations

from typing import Any

from src.alignment.zh_note_alignment import align_zh_notes, save_zh_alignment
from src.dataset.opencpop_loader import OpenCpopDatasetLoader
from src.lyrics.zh_phrase_builder import build_zh_phrases, save_zh_phrases
from src.melody.import_opencpop_midi import OpenCpopMidiImporter
from src.phoneme.zh_phonemizer import phonemize_zh, save_zh_phonemes
from src.svs.build_opencpop_svs_score import build_opencpop_svs_score, save_opencpop_svs_score
from src.svs.checkpoint_manager import check_svs_checkpoints, save_checkpoint_status
from src.svs.diffsinger_opencpop_exporter import build_diffsinger_export_plan, save_diffsinger_export_plan
from src.svs.opencpop_neural_svs_backend import build_neural_svs_render_plan, save_neural_svs_render_plan
from src.svs.rvc_voice_conversion import apply_rvc_voice_conversion


def _merge_result(outputs: dict[str, str], warnings: list[str], result: dict[str, Any]) -> None:
    for key, value in result.get("outputs", {}).items():
        outputs[key] = str(value)
    for warning in result.get("warnings", []):
        if warning not in warnings:
            warnings.append(warning)


def _merge_warnings(warnings: list[str], payload: dict[str, Any]) -> None:
    for warning in payload.get("warnings", []):
        if warning not in warnings:
            warnings.append(warning)


class OpenCpopZhSvsRunner:
    """Run the Chinese OpenCpop SVS inference pipeline."""

    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        return run_opencpop_svs(self.config)


def run_opencpop_svs(config: dict) -> dict:
    outputs: dict[str, str] = {}
    warnings: list[str] = []

    dataset_result = OpenCpopDatasetLoader(config).run()
    _merge_result(outputs, warnings, dataset_result)

    melody_result = OpenCpopMidiImporter(config).run()
    _merge_result(outputs, warnings, melody_result)

    phrases = build_zh_phrases(config)
    phrase_path = save_zh_phrases(config, phrases)
    outputs["lyrics_zh_phrases"] = str(phrase_path)
    _merge_warnings(warnings, phrases)

    phonemes = phonemize_zh(config)
    phoneme_path = save_zh_phonemes(config, phonemes)
    outputs["phonemes_zh"] = str(phoneme_path)
    _merge_warnings(warnings, phonemes)

    alignment = align_zh_notes(config)
    alignment_path = save_zh_alignment(config, alignment)
    outputs["alignment_zh"] = str(alignment_path)
    _merge_warnings(warnings, alignment)

    score = build_opencpop_svs_score(config)
    score_path = save_opencpop_svs_score(config, score)
    outputs["svs_score"] = str(score_path)
    _merge_warnings(warnings, score)

    export_plan = build_diffsinger_export_plan(config)
    export_plan_path = save_diffsinger_export_plan(config, export_plan)
    outputs["diffsinger_export_plan"] = str(export_plan_path)
    for key, value in export_plan.get("expected_files", {}).items():
        outputs[f"diffsinger_{key}"] = str(value)
    _merge_warnings(warnings, export_plan)

    checkpoint_status = check_svs_checkpoints(config)
    checkpoint_status_path = save_checkpoint_status(config, checkpoint_status)
    outputs["checkpoint_status"] = str(checkpoint_status_path)
    _merge_warnings(warnings, checkpoint_status)

    render_plan = build_neural_svs_render_plan(config)
    render_plan_path = save_neural_svs_render_plan(config, render_plan)
    outputs["render_plan"] = str(render_plan_path)
    if render_plan.get("output_audio"):
        outputs["svs_vocal"] = str(render_plan["output_audio"])
    if render_plan.get("infer_report"):
        outputs["infer_report"] = str(render_plan["infer_report"])
    _merge_warnings(warnings, render_plan)

    status = str(render_plan.get("status") or "failed")
    if status == "success" and bool(config.get("svs", {}).get("rvc", {}).get("enabled", False)):
        rvc_result = apply_rvc_voice_conversion(config, render_plan.get("output_audio"))
        _merge_result(outputs, warnings, rvc_result)
        status = str(rvc_result.get("status") or status)

    return {
        "status": status,
        "outputs": outputs,
        "warnings": warnings,
        "message": (
            "OpenCpop Chinese SVS inference finished."
            if status == "success"
            else "OpenCpop Chinese SVS inference did not produce wav. Check warnings/report."
        ),
    }
