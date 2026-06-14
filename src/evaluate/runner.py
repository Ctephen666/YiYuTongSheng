from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.evaluate.audio_metrics import audio_quality_metrics, waveform_difference_metrics
from src.evaluate.pitch_metrics import pitch_preservation_metrics
from src.evaluate.report import compute_overall, write_json_report, write_markdown_report
from src.evaluate.speaker_metrics import speaker_similarity_metrics
from src.evaluate.text_metrics import intelligibility_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(path: str | Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read evaluate config.")
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Evaluate config is not a mapping: {path}")
    return data


def _resolve(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _collect_warnings(*sections: Any) -> list[str]:
    warnings: list[str] = []
    for section in sections:
        if isinstance(section, dict):
            for item in section.get("warnings", []) or []:
                if item not in warnings:
                    warnings.append(str(item))
            for value in section.values():
                if isinstance(value, dict):
                    for item in value.get("warnings", []) or []:
                        if item not in warnings:
                            warnings.append(str(item))
    return warnings


def run_evaluate(config_path: str = "configs/evaluate.yaml", overrides: dict | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    config = _load_config(config_path)
    if overrides:
        config = _deep_update(config, overrides)

    inputs = config.get("inputs", {}) if isinstance(config.get("inputs", {}), dict) else {}
    output = config.get("output", {}) if isinstance(config.get("output", {}), dict) else {}
    pitch_cfg = config.get("pitch", {}) if isinstance(config.get("pitch", {}), dict) else {}
    text_cfg = config.get("text", {}) if isinstance(config.get("text", {}), dict) else {}
    speaker_cfg = config.get("speaker", {}) if isinstance(config.get("speaker", {}), dict) else {}

    svs_wav = _resolve(inputs.get("svs_wav"))
    svc_wav = _resolve(inputs.get("svc_wav"))
    final_wav = _resolve(inputs.get("final_wav"))
    score_json = _resolve(inputs.get("score_json"))
    reference_wav = _resolve(inputs.get("reference_wav"))
    json_report = _resolve(output.get("json_report") or "data/evaluate/evaluate_report.json")
    markdown_path = _resolve(output.get("markdown_report") or "data/evaluate/evaluate_report.md")

    audio_quality = {}
    for name, path in [("reference", reference_wav), ("svs", svs_wav), ("svc", svc_wav), ("final", final_wav)]:
        if path is None:
            audio_quality[name] = {"is_valid_audio": False, "warnings": [f"{name} path is not configured."]}
        else:
            try:
                audio_quality[name] = audio_quality_metrics(path)
            except Exception as exc:  # noqa: BLE001
                audio_quality[name] = {"is_valid_audio": False, "warnings": [f"{name} audio metric failed: {exc}"]}

    try:
        conversion_integrity = waveform_difference_metrics(svs_wav or '', svc_wav or '')
    except Exception as exc:  # noqa: BLE001
        conversion_integrity = {'warnings': [f'Conversion integrity metric failed: {exc}'], 'likely_same_audio': None}

    try:
        pitch = pitch_preservation_metrics(
            svs_wav or "",
            svc_wav or "",
            frame_ms=float(pitch_cfg.get("frame_ms", 20)),
            min_f0=float(pitch_cfg.get("min_f0", 50)),
            max_f0=float(pitch_cfg.get("max_f0", 1100)),
        )
    except Exception as exc:  # noqa: BLE001
        pitch = {"warnings": [f"Pitch preservation metric failed: {exc}"]}

    try:
        intelligibility = intelligibility_metrics(
            score_json or "",
            recognized_text=text_cfg.get("recognized_text"),
            use_asr=bool(text_cfg.get("use_asr", False)),
        )
    except Exception as exc:  # noqa: BLE001
        intelligibility = {"warnings": [f"Intelligibility metric failed: {exc}"], "asr_available": False}

    try:
        speaker = speaker_similarity_metrics(
            speaker_cfg.get("target_reference_dir", "data/svc/target_voice_reference"),
            reference_wav or "",
            svc_wav or "",
        )
    except Exception as exc:  # noqa: BLE001
        speaker = {"warnings": [f"Speaker similarity metric failed: {exc}"], "speaker_embedding_available": False}

    report = {
        "format": "yiyutongsheng_evaluate_report.v1",
        "status": "success",
        "inputs": {key: (value if key == "song_id" else str(_resolve(value)) if isinstance(value, str) else value) for key, value in inputs.items()},
        "audio_quality": audio_quality,
        "conversion_integrity": conversion_integrity,
        "pitch_preservation": pitch,
        "intelligibility": intelligibility,
        "speaker_similarity": speaker,
        "overall": {},
        "warnings": [],
        "errors": errors,
    }
    warnings.extend(_collect_warnings(audio_quality, conversion_integrity, pitch, intelligibility, speaker))
    report["warnings"] = warnings
    report["overall"] = compute_overall(report)

    try:
        if json_report is not None:
            write_json_report(report, json_report)
        if markdown_path is not None:
            write_markdown_report(report, markdown_path)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "partial"
        report["errors"].append(f"Failed to write report: {exc}")
    return report
