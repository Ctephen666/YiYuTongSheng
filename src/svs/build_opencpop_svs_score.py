from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent


def _root(config: dict) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def _resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _root(config) / path


def _config_path(config: dict, key: str, default: str) -> Path:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs", {}), dict) else {}
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    return _resolve(config, outputs.get(key) or paths.get(key) or default)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_opencpop_svs_score(config: dict) -> dict:
    alignment_path = _config_path(config, "alignment_zh", "data/alignment/zh_note_alignment.json")
    item_path = _config_path(config, "opencpop_item", "data/dataset_manifest/opencpop_item_2001.json")
    alignment = _read_json(alignment_path, {})
    item = _read_json(item_path, {})

    warnings = []
    if isinstance(alignment, dict):
        warnings.extend(alignment.get("warnings", []))
    if isinstance(item, dict):
        warnings.extend(item.get("warnings", []))

    phrases: list[dict] = []
    for phrase in alignment.get("phrases", []) if isinstance(alignment, dict) else []:
        if not isinstance(phrase, dict):
            continue
        notes = []
        for note in phrase.get("notes", []):
            if not isinstance(note, dict):
                continue
            notes.append(
                {
                    "pitch": note.get("pitch"),
                    "midi": note.get("midi"),
                    "start": note.get("start"),
                    "end": note.get("end"),
                    "duration": note.get("duration"),
                    "lyric": note.get("lyric", ""),
                    "phoneme": note.get("phoneme", ""),
                    "is_rest": bool(note.get("is_rest")),
                    "is_slur": bool(note.get("is_slur")),
                }
            )
        phrases.append(
            {
                "id": phrase.get("id", len(phrases) + 1),
                "zh": phrase.get("zh", ""),
                "start": phrase.get("start"),
                "end": phrase.get("end"),
                "notes": notes,
            }
        )

    seen_warnings: list[str] = []
    for warning in warnings:
        if warning not in seen_warnings:
            seen_warnings.append(warning)

    dataset_root = config.get("dataset", {}).get("root") or config.get("inputs", {}).get("opencpop_root")
    return {
        "format": "opencpop_svs_score.v1",
        "dataset": "opencpop",
        "data_source": "opencpop",
        "dataset_root": str(dataset_root or item.get("dataset_root", "/data/dataset/opencpop")),
        "song_id": alignment.get("song_id") if isinstance(alignment, dict) else item.get("song_id"),
        "strict_dataset_source": bool(config.get("dataset", {}).get("strict_dataset_source", True)),
        "language": "zh",
        "timing_unit": "seconds",
        "score_type": "diffsinger_inference_score",
        "midi_path": item.get("midi_path", "") if isinstance(item, dict) else "",
        "wav_path": item.get("wav_path", "") if isinstance(item, dict) else "",
        "textgrid_path": item.get("textgrid_path", "") if isinstance(item, dict) else "",
        "phrase_count": len(phrases),
        "phrases": phrases,
        "warnings": seen_warnings,
    }


def save_opencpop_svs_score(config: dict, payload: dict) -> Path:
    output_path = _config_path(config, "svs_score", "data/svs/opencpop_svs_score.json")
    _write_json(output_path, payload)
    return output_path


class OpenCpopSvsScoreBuilder:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = build_opencpop_svs_score(self.config)
        output_path = save_opencpop_svs_score(self.config, payload)
        status = "success" if payload.get("phrase_count") else "warning"
        return {
            "status": status,
            "outputs": {"svs_score": str(output_path)},
            "warnings": payload.get("warnings", []),
            "message": f"Built OpenCpop Chinese SVS score with {payload.get('phrase_count', 0)} phrases.",
        }
