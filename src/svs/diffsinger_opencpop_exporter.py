from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent


def _root(config: dict) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def _resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        text = str(value).replace("\\", "/")
        if text.startswith("/data/"):
            project_relative = _root(config) / text.lstrip("/")
            if project_relative.exists():
                return project_relative
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


def _midi_to_note_name(value: Any) -> str:
    try:
        midi = int(value)
    except (TypeError, ValueError):
        return "rest"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _note_pitch(note: dict) -> str:
    if note.get("is_rest"):
        return "rest"
    pitch = str(note.get("pitch") or "").strip()
    if pitch and pitch.lower() not in {"none", "null"}:
        return pitch
    return _midi_to_note_name(note.get("midi"))


def _duration_value(note: dict) -> float:
    try:
        duration = float(note.get("duration"))
    except (TypeError, ValueError):
        try:
            duration = float(note.get("end")) - float(note.get("start"))
        except (TypeError, ValueError):
            duration = 0.0
    return max(duration, 0.001)


def _duration_text(duration: float) -> str:
    return f"{max(float(duration), 0.001):.6f}"


def _item_name(song_id: Any, phrase_id: Any, phrase_index: int) -> str:
    if str(phrase_id).isdigit():
        return f"opencpop_{song_id or 'sample'}_phrase_{int(phrase_id):03d}"
    return f"opencpop_{song_id or 'sample'}_phrase_{phrase_index:03d}"


def _build_word_level_phrase_item(score: dict, phrase: dict, phrase_index: int) -> tuple[dict | None, list[dict], list[str]]:
    warnings: list[str] = []
    rows: list[dict] = []
    text_parts: list[str] = []
    note_groups: list[list[str]] = []
    duration_groups: list[list[str]] = []

    phrase_id = phrase.get("id") or phrase_index
    song_id = score.get("song_id") if isinstance(score, dict) else "sample"

    if not isinstance(phrase, dict):
        return None, rows, warnings

    for note in phrase.get("notes", []):
        if not isinstance(note, dict):
            continue

        pitch = _note_pitch(note)
        duration = _duration_text(_duration_value(note))
        lyric = str(note.get("lyric") or "").strip()
        is_rest = bool(note.get("is_rest")) or pitch == "rest"
        is_slur = bool(note.get("is_slur"))

        if is_slur and note_groups and not is_rest:
            note_groups[-1].append(pitch)
            duration_groups[-1].append(duration)
            word_index = len(note_groups) - 1
        else:
            if is_rest:
                pitch = "rest"
                text_token = lyric if lyric in {"AP", "SP"} else "SP"
            else:
                if not lyric:
                    warnings.append(f"Skipping note with empty lyric in phrase {phrase_id}: {note}")
                    continue
                text_token = lyric

            text_parts.append(text_token)
            note_groups.append([pitch])
            duration_groups.append([duration])
            word_index = len(note_groups) - 1

        rows.append(
            {
                "phrase_id": phrase_id,
                "word_index": word_index,
                "lyric": lyric,
                "note": pitch,
                "duration": duration,
                "is_rest": is_rest,
                "is_slur": is_slur,
            }
        )

    if not note_groups:
        warnings.append(f"No word-level DiffSinger note groups were generated for phrase {phrase_id}.")
        return None, rows, warnings

    duration_total = sum(float(value) for group in duration_groups for value in group)
    return (
        {
            "item_name": _item_name(song_id, phrase_id, phrase_index),
            "text": "".join(text_parts),
            "notes": " | ".join(" ".join(group) for group in note_groups),
            "notes_duration": " | ".join(" ".join(group) for group in duration_groups),
            "start": float(phrase.get("start", 0.0) or 0.0),
            "end": float(phrase.get("end", 0.0) or 0.0),
            "phrase_id": phrase_id,
            "input_type": "word",
            "duration_total": round(duration_total, 6),
        },
        rows,
        warnings,
    )


def _build_diffsinger_inputs(score: dict) -> tuple[list[dict], list[dict], list[str]]:
    warnings: list[str] = []
    inputs: list[dict] = []
    rows: list[dict] = []

    phrases = score.get("phrases", []) if isinstance(score, dict) else []
    for phrase_index, phrase in enumerate(phrases, start=1):
        item, phrase_rows, phrase_warnings = _build_word_level_phrase_item(score, phrase, phrase_index)
        rows.extend(phrase_rows)
        warnings.extend(phrase_warnings)
        if item is not None:
            inputs.append(item)

    if not inputs:
        warnings.append("No word-level phrase DiffSinger inputs were generated from opencpop_svs_score.json.")

    return inputs, rows, warnings


def build_diffsinger_export_plan(config: dict) -> dict:
    score_path = _config_path(config, "svs_score", "data/svs/opencpop_svs_score.json")
    score = _read_json(score_path, {})
    diffsinger_config = config.get("svs", {}).get("diffsinger", {}) if isinstance(config.get("svs", {}), dict) else {}

    target_dir = diffsinger_config.get("export_dir") or "data/svs/diffsinger_opencpop"
    target_dir_path = _resolve(config, target_dir)
    song_id = score.get("song_id", "sample") if isinstance(score, dict) else "sample"
    input_path = _config_path(config, "diffsinger_input", f"data/svs/diffsinger_opencpop/diffsinger_input_{song_id}.json")
    word_tsv_path = _config_path(config, "diffsinger_phoneme_tsv", f"data/svs/diffsinger_opencpop/phonemes_{song_id}.tsv")
    notes_csv_path = _config_path(config, "diffsinger_notes_csv", f"data/svs/diffsinger_opencpop/notes_{song_id}.csv")

    diffsinger_inputs, rows, input_warnings = _build_diffsinger_inputs(score)

    warnings = []
    if isinstance(score, dict):
        warnings.extend(score.get("warnings", []))
    warnings.extend(input_warnings)

    return {
        "format": "diffsinger_opencpop_export_plan.v3",
        "dry_run": False,
        "executed": True,
        "dataset": "opencpop",
        "song_id": score.get("song_id") if isinstance(score, dict) else None,
        "language": "zh",
        "score_input": str(score_path),
        "target_export_dir": str(target_dir_path),
        "diffsinger_inputs": diffsinger_inputs,
        "word_rows": rows,
        "expected_files": {
            "input_json": str(input_path),
            "phoneme_tsv": str(word_tsv_path),
            "notes_csv": str(notes_csv_path),
            "score_json": str(score_path),
        },
        "diffsinger_config": diffsinger_config,
        "warnings": list(dict.fromkeys(warnings)),
    }


def save_diffsinger_export_plan(config: dict, payload: dict) -> Path:
    input_path = Path(payload["expected_files"]["input_json"])
    word_tsv_path = Path(payload["expected_files"]["phoneme_tsv"])
    notes_csv_path = Path(payload["expected_files"]["notes_csv"])

    _write_json(input_path, {"diffsinger_inputs": payload.get("diffsinger_inputs", [])})

    ensure_parent(word_tsv_path)
    with word_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("index\tphrase_id\tword_index\tlyric\tnote\tduration\tis_rest\tis_slur\n")
        for index, row in enumerate(payload.get("word_rows", [])):
            handle.write(
                f"{index}\t{row.get('phrase_id','')}\t{row.get('word_index','')}\t{row.get('lyric','')}\t"
                f"{row.get('note','')}\t{row.get('duration','')}\t{row.get('is_rest','')}\t{row.get('is_slur','')}\n"
            )

    ensure_parent(notes_csv_path)
    with notes_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "phrase_id", "word_index", "lyric", "note", "duration", "is_rest", "is_slur"],
        )
        writer.writeheader()
        for index, row in enumerate(payload.get("word_rows", [])):
            writer.writerow({"index": index, **row})

    output_path = _config_path(config, "diffsinger_export_plan", "data/svs/diffsinger_opencpop_export_plan.json")
    plan_payload = dict(payload)
    plan_payload.pop("word_rows", None)
    _write_json(output_path, plan_payload)
    return output_path


class DiffSingerOpenCpopExporter:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = build_diffsinger_export_plan(self.config)
        output_path = save_diffsinger_export_plan(self.config, payload)
        generated = payload.get("expected_files", {})
        status = "success" if payload.get("diffsinger_inputs") else "warning"
        return {
            "status": status,
            "outputs": {
                "diffsinger_export_plan": str(output_path),
                "diffsinger_inputs": generated.get("input_json"),
                "diffsinger_word_tsv": generated.get("phoneme_tsv"),
                "diffsinger_notes_csv": generated.get("notes_csv"),
            },
            "warnings": payload.get("warnings", []),
            "message": "Prepared DiffSinger word-level inference input.",
        }
