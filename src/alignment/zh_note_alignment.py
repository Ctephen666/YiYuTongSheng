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


def _is_slur_note(note: dict) -> bool:
    label = str(note.get("lyric") or note.get("syllable") or note.get("phoneme") or "").strip()
    return bool(note.get("is_slur") or note.get("prolongation")) or label in {"_", "-"}


def _is_rest(note: dict) -> bool:
    if _is_slur_note(note):
        return False
    pitch = str(note.get("pitch") or "").upper()
    return bool(note.get("is_rest")) or pitch in {"REST", "R", "SIL"} or note.get("midi") is None


def _duration(note: dict) -> float:
    if "duration" in note:
        try:
            return float(note["duration"])
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(note.get("end", 0.0)) - float(note.get("start", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _note_item(note: dict, index: int, token: dict | None, slur: bool) -> dict:
    lyric = "" if token is None else str(token.get("char") or "")
    phoneme = "" if token is None else str(token.get("phoneme") or token.get("pinyin") or "")
    return {
        "note_index": index,
        "pitch": note.get("pitch"),
        "midi": note.get("midi"),
        "start": note.get("start"),
        "end": note.get("end"),
        "duration": _duration(note),
        "lyric": lyric,
        "phoneme": phoneme,
        "is_rest": token is None,
        "is_slur": bool(slur),
        "prolongation": bool(slur),
        "alignment_rule": "one_char_one_note_with_slur" if token is not None else "rest_or_unassigned",
    }


def _melody_phrases(melody: Any) -> list[dict]:
    if not isinstance(melody, dict):
        return []
    phrases = melody.get("phrases", [])
    if isinstance(phrases, list):
        return [item for item in phrases if isinstance(item, dict)]
    notes = melody.get("notes") or melody.get("items") or []
    if isinstance(notes, list):
        return [{"id": 1, "notes": [item for item in notes if isinstance(item, dict)]}]
    return []


def _find_melody_phrase(melody_phrases: list[dict], phrase_id: Any, index: int) -> dict | None:
    for phrase in melody_phrases:
        if phrase.get("id") == phrase_id or str(phrase.get("id")) == str(phrase_id):
            return phrase
    if index < len(melody_phrases):
        return melody_phrases[index]
    return None


def _align_phrase(phrase: dict, melody_phrase: dict | None, phrase_index: int) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    tokens = [token for token in phrase.get("tokens", []) if isinstance(token, dict)]
    notes = []
    if melody_phrase and isinstance(melody_phrase.get("notes"), list):
        notes = [note for note in melody_phrase["notes"] if isinstance(note, dict)]

    if not notes:
        warnings.append(f"Phrase {phrase.get('id', phrase_index + 1)} has no notes to align.")
    if not tokens:
        warnings.append(f"Phrase {phrase.get('id', phrase_index + 1)} has no Chinese lyric tokens to align.")

    assignments: list[dict] = []
    token_index = 0
    last_token: dict | None = None
    for note_index, note in enumerate(notes):
        if _is_rest(note):
            assignments.append(_note_item(note, note_index, None, False))
            continue

        if _is_slur_note(note) and last_token is not None:
            assignments.append(_note_item(note, note_index, last_token, True))
            continue

        if token_index < len(tokens):
            token = tokens[token_index]
            last_token = token
            token_index += 1
            assignments.append(_note_item(note, note_index, token, False))
        elif last_token is not None:
            assignments.append(_note_item(note, note_index, last_token, True))
        else:
            assignments.append(_note_item(note, note_index, None, False))

    if token_index < len(tokens):
        warnings.append(
            f"Phrase {phrase.get('id', phrase_index + 1)} has {len(tokens) - token_index} lyric tokens without notes."
        )

    start_values = [item.get("start") for item in assignments if item.get("start") is not None]
    end_values = [item.get("end") for item in assignments if item.get("end") is not None]
    return {
        "id": phrase.get("id", phrase_index + 1),
        "zh": phrase.get("zh", ""),
        "note_count": len(notes),
        "token_count": len(tokens),
        "assigned_token_count": min(token_index, len(tokens)),
        "start": min(start_values) if start_values else melody_phrase.get("start") if melody_phrase else None,
        "end": max(end_values) if end_values else melody_phrase.get("end") if melody_phrase else None,
        "notes": assignments,
    }, warnings


def align_zh_notes(config: dict) -> dict:
    phoneme_path = _config_path(config, "phonemes_zh", "data/phoneme/phonemes_zh.json")
    melody_path = _config_path(config, "melody_notes", "data/score/melody_notes.json")
    phonemes = _read_json(phoneme_path, {})
    melody = _read_json(melody_path, {})
    melody_phrases = _melody_phrases(melody)

    warnings = []
    if isinstance(phonemes, dict):
        warnings.extend(phonemes.get("warnings", []))
    if isinstance(melody, dict):
        warnings.extend(melody.get("warnings", []))
    if not melody_phrases:
        warnings.append(f"No melody phrases found in {melody_path}.")

    aligned_phrases: list[dict] = []
    source_phrases = phonemes.get("phrases", []) if isinstance(phonemes, dict) else []
    for index, phrase in enumerate(source_phrases):
        if not isinstance(phrase, dict):
            continue
        source_notes = phrase.get("source_notes") if isinstance(phrase.get("source_notes"), list) else []
        if source_notes:
            melody_phrase = {
                "id": phrase.get("id"),
                "start": phrase.get("start"),
                "end": phrase.get("end"),
                "notes": source_notes,
            }
        else:
            melody_phrase = _find_melody_phrase(melody_phrases, phrase.get("id"), index)
        aligned, phrase_warnings = _align_phrase(phrase, melody_phrase, index)
        aligned_phrases.append(aligned)
        warnings.extend(phrase_warnings)

    seen_warnings: list[str] = []
    for warning in warnings:
        if warning not in seen_warnings:
            seen_warnings.append(warning)

    return {
        "dataset": "opencpop",
        "song_id": phonemes.get("song_id") if isinstance(phonemes, dict) else None,
        "strict_dataset_source": bool(phonemes.get("strict_dataset_source", True)) if isinstance(phonemes, dict) else True,
        "alignment_strategy": "zh_one_char_one_note_with_slur",
        "phrase_count": len(aligned_phrases),
        "phrases": aligned_phrases,
        "warnings": seen_warnings,
    }


def save_zh_alignment(config: dict, payload: dict) -> Path:
    output_path = _config_path(config, "alignment_zh", "data/alignment/zh_note_alignment.json")
    _write_json(output_path, payload)
    return output_path


class ZhNoteAligner:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = align_zh_notes(self.config)
        output_path = save_zh_alignment(self.config, payload)
        status = "success" if payload.get("phrases") else "warning"
        return {
            "status": status,
            "outputs": {"alignment_zh": str(output_path)},
            "warnings": payload.get("warnings", []),
            "message": f"Prepared Chinese note alignment for {payload.get('phrase_count', 0)} phrases.",
        }
