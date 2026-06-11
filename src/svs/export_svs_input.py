from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import path_from_config
from src.common.json_utils import read_json, write_json


class SVSInputExporter:
    """Export syllable-group alignment into a compact SVS input file."""

    def __init__(self, config: dict):
        self.config = config

    def _config_path_or_default(self, key: str, default: str) -> Path:
        try:
            return Path(path_from_config(self.config, key))
        except Exception:
            return Path(default)

    def _alignment_path(self) -> Path:
        return self._config_path_or_default(
            "note_lyric_alignment",
            "data/alignment/note_lyric_alignment.json",
        )

    def _output_path(self) -> Path:
        return self._config_path_or_default(
            "svs_input",
            "data/svs/svs_input.json",
        )

    def _float_or_zero(self, value: Any) -> float:
        try:
            return round(float(value), 5)
        except Exception:
            return 0.0

    def _int_or_none(self, value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _clean_note(self, note: dict) -> dict:
        start = self._float_or_zero(note.get("start", 0.0))
        end = self._float_or_zero(note.get("end", 0.0))
        duration = note.get("duration", end - start)

        return {
            "index": self._int_or_none(note.get("index")),
            "pitch": note.get("pitch", ""),
            "midi": self._int_or_none(note.get("midi")),
            "start": start,
            "end": end,
            "duration": self._float_or_zero(duration),
            "source_lyric": note.get("lyric", ""),
            "source_syllable": note.get("syllable", ""),
        }

    def _clean_merged_units(self, merged_units: list[dict]) -> list[dict]:
        cleaned = []

        for unit in merged_units:
            cleaned.append(
                {
                    "unit": unit.get("unit", ""),
                    "word": unit.get("word", ""),
                    "word_index": unit.get("word_index"),
                    "syllable_index": unit.get("syllable_index"),
                    "phonemes": unit.get("phonemes", []),
                    "phoneme_source": unit.get("phoneme_source", ""),
                }
            )

        return cleaned

    def _clean_unit(self, item: dict, unit_index: int) -> dict:
        notes = [self._clean_note(note) for note in item.get("notes", [])]

        start = notes[0]["start"] if notes else 0.0
        end = notes[-1]["end"] if notes else 0.0
        duration = round(end - start, 5) if notes else 0.0

        return {
            "unit_index": unit_index,
            "word": item.get("word", ""),
            "syllable": item.get("unit", ""),
            "phonemes": item.get("phonemes", []),
            "merged_units": self._clean_merged_units(item.get("merged_units", [])),
            "note_count": len(notes),
            "start": start,
            "end": end,
            "duration": duration,
            "notes": notes,
        }

    def _phrase_time_range(self, units: list[dict]) -> tuple[float, float, float]:
        starts = [unit["start"] for unit in units if unit.get("note_count", 0) > 0]
        ends = [unit["end"] for unit in units if unit.get("note_count", 0) > 0]

        if not starts or not ends:
            return 0.0, 0.0, 0.0

        start = min(starts)
        end = max(ends)
        duration = round(end - start, 5)
        return start, end, duration

    def _flatten_notes(self, units: list[dict]) -> list[dict]:
        flat = []

        for unit in units:
            for note in unit.get("notes", []):
                flat.append(
                    {
                        **note,
                        "word": unit.get("word", ""),
                        "syllable": unit.get("syllable", ""),
                        "phonemes": unit.get("phonemes", []),
                    }
                )

        return flat

    def _convert_phrase(self, phrase: dict) -> dict:
        items = phrase.get("items", [])

        if not isinstance(items, list):
            items = []

        units = [
            self._clean_unit(item, unit_index=index)
            for index, item in enumerate(items)
        ]

        start, end, duration = self._phrase_time_range(units)
        flat_notes = self._flatten_notes(units)

        return {
            "id": phrase.get("id"),
            "zh": phrase.get("zh", ""),
            "text": phrase.get("text", ""),
            "unit_type": "syllable_group",
            "status": phrase.get("status", "success"),
            "start": start,
            "end": end,
            "duration": duration,
            "unit_count": len(units),
            "note_count": len(flat_notes),
            "raw_unit_count": phrase.get("raw_unit_count", phrase.get("unit_count", len(units))),
            "merge_info": phrase.get("merge_info", {}),
            "units": units,
            "notes": flat_notes,
        }

    def run(self) -> dict:
        alignment_path = self._alignment_path()
        output_path = self._output_path()

        alignment_data = read_json(alignment_path, {"phrases": []})
        phrases = alignment_data.get("phrases", [])

        if not phrases:
            raise RuntimeError(
                f"note_lyric_alignment.json contains no phrases: {alignment_path}"
            )

        output_phrases = [
            self._convert_phrase(phrase)
            for phrase in phrases
        ]

        output = {
            "source": "note_lyric_alignment",
            "format": "svs_input_v1",
            "language": "en",
            "unit_type": "syllable_group",
            "description": (
                "SVS-ready input exported from English syllable-group alignment. "
                "Each unit contains word, syllable, phonemes, and aligned note timing."
            ),
            "phrases": output_phrases,
        }

        write_json(output_path, output, self.config)

        return {
            "status": "success",
            "outputs": {
                "svs_input": str(output_path),
            },
            "message": "Exported SVS input from note-lyric alignment.",
        }