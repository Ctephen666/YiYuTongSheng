from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, path_from_config, should_write
from src.common.json_utils import read_json, write_json


def _pitch_name(midi_number: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_number // 12 - 1
    return f"{names[midi_number % 12]}{octave}"


def _round_time(value: float) -> float:
    return round(float(value), 6)


class OpenCpopMidiImporter:
    """Import OpenCpop MIDI as phrase-level melody notes."""

    def __init__(self, config: dict):
        self.config = config

    def _project_root(self) -> Path:
        return Path(self.config.get("_project_root", ".")).resolve()

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self._project_root() / path

    def _config_path(self, key: str, default: str | None = None) -> Path:
        outputs = self.config.get("outputs", {}) if isinstance(self.config.get("outputs", {}), dict) else {}
        paths = self.config.get("paths", {}) if isinstance(self.config.get("paths", {}), dict) else {}
        value = outputs.get(key) or paths.get(key) or default
        if value is None:
            return path_from_config(self.config, key)
        return self._resolve(value)

    def _path_label(self, key: str) -> str:
        outputs = self.config.get("outputs", {}) if isinstance(self.config.get("outputs", {}), dict) else {}
        paths = self.config.get("paths", {}) if isinstance(self.config.get("paths", {}), dict) else {}
        return str(outputs.get(key) or paths.get(key) or key)

    def _strict(self) -> bool:
        return bool(self.config.get("dataset", {}).get("strict_dataset_source", True))

    def _load_item(self) -> dict:
        item_path = self._config_path("opencpop_item", "data/dataset_manifest/opencpop_item_2001.json")
        if not item_path.exists():
            return {}
        try:
            return json.loads(item_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _select_midi_path(self) -> tuple[Path | None, str, list[str]]:
        warnings: list[str] = []
        item = self._load_item()
        item_midi = str(item.get("midi_path") or "").strip() if isinstance(item, dict) else ""
        if item_midi:
            return Path(item_midi), item_midi, warnings

        if self._strict():
            warnings.append(
                "Strict dataset source is enabled; OpenCpop item has no midi_path and paths.opencpop_midi fallback was not used."
            )
            return None, "", warnings

        fallback = self._config_path("opencpop_midi", "data/dataset/opencpop/midis/2001.midi")
        warnings.append("Strict dataset source is disabled; using paths.opencpop_midi fallback.")
        return fallback, self._path_label("opencpop_midi"), warnings

    def _load_pretty_midi(self):
        try:
            import pretty_midi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pretty_midi is not installed. Install dependencies with: "
                "pip install pretty_midi librosa soundfile numpy"
            ) from exc
        return pretty_midi

    def _load_notes(self, midi_path: Path) -> list[dict]:
        pretty_midi = self._load_pretty_midi()
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                start = float(note.start)
                end = float(note.end)
                if end <= start:
                    continue
                notes.append(
                    {
                        "pitch": _pitch_name(int(note.pitch)),
                        "midi": int(note.pitch),
                        "start": _round_time(start),
                        "end": _round_time(end),
                        "duration": _round_time(end - start),
                        "velocity": int(note.velocity),
                    }
                )

        notes.sort(key=lambda item: (item["start"], item["end"], item["midi"]))
        return notes

    def _phrase_map_slices(self, notes: list[dict]) -> list[tuple[int, int, int]]:
        if self._strict():
            return []

        phrase_map_path = self._config_path("phrase_map", "data/lyrics/phrase_map.json")
        phrase_map = read_json(phrase_map_path, default={})
        phrases = phrase_map.get("phrases", []) if isinstance(phrase_map, dict) else []

        slices = []
        for index, phrase in enumerate(phrases, start=1):
            if not isinstance(phrase, dict):
                continue
            if "note_start" not in phrase or "note_end" not in phrase:
                return []

            try:
                start = int(phrase["note_start"])
                end = int(phrase["note_end"])
            except (TypeError, ValueError):
                return []

            start = max(0, min(start, len(notes)))
            end = max(start + 1, min(end, len(notes)))
            if start >= len(notes) or end <= start:
                continue

            phrase_id = phrase.get("id", index)
            try:
                phrase_id = int(phrase_id)
            except (TypeError, ValueError):
                phrase_id = index

            slices.append((phrase_id, start, end))

        return slices

    def _auto_slices(self, notes: list[dict]) -> list[tuple[int, int, int]]:
        if not notes:
            return []

        slices = []
        start_index = 0
        phrase_id = 1

        for index in range(1, len(notes)):
            gap = float(notes[index]["start"]) - float(notes[index - 1]["end"])
            if gap > 0.6:
                slices.append((phrase_id, start_index, index))
                phrase_id += 1
                start_index = index

        slices.append((phrase_id, start_index, len(notes)))
        return slices

    def _build_phrase(self, phrase_id: int, start_index: int, end_index: int, notes: list[dict]) -> dict:
        phrase_notes = notes[start_index:end_index]
        start = min(float(note["start"]) for note in phrase_notes)
        end = max(float(note["end"]) for note in phrase_notes)
        return {
            "id": phrase_id,
            "start": _round_time(start),
            "end": _round_time(end),
            "duration": _round_time(end - start),
            "note_start": start_index,
            "note_end": end_index,
            "notes": phrase_notes,
        }

    def _write_empty_melody(self, midi_label: str, warnings: list[str]) -> dict:
        melody_notes = self._config_path("melody_notes", "data/score/melody_notes.json")
        write_json(
            melody_notes,
            {
                "source": "opencpop_midi_missing",
                "midi": midi_label,
                "note_count": 0,
                "phrases": [],
                "warnings": warnings,
            },
            self.config,
        )
        return {
            "status": "warning",
            "outputs": {"melody_notes": str(melody_notes)},
            "warnings": warnings,
            "message": "OpenCpop MIDI was not available; wrote empty melody structure so downstream validation can report the missing input.",
        }

    def run(self) -> dict:
        midi_path, midi_label, warnings = self._select_midi_path()
        if midi_path is None:
            return self._write_empty_melody(midi_label, warnings)

        if not midi_path.exists():
            warnings.append(f"Missing OpenCpop MIDI in dataset: {midi_path}")
            if self._strict():
                return self._write_empty_melody(str(midi_path), warnings)
            raise FileNotFoundError(f"Missing OpenCpop MIDI: {midi_path}")

        notes = self._load_notes(midi_path)
        if not notes:
            warnings.append(f"OpenCpop MIDI contains no non-drum notes: {midi_path}")
            return self._write_empty_melody(str(midi_path), warnings)

        slices = self._phrase_map_slices(notes) or self._auto_slices(notes)
        phrases = [
            self._build_phrase(phrase_id, start, end, notes)
            for phrase_id, start, end in slices
            if end > start
        ]

        if not phrases:
            warnings.append(f"OpenCpop MIDI produced no phrases: {midi_path}")
            return self._write_empty_melody(str(midi_path), warnings)

        melody_midi = self._config_path("melody_midi", "data/score/melody.mid")
        if should_write(melody_midi, self.config):
            ensure_parent(melody_midi)
            shutil.copyfile(midi_path, melody_midi)

        melody_notes = self._config_path("melody_notes", "data/score/melody_notes.json")
        write_json(
            melody_notes,
            {
                "source": "opencpop_midi",
                "midi": str(midi_path),
                "note_count": len(notes),
                "phrases": phrases,
                "warnings": warnings,
            },
            self.config,
        )

        return {
            "status": "success" if not warnings else "warning",
            "outputs": {
                "melody_midi": str(melody_midi),
                "melody_notes": str(melody_notes),
            },
            "warnings": warnings,
            "message": f"Imported {len(notes)} MIDI notes into {len(phrases)} phrases.",
        }
