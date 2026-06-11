from __future__ import annotations

import shutil
from pathlib import Path

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

    def _path_label(self, key: str) -> str:
        return str(self.config.get("paths", {}).get(key, key))

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
        phrase_map_path = path_from_config(self.config, "phrase_map")
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

    def run(self) -> dict:
        midi_path = path_from_config(self.config, "opencpop_midi")
        if not midi_path.exists():
            raise FileNotFoundError(f"Missing OpenCpop MIDI: {self._path_label('opencpop_midi')}")

        notes = self._load_notes(midi_path)
        if not notes:
            raise RuntimeError(f"OpenCpop MIDI contains no non-drum notes: {midi_path}")

        slices = self._phrase_map_slices(notes) or self._auto_slices(notes)
        phrases = [
            self._build_phrase(phrase_id, start, end, notes)
            for phrase_id, start, end in slices
            if end > start
        ]

        if not phrases:
            raise RuntimeError(f"OpenCpop MIDI produced no phrases: {midi_path}")

        melody_midi = path_from_config(self.config, "melody_midi")
        if should_write(melody_midi, self.config):
            ensure_parent(melody_midi)
            shutil.copyfile(midi_path, melody_midi)

        melody_notes = path_from_config(self.config, "melody_notes")
        write_json(
            melody_notes,
            {
                "source": "opencpop_midi",
                "midi": self._path_label("opencpop_midi"),
                "note_count": len(notes),
                "phrases": phrases,
            },
            self.config,
        )

        return {
            "status": "success",
            "outputs": {
                "melody_midi": str(melody_midi),
                "melody_notes": str(melody_notes),
            },
            "message": f"Imported {len(notes)} MIDI notes into {len(phrases)} phrases.",
        }
