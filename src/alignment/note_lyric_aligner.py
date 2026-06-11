from __future__ import annotations

import re

from src.common.io_utils import path_from_config, project_root, require_or_mock_input, write_text
from src.common.json_utils import read_json, write_json


ELONGATION_WORDS = {"reach", "you", "dream", "sky", "night", "madness"}


class NoteLyricAligner:
    """Align target-language lyric words to melody notes."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with melody notes, phonemes, and alignment output paths.
        Output:
            NoteLyricAligner instance.
        TODO:
            Add dynamic programming constraints for stress, duration, and note splitting.
        """
        self.config = config

    def split_units(self, text: str) -> list[str]:
        """Split selected English lyric text into space-free word units."""
        units = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        split_units: list[str] = []
        for unit in units:
            split_units.extend(part for part in unit.split() if part)
        return split_units

    def selected_text(self, phrase: dict) -> str:
        """Return selected lyric text from current string or legacy dict format."""
        selected = phrase.get("selected", "")
        if isinstance(selected, dict):
            selected = selected.get("text", "")
        return str(phrase.get("text") or selected)

    def normalized_unit(self, unit: str) -> str:
        """Normalize an English unit for elongation-word matching."""
        return re.sub(r"[^a-z]", "", unit.lower())

    def elongation_order(self, units: list[str]) -> list[int]:
        """Return unit indexes that should receive extra notes first."""
        late_start = max(0, len(units) - 4)
        preferred = [
            index
            for index in range(len(units) - 1, -1, -1)
            if index >= late_start and self.normalized_unit(units[index]) in ELONGATION_WORDS
        ]
        if units and (len(units) - 1) not in preferred:
            preferred.append(len(units) - 1)
        if not preferred:
            preferred = [len(units) - 1]
        return preferred

    def note_counts_per_unit(self, units: list[str], note_count: int) -> list[int]:
        """Allocate at least one note per unit, then distribute extra notes to stretchable words."""
        counts = [1 for _ in units]
        extra_notes = note_count - len(units)
        if extra_notes <= 0:
            return counts

        stretch_indexes = self.elongation_order(units)
        for extra_index in range(extra_notes):
            counts[stretch_indexes[extra_index % len(stretch_indexes)]] += 1
        return counts

    def align_phrase(self, units: list[str], notes: list[dict]) -> dict:
        """Align one phrase by assigning word units to ordered melody notes.

        Input:
            units: Space-free English word units.
            notes: Melody note dictionaries.
        Output:
            Alignment dict with status, items, or failure reason.
        TODO:
            Replace with dynamic programming and phoneme duration prediction.
        """
        unit_count = len(units)
        note_count = len(notes)
        if unit_count > note_count:
            return {
                "status": "failed",
                "reason": "too_many_units",
                "suggestion": "rewrite target lyric shorter",
                "unit_count": unit_count,
                "note_count": note_count,
                "items": [],
            }

        counts = self.note_counts_per_unit(units, note_count)
        items = []
        note_index = 0
        for unit, count in zip(units, counts):
            assigned = notes[note_index : note_index + count]
            note_index += count
            items.append({"unit": unit, "notes": assigned})

        return {
            "status": "success",
            "unit_count": unit_count,
            "note_count": note_count,
            "items": items,
        }

    def note_preview(self, note: dict) -> str:
        """Format one note for human-readable preview output."""
        pitch = note.get("pitch", "?")
        start = float(note.get("start", 0.0))
        end = float(note.get("end", 0.0))
        return f"{pitch} {start:.3f}-{end:.3f}"

    def item_preview(self, item: dict) -> str:
        """Format one aligned unit and its note assignment."""
        notes = item.get("notes", [])
        if not notes:
            return f"{item.get('unit', '')} -> no note"
        line = f"{item.get('unit', '')} -> {self.note_preview(notes[0])}"
        if len(notes) > 1:
            extras = ", ".join(self.note_preview(note) for note in notes[1:])
            line = f"{line} + extra notes {extras}"
        return line

    def build_preview(self, alignments: list[dict]) -> str:
        """Build alignment_preview.txt content."""
        lines: list[str] = []
        for phrase in alignments:
            lines.append(f"[{phrase.get('id')}] {phrase.get('zh', '')}")
            lines.append(f"EN: {phrase.get('text', '')}")
            if phrase.get("status") == "failed":
                lines.append(
                    f"FAILED: {phrase.get('reason')} | {phrase.get('suggestion')}"
                )
            else:
                lines.extend(self.item_preview(item) for item in phrase.get("items", []))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def run(self) -> dict:
        """Create note_lyric_alignment.json and alignment_preview.txt.

        Input:
            data/score/melody_notes.json and data/phoneme/phonemes_target.json.
        Output:
            data/alignment/note_lyric_alignment.json and data/alignment/alignment_preview.txt.
        TODO:
            Support many-to-one and one-to-many note/unit strategies with scoring.
        """
        melody_path = path_from_config(self.config, "melody_notes")
        phoneme_path = path_from_config(self.config, "phonemes_target")
        phrase_map_path = path_from_config(self.config, "phrase_map")
        melody_status = require_or_mock_input(melody_path, self.config, "melody notes")
        phoneme_status = require_or_mock_input(phoneme_path, self.config, "target phonemes")

        melody = read_json(
            melody_path,
            {"phrases": [{"id": 1, "notes": [{"pitch": "C4"}, {"pitch": "D4"}, {"pitch": "E4"}]}]},
        )
        phonemes = read_json(
            phoneme_path,
            {"phrases": [{"id": 1, "text": "go see sea"}]},
        )
        phrase_map = read_json(phrase_map_path, {"phrases": []})
        notes_by_id = {phrase.get("id"): phrase.get("notes", []) for phrase in melody.get("phrases", [])}
        zh_by_id = {phrase.get("id"): phrase.get("zh", "") for phrase in phrase_map.get("phrases", [])}

        alignments = []
        overall_status = "success"
        for phrase in phonemes.get("phrases", []):
            text = self.selected_text(phrase)
            units = self.split_units(text)
            result = self.align_phrase(units, notes_by_id.get(phrase.get("id"), []))
            if result["status"] == "failed":
                overall_status = "failed"
            alignments.append(
                {
                    "id": phrase.get("id"),
                    "zh": phrase.get("zh") or zh_by_id.get(phrase.get("id"), ""),
                    "text": text,
                    "units": units,
                    **result,
                }
            )

        output = path_from_config(self.config, "note_lyric_alignment")
        preview_output = project_root(self.config) / "data" / "alignment" / "alignment_preview.txt"
        write_json(output, {"phrases": alignments}, self.config)
        write_text(preview_output, self.build_preview(alignments), self.config)

        status = "mock" if "mock" in {melody_status, phoneme_status} else overall_status
        return {
            "status": status,
            "outputs": {
                "note_lyric_alignment": str(output),
                "alignment_preview": str(preview_output),
            },
            "message": "Generated word-level lyric-note alignment.",
        }
