from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json


class NoteLyricAligner:
    """Align target-language syllables to melody notes."""

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

    def align_phrase(self, syllables: list[str], notes: list[dict]) -> dict:
        """Align one phrase using a simple rule.

        Input:
            syllables: Target-language syllables or word proxies.
            notes: Melody note dictionaries.
        Output:
            Alignment dict with status, items, or failure reason.
        TODO:
            Replace with dynamic programming and phoneme duration prediction.
        """
        if len(syllables) > len(notes):
            return {"status": "failed", "reason": "too_many_syllables", "items": []}

        items = []
        for index, syllable in enumerate(syllables):
            assigned = [notes[index]]
            if index == len(syllables) - 1 and len(notes) > len(syllables):
                assigned.extend(notes[index + 1 :])
            items.append({"syllable": syllable, "notes": assigned})
        return {"status": "success", "items": items}

    def run(self) -> dict:
        """Create note_lyric_alignment.json.

        Input:
            data/score/melody_notes.json and data/phoneme/phonemes_target.json.
        Output:
            data/alignment/note_lyric_alignment.json.
        TODO:
            Support many-to-one and one-to-many note/syllable strategies with scoring.
        """
        melody_path = path_from_config(self.config, "melody_notes")
        phoneme_path = path_from_config(self.config, "phonemes_target")
        melody_status = require_or_mock_input(melody_path, self.config, "melody notes")
        phoneme_status = require_or_mock_input(phoneme_path, self.config, "target phonemes")

        melody = read_json(melody_path, {"phrases": [{"id": 1, "notes": [{"pitch": "C4"}, {"pitch": "D4"}, {"pitch": "E4"}]}]})
        phonemes = read_json(phoneme_path, {"phrases": [{"id": 1, "syllables": ["go", "see", "sea"]}]})
        notes_by_id = {phrase.get("id"): phrase.get("notes", []) for phrase in melody.get("phrases", [])}
        alignments = []
        overall_status = "success"
        for phrase in phonemes.get("phrases", []):
            result = self.align_phrase(phrase.get("syllables", []), notes_by_id.get(phrase.get("id"), []))
            if result["status"] == "failed":
                overall_status = "failed"
            alignments.append({"id": phrase.get("id"), **result})

        output = path_from_config(self.config, "note_lyric_alignment")
        write_json(output, {"phrases": alignments}, self.config)
        status = "mock" if "mock" in {melody_status, phoneme_status} else overall_status
        return {
            "status": status,
            "outputs": {"note_lyric_alignment": str(output)},
            "message": "TODO: replace simple syllable-note assignment with robust alignment.",
        }
