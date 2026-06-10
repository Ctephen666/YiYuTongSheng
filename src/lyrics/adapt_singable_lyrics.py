from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json
from src.lyrics.lyric_scorer import score_singable_lyric


class SingableLyricAdapter:
    """Adapt literal translations into singable target-language lyrics."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with literal lyrics, melody notes, and singable lyric output path.
        Output:
            SingableLyricAdapter instance.
        TODO:
            Add target-language prosody, rhyme, stress, and semantic constraints.
        """
        self.config = config

    def generate_candidates(self, zh: str, literal: str, note_count: int) -> list[str]:
        """Generate mock singable lyric candidates.

        Input:
            zh: Source Chinese lyric.
            literal: Literal target-language translation.
            note_count: Available notes for the phrase.
        Output:
            List of candidate lyric strings.
        TODO:
            Replace with LLM-based candidate generation constrained by note count.
        """
        if literal == "I want to go see the sea with you":
            return [
                "Let's go see the sea",
                "I want the sea with you",
                "Go see the sea with me",
            ]
        if self.config.get("project", {}).get("target_language") == "ja":
            return [literal, "海を見に行こう", "君と海へ"]
        return [literal, "TODO singable lyric", zh]

    def run(self) -> dict:
        """Select the highest-scoring singable lyric candidate per phrase.

        Input:
            data/lyrics/lyrics_literal.json and data/score/melody_notes.json.
        Output:
            data/lyrics/lyrics_singable.json.
        TODO:
            Add candidate diversity controls and human-editable review fields.
        """
        literal_path = path_from_config(self.config, "lyrics_literal")
        notes_path = path_from_config(self.config, "melody_notes")
        literal_status = require_or_mock_input(literal_path, self.config, "literal lyrics")
        notes_status = require_or_mock_input(notes_path, self.config, "melody notes")

        literal_data = read_json(literal_path, {"phrases": [{"id": 1, "zh": "我想和你去看海", "literal": "I want to go see the sea with you", "start": 0.0, "end": 3.2}]})
        melody_data = read_json(notes_path, {"phrases": [{"id": 1, "notes": [{"pitch": "C4"}, {"pitch": "D4"}, {"pitch": "E4"}]}]})
        melody_by_id = {phrase.get("id"): phrase for phrase in melody_data.get("phrases", [])}

        adapted = []
        for phrase in literal_data.get("phrases", []):
            melody_phrase = melody_by_id.get(phrase.get("id"), {})
            note_count = len(melody_phrase.get("notes", [])) or max(1, phrase.get("note_end", 3) - phrase.get("note_start", 0))
            duration = float(phrase.get("end", 3.2)) - float(phrase.get("start", 0.0))
            candidates = self.generate_candidates(phrase.get("zh", ""), phrase.get("literal", ""), note_count)
            scored = [
                score_singable_lyric(candidate, note_count=note_count, phrase_duration=duration)
                for candidate in candidates
            ]
            best = max(scored, key=lambda item: item["final_score"])
            adapted.append({**phrase, "note_count": note_count, "candidates": scored, "selected": best})

        output = path_from_config(self.config, "lyrics_singable")
        write_json(output, {"phrases": adapted}, self.config)
        return {
            "status": "mock" if "mock" in {literal_status, notes_status} else "success",
            "outputs": {"lyrics_singable": str(output)},
            "message": "TODO: replace heuristic candidate scoring with real singability adaptation.",
        }
