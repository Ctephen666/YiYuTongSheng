from __future__ import annotations

from src.common.io_utils import path_from_config, project_root, require_or_mock_input
from src.common.json_utils import read_json, write_json
from src.lyrics.lyric_scorer import score_singable_lyric
from src.lyrics.syllable_counter import count_english_syllables


SHORT_CANDIDATES = [
    "Stay with me",
    "Do not let go",
    "Shine on me",
]

MEDIUM_CANDIDATES = [
    "I still dream of you",
    "Stay with me through the night",
    "You are the light I need",
]

LONG_CANDIDATES = [
    "You are the heaven I cannot reach",
    "I keep chasing your light",
    "Let my heart run back to you",
]

EXTENDED_CANDIDATES = [
    "You are the heaven I cannot reach tonight",
    "I keep chasing the light I cannot hold",
    "Let my heart cross the sky to you",
]


class SingableLyricAdapter:
    """Adapt literal translations into singable target-language lyrics."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with phrase_map, literal lyrics, melody notes, and singable lyric output path.
        Output:
            SingableLyricAdapter instance.
        TODO:
            Add target-language prosody, rhyme, stress, and semantic constraints.
        """
        self.config = config

    def target_syllable_range(self, note_count: int) -> tuple[int, int]:
        """Return a practical English syllable range for the phrase note count."""
        if note_count <= 4:
            return 2, 4
        if note_count <= 8:
            return 4, 8
        if note_count <= 12:
            return 6, 12
        return 8, 14

    def target_syllables(self, note_count: int) -> int:
        """Return one preferred syllable count inside the allowed range."""
        low, high = self.target_syllable_range(note_count)
        return max(low, min(high, note_count))

    def generate_candidates(self, zh: str, literal: str, note_count: int) -> list[str]:
        """Generate rule-based singable lyric candidates close to the note count.

        Input:
            zh: Source Chinese lyric.
            literal: Literal target-language translation.
            note_count: Available notes for the phrase.
        Output:
            List of candidate lyric strings.
        TODO:
            Replace with LLM-based candidate generation constrained by note count.
        """
        low, high = self.target_syllable_range(note_count)
        pool = [literal] if literal else []

        if note_count <= 4:
            pool.extend(SHORT_CANDIDATES)
        elif note_count <= 8:
            pool.extend(MEDIUM_CANDIDATES)
        elif note_count <= 12:
            pool.extend(LONG_CANDIDATES)
        else:
            pool.extend(EXTENDED_CANDIDATES)

        in_range = [
            candidate
            for candidate in pool
            if low <= count_english_syllables(candidate) <= high
        ]
        candidates = in_range or pool

        unique = []
        seen = set()
        for candidate in candidates:
            text = candidate.strip()
            if text and text.lower() not in seen:
                unique.append(text)
                seen.add(text.lower())
        return unique

    def _phrase_duration(self, phrase: dict) -> float:
        return round(float(phrase.get("end", 0.0)) - float(phrase.get("start", 0.0)), 3)

    def _note_count(self, phrase: dict, melody_by_id: dict) -> int:
        note_start = phrase.get("note_start")
        note_end = phrase.get("note_end")
        if note_start is not None and note_end is not None:
            return max(1, int(note_end) - int(note_start))

        melody_phrase = melody_by_id.get(phrase.get("id"), {})
        return max(1, len(melody_phrase.get("notes", [])))

    def _literal_by_id(self, literal_data: dict) -> dict:
        return {phrase.get("id"): phrase for phrase in literal_data.get("phrases", [])}

    def _manual_output(self, manual_data: dict, phrase_by_id: dict, literal_by_id: dict, melody_by_id: dict) -> list[dict]:
        adapted = []
        for manual_phrase in manual_data.get("phrases", []):
            phrase_id = manual_phrase.get("id")
            phrase = phrase_by_id.get(phrase_id, {})
            literal_phrase = literal_by_id.get(phrase_id, {})
            note_count = int(
                manual_phrase.get("note_count")
                or self._note_count(phrase, melody_by_id)
            )
            adapted.append(
                {
                    "id": phrase_id,
                    "zh": manual_phrase.get("zh") or phrase.get("zh", ""),
                    "literal_en": literal_phrase.get("literal_en") or literal_phrase.get("literal", ""),
                    "note_count": note_count,
                    "phrase_duration": self._phrase_duration(phrase),
                    "target_syllables": int(
                        manual_phrase.get("target_syllables")
                        or self.target_syllables(note_count)
                    ),
                    "candidates": [],
                    "selected": manual_phrase.get("selected", ""),
                }
            )
        return adapted

    def run(self) -> dict:
        """Select the highest-scoring singable lyric candidate per phrase.

        Input:
            data/lyrics/phrase_map.json, data/score/melody_notes.json, and data/lyrics/lyrics_literal.json.
        Output:
            data/lyrics/lyrics_singable.json.
        TODO:
            Add candidate diversity controls and human-editable review fields.
        """
        phrase_map_path = path_from_config(self.config, "phrase_map")
        notes_path = path_from_config(self.config, "melody_notes")
        literal_path = path_from_config(self.config, "lyrics_literal")
        output = path_from_config(self.config, "lyrics_singable")
        manual_path = project_root(self.config) / "data" / "lyrics" / "lyrics_singable_manual.json"

        phrase_status = require_or_mock_input(phrase_map_path, self.config, "phrase map")
        notes_status = require_or_mock_input(notes_path, self.config, "melody notes")
        literal_status = require_or_mock_input(literal_path, self.config, "literal lyrics")

        phrase_data = read_json(phrase_map_path, {"phrases": [{"id": 1, "zh": "", "start": 0.0, "end": 3.2, "note_start": 0, "note_end": 3}]})
        melody_data = read_json(notes_path, {"phrases": [{"id": 1, "notes": [{"pitch": "C4"}, {"pitch": "D4"}, {"pitch": "E4"}]}]})
        literal_data = read_json(literal_path, {"phrases": [{"id": 1, "literal": "I want to sing with you", "literal_en": "I want to sing with you"}]})

        melody_by_id = {phrase.get("id"): phrase for phrase in melody_data.get("phrases", [])}
        phrase_by_id = {phrase.get("id"): phrase for phrase in phrase_data.get("phrases", [])}
        literal_by_id = self._literal_by_id(literal_data)

        if manual_path.exists():
            manual_data = read_json(manual_path, {"phrases": []})
            adapted = self._manual_output(manual_data, phrase_by_id, literal_by_id, melody_by_id)
            write_json(output, {"phrases": adapted}, self.config)
            return {
                "status": "manual",
                "outputs": {"lyrics_singable": str(output)},
                "message": "Manual singable lyrics detected. Generated lyrics_singable.json from manual selections.",
            }

        adapted = []
        for phrase in phrase_data.get("phrases", []):
            phrase_id = phrase.get("id")
            literal_phrase = literal_by_id.get(phrase_id, {})
            literal = literal_phrase.get("literal_en") or literal_phrase.get("literal", "")
            note_count = self._note_count(phrase, melody_by_id)
            phrase_duration = self._phrase_duration(phrase)
            candidates = self.generate_candidates(phrase.get("zh", ""), literal, note_count)
            scored = [
                score_singable_lyric(candidate, note_count=note_count, phrase_duration=phrase_duration)
                for candidate in candidates
            ]
            best = max(scored, key=lambda item: item["final_score"])
            adapted.append(
                {
                    "id": phrase_id,
                    "zh": phrase.get("zh", ""),
                    "literal_en": literal,
                    "note_count": note_count,
                    "phrase_duration": phrase_duration,
                    "candidates": scored,
                    "selected": best["text"],
                }
            )

        write_json(output, {"phrases": adapted}, self.config)
        return {
            "status": "mock" if "mock" in {phrase_status, literal_status, notes_status} else "success",
            "outputs": {"lyrics_singable": str(output)},
            "message": "Generated rule-based singable English lyric candidates from phrase_map note counts.",
        }
