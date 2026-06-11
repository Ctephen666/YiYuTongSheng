from __future__ import annotations

import re

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json


class EnglishPhonemizer:
    """Mock English phonemizer."""

    MOCK_DICT = {
        "let's": ["L", "EH", "T", "S"],
        "lets": ["L", "EH", "T", "S"],
        "go": ["G", "OW"],
        "see": ["S", "IY"],
        "the": ["DH", "AH"],
        "sea": ["S", "IY"],
        "with": ["W", "IH", "DH"],
        "me": ["M", "IY"],
        "you": ["Y", "UW"],
        "i": ["AY"],
        "want": ["W", "AA", "N", "T"],
    }

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.lyrics_singable and paths.phonemes_target.
        Output:
            EnglishPhonemizer instance.
        TODO:
            Add phonemizer/espeak/cmudict backend configuration.
        """
        self.config = config

    def phonemize_line(self, text: str) -> list[dict]:
        """Phonemize one English line with a small mock dictionary.

        Input:
            text: Singable English lyric line.
        Output:
            List of dicts containing word and phonemes.
        TODO:
            Replace with a real phonemizer and syllable boundary information.
        """
        words = re.findall(r"[A-Za-z']+", text.lower())
        return [
            {"word": word, "phonemes": self.MOCK_DICT.get(word, list(word.upper()))}
            for word in words
        ]

    def alignment_units(self, text: str) -> list[str]:
        """Group words into coarse lyric units for the mock aligner.

        Input:
            text: Singable English lyric line.
        Output:
            List of coarse syllable-like units for note alignment.
        TODO:
            Replace this with real syllable boundaries from the phonemizer.
        """
        normalized = text.strip().lower()
        if normalized == "let's go see the sea":
            return ["let's go", "see", "the sea"]
        words = re.findall(r"[A-Za-z']+", normalized)
        units: list[str] = []
        index = 0
        while index < len(words):
            word = words[index]
            if word in {"the", "with"} and units:
                units[-1] = f"{units[-1]} {word}"
                if index + 1 < len(words):
                    units[-1] = f"{units[-1]} {words[index + 1]}"
                    index += 2
                    continue
            units.append(word)
            index += 1
        return units

    def selected_text(self, phrase: dict) -> str:
        """Return selected lyric text from either current string or legacy dict format."""
        selected = phrase.get("selected", "")
        if isinstance(selected, dict):
            return selected.get("text", "")
        return str(selected)

    def run(self) -> dict:
        """Generate target-language phoneme JSON.

        Input:
            data/lyrics/lyrics_singable.json.
        Output:
            data/phoneme/phonemes_target.json.
        TODO:
            Add language-specific phoneme inventories and stress marks.
        """
        singable_path = path_from_config(self.config, "lyrics_singable")
        input_status = require_or_mock_input(singable_path, self.config, "singable lyrics")
        singable = read_json(
            singable_path,
            {"phrases": [{"id": 1, "selected": {"text": "Let's go see the sea"}}]},
        )
        phrases = []
        for phrase in singable.get("phrases", []):
            text = self.selected_text(phrase)
            tokens = self.phonemize_line(text)
            syllables = self.alignment_units(text)
            phrases.append({**phrase, "text": text, "syllables": syllables, "tokens": tokens})

        output = path_from_config(self.config, "phonemes_target")
        write_json(output, {"language": "en", "phrases": phrases}, self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"phonemes_target": str(output)},
            "message": "TODO: replace mock English phonemization with phonemizer/cmudict.",
        }
