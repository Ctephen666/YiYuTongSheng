from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json


LITERAL_EN_BANK = [
    "You are the heaven I cannot reach",
    "I still keep longing for you",
    "Let my dreams cover the sky",
    "This unknown road",
    "Keeps calling me onward",
    "I carry your light in my heart",
    "Every moment turns back to you",
    "I cannot stop imagining",
]


class LyricTranslator:
    """Rule-based placeholder lyric translator."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.phrase_map and paths.lyrics_literal.
        Output:
            LyricTranslator instance.
        TODO:
            Add LLM or translation API provider settings.
        """
        self.config = config

    def translate_line(self, zh_line: str, target_language: str, phrase_id: int | None = None) -> str:
        """Translate one source phrase with a deterministic placeholder rule.

        Input:
            zh_line: Source Chinese lyric line.
            target_language: Target language code, such as en or ja.
            phrase_id: Optional phrase id for stable placeholder selection.
        Output:
            Placeholder literal lyric.
        TODO:
            Replace with glossary-aware translation and preserve imagery/rhyme hints.
        """
        if target_language == "ja":
            return zh_line

        if phrase_id is None:
            return "A literal lyric translation"

        return LITERAL_EN_BANK[(phrase_id - 1) % len(LITERAL_EN_BANK)]

    def run(self) -> dict:
        """Translate phrase_map.json into literal target-language lyrics.

        Input:
            data/lyrics/phrase_map.json.
        Output:
            data/lyrics/lyrics_literal.json.
        TODO:
            Add batch translation, manual review fields, and prompt templates.
        """
        phrase_map = path_from_config(self.config, "phrase_map")
        input_status = require_or_mock_input(phrase_map, self.config, "phrase map")
        phrase_data = read_json(phrase_map, {"phrases": [{"id": 1, "zh": ""}]})
        target_language = self.config.get("project", {}).get("target_language", "en")

        translated = []
        for phrase in phrase_data.get("phrases", []):
            phrase_id = int(phrase.get("id", len(translated) + 1))
            literal = self.translate_line(phrase.get("zh", ""), target_language, phrase_id=phrase_id)
            translated.append(
                {
                    **phrase,
                    "target_language": target_language,
                    "literal": literal,
                    "literal_en": literal if target_language == "en" else "",
                }
            )

        output = path_from_config(self.config, "lyrics_literal")
        write_json(
            output,
            {
                "source": phrase_data.get("source", "phrase_map"),
                "phrases": translated,
            },
            self.config,
        )
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"lyrics_literal": str(output)},
            "message": "Generated placeholder literal lyrics from phrase_map.",
        }
