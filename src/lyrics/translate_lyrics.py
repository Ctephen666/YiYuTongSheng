from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json


class LyricTranslator:
    """Mock lyric translator."""

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

    def translate_line(self, zh_line: str, target_language: str) -> str:
        """Translate one Chinese line with a tiny mock rule table.

        Input:
            zh_line: Source Chinese lyric line.
            target_language: Target language code, such as en or ja.
        Output:
            Mock translated lyric.
        TODO:
            Replace with glossary-aware translation and preserve imagery/rhyme hints.
        """
        if target_language == "ja":
            return "君と海を見に行きたい"
        if zh_line == "我想和你去看海":
            return "I want to go see the sea with you"
        return "TODO translated lyric"

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
        phrase_data = read_json(phrase_map, {"phrases": [{"id": 1, "zh": "我想和你去看海"}]})
        target_language = self.config.get("project", {}).get("target_language", "en")
        translated = []
        for phrase in phrase_data.get("phrases", []):
            literal = self.translate_line(phrase.get("zh", ""), target_language)
            translated.append({**phrase, "target_language": target_language, "literal": literal})

        output = path_from_config(self.config, "lyrics_literal")
        write_json(output, {"phrases": translated}, self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"lyrics_literal": str(output)},
            "message": "TODO: replace mock translation with controlled lyric translation.",
        }
