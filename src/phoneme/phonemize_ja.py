from __future__ import annotations

from src.common.io_utils import path_from_config, require_or_mock_input
from src.common.json_utils import read_json, write_json


class JapanesePhonemizer:
    """Mock Japanese phonemizer."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.lyrics_singable and paths.phonemes_target.
        Output:
            JapanesePhonemizer instance.
        TODO:
            Add pyopenjtalk or OpenUtau-compatible phonemization backend.
        """
        self.config = config

    def phonemize_line(self, text: str) -> list[dict]:
        """Phonemize one Japanese line with placeholder mora tokens.

        Input:
            text: Singable Japanese lyric line.
        Output:
            List of mock mora/phoneme dicts.
        TODO:
            Replace character splitting with proper kana, mora, and phoneme conversion.
        """
        chars = [char for char in text if not char.isspace()]
        return [{"mora": char, "phonemes": [char]} for char in chars]

    def selected_text(self, phrase: dict) -> str:
        """Return selected lyric text from either current string or legacy dict format."""
        selected = phrase.get("selected", "")
        if isinstance(selected, dict):
            return selected.get("text", "")
        return str(selected)

    def run(self) -> dict:
        """Generate mock Japanese phoneme JSON.

        Input:
            data/lyrics/lyrics_singable.json.
        Output:
            data/phoneme/phonemes_target.json.
        TODO:
            Add OpenJTalk dictionary support and singer-specific phoneme mapping.
        """
        singable_path = path_from_config(self.config, "lyrics_singable")
        input_status = require_or_mock_input(singable_path, self.config, "singable lyrics")
        singable = read_json(singable_path, {"phrases": [{"id": 1, "selected": {"text": "海を見に行こう"}}]})
        phrases = []
        for phrase in singable.get("phrases", []):
            text = self.selected_text(phrase)
            tokens = self.phonemize_line(text)
            phrases.append({**phrase, "text": text, "syllables": [item["mora"] for item in tokens], "tokens": tokens})

        output = path_from_config(self.config, "phonemes_target")
        write_json(output, {"language": "ja", "phrases": phrases}, self.config)
        return {
            "status": "mock" if input_status == "mock" else "success",
            "outputs": {"phonemes_target": str(output)},
            "message": "TODO: replace mock Japanese phonemization with pyopenjtalk/OpenUtau mapping.",
        }
