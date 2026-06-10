from __future__ import annotations

from src.common.io_utils import path_from_config, project_root, write_text
from src.common.json_utils import write_json


class PhraseMapper:
    """Map source lyrics to phrase-level melody regions."""

    def __init__(self, config: dict):
        """Store pipeline config.

        Input:
            config: Pipeline config with paths.lyrics_zh and paths.phrase_map.
        Output:
            PhraseMapper instance.
        TODO:
            Add punctuation-aware Chinese segmentation and manual timing import.
        """
        self.config = config

    def run(self) -> dict:
        """Create phrase_map.json from Chinese lyrics.

        Input:
            data/lyrics/lyrics_zh.txt, auto-created from examples when missing.
        Output:
            data/lyrics/phrase_map.json.
        TODO:
            Align each source phrase to detected melody phrase boundaries.
        """
        lyrics_zh = path_from_config(self.config, "lyrics_zh")
        example = project_root(self.config) / "examples" / "lyrics_zh.example.txt"
        if not lyrics_zh.exists():
            example_text = example.read_text(encoding="utf-8") if example.exists() else "我想和你去看海\n"
            write_text(lyrics_zh, example_text, self.config)

        lines = [line.strip() for line in lyrics_zh.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            lines = ["我想和你去看海"]

        phrases = []
        for index, line in enumerate(lines, start=1):
            start = round((index - 1) * 3.2, 2)
            end = round(start + 3.2, 2)
            phrases.append(
                {
                    "id": index,
                    "zh": line,
                    "start": start,
                    "end": end,
                    "note_start": 0,
                    "note_end": 3,
                }
            )

        phrase_map = path_from_config(self.config, "phrase_map")
        write_json(phrase_map, {"phrases": phrases}, self.config)
        return {
            "status": "mock",
            "outputs": {"phrase_map": str(phrase_map), "lyrics_zh": str(lyrics_zh)},
            "message": "TODO: replace rule-based phrase mapping with timed lyric segmentation.",
        }
