from __future__ import annotations

from pathlib import Path

from src.common.io_utils import ensure_parent, path_from_config, project_root
from src.common.json_utils import write_json


class OpenCpopTextGridImporter:
    """Import Chinese lyrics and phrase map from the matching OpenCpop TextGrid."""

    def __init__(self, config: dict):
        self.config = config

    def _tool(self):
        try:
            import tools.extract_opencpop_textgrid as tool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "praatio is not installed, so OpenCpop TextGrid lyrics cannot be extracted. "
                "Install dependencies with: pip install praatio"
            ) from exc
        return tool

    def _sample_id(self) -> str:
        explicit = self.config.get("opencpop", {}).get("sample_id")
        if explicit:
            return str(explicit)

        midi_path = path_from_config(self.config, "opencpop_midi")
        return midi_path.stem

    def _textgrid_path(self) -> Path:
        paths = self.config.get("paths", {})
        if paths.get("opencpop_textgrid"):
            return path_from_config(self.config, "opencpop_textgrid")

        return project_root(self.config) / "data" / "dataset" / "opencpop" / "textgrids" / f"{self._sample_id()}.TextGrid"

    def run(self) -> dict:
        tool = self._tool()
        textgrid_path = tool.find_textgrid_path(self._textgrid_path())
        sample_id = self._sample_id()

        tg = tool.textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
        required_tiers = ["句子", "汉字", "音节", "音高", "音长"]
        legacy_required_tiers = ["鍙ュ瓙", "姹夊瓧", "闊宠妭", "闊抽珮", "闊抽暱"]
        tier_names = set(tg.tierNames)

        if all(name in tier_names for name in required_tiers):
            sentence_tier, char_tier, syllable_tier, pitch_tier, duration_tier = required_tiers
        elif all(name in tier_names for name in legacy_required_tiers):
            sentence_tier, char_tier, syllable_tier, pitch_tier, duration_tier = legacy_required_tiers
        else:
            raise RuntimeError(
                f"OpenCpop TextGrid missing required tiers: {textgrid_path}. "
                f"Available tiers: {tg.tierNames}"
            )

        sentence_entries = tool.get_tier_entries(tg, sentence_tier)
        char_entries = tool.get_tier_entries(tg, char_tier)
        syllable_entries = tool.get_tier_entries(tg, syllable_tier)
        pitch_entries = tool.get_tier_entries(tg, pitch_tier)
        duration_entries = tool.get_tier_entries(tg, duration_tier)

        phrases = tool.extract_sentence_phrases(sentence_entries)
        notes = tool.build_notes(
            char_entries=char_entries,
            syllable_entries=syllable_entries,
            pitch_entries=pitch_entries,
            duration_entries=duration_entries,
        )
        phrases = tool.attach_notes_to_phrases(phrases, notes)

        if not phrases:
            raise RuntimeError(f"No Chinese lyric phrases extracted from OpenCpop TextGrid: {textgrid_path}")

        lyrics_path = path_from_config(self.config, "lyrics_zh")
        phrase_map_path = path_from_config(self.config, "phrase_map")

        ensure_parent(lyrics_path)
        lyrics_path.write_text("\n".join(phrase["zh"] for phrase in phrases) + "\n", encoding="utf-8")

        phrase_map = {
            "source": "opencpop_textgrid",
            "sample_id": sample_id,
            "textgrid": str(textgrid_path),
            "note_index_rule": "note_start inclusive, note_end exclusive",
            "phrases": phrases,
        }
        write_json(phrase_map_path, phrase_map, self.config)

        return {
            "status": "success",
            "outputs": {
                "lyrics_zh": str(lyrics_path),
                "phrase_map": str(phrase_map_path),
            },
            "message": f"Extracted {len(phrases)} OpenCpop lyric phrases from sample {sample_id}.",
        }
