from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent


def _root(config: dict) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def _resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _root(config) / path


def _config_path(config: dict, key: str, default: str) -> Path:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs", {}), dict) else {}
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    return _resolve(config, outputs.get(key) or paths.get(key) or default)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _strict(config: dict) -> bool:
    return bool(config.get("dataset", {}).get("strict_dataset_source", True))


def _song_id(config: dict, item: dict | None = None) -> str:
    if item and item.get("song_id"):
        return str(item["song_id"])
    return str(
        config.get("dataset", {}).get("default_song_id")
        or config.get("inputs", {}).get("opencpop_default_song_id")
        or config.get("opencpop", {}).get("sample_id")
        or "2001"
    )


def _phrase_text(phrase: Any) -> str:
    if isinstance(phrase, dict):
        return str(phrase.get("zh") or phrase.get("text") or phrase.get("lyric") or phrase.get("sentence") or "")
    return str(phrase or "")


def _normalise_phrases(raw_phrases: list[Any], source: str) -> list[dict]:
    phrases: list[dict] = []
    for index, raw in enumerate(raw_phrases, start=1):
        text = _phrase_text(raw).strip()
        if not text:
            continue
        phrase = {
            "id": raw.get("id", index) if isinstance(raw, dict) else index,
            "zh": text,
            "characters": [char for char in text if not char.isspace()],
            "source": source,
        }
        if isinstance(raw, dict):
            for key in ("start", "end", "duration", "note_start", "note_end", "notes"):
                if key in raw:
                    phrase[key] = raw[key]
        phrases.append(phrase)
    return phrases


def _phrases_from_text(text: str, source: str) -> list[dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines and text.strip():
        lines = [text.strip()]
    return _normalise_phrases(lines, source)


def _extract_textgrid_phrases(textgrid_path: Path, sample_id: str) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    if not textgrid_path.exists():
        return [], [f"OpenCpop TextGrid does not exist: {textgrid_path}"]

    try:
        import tools.extract_opencpop_textgrid as tool
    except ImportError as exc:
        return [], [f"Could not import tools.extract_opencpop_textgrid: {exc}"]

    try:
        resolved = tool.find_textgrid_path(textgrid_path)
        tg = tool.textgrid.openTextgrid(str(resolved), includeEmptyIntervals=False)
        required_tiers = ["句子", "汉字", "音节", "音高", "音长"]
        legacy_required_tiers = ["鍙ュ瓙", "姹夊瓧", "闊宠妭", "闊抽珮", "闊抽暱"]
        tier_names = set(tg.tierNames)
        if all(name in tier_names for name in required_tiers):
            sentence_tier, char_tier, syllable_tier, pitch_tier, duration_tier = required_tiers
        elif all(name in tier_names for name in legacy_required_tiers):
            sentence_tier, char_tier, syllable_tier, pitch_tier, duration_tier = legacy_required_tiers
        else:
            return [], [f"OpenCpop TextGrid missing required tiers: {resolved}; available={tg.tierNames}"]

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
        notes_by_index = {
            int(note.get("index", -1)): note
            for note in notes
            if isinstance(note, dict)
        }
        for phrase in phrases:
            try:
                start_index = int(phrase.get("note_start", 0))
                end_index = int(phrase.get("note_end", 0))
            except (TypeError, ValueError):
                start_index = 0
                end_index = 0
            phrase_notes = []
            for note_index in range(start_index, end_index):
                note = notes_by_index.get(note_index)
                if note is None:
                    continue
                note_payload = dict(note)
                label = str(note_payload.get("lyric") or note_payload.get("syllable") or "").strip()
                note_payload["is_slur"] = label in {"_", "-"}
                phrase_notes.append(note_payload)
            phrase["notes"] = phrase_notes
        normalised = _normalise_phrases(phrases, "opencpop_textgrid")
        for phrase in normalised:
            phrase["sample_id"] = sample_id
            phrase["textgrid_path"] = str(resolved)
        return normalised, warnings
    except Exception as exc:  # pragma: no cover - depends on TextGrid parser and dataset content
        return [], [f"Could not extract OpenCpop TextGrid lyrics: {exc}"]


def _legacy_fallback(config: dict) -> tuple[list[dict], list[str]]:
    warnings = ["Strict dataset source is disabled; legacy lyric fallback is allowed."]
    phrase_map_path = _config_path(config, "phrase_map", "data/lyrics/phrase_map.json")
    if phrase_map_path.exists():
        phrase_map = _read_json(phrase_map_path, {})
        phrases = _normalise_phrases(phrase_map.get("phrases", []), "legacy_phrase_map")
        if phrases:
            return phrases, warnings

    lyrics_path = _config_path(config, "lyrics_zh", "data/lyrics/lyrics_zh.txt")
    if lyrics_path.exists():
        phrases = _phrases_from_text(lyrics_path.read_text(encoding="utf-8"), "legacy_lyrics_zh")
        if phrases:
            return phrases, warnings

    warnings.append("Legacy lyric fallback files were not found.")
    return [], warnings


def build_zh_phrases(config: dict) -> dict:
    item_path = _config_path(config, "opencpop_item", "data/dataset_manifest/opencpop_item_2001.json")
    item = _read_json(item_path, {})
    warnings: list[str] = []
    strict = _strict(config)
    song_id = _song_id(config, item if isinstance(item, dict) else None)

    phrases: list[dict] = []
    source = "dataset_missing"
    if isinstance(item, dict):
        lyrics = item.get("lyrics", {}) if isinstance(item.get("lyrics", {}), dict) else {}
        raw_phrases = lyrics.get("phrases", []) if isinstance(lyrics.get("phrases", []), list) else []
        phrases = _normalise_phrases(raw_phrases, "opencpop_dataset_lyrics")
        if not phrases and str(lyrics.get("text") or "").strip():
            phrases = _phrases_from_text(str(lyrics.get("text")), "opencpop_dataset_lyrics")
        if phrases:
            source = "opencpop_dataset_lyrics"
        elif item.get("textgrid_path"):
            extracted, tg_warnings = _extract_textgrid_phrases(Path(str(item["textgrid_path"])), song_id)
            warnings.extend(tg_warnings)
            if extracted:
                phrases = extracted
                source = "opencpop_textgrid"
    else:
        warnings.append(f"OpenCpop item JSON is missing or invalid: {item_path}")

    if not phrases:
        if strict:
            warnings.append(
                "Strict dataset source is enabled; no legacy lyrics_zh.txt or phrase_map.json fallback was used."
            )
        else:
            phrases, fallback_warnings = _legacy_fallback(config)
            warnings.extend(fallback_warnings)
            source = phrases[0].get("source", "legacy_fallback") if phrases else "legacy_missing"

    return {
        "dataset": "opencpop",
        "song_id": song_id,
        "strict_dataset_source": strict,
        "source": source,
        "phrase_count": len(phrases),
        "phrases": phrases,
        "warnings": warnings,
    }


def save_zh_phrases(config: dict, payload: dict) -> Path:
    output_path = _config_path(config, "lyrics_zh_phrases", "data/lyrics/lyrics_zh_phrases.json")
    _write_json(output_path, payload)
    return output_path


class ZhPhraseBuilder:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = build_zh_phrases(self.config)
        output_path = save_zh_phrases(self.config, payload)
        status = "success" if payload.get("phrases") else "warning"
        return {
            "status": status,
            "outputs": {"lyrics_zh_phrases": str(output_path)},
            "warnings": payload.get("warnings", []),
            "message": f"Prepared {payload.get('phrase_count', 0)} Chinese lyric phrases from {payload.get('source')}.",
        }
