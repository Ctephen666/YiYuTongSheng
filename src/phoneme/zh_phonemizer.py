from __future__ import annotations

import json
import re
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


def _load_pypinyin():
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return None, None
    return Style, lazy_pinyin


def _normalise_syllable(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[1-5]", "", text)
    text = text.replace("u:", "v").replace("ü", "v")
    if len(text) >= 2 and text[0] in "jqxy" and text[1] == "v":
        text = text[0] + "u" + text[2:]
    return text


def _split_initial_final(pinyin: str) -> tuple[str, str]:
    initials = (
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "r", "z", "c", "s", "y", "w",
    )
    for initial in initials:
        if pinyin.startswith(initial) and pinyin != initial:
            return initial, pinyin[len(initial):]
    return "", pinyin


def _annotation_tokens(annotations: list[Any], phrase_id: Any) -> list[dict]:
    tokens: list[dict] = []
    for index, item in enumerate(annotations, start=1):
        if not isinstance(item, dict):
            continue
        item_phrase = item.get("phrase_id") or item.get("sentence_id") or item.get("id")
        if item_phrase not in (None, phrase_id, str(phrase_id)):
            continue
        text = str(item.get("char") or item.get("text") or item.get("zh") or "")
        phoneme = str(item.get("phoneme") or item.get("pinyin") or item.get("syllable") or text)
        if not text and not phoneme:
            continue
        onset, final = _split_initial_final(_normalise_syllable(phoneme))
        tokens.append(
            {
                "index": len(tokens),
                "char": text,
                "phoneme": phoneme,
                "pinyin": phoneme,
                "initial": onset,
                "final": final,
                "source": "opencpop_annotation",
            }
        )
    return tokens


def _textgrid_note_tokens(phrase: dict) -> list[dict]:
    tokens: list[dict] = []
    notes = phrase.get("notes", [])
    if not isinstance(notes, list):
        return tokens

    for note in notes:
        if not isinstance(note, dict):
            continue
        lyric = str(note.get("lyric") or "").strip()
        syllable = _normalise_syllable(note.get("syllable") or note.get("phoneme") or note.get("pinyin") or "")
        if not lyric or lyric in {"_", "SP", "AP"}:
            continue
        if not syllable or syllable in {"_", "sp", "ap", "sil", "rest", "silence"}:
            continue
        initial, final = _split_initial_final(syllable)
        tokens.append(
            {
                "index": len(tokens),
                "char": lyric,
                "phoneme": syllable,
                "pinyin": syllable,
                "initial": initial,
                "final": final,
                "source": "opencpop_textgrid_syllable",
            }
        )
    return tokens


def _pinyin_tokens(text: str, source: str) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    Style, lazy_pinyin = _load_pypinyin()
    chars = [char for char in text if not char.isspace()]
    tokens: list[dict] = []

    if lazy_pinyin is None:
        warnings.append("pypinyin is not installed; Chinese phoneme output uses character-level fallback tokens.")
        for index, char in enumerate(chars):
            tokens.append(
                {
                    "index": index,
                    "char": char,
                    "phoneme": char,
                    "pinyin": char,
                    "initial": "",
                    "final": char,
                    "source": "char_fallback",
                }
            )
        return tokens, warnings

    syllables = lazy_pinyin(chars, style=Style.TONE3, neutral_tone_with_five=True, errors="default")
    for index, (char, pinyin) in enumerate(zip(chars, syllables)):
        pinyin_text = _normalise_syllable(pinyin)
        initial, final = _split_initial_final(pinyin_text)
        tokens.append(
            {
                "index": index,
                "char": char,
                "phoneme": pinyin_text,
                "pinyin": pinyin_text,
                "initial": initial,
                "final": final,
                "source": source,
            }
        )
    return tokens, warnings


def phonemize_zh(config: dict) -> dict:
    phrases_path = _config_path(config, "lyrics_zh_phrases", "data/lyrics/lyrics_zh_phrases.json")
    item_path = _config_path(config, "opencpop_item", "data/dataset_manifest/opencpop_item_2001.json")
    phrases_payload = _read_json(phrases_path, {})
    item = _read_json(item_path, {})

    annotations = []
    if isinstance(item, dict):
        ann_payload = item.get("phoneme_annotations", {})
        if isinstance(ann_payload, dict) and isinstance(ann_payload.get("items"), list):
            annotations = ann_payload["items"]

    warnings = list(phrases_payload.get("warnings", [])) if isinstance(phrases_payload, dict) else []
    phrase_outputs: list[dict] = []
    for phrase in phrases_payload.get("phrases", []) if isinstance(phrases_payload, dict) else []:
        if not isinstance(phrase, dict):
            continue
        phrase_id = phrase.get("id", len(phrase_outputs) + 1)
        text = str(phrase.get("zh") or phrase.get("text") or "")
        tokens = _annotation_tokens(annotations, phrase_id)
        if tokens:
            token_source = "opencpop_annotation"
        else:
            tokens = _textgrid_note_tokens(phrase)
            if tokens:
                token_source = "opencpop_textgrid_syllable"
            else:
                tokens, token_warnings = _pinyin_tokens(text, "pypinyin")
                warnings.extend(token_warnings)
                token_source = tokens[0].get("source", "empty") if tokens else "empty"
        phrase_outputs.append(
            {
                "id": phrase_id,
                "zh": text,
                "source_phrase": phrase.get("source"),
                "token_source": token_source,
                "start": phrase.get("start"),
                "end": phrase.get("end"),
                "source_notes": phrase.get("notes", []),
                "tokens": tokens,
            }
        )

    seen_warnings: list[str] = []
    for warning in warnings:
        if warning not in seen_warnings:
            seen_warnings.append(warning)

    return {
        "dataset": "opencpop",
        "song_id": phrases_payload.get("song_id") if isinstance(phrases_payload, dict) else None,
        "strict_dataset_source": bool(phrases_payload.get("strict_dataset_source", True)) if isinstance(phrases_payload, dict) else True,
        "source": "opencpop_annotation_or_pypinyin",
        "phrase_count": len(phrase_outputs),
        "phrases": phrase_outputs,
        "warnings": seen_warnings,
    }


def save_zh_phonemes(config: dict, payload: dict) -> Path:
    output_path = _config_path(config, "phonemes_zh", "data/phoneme/phonemes_zh.json")
    _write_json(output_path, payload)
    return output_path


class ZhPhonemizer:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = phonemize_zh(self.config)
        output_path = save_zh_phonemes(self.config, payload)
        status = "success" if payload.get("phrases") else "warning"
        return {
            "status": status,
            "outputs": {"phonemes_zh": str(output_path)},
            "warnings": payload.get("warnings", []),
            "message": f"Prepared Chinese phoneme tokens for {payload.get('phrase_count', 0)} phrases.",
        }
