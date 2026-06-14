from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def levenshtein_distance(a: list[str] | str, b: list[str] | str) -> int:
    left = list(a)
    right = list(b)
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        cur = [i]
        for j, cb in enumerate(right, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def char_error_rate(reference: str, recognized: str) -> float | None:
    ref = re.sub(r"\s+", "", reference or "")
    hyp = re.sub(r"\s+", "", recognized or "")
    if not ref:
        return None
    return levenshtein_distance(ref, hyp) / float(len(ref))


def word_error_rate(reference: str, recognized: str) -> float | None:
    ref_words = (reference or "").split()
    hyp_words = (recognized or "").split()
    if not ref_words:
        return None
    return levenshtein_distance(ref_words, hyp_words) / float(len(ref_words))


def _collect_text(value: Any, parts: list[str]) -> None:
    if isinstance(value, dict):
        for key in ("zh", "text", "lyric", "lyrics"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                break
        for item in value.values():
            _collect_text(item, parts)
    elif isinstance(value, list):
        for item in value:
            _collect_text(item, parts)


def reference_text_from_score(score_json: str | Path) -> tuple[str | None, list[str]]:
    path = Path(score_json)
    if not path.exists():
        return None, [f"Score JSON not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, [f"Failed to read score JSON {path}: {exc}"]
    parts: list[str] = []
    _collect_text(data, parts)
    text = "".join(parts)
    return (text or None), []


def intelligibility_metrics(score_json: str | Path, recognized_text: str | None = None, use_asr: bool = False) -> dict:
    reference_text, warnings = reference_text_from_score(score_json)
    result = {
        "reference_text": reference_text,
        "recognized_text": recognized_text,
        "cer": None,
        "wer": None,
        "asr_available": False,
        "warnings": warnings,
    }
    if recognized_text is not None:
        result["cer"] = char_error_rate(reference_text or "", recognized_text)
        result["wer"] = word_error_rate(reference_text or "", recognized_text)
        return result
    if not use_asr:
        result["warnings"].append("ASR not available; intelligibility metric skipped.")
        return result
    result["warnings"].append("ASR requested but no local ASR adapter is configured; intelligibility metric skipped.")
    return result
