from __future__ import annotations

import re


def count_english_syllables(text: str) -> int:
    """Estimate English syllable count with a simple vowel-group rule.

    Input:
        text: English text candidate.
    Output:
        Estimated syllable count as an integer.
    TODO:
        Replace with a syllable dictionary or phoneme-based counter.
    """
    words = re.findall(r"[A-Za-z']+", text.lower())
    count = 0
    for word in words:
        cleaned = word.strip("'")
        groups = re.findall(r"[aeiouy]+", cleaned)
        word_count = len(groups)
        if cleaned.endswith("e") and word_count > 1 and not cleaned.endswith(("le", "ye")):
            word_count -= 1
        count += max(1, word_count)
    return count
