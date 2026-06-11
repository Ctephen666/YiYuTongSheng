from __future__ import annotations

import re


VOWELS = "aeiouy"


def _count_word_syllables(word: str) -> int:
    """Estimate syllables in one English word."""
    cleaned = word.lower().replace("'", "")
    cleaned = re.sub(r"[^a-z]", "", cleaned)
    if not cleaned:
        return 0

    groups = re.findall(rf"[{VOWELS}]+", cleaned)
    count = len(groups)

    if cleaned.endswith("e") and count > 1 and not cleaned.endswith(("le", "ye")):
        count -= 1

    return max(1, count)


def count_english_syllables(text: str) -> int:
    """Estimate English syllables with a stable word-level vowel-group rule.

    Input:
        text: English text candidate, including contractions such as let's, you're, or can't.
    Output:
        Estimated syllable count as an integer.
    TODO:
        Replace with a syllable dictionary or phoneme-based counter.
    """
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return sum(_count_word_syllables(word) for word in words)
