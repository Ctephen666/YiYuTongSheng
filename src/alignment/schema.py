from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentItem:
    """One lyric syllable aligned to one or more notes.

    Input:
        syllable: Target-language syllable or word proxy.
        notes: Note dictionaries assigned to the syllable.
    Output:
        Dataclass used by alignment artifacts.
    TODO:
        Add phoneme-level durations and consonant/vowel timing.
    """

    syllable: str
    notes: list[dict]
