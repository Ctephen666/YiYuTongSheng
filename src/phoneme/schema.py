from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhonemeToken:
    """A phoneme token assigned to a word.

    Input:
        word: Original target-language word.
        phonemes: List of phoneme symbols.
    Output:
        Dataclass for phonemization artifacts.
    TODO:
        Add syllable grouping and phoneme duration priors.
    """

    word: str
    phonemes: list[str]
