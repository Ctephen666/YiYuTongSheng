from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LyricPhrase:
    """A phrase-level lyric segment.

    Input:
        id: Phrase id.
        zh: Source Chinese lyric.
        start: Phrase start time in seconds.
        end: Phrase end time in seconds.
    Output:
        Dataclass for phrase mapping artifacts.
    TODO:
        Add source timestamps from forced alignment or manual annotation.
    """

    id: int
    zh: str
    start: float
    end: float
