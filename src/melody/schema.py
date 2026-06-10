from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Note:
    """A simple note representation.

    Input:
        pitch: Scientific pitch name.
        start: Start time in seconds.
        duration: Duration in seconds.
    Output:
        Dataclass used by mock melody artifacts.
    TODO:
        Add MIDI pitch, lyric syllable, and confidence fields.
    """

    pitch: str
    start: float
    duration: float
