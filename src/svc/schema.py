from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RvcParams:
    """RVC inference parameters.

    Input:
        model_path: RVC model path.
        index_path: Optional feature index path.
        pitch_shift: Semitone shift.
    Output:
        Dataclass describing planned RVC parameters.
    TODO:
        Add f0 method, index rate, protect, and filter radius fields.
    """

    model_path: str
    index_path: str
    pitch_shift: int = 0
