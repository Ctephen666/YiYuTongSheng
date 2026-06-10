from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SvsRenderPlan:
    """A plan for rendering target-language singing.

    Input:
        melody_midi: MIDI path.
        lyrics_singable: Singable lyrics path.
        alignment: Alignment path.
    Output:
        Dataclass describing an external SVS render request.
    TODO:
        Add singer, voicebank, tempo map, and render command fields.
    """

    melody_midi: str
    lyrics_singable: str
    alignment: str
