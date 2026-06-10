from __future__ import annotations


def compute_rhythm_alignment(alignment_path: str) -> dict:
    """Return a placeholder rhythm alignment score.

    Input:
        alignment_path: note_lyric_alignment.json path.
    Output:
        Dict with mock rhythm score.
    TODO:
        Compare syllable onsets, note durations, and phrase boundaries.
    """
    return {"status": "mock", "score": 0.0, "alignment_path": alignment_path}
