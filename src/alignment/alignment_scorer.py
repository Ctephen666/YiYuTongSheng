from __future__ import annotations


def score_alignment(alignment: dict) -> dict:
    """Score a note-lyric alignment artifact.

    Input:
        alignment: Alignment dictionary from NoteLyricAligner.
    Output:
        Dict containing a rough score and status.
    TODO:
        Evaluate note coverage, syllable stress, phoneme duration, and rests.
    """
    phrases = alignment.get("phrases", [])
    if not phrases:
        return {"status": "mock", "score": 0.0}
    failed = sum(1 for phrase in phrases if phrase.get("status") == "failed")
    return {"status": "mock", "score": round(1.0 - failed / max(len(phrases), 1), 3)}
