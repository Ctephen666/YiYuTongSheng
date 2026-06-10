from __future__ import annotations


def evaluate_lyric_adaptation(lyrics_singable_path: str) -> dict:
    """Return a placeholder lyric adaptation evaluation.

    Input:
        lyrics_singable_path: Singable lyric JSON path.
    Output:
        Dict with mock score and TODO message.
    TODO:
        Evaluate semantics, syllable count, rhyme, stress, and human edit distance.
    """
    return {"status": "mock", "score": 0.0, "lyrics_singable_path": lyrics_singable_path}
