from __future__ import annotations


def compute_speaker_similarity(source_vocal: str, converted_vocal: str) -> dict:
    """Return a placeholder speaker similarity score.

    Input:
        source_vocal: Original or reference singer vocal path.
        converted_vocal: RVC converted vocal path.
    Output:
        Dict with mock score.
    TODO:
        Use a speaker embedding model to compare timbre similarity.
    """
    return {"status": "mock", "score": 0.0, "source_vocal": source_vocal, "converted_vocal": converted_vocal}
