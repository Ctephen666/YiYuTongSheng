from __future__ import annotations

from src.lyrics.syllable_counter import count_english_syllables


def score_singable_lyric(
    candidate: str,
    note_count: int,
    phrase_duration: float,
    semantic_score: float = 0.8,
) -> dict:
    """Score a lyric candidate for rough singability.

    Input:
        candidate: Target-language lyric candidate.
        note_count: Number of melody notes available for the phrase.
        phrase_duration: Phrase duration in seconds.
        semantic_score: Mock semantic preservation score.
    Output:
        Dict with text, syllables, semantic_score, rhythm_score, singability_score, and final_score.
    TODO:
        Replace heuristic scoring with LLM, phoneme duration, stress, and rhyme features.
    """
    syllables = count_english_syllables(candidate)
    if note_count <= 0:
        rhythm_score = 0.0
    else:
        rhythm_score = max(0.0, 1.0 - abs(syllables - note_count) / max(note_count, 1))

    seconds_per_syllable = phrase_duration / max(syllables, 1)
    singability_score = 1.0 if 0.25 <= seconds_per_syllable <= 0.9 else 0.6
    ending_score = 1.0 if candidate.strip().lower().endswith(("sea", "me", "you")) else 0.7
    final_score = (
        0.4 * semantic_score
        + 0.3 * rhythm_score
        + 0.2 * singability_score
        + 0.1 * ending_score
    )
    return {
        "text": candidate,
        "syllables": syllables,
        "semantic_score": round(semantic_score, 3),
        "rhythm_score": round(rhythm_score, 3),
        "singability_score": round(singability_score, 3),
        "ending_score": round(ending_score, 3),
        "final_score": round(final_score, 3),
    }
