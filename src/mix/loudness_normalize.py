from __future__ import annotations


def normalize_loudness_plan(vocal_path: str, accompaniment_path: str) -> dict:
    """Return a placeholder loudness normalization plan.

    Input:
        vocal_path: Converted vocal path.
        accompaniment_path: Accompaniment path.
    Output:
        Dict describing target loudness and TODO details.
    TODO:
        Implement LUFS normalization with ffmpeg or pyloudnorm.
    """
    return {
        "vocal_path": vocal_path,
        "accompaniment_path": accompaniment_path,
        "target_lufs": -14,
        "status": "mock",
    }
