from __future__ import annotations


def compute_f0_similarity(reference_f0: str, converted_f0: str) -> dict:
    """Return a placeholder F0 similarity score.

    Input:
        reference_f0: Reference F0 CSV path.
        converted_f0: Converted vocal F0 CSV path.
    Output:
        Dict with mock score and TODO message.
    TODO:
        Extract converted F0 and compare pitch contours with DTW or correlation.
    """
    return {"status": "mock", "score": 0.0, "reference_f0": reference_f0, "converted_f0": converted_f0}
