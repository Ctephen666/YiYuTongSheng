from __future__ import annotations

from pathlib import Path

from src.common.io_utils import write_placeholder


def write_placeholder_audio(path: Path, label: str, config: dict) -> None:
    """Create a mock audio artifact.

    Input:
        path: Destination audio path.
        label: Description of the mock audio.
        config: Pipeline config.
    Output:
        None.
    TODO:
        Generate valid silent WAV files with soundfile when downstream tools need them.
    """
    write_placeholder(path, label, config)


def audio_exists(path: Path) -> bool:
    """Check whether an audio artifact exists.

    Input:
        path: Audio file path.
    Output:
        True when the path exists.
    TODO:
        Validate duration, sample rate, channel count, and decode errors.
    """
    return path.exists()
