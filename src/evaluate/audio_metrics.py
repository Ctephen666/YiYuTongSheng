from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _json_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_with_soundfile(path: Path):
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        return None
    data, sample_rate = sf.read(str(path), always_2d=True)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def _read_wav_builtin(path: Path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = bytes_[:, 0].astype(np.int32) | (bytes_[:, 1].astype(np.int32) << 8) | (bytes_[:, 2].astype(np.int32) << 16)
        ints = np.where(ints & 0x800000, ints | ~0xFFFFFF, ints)
        data = ints.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data, int(sample_rate)


def read_audio(path: str | Path) -> tuple[np.ndarray | None, int | None, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    if not path.exists():
        return None, None, [f"Audio file not found: {path}"]
    if path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
        return None, None, [f"Unsupported audio extension: {path.suffix}"]

    try:
        loaded = _read_with_soundfile(path)
        if loaded is None:
            if path.suffix.lower() != ".wav":
                return None, None, [f"soundfile is unavailable; only WAV fallback can be read: {path}"]
            loaded = _read_wav_builtin(path)
        data, sample_rate = loaded
    except Exception as exc:  # noqa: BLE001
        return None, None, [f"Failed to read audio {path}: {exc}"]

    if data.size == 0:
        return None, sample_rate, [f"Audio is empty: {path}"]
    return data.astype(np.float32, copy=False), sample_rate, warnings


def _frame_view(audio: np.ndarray, frame_size: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((0, frame_size), dtype=np.float32)
    frame_size = max(1, int(frame_size))
    frame_count = int(np.ceil(audio.size / frame_size))
    padded = np.zeros(frame_count * frame_size, dtype=np.float32)
    padded[: audio.size] = audio
    return padded.reshape(frame_count, frame_size)


def _spectral_ratios(mono: np.ndarray, sample_rate: int) -> tuple[float | None, float | None, float | None]:
    if mono.size < 2 or sample_rate <= 0:
        return None, None, None
    n = min(int(sample_rate), mono.size)
    segment = mono[:n].astype(np.float32)
    if segment.size < 2:
        return None, None, None
    window = np.hanning(segment.size).astype(np.float32)
    spectrum = np.fft.rfft(segment * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0, 0.0, 0.0
    centroid = float(np.sum(freqs * power) / total)
    high = float(np.sum(power[freqs >= 8000.0]) / total)
    low = float(np.sum(power[freqs <= 80.0]) / total)
    return centroid, high, low


def audio_quality_metrics(path: str | Path) -> dict:
    result = {
        "path": str(path),
        "exists": Path(path).exists(),
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "peak": None,
        "rms": None,
        "clipping_ratio": None,
        "silence_ratio": None,
        "zero_crossing_rate": None,
        "spectral_centroid": None,
        "high_freq_ratio": None,
        "low_freq_ratio": None,
        "nan_or_inf": None,
        "is_valid_audio": False,
        "warnings": [],
    }
    audio, sample_rate, warnings = read_audio(path)
    result["warnings"].extend(warnings)
    if audio is None or sample_rate is None:
        return result

    channels = int(audio.shape[1]) if audio.ndim == 2 else 1
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio.reshape(-1)
    nan_or_inf = bool(np.isnan(mono).any() or np.isinf(mono).any())
    finite = mono[np.isfinite(mono)]
    if finite.size == 0:
        result["warnings"].append(f"Audio contains no finite samples: {path}")
        result["nan_or_inf"] = nan_or_inf
        return result

    frame_size = max(1, int(round(sample_rate * 0.02)))
    frames = _frame_view(finite, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1)) if frames.size else np.asarray([], dtype=np.float32)
    silence_ratio = float(np.mean(frame_rms < 1e-4)) if frame_rms.size else None
    signs = np.signbit(finite)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if finite.size > 1 else 0.0
    spectral_centroid, high_ratio, low_ratio = _spectral_ratios(finite, sample_rate)

    result.update(
        {
            "duration_sec": _json_float(finite.size / float(sample_rate)),
            "sample_rate": int(sample_rate),
            "channels": channels,
            "peak": _json_float(np.max(np.abs(finite))),
            "rms": _json_float(np.sqrt(np.mean(finite * finite))),
            "clipping_ratio": _json_float(np.mean(np.abs(finite) >= 0.99)),
            "silence_ratio": _json_float(silence_ratio),
            "zero_crossing_rate": _json_float(zcr),
            "spectral_centroid": _json_float(spectral_centroid),
            "high_freq_ratio": _json_float(high_ratio),
            "low_freq_ratio": _json_float(low_ratio),
            "nan_or_inf": nan_or_inf,
            "is_valid_audio": True,
        }
    )
    return result
