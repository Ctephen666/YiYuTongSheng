from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.evaluate.audio_metrics import read_audio


def _to_mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1) if audio.ndim == 2 else audio.reshape(-1)


def _autocorr_f0(mono: np.ndarray, sample_rate: int, frame_ms: float, min_f0: float, max_f0: float) -> np.ndarray:
    frame_size = max(64, int(round(sample_rate * frame_ms / 1000.0)))
    hop = frame_size
    if mono.size < frame_size:
        return np.zeros(0, dtype=np.float32)
    min_lag = max(1, int(sample_rate / max_f0))
    max_lag = min(frame_size - 2, int(sample_rate / min_f0))
    if max_lag <= min_lag:
        return np.zeros(0, dtype=np.float32)
    window = np.hanning(frame_size).astype(np.float32)
    values = []
    for start in range(0, mono.size - frame_size + 1, hop):
        frame = mono[start : start + frame_size].astype(np.float32)
        frame = frame - float(np.mean(frame))
        energy = float(np.sqrt(np.mean(frame * frame)))
        if energy < 1e-4:
            values.append(0.0)
            continue
        corr = np.correlate(frame * window, frame * window, mode="full")[frame_size - 1 :]
        if corr[0] <= 1e-9:
            values.append(0.0)
            continue
        search = corr[min_lag : max_lag + 1]
        lag = int(np.argmax(search) + min_lag)
        confidence = float(corr[lag] / corr[0])
        values.append(float(sample_rate / lag) if confidence >= 0.25 else 0.0)
    return np.asarray(values, dtype=np.float32)


def extract_f0(path: str | Path, frame_ms: float = 20.0, min_f0: float = 50.0, max_f0: float = 1100.0) -> tuple[np.ndarray, list[str]]:
    audio, sample_rate, warnings = read_audio(path)
    if audio is None or sample_rate is None:
        return np.zeros(0, dtype=np.float32), warnings
    mono = _to_mono(audio)
    finite = mono[np.isfinite(mono)]
    if finite.size == 0:
        return np.zeros(0, dtype=np.float32), warnings + [f"No finite samples for F0 extraction: {path}"]

    try:
        import pyworld as pw  # type: ignore

        frame_period = float(frame_ms)
        f0, _ = pw.dio(finite.astype(np.float64), int(sample_rate), frame_period=frame_period, f0_floor=min_f0, f0_ceil=max_f0)
        f0 = pw.stonemask(finite.astype(np.float64), f0, np.arange(len(f0)) * frame_period / 1000.0, int(sample_rate))
        return np.asarray(f0, dtype=np.float32), warnings
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        hop_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
        f0 = librosa.yin(finite.astype(np.float32), fmin=min_f0, fmax=max_f0, sr=sample_rate, hop_length=hop_length)
        f0 = np.where(np.isfinite(f0), f0, 0.0)
        return np.asarray(f0, dtype=np.float32), warnings
    except Exception:
        pass

    try:
        f0 = _autocorr_f0(finite, int(sample_rate), frame_ms, min_f0, max_f0)
        return f0, warnings + ["Using autocorrelation F0 fallback."]
    except Exception as exc:  # noqa: BLE001
        return np.zeros(0, dtype=np.float32), warnings + [f"F0 extraction failed: {exc}"]


def _safe_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def pitch_preservation_metrics(source_path: str | Path, converted_path: str | Path, frame_ms: float = 20.0, min_f0: float = 50.0, max_f0: float = 1100.0) -> dict:
    warnings: list[str] = []
    source_f0, source_warnings = extract_f0(source_path, frame_ms, min_f0, max_f0)
    converted_f0, converted_warnings = extract_f0(converted_path, frame_ms, min_f0, max_f0)
    warnings.extend(source_warnings)
    warnings.extend(converted_warnings)

    length = min(len(source_f0), len(converted_f0))
    result = {
        "source_path": str(source_path),
        "converted_path": str(converted_path),
        "f0_rmse_hz": None,
        "f0_rmse_cents": None,
        "f0_mae_hz": None,
        "f0_mae_cents": None,
        "f0_correlation": None,
        "voiced_unvoiced_accuracy": None,
        "voiced_frame_count": 0,
        "compared_frame_count": 0,
        "source_f0_mean": None,
        "converted_f0_mean": None,
        "source_f0_min": None,
        "source_f0_max": None,
        "converted_f0_min": None,
        "converted_f0_max": None,
        "warnings": warnings,
    }
    if length == 0:
        result["warnings"].append("No F0 frames extracted.")
        return result

    src = source_f0[:length].astype(np.float64)
    conv = converted_f0[:length].astype(np.float64)
    finite = np.isfinite(src) & np.isfinite(conv)
    src_voiced = src > 0
    conv_voiced = conv > 0
    voiced = finite & src_voiced & conv_voiced
    result["voiced_unvoiced_accuracy"] = _safe_float(np.mean(src_voiced[finite] == conv_voiced[finite])) if np.any(finite) else None
    result["voiced_frame_count"] = int(np.sum(voiced))
    result["compared_frame_count"] = int(np.sum(finite))

    if np.sum(voiced) < 5:
        result["warnings"].append("Too few voiced frames for reliable pitch comparison.")
        return result

    src_v = src[voiced]
    conv_v = conv[voiced]
    hz_error = conv_v - src_v
    cents_error = 1200.0 * np.log2(conv_v / src_v)
    corr = np.corrcoef(src_v, conv_v)[0, 1] if len(src_v) > 1 else None
    result.update(
        {
            "f0_rmse_hz": _safe_float(np.sqrt(np.mean(hz_error * hz_error))),
            "f0_rmse_cents": _safe_float(np.sqrt(np.mean(cents_error * cents_error))),
            "f0_mae_hz": _safe_float(np.mean(np.abs(hz_error))),
            "f0_mae_cents": _safe_float(np.mean(np.abs(cents_error))),
            "f0_correlation": _safe_float(corr),
            "source_f0_mean": _safe_float(np.mean(src_v)),
            "converted_f0_mean": _safe_float(np.mean(conv_v)),
            "source_f0_min": _safe_float(np.min(src_v)),
            "source_f0_max": _safe_float(np.max(src_v)),
            "converted_f0_min": _safe_float(np.min(conv_v)),
            "converted_f0_max": _safe_float(np.max(conv_v)),
        }
    )
    return result
