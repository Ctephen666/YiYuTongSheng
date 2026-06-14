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



def _sha256_file(path: Path) -> str | None:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio.reshape(-1)
    return mono.astype(np.float32, copy=False)


def waveform_difference_metrics(source_path: str | Path, converted_path: str | Path) -> dict:
    """Check whether SVC output is actually different from SVS input.

    This is intentionally lightweight and dependency-free. It catches the common
    pipeline bug where RVC/SVC writes back to the SVS path and the evaluator then
    compares two copies of the same file.
    """
    source = Path(source_path)
    converted = Path(converted_path)
    result = {
        'source_path': str(source),
        'converted_path': str(converted),
        'source_exists': source.exists(),
        'converted_exists': converted.exists(),
        'same_resolved_path': False,
        'same_file_size': None,
        'sha256_equal': None,
        'sample_rate_match': None,
        'duration_diff_sec': None,
        'compared_samples': 0,
        'mean_abs_diff': None,
        'max_abs_diff': None,
        'rms_diff': None,
        'waveform_correlation': None,
        'likely_same_audio': None,
        'warnings': [],
    }
    if not source.exists() or not converted.exists():
        result['warnings'].append('Cannot compare SVS and SVC audio because one or both files are missing.')
        return result
    try:
        result['same_resolved_path'] = source.resolve() == converted.resolve()
    except Exception:
        result['same_resolved_path'] = False
    try:
        result['same_file_size'] = source.stat().st_size == converted.stat().st_size
    except Exception:
        pass

    source_hash = _sha256_file(source)
    converted_hash = _sha256_file(converted)
    if source_hash is not None and converted_hash is not None:
        result['sha256_equal'] = source_hash == converted_hash

    src_audio, src_sr, src_warnings = read_audio(source)
    conv_audio, conv_sr, conv_warnings = read_audio(converted)
    result['warnings'].extend(src_warnings)
    result['warnings'].extend(conv_warnings)
    if src_audio is None or conv_audio is None or src_sr is None or conv_sr is None:
        result['warnings'].append('Cannot compute waveform difference because audio decoding failed.')
        return result

    src = _to_mono_float(src_audio)
    conv = _to_mono_float(conv_audio)
    result['sample_rate_match'] = int(src_sr) == int(conv_sr)
    result['duration_diff_sec'] = _json_float(abs((src.size / float(src_sr)) - (conv.size / float(conv_sr))))

    if src_sr != conv_sr:
        result['warnings'].append(
            f'Sample rates differ ({src_sr} vs {conv_sr}); waveform comparison uses raw aligned samples only.'
        )

    length = min(src.size, conv.size)
    result['compared_samples'] = int(length)
    if length == 0:
        result['warnings'].append('No samples available for waveform comparison.')
        return result
    src = src[:length]
    conv = conv[:length]
    finite = np.isfinite(src) & np.isfinite(conv)
    if not np.any(finite):
        result['warnings'].append('No finite aligned samples for waveform comparison.')
        return result
    src = src[finite].astype(np.float64)
    conv = conv[finite].astype(np.float64)
    diff = conv - src
    result['mean_abs_diff'] = _json_float(np.mean(np.abs(diff)))
    result['max_abs_diff'] = _json_float(np.max(np.abs(diff)))
    result['rms_diff'] = _json_float(np.sqrt(np.mean(diff * diff)))
    if src.size > 1 and np.std(src) > 1e-12 and np.std(conv) > 1e-12:
        result['waveform_correlation'] = _json_float(np.corrcoef(src, conv)[0, 1])

    mean_abs = result['mean_abs_diff'] if result['mean_abs_diff'] is not None else 1.0
    max_abs = result['max_abs_diff'] if result['max_abs_diff'] is not None else 1.0
    corr = result['waveform_correlation'] if result['waveform_correlation'] is not None else 0.0
    likely_same = bool(
        result['same_resolved_path']
        or result['sha256_equal']
        or (mean_abs < 1e-7 and max_abs < 1e-6)
        or (mean_abs < 1e-6 and corr > 0.99999)
    )
    result['likely_same_audio'] = likely_same
    if likely_same:
        result['warnings'].append(
            'SVC output is nearly identical to SVS input; voice conversion may not have run or may have overwritten the SVS file.'
        )
    return result
