from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from src.evaluate.audio_metrics import audio_quality_metrics
from src.evaluate.pitch_metrics import pitch_preservation_metrics
from src.evaluate.report import compute_overall
from src.evaluate.speaker_metrics import cosine_similarity
from src.evaluate.text_metrics import char_error_rate


def _write_sine(path: Path, freq: float = 440.0, sr: int = 16000, duration: float = 0.5) -> None:
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    audio = 0.2 * np.sin(2.0 * np.pi * freq * t)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def test_audio_quality_sine(tmp_path: Path):
    wav = tmp_path / "sine.wav"
    _write_sine(wav)
    metrics = audio_quality_metrics(wav)
    assert metrics["is_valid_audio"] is True
    assert metrics["sample_rate"] == 16000
    assert metrics["rms"] is not None


def test_pitch_rmse_cents_nearby_sines(tmp_path: Path):
    source = tmp_path / "source.wav"
    converted = tmp_path / "converted.wav"
    _write_sine(source, 440.0)
    _write_sine(converted, 445.0)
    metrics = pitch_preservation_metrics(source, converted, frame_ms=40, min_f0=80, max_f0=800)
    assert metrics["compared_frame_count"] >= 1
    assert "warnings" in metrics


def test_cer():
    assert char_error_rate("你好世界", "你好世") == 0.25


def test_cosine_similarity():
    value = cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert value == 1.0


def test_missing_file_does_not_crash(tmp_path: Path):
    metrics = audio_quality_metrics(tmp_path / "missing.wav")
    assert metrics["is_valid_audio"] is False
    assert metrics["warnings"]


def test_report_json_serializable():
    report = {
        "audio_quality": {"svc": {"is_valid_audio": True, "clipping_ratio": 0.0, "silence_ratio": 0.1, "high_freq_ratio": 0.1, "peak": 0.5, "rms": 0.08}},
        "pitch_preservation": {"f0_rmse_cents": 20.0, "f0_correlation": 0.9, "voiced_unvoiced_accuracy": 0.95},
        "intelligibility": {"cer": None},
        "speaker_similarity": {"speaker_similarity_to_target": None},
    }
    report["overall"] = compute_overall(report)
    json.dumps(report)
