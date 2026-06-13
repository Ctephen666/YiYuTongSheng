from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def midi_to_hz(midi: int | float | None) -> float:
    if midi is None:
        return 0.0
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _note_start(note: dict) -> float:
    return _float_value(note.get("start"), 0.0)


def _note_end(note: dict) -> float:
    start = _note_start(note)
    end = _float_value(note.get("end"), start)
    if end <= start:
        end = start + _float_value(note.get("duration"), 0.0)
    return end


def flatten_melody_notes(melody_data: dict | list[dict]) -> list[dict]:
    if isinstance(melody_data, list):
        return [note for note in melody_data if isinstance(note, dict)]
    notes = []
    for phrase in melody_data.get("phrases", []):
        for note in phrase.get("notes", []):
            if isinstance(note, dict):
                notes.append(note)
    return notes


def build_midi_f0_from_melody(frame_times, melody_notes: dict | list[dict], midi_transpose: int | float = 0):
    import numpy as np

    times = np.asarray(frame_times, dtype=np.float64)
    midi_f0 = np.zeros_like(times, dtype=np.float64)
    for note in flatten_melody_notes(melody_notes):
        midi = note.get("midi")
        if midi is None:
            continue
        start = _note_start(note)
        end = _note_end(note)
        if end <= start:
            continue
        hz = midi_to_hz(float(midi) + float(midi_transpose or 0))
        midi_f0[(times >= start) & (times < end)] = hz
    return midi_f0


def smooth_voiced_f0(f0, window_size: int):
    import numpy as np

    values = np.asarray(f0, dtype=np.float64)
    if window_size <= 1:
        return values.copy()
    smoothed = values.copy()
    voiced = values > 0
    start = None
    kernel = np.ones(int(window_size), dtype=np.float64) / float(window_size)
    for index, is_voiced in enumerate(voiced.tolist() + [False]):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            end = index
            if end - start >= window_size:
                smoothed[start:end] = np.convolve(values[start:end], kernel, mode="same")
            start = None
    return smoothed


def extract_source_f0(vocals_path: str | Path, sample_rate: int, frame_period: float, method: str = "harvest"):
    import librosa
    import numpy as np
    import pyworld as pw

    wav, _ = librosa.load(str(vocals_path), sr=int(sample_rate), mono=True)
    wav64 = np.asarray(wav, dtype=np.float64)
    method = str(method or "harvest").lower()
    if method == "dio":
        raw_f0, times = pw.dio(wav64, int(sample_rate), frame_period=float(frame_period))
        f0 = pw.stonemask(wav64, raw_f0, times, int(sample_rate))
    else:
        f0, times = pw.harvest(wav64, int(sample_rate), frame_period=float(frame_period))
    return times, f0


def refine_source_f0_with_midi(
    source_f0,
    midi_f0,
    max_source_delta_semitones: float = 2.5,
    source_f0_smooth_window: int = 5,
    source_f0_detail_strength: float = 1.0,
    midi_anchor_strength: float = 0.0,
):
    import numpy as np

    source = np.asarray(source_f0, dtype=np.float64)
    midi = np.asarray(midi_f0, dtype=np.float64)
    refined = np.zeros_like(source, dtype=np.float64)
    delta = np.zeros_like(source, dtype=np.float64)
    delta_clamped = np.zeros_like(source, dtype=np.float64)
    max_delta = float(max_source_delta_semitones)
    detail_weight = max(0.0, float(source_f0_detail_strength))
    midi_weight = max(0.0, float(midi_anchor_strength))
    weight_sum = detail_weight + midi_weight

    for index, (source_value, midi_value) in enumerate(zip(source, midi)):
        if midi_value <= 0:
            refined[index] = 0.0
            continue
        if source_value <= 0:
            refined[index] = midi_value
            continue
        current_delta = 12.0 * np.log2(source_value / midi_value)
        current_clamped = float(np.clip(current_delta, -max_delta, max_delta))
        source_guided = midi_value * (2.0 ** (current_clamped / 12.0))
        delta[index] = float(current_delta)
        delta_clamped[index] = current_clamped
        refined[index] = source_guided if weight_sum <= 0 else (source_guided * detail_weight + midi_value * midi_weight) / weight_sum

    refined = smooth_voiced_f0(refined, int(source_f0_smooth_window or 1))
    return refined, delta, delta_clamped


def write_source_f0_csv(path: str | Path, times, source_f0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "source_f0"])
        for time, f0 in zip(times, source_f0):
            writer.writerow([f"{float(time):.6f}", f"{float(f0):.6f}"])


def write_refined_f0_csv(path: str | Path, times, source_f0, midi_f0, delta, delta_clamped, refined_f0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "source_f0", "midi_f0", "delta_semitone", "delta_clamped", "refined_f0"])
        for row in zip(times, source_f0, midi_f0, delta, delta_clamped, refined_f0):
            writer.writerow([f"{float(value):.6f}" for value in row])


def read_refined_f0_csv(path: str | Path):
    import numpy as np

    times = []
    source_f0 = []
    midi_f0 = []
    refined_f0 = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            times.append(_float_value(row.get("time"), 0.0))
            source_f0.append(_float_value(row.get("source_f0"), 0.0))
            midi_f0.append(_float_value(row.get("midi_f0"), 0.0))
            refined_f0.append(_float_value(row.get("refined_f0"), 0.0))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(source_f0, dtype=np.float64),
        np.asarray(midi_f0, dtype=np.float64),
        np.asarray(refined_f0, dtype=np.float64),
    )