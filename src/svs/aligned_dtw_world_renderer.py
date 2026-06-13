from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, path_from_config, project_root, resolve_path, write_text
from src.common.json_utils import read_json, write_json


def midi_to_hz(midi: int | float | None) -> float:
    if midi is None:
        return 0.0
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


class AlignedDTWWorldRenderer:
    """Use NoteLyricAligner syllable slots, local DTW, and soft WORLD F0 replacement."""

    WARNING = "This is syllable-slot DTW plus MIDI F0 replacement. It is still not full SVS."

    def __init__(self, config: dict):
        self.config = config

    def _svs_config(self) -> dict:
        return self.config.get("svs", {})

    def _cfg(self, key: str, default):
        return self._svs_config().get(key, default)

    def _sample_rate(self) -> int:
        return int(self.config.get("project", {}).get("sample_rate", 44100))

    def _frame_period(self) -> float:
        return float(self._cfg("world_frame_period", 5.0) or 5.0)

    def _flat_vocal_path(self) -> Path:
        paths = self.config.get("paths", {})
        if paths.get("svs_vocal_flat"):
            return path_from_config(self.config, "svs_vocal_flat")
        if self._cfg("flat_vocal", None):
            return resolve_path(self.config, self._cfg("flat_vocal", "data/svs/target_language_vocal_flat.wav"))
        return project_root(self.config) / "data" / "svs" / "target_language_vocal_flat.wav"

    def _melotts_report_path(self) -> Path:
        return self._flat_vocal_path().parent / "melotts_render_report.json"

    def _dtw_alignment_report_path(self) -> Path:
        if self.config.get("paths", {}).get("dtw_syllable_alignment"):
            return path_from_config(self.config, "dtw_syllable_alignment")
        return project_root(self.config) / "data" / "alignment" / "dtw_syllable_alignment.json"

    def _world_report_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "aligned_dtw_world_report.json"

    def _debug_f0_dir(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "debug_f0"

    def _load_dependencies(self):
        try:
            import librosa
            import numpy as np
            import pyworld as pw
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyworld/scipy dependencies are not installed. Install with: pip install pyworld scipy") from exc
        return librosa, np, pw, sf

    def _float_value(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _note_start(self, note: dict) -> float:
        return self._float_value(note.get("start"), 0.0)

    def _note_end(self, note: dict) -> float:
        start = self._note_start(note)
        end = self._float_value(note.get("end"), start)
        if end <= start:
            end = start + self._float_value(note.get("duration"), 0.0)
        return end

    def _note_midi(self, note: dict) -> float | None:
        try:
            return float(note.get("midi")) + float(self._cfg("midi_transpose", -12) or 0)
        except (TypeError, ValueError):
            return None

    def _phrase_by_id(self, phrases: list[dict]) -> dict:
        result = {}
        for phrase in phrases:
            phrase_id = phrase.get("id")
            if phrase_id is None:
                continue
            result[phrase_id] = phrase
            result[str(phrase_id)] = phrase
        return result

    def _matched_phrase(self, phrases_by_id: dict, phrases: list[dict], phrase_id: Any, index: int) -> dict:
        if phrase_id in phrases_by_id:
            return phrases_by_id[phrase_id]
        if str(phrase_id) in phrases_by_id:
            return phrases_by_id[str(phrase_id)]
        if index < len(phrases):
            return phrases[index]
        return {}

    def _resolve_report_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return project_root(self.config) / path

    def _fade(self, wav, sample_rate: int, np):
        fade_ms = float(self._cfg("phrase_fade_ms", 25) or 0)
        fade_samples = int(sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(wav) // 2)
        if fade_samples <= 0:
            return wav
        wav[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
        wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
        return wav

    def _fit_length(self, wav, target_len: int, np):
        if len(wav) > target_len:
            return wav[:target_len]
        if len(wav) < target_len:
            return np.pad(wav, (0, target_len - len(wav)))
        return wav

    def _normalize_feature(self, values, np):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        values = np.nan_to_num(values)
        return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-8)

    def _linear_mapping(self, source_frame_count: int, slot_duration: float, np):
        if source_frame_count <= 1:
            return np.asarray([0.0], dtype=np.float64)
        return np.linspace(0.0, slot_duration, source_frame_count, dtype=np.float64)

    def _target_energy_template(self, notes: list[dict], slot_start_global: float, slot_duration: float, target_frame_count: int, np):
        target = np.full(target_frame_count, 0.7, dtype=np.float64)
        if target_frame_count <= 1 or slot_duration <= 0:
            return target
        target_times = np.linspace(0.0, slot_duration, target_frame_count, endpoint=True, dtype=np.float64)
        for note in notes:
            local_start = max(0.0, self._note_start(note) - slot_start_global)
            local_end = min(slot_duration, self._note_end(note) - slot_start_global)
            duration = local_end - local_start
            if duration <= 0:
                continue
            mask = (target_times >= local_start) & (target_times < local_end)
            if not bool(np.any(mask)):
                continue
            progress = (target_times[mask] - local_start) / max(duration, 1e-8)
            envelope = np.ones_like(progress, dtype=np.float64)
            attack = progress < 0.15
            release = progress > 0.80
            envelope[attack] = 0.5 + (progress[attack] / 0.15) * 0.5
            envelope[release] = 1.0 - ((progress[release] - 0.80) / 0.20) * 0.3
            target[mask] = envelope
        return target

    def _run_energy_dtw(self, item_wav, notes: list[dict], slot_start_global: float, slot_duration: float, sample_rate: int, librosa, np):
        hop_length = 256
        fps = float(self._cfg("dtw_energy_frames_per_second", 200) or 200)
        target_frame_count = max(4, int(slot_duration * fps))
        if len(item_wav) < hop_length or slot_duration <= 0:
            return {"source_to_target_time": self._linear_mapping(1, max(slot_duration, 0.0), np), "source_frame_count": 1, "target_frame_count": target_frame_count, "dtw_cost": None, "used_fallback": True}
        src_rms = librosa.feature.rms(y=item_wav, frame_length=1024, hop_length=hop_length)[0]
        src = self._normalize_feature(np.log1p(src_rms), np)
        source_frame_count = int(len(src))
        target = self._normalize_feature(self._target_energy_template(notes, slot_start_global, slot_duration, target_frame_count, np), np)
        if source_frame_count < 2 or target_frame_count < 2:
            return {"source_to_target_time": self._linear_mapping(max(source_frame_count, 1), slot_duration, np), "source_frame_count": max(source_frame_count, 1), "target_frame_count": target_frame_count, "dtw_cost": None, "used_fallback": True}
        try:
            D, wp = librosa.sequence.dtw(X=src.reshape(1, -1), Y=target.reshape(1, -1), metric="euclidean", global_constraints=True, band_rad=float(self._cfg("dtw_band_rad", 0.15) or 0.15))
            wp = wp[::-1]
            grouped: dict[int, list[int]] = {}
            for source_index, target_index in wp:
                source_i = int(source_index)
                target_i = int(target_index)
                if 0 <= source_i < source_frame_count:
                    grouped.setdefault(source_i, []).append(target_i)
            source_to_target_time = self._linear_mapping(source_frame_count, slot_duration, np)
            for source_i, target_indices in grouped.items():
                source_to_target_time[source_i] = float(np.mean(target_indices)) / float(max(target_frame_count - 1, 1)) * slot_duration
            return {"source_to_target_time": source_to_target_time, "source_frame_count": source_frame_count, "target_frame_count": target_frame_count, "dtw_cost": float(D[-1, -1]), "used_fallback": False}
        except Exception:
            return {"source_to_target_time": self._linear_mapping(source_frame_count, slot_duration, np), "source_frame_count": source_frame_count, "target_frame_count": target_frame_count, "dtw_cost": None, "used_fallback": True}
    def _pitch_at_time(self, notes: list[dict], target_time_global: float) -> float | None:
        nearest_midi = None
        nearest_distance = None
        for note in notes:
            midi = self._note_midi(note)
            if midi is None:
                continue
            start = self._note_start(note)
            end = self._note_end(note)
            if start <= target_time_global < end:
                return midi
            distance = abs(((start + end) / 2.0) - target_time_global)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_midi = midi
        return nearest_midi

    def _valid_original_f0(self, value: float) -> bool:
        if value <= 0:
            return False
        return float(self._cfg("min_voiced_f0", 70.0)) <= value <= float(self._cfg("max_voiced_f0", 900.0))

    def _blend_f0(self, original_f0: float, midi_f0: float, strength: float) -> float:
        if original_f0 <= 0 or midi_f0 <= 0:
            return 0.0
        strength = min(1.0, max(0.0, float(strength)))
        return float(original_f0) * (1.0 - strength) + float(midi_f0) * strength

    def _replacement_strength(self, item: dict, target_time_global: float, slot_start_global: float, slot_end_global: float) -> float:
        strong = float(self._cfg("f0_replace_strength", 0.65) or 0.65)
        weak = float(self._cfg("consonant_f0_strength", 0.15) or 0.15)
        if not bool(self._cfg("enable_vowel_centered_f0", True)):
            return strong
        if not bool(item.get("has_vowel", True)):
            return weak
        vowel_start = item.get("recommended_vowel_start")
        vowel_end = item.get("recommended_vowel_end")
        try:
            vowel_start = float(vowel_start)
            vowel_end = float(vowel_end)
        except (TypeError, ValueError):
            duration = max(1e-8, slot_end_global - slot_start_global)
            ratio = min(0.95, max(0.1, float(self._cfg("vowel_center_ratio", 0.65) or 0.65)))
            pad = (1.0 - ratio) * duration / 2.0
            vowel_start = slot_start_global + pad
            vowel_end = slot_end_global - pad
        return strong if vowel_start <= target_time_global <= vowel_end else weak

    def _all_item_notes(self, items: list[dict]) -> list[dict]:
        notes = []
        for item in items:
            item_notes = item.get("notes", [])
            if isinstance(item_notes, list):
                notes.extend(item_notes)
        return notes

    def _item_slot(self, item: dict, phrase_start: float, phrase_duration: float) -> tuple[float, float, float, float, list[dict]] | None:
        raw_notes = item.get("notes", [])
        notes = [note for note in raw_notes if isinstance(note, dict) and self._note_midi(note) is not None]
        if not notes:
            return None
        slot_start_global = self._float_value(item.get("slot_start"), min(self._note_start(note) for note in notes))
        slot_end_global = self._float_value(item.get("slot_end"), max(self._note_end(note) for note in notes))
        slot_start_local = max(0.0, slot_start_global - phrase_start)
        slot_end_local = min(phrase_duration, slot_end_global - phrase_start)
        if slot_end_local <= slot_start_local:
            return None
        return slot_start_global, slot_end_global, slot_start_local, slot_end_local, notes

    def _apply_portamento_to_midi_f0_curve(self, hard_f0, frame_times, items: list[dict], phrase_start: float, np):
        if not bool(self._cfg("enable_portamento", True)):
            return hard_f0
        half_window = float(self._cfg("portamento_ms", 60) or 60) / 1000.0 / 2.0
        if half_window <= 0:
            return hard_f0
        smoothed = hard_f0.copy()
        for item in items:
            notes = [note for note in item.get("notes", []) if isinstance(note, dict) and self._note_midi(note) is not None]
            notes = sorted(notes, key=self._note_start)
            for previous, current in zip(notes, notes[1:]):
                previous_f0 = midi_to_hz(self._note_midi(previous))
                current_f0 = midi_to_hz(self._note_midi(current))
                if previous_f0 <= 0 or current_f0 <= 0:
                    continue
                boundary = self._note_start(current) - phrase_start
                mask = (frame_times >= boundary - half_window) & (frame_times <= boundary + half_window) & (smoothed > 0)
                if not bool(np.any(mask)):
                    continue
                x = (frame_times[mask] - (boundary - half_window)) / max(2.0 * half_window, 1e-8)
                alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))
                smoothed[mask] = previous_f0 * (1.0 - alpha) + current_f0 * alpha
        return smoothed

    def _limit_f0_jumps(self, f0, max_jump_semitones: float, np):
        if max_jump_semitones <= 0:
            return f0
        limited = f0.copy()
        voiced = limited > 0
        start = None
        for index, is_voiced in enumerate(voiced.tolist() + [False]):
            if is_voiced and start is None:
                start = index
            elif not is_voiced and start is not None:
                end = index
                for frame_index in range(start + 1, end):
                    prev = limited[frame_index - 1]
                    current = limited[frame_index]
                    if prev <= 0 or current <= 0:
                        continue
                    diff = 12.0 * np.log2(current / prev)
                    if abs(diff) > max_jump_semitones:
                        limited[frame_index] = prev * (2.0 ** (np.sign(diff) * max_jump_semitones / 12.0))
                start = None
        return limited

    def _max_f0_jump(self, f0, np) -> float:
        max_jump = 0.0
        previous = 0.0
        for value in f0:
            value = float(value)
            if value <= 0:
                previous = 0.0
                continue
            if previous > 0:
                max_jump = max(max_jump, abs(12.0 * np.log2(value / previous)))
            previous = value
        return float(max_jump)

    def _smooth_voiced_f0(self, f0, window_size: int, np):
        if window_size <= 1:
            return f0
        smoothed = f0.copy()
        voiced = f0 > 0
        start = None
        kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
        for index, is_voiced in enumerate(voiced.tolist() + [False]):
            if is_voiced and start is None:
                start = index
            elif not is_voiced and start is not None:
                end = index
                if end - start >= window_size:
                    smoothed[start:end] = np.convolve(f0[start:end], kernel, mode="same")
                start = None
        return smoothed

    def _apply_vibrato(self, target_f0, items: list[dict], phrase_start: float, frame_times, np):
        if not bool(self._cfg("enable_vibrato", False)):
            return target_f0
        rate = float(self._cfg("vibrato_rate", 5.0) or 5.0)
        depth = float(self._cfg("vibrato_depth", 0.003) or 0.003)
        min_duration = float(self._cfg("min_vibrato_note_duration", 0.5) or 0.5)
        for item in items:
            for note in item.get("notes", []):
                start_global = self._note_start(note)
                end_global = self._note_end(note)
                if end_global - start_global < min_duration:
                    continue
                local_start = start_global - phrase_start
                local_end = end_global - phrase_start
                mask = (frame_times >= local_start) & (frame_times < local_end) & (target_f0 > 0)
                if not bool(np.any(mask)):
                    continue
                local_t = frame_times[mask] - local_start
                target_f0[mask] *= 1.0 + depth * np.sin(2.0 * np.pi * rate * local_t)
        return target_f0

    def _build_target_f0(self, wav, f0, frame_times, phrase: dict, phrase_report: dict, sample_rate: int, librosa, np):
        phrase_start = float(phrase_report.get("start", 0.0))
        phrase_duration = float(phrase_report.get("duration", len(wav) / sample_rate if sample_rate else 0.0))
        items = phrase.get("items", []) if isinstance(phrase.get("items", []), list) else []
        hard_midi_f0 = np.zeros_like(f0, dtype=np.float64)
        strength_curve = np.zeros_like(f0, dtype=np.float64)
        item_reports = []
        any_slot = False
        for item in items:
            slot = self._item_slot(item, phrase_start, phrase_duration)
            if slot is None:
                continue
            slot_start_global, slot_end_global, slot_start_local, slot_end_local, notes = slot
            slot_duration = slot_end_local - slot_start_local
            start_sample = max(0, int(slot_start_local * sample_rate))
            end_sample = min(len(wav), max(start_sample + 1, int(slot_end_local * sample_rate)))
            any_slot = True
            dtw_result = self._run_energy_dtw(wav[start_sample:end_sample], notes, slot_start_global, slot_duration, sample_rate, librosa, np)
            source_to_target_time = dtw_result["source_to_target_time"]
            source_frame_count = int(dtw_result["source_frame_count"])
            rms_hop_seconds = 256.0 / float(sample_rate)
            slot_indices = np.where((frame_times >= slot_start_local) & (frame_times < slot_end_local))[0]
            for frame_index in slot_indices:
                original = float(f0[frame_index])
                if not self._valid_original_f0(original):
                    continue
                source_frame_index = int(round((float(frame_times[frame_index]) - slot_start_local) / rms_hop_seconds))
                if 0 <= source_frame_index < source_frame_count and not (dtw_result.get("used_fallback") and source_frame_count <= 1):
                    target_time = float(source_to_target_time[source_frame_index])
                else:
                    target_time = (float(frame_times[frame_index]) - slot_start_local) / max(slot_duration, 1e-8) * slot_duration
                target_time_global = slot_start_global + min(max(target_time, 0.0), slot_duration)
                hard_midi_f0[frame_index] = midi_to_hz(self._pitch_at_time(notes, target_time_global))
                strength_curve[frame_index] = self._replacement_strength(item, target_time_global, slot_start_global, slot_end_global)
            item_reports.append({
                "unit": item.get("unit", ""), "word": item.get("word", ""),
                "slot_start": slot_start_global, "slot_end": slot_end_global,
                "slot_start_local": slot_start_local, "slot_end_local": slot_end_local,
                "slot_duration": slot_duration,
                "recommended_vowel_start": item.get("recommended_vowel_start"),
                "recommended_vowel_end": item.get("recommended_vowel_end"),
                "note_count": len(notes), "source_frame_count": dtw_result["source_frame_count"],
                "target_frame_count": dtw_result["target_frame_count"], "dtw_cost": dtw_result["dtw_cost"],
                "used_fallback": dtw_result["used_fallback"], "notes": notes,
            })
        phrase_notes = self._all_item_notes(items)
        if not any_slot or not phrase_notes:
            raise RuntimeError("Phrase has no usable alignment items with MIDI notes.")
        missing_voiced = (f0 > 0) & (hard_midi_f0 <= 0)
        for frame_index in np.where(missing_voiced)[0]:
            original = float(f0[frame_index])
            if not self._valid_original_f0(original):
                continue
            hard_midi_f0[frame_index] = midi_to_hz(self._pitch_at_time(phrase_notes, phrase_start + float(frame_times[frame_index])))
            strength_curve[frame_index] = float(self._cfg("consonant_f0_strength", 0.15) or 0.15)
        hard_midi_f0 = self._apply_portamento_to_midi_f0_curve(hard_midi_f0, frame_times, items, phrase_start, np)
        target_f0 = np.zeros_like(f0, dtype=np.float64)
        hard_mode = str(self._cfg("f0_mode", "soft") or "soft").lower() == "hard"
        for index, original in enumerate(f0):
            original = float(original)
            midi_f0 = float(hard_midi_f0[index])
            if not self._valid_original_f0(original):
                target_f0[index] = 0.0
                continue
            target_f0[index] = midi_f0 if hard_mode and midi_f0 > 0 else self._blend_f0(original, midi_f0, strength_curve[index])
        max_jump_before = self._max_f0_jump(target_f0, np)
        target_f0 = self._limit_f0_jumps(target_f0, float(self._cfg("max_f0_jump_semitones", 5.0) or 5.0), np)
        target_f0 = self._smooth_voiced_f0(target_f0, int(self._cfg("f0_smooth_window", 9) or 9), np)
        target_f0 = self._apply_vibrato(target_f0, items, phrase_start, frame_times, np)
        target_f0 = self._limit_f0_jumps(target_f0, float(self._cfg("max_f0_jump_semitones", 5.0) or 5.0), np)
        valid_original = np.asarray([self._valid_original_f0(float(value)) for value in f0], dtype=bool)
        stats = {
            "mean_original_f0": float(np.mean(f0[valid_original])) if bool(np.any(valid_original)) else 0.0,
            "mean_target_f0": float(np.mean(target_f0[target_f0 > 0])) if bool(np.any(target_f0 > 0)) else 0.0,
            "max_f0_jump_before": max_jump_before,
            "max_f0_jump_after": self._max_f0_jump(target_f0, np),
        }
        return target_f0, hard_midi_f0, strength_curve, item_reports, stats
    def _postprocess_wave(self, wav, sample_rate: int, np):
        processed = np.asarray(wav, dtype=np.float64)
        if bool(self._cfg("enable_de_metal_filter", True)):
            try:
                from scipy import signal
                cutoff = float(self._cfg("de_metal_lowpass_hz", 8500) or 8500)
                nyquist = sample_rate / 2.0
                if 20.0 < cutoff < nyquist * 0.98:
                    sos = signal.butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
                    processed = signal.sosfiltfilt(sos, processed) if len(processed) > 32 else signal.sosfilt(sos, processed)
            except Exception:
                pass
        if bool(self._cfg("enable_soft_limiter", True)):
            threshold = float(self._cfg("limiter_threshold", 0.95) or 0.95)
            if threshold > 0:
                processed = np.tanh(processed / threshold) * threshold
        return np.nan_to_num(processed)

    def _write_f0_debug_csv(self, phrase_number: int, frame_times, original_f0, hard_midi_f0, target_f0, strength_curve) -> str:
        if phrase_number > 5:
            return ""
        debug_path = self._debug_f0_dir() / f"phrase_{phrase_number:03d}_f0.csv"
        rows = ["time,original_f0,hard_midi_f0,soft_target_f0,strength"]
        for time, original, hard, soft, strength in zip(frame_times, original_f0, hard_midi_f0, target_f0, strength_curve):
            rows.append(f"{float(time):.6f},{float(original):.6f},{float(hard):.6f},{float(soft):.6f},{float(strength):.6f}")
        write_text(debug_path, "\n".join(rows) + "\n", self.config)
        return str(debug_path)

    def _synthesize_phrase(self, aligned_path: Path, phrase: dict, phrase_report: dict, phrase_number: int, sample_rate: int, librosa, np, pw):
        wav, _ = librosa.load(aligned_path, sr=sample_rate, mono=True)
        target_len = len(wav)
        wav64 = np.asarray(wav, dtype=np.float64)
        frame_period = self._frame_period()
        f0, frame_times = pw.harvest(wav64, sample_rate, frame_period=frame_period)
        sp = pw.cheaptrick(wav64, f0, frame_times, sample_rate)
        ap = pw.d4c(wav64, f0, frame_times, sample_rate)
        target_f0, hard_midi_f0, strength_curve, item_reports, stats = self._build_target_f0(wav64, f0, frame_times, phrase, phrase_report, sample_rate, librosa, np)
        synth = pw.synthesize(target_f0.astype(np.float64), sp, ap, sample_rate, frame_period=frame_period)
        synth = self._fit_length(synth, target_len, np)
        synth = self._postprocess_wave(synth, sample_rate, np)
        synth = self._fade(synth, sample_rate, np)
        debug_path = self._write_f0_debug_csv(phrase_number, frame_times, f0, hard_midi_f0, target_f0, strength_curve)
        return synth, target_f0, item_reports, stats, debug_path

    def _metal_reduction_report(self) -> dict:
        return {
            "midi_transpose": self._cfg("midi_transpose", -12),
            "f0_mode": self._cfg("f0_mode", "soft"),
            "f0_replace_strength": self._cfg("f0_replace_strength", 0.65),
            "consonant_f0_strength": self._cfg("consonant_f0_strength", 0.15),
            "enable_vowel_centered_f0": self._cfg("enable_vowel_centered_f0", True),
            "enable_portamento": self._cfg("enable_portamento", True),
            "portamento_ms": self._cfg("portamento_ms", 60),
            "enable_vibrato": self._cfg("enable_vibrato", False),
            "enable_de_metal_filter": self._cfg("enable_de_metal_filter", True),
        }

    def run(self) -> dict:
        alignment_path = path_from_config(self.config, "note_lyric_alignment")
        melotts_report_path = self._melotts_report_path()
        flat_vocal = self._flat_vocal_path()
        output_path = path_from_config(self.config, "svs_vocal")
        if not alignment_path.exists():
            raise FileNotFoundError("Missing note_lyric_alignment.json. Run first: python app.py --step alignment --target-language en")
        if not melotts_report_path.exists():
            raise FileNotFoundError("Missing melotts_render_report.json. Run first: python app.py --step svs --target-language en")
        librosa, np, pw, sf = self._load_dependencies()
        sample_rate = self._sample_rate()
        normalize_peak = float(self._cfg("normalize_peak", 0.95) or 0.95)
        alignment_data = read_json(alignment_path, {"phrases": []})
        melotts_report = read_json(melotts_report_path, {"phrases": []})
        alignment_phrases = alignment_data.get("phrases", [])
        report_phrases = melotts_report.get("phrases", [])
        alignment_by_id = self._phrase_by_id(alignment_phrases)
        if not report_phrases:
            raise RuntimeError("melotts_render_report.json contains no phrases.")
        if not alignment_phrases:
            raise RuntimeError("note_lyric_alignment.json contains no phrases.")
        final_duration = max(float(item.get("start", 0.0)) + float(item.get("duration", 0.0)) for item in report_phrases)
        final = np.zeros(int((final_duration + 0.5) * sample_rate), dtype=np.float64)
        world_phrases = []
        dtw_phrases = []
        for index, phrase_report in enumerate(report_phrases):
            phrase_id = phrase_report.get("id", index + 1)
            phrase = self._matched_phrase(alignment_by_id, alignment_phrases, phrase_id, index)
            aligned_path = self._resolve_report_path(phrase_report.get("aligned_path") or "")
            error = ""
            item_reports = []
            debug_path = ""
            stats = {"mean_original_f0": 0.0, "mean_target_f0": 0.0, "max_f0_jump_before": 0.0, "max_f0_jump_after": 0.0}
            fallback = False
            try:
                if not aligned_path.exists():
                    raise FileNotFoundError(f"Missing aligned phrase wav: {aligned_path}")
                try:
                    phrase_number = int(phrase_id)
                except (TypeError, ValueError):
                    phrase_number = index + 1
                phrase_wav, target_f0, item_reports, stats, debug_path = self._synthesize_phrase(aligned_path, phrase, phrase_report, phrase_number, sample_rate, librosa, np, pw)
            except Exception as exc:
                error = str(exc)
                fallback = True
                if aligned_path.exists():
                    phrase_wav, _ = librosa.load(aligned_path, sr=sample_rate, mono=True)
                    phrase_wav = np.asarray(phrase_wav, dtype=np.float64)
                else:
                    fallback_len = max(1, int(float(phrase_report.get("duration", 0.0)) * sample_rate))
                    phrase_wav = np.zeros(fallback_len, dtype=np.float64)
                target_f0 = np.zeros(0, dtype=np.float64)
            start_sample = max(0, int(float(phrase_report.get("start", 0.0)) * sample_rate))
            end_sample = min(len(final), start_sample + len(phrase_wav))
            if end_sample > start_sample:
                final[start_sample:end_sample] += phrase_wav[: end_sample - start_sample]
            dtw_phrases.append({"id": phrase_id, "text": phrase_report.get("text", phrase.get("text", "")), "items": item_reports})
            world_phrases.append({
                "id": phrase_id,
                "text": phrase_report.get("text", phrase.get("text", "")),
                "aligned_path": str(aligned_path),
                "item_count": len(item_reports),
                "voiced_frame_count": int(np.sum(target_f0 > 0)) if len(target_f0) else 0,
                "mean_original_f0": stats["mean_original_f0"],
                "mean_target_f0": stats["mean_target_f0"],
                "max_f0_jump_before": stats["max_f0_jump_before"],
                "max_f0_jump_after": stats["max_f0_jump_after"],
                "debug_f0_path": debug_path,
                "fallback": fallback,
                "error": error,
            })
        peak = float(np.max(np.abs(final))) if final.size else 0.0
        if peak > normalize_peak > 0:
            final = final / peak * normalize_peak
        ensure_parent(output_path)
        sf.write(str(output_path), final.astype(np.float32), sample_rate)
        dtw_report_path = self._dtw_alignment_report_path()
        write_json(dtw_report_path, {"phrases": dtw_phrases}, self.config)
        world_report_path = self._world_report_path()
        write_json(world_report_path, {
            "backend": "MeloTTS + NoteLyricAligner + local DTW + WORLD",
            "input_flat_vocal": str(flat_vocal),
            "output": str(output_path),
            "phrase_count": len(world_phrases),
            "sample_rate": sample_rate,
            "frame_period": self._frame_period(),
            "metal_reduction": self._metal_reduction_report(),
            "phrases": world_phrases,
            "warning": self.WARNING,
        }, self.config)
        return {
            "status": "success",
            "outputs": {"svs_vocal": str(output_path), "dtw_syllable_alignment": str(dtw_report_path), "aligned_dtw_world_report": str(world_report_path)},
            "message": self.WARNING,
        }