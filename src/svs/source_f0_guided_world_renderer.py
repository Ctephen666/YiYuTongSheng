from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, path_from_config, project_root, resolve_path, write_text
from src.common.json_utils import read_json, write_json
from src.svs.source_f0 import (
    build_midi_f0_from_melody,
    extract_source_f0,
    midi_to_hz,
    refine_source_f0_with_midi,
    write_refined_f0_csv,
    write_source_f0_csv,
)


class SourceF0GuidedWorldRenderer:
    """Render English vocals with source-singer F0 details constrained by MIDI and alignment slots."""

    WARNING = "This is source-F0-guided WORLD resynthesis. It aims to preserve original singing pitch details but is still not full SVS."

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

    def _path(self, key: str, default: str) -> Path:
        if self.config.get("paths", {}).get(key):
            return path_from_config(self.config, key)
        return project_root(self.config) / default

    def _vocals_path(self) -> Path:
        return self._path("vocals", "data/stems/vocals.wav")

    def _source_f0_csv_path(self) -> Path:
        return self._path("source_f0_csv", "data/score/source_f0.csv")

    def _source_f0_refined_csv_path(self) -> Path:
        return self._path("source_f0_refined_csv", "data/score/source_f0_refined.csv")

    def _source_output_path(self) -> Path:
        return self._path("svs_vocal_source_f0", "data/svs/target_language_vocal_source_f0.wav")

    def _default_output_path(self) -> Path:
        return path_from_config(self.config, "svs_vocal")

    def _flat_vocal_path(self) -> Path:
        if self.config.get("paths", {}).get("svs_vocal_flat"):
            return path_from_config(self.config, "svs_vocal_flat")
        if self._cfg("flat_vocal", None):
            return resolve_path(self.config, self._cfg("flat_vocal", "data/svs/target_language_vocal_flat.wav"))
        return project_root(self.config) / "data" / "svs" / "target_language_vocal_flat.wav"

    def _melotts_report_path(self) -> Path:
        return self._flat_vocal_path().parent / "melotts_render_report.json"

    def _report_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "source_f0_guided_report.json"

    def _debug_dir(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "debug_source_f0"

    def _load_dependencies(self):
        try:
            import librosa
            import numpy as np
            import pyworld as pw
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyworld/librosa/numpy/soundfile dependencies are not installed. Install with: pip install pyworld librosa soundfile numpy scipy") from exc
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
            return float(note.get("midi")) + float(self._cfg("midi_transpose", 0) or 0)
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
    def _extract_and_refine_source_f0(self, melody_data: dict):
        vocals_path = self._vocals_path()
        if not vocals_path.exists():
            configured = self.config.get("paths", {}).get("vocals", "data/stems/vocals.wav")
            raise FileNotFoundError(f"Missing source vocals for source-F0-guided rendering: {configured}")
        times, source_f0 = extract_source_f0(
            vocals_path,
            sample_rate=self._sample_rate(),
            frame_period=self._frame_period(),
            method=str(self._cfg("source_f0_method", "harvest") or "harvest"),
        )
        source_csv = self._source_f0_csv_path()
        write_source_f0_csv(source_csv, times, source_f0)
        midi_f0 = build_midi_f0_from_melody(times, melody_data, midi_transpose=float(self._cfg("midi_transpose", 0) or 0))
        refined_f0, delta, delta_clamped = refine_source_f0_with_midi(
            source_f0,
            midi_f0,
            max_source_delta_semitones=float(self._cfg("max_source_delta_semitones", 2.5) or 2.5),
            source_f0_smooth_window=int(self._cfg("source_f0_smooth_window", 5) or 5),
        )
        refined_csv = self._source_f0_refined_csv_path()
        write_refined_f0_csv(refined_csv, times, source_f0, midi_f0, delta, delta_clamped, refined_f0)
        return times, source_f0, midi_f0, refined_f0, source_csv, refined_csv

    def _valid_original_f0(self, value: float) -> bool:
        if value <= 0:
            return False
        return float(self._cfg("min_voiced_f0", 70.0)) <= value <= float(self._cfg("max_voiced_f0", 900.0))

    def _interp_curve(self, times, values, query_times, np):
        if len(times) == 0:
            return np.zeros_like(query_times, dtype=np.float64)
        return np.interp(query_times, times, values, left=0.0, right=0.0)

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

    def _apply_portamento(self, f0, frame_times, items: list[dict], phrase_start: float, np):
        if not bool(self._cfg("enable_portamento", True)):
            return f0
        half_window = float(self._cfg("portamento_ms", 30) or 30) / 1000.0 / 2.0
        if half_window <= 0:
            return f0
        smoothed = f0.copy()
        for item in items:
            notes = [note for note in item.get("notes", []) if isinstance(note, dict) and self._note_midi(note) is not None]
            notes = sorted(notes, key=self._note_start)
            for previous, current in zip(notes, notes[1:]):
                boundary = self._note_start(current) - phrase_start
                mask = (frame_times >= boundary - half_window) & (frame_times <= boundary + half_window) & (smoothed > 0)
                if not bool(np.any(mask)):
                    continue
                first = smoothed[mask][0]
                last = smoothed[mask][-1]
                if first <= 0 or last <= 0:
                    continue
                x = (frame_times[mask] - (boundary - half_window)) / max(2.0 * half_window, 1e-8)
                alpha = 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))
                smoothed[mask] = first * (1.0 - alpha) + last * alpha
        return smoothed

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

    def _postprocess_wave(self, wav, sample_rate: int, np):
        processed = np.asarray(wav, dtype=np.float64)
        if bool(self._cfg("enable_de_metal_filter", True)):
            try:
                from scipy import signal
                cutoff = float(self._cfg("de_metal_lowpass_hz", 10000) or 10000)
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

    def _vowel_region(self, item: dict, slot_start_global: float, slot_end_global: float) -> tuple[float, float]:
        try:
            return float(item["recommended_vowel_start"]), float(item["recommended_vowel_end"])
        except (KeyError, TypeError, ValueError):
            ratio = float(self._cfg("vowel_center_ratio", 0.65) or 0.65)
            duration = max(0.0, slot_end_global - slot_start_global)
            pad = (1.0 - min(max(ratio, 0.1), 0.95)) * duration / 2.0
            return slot_start_global + pad, slot_end_global - pad

    def _midi_f0_at_time(self, notes: list[dict], time_global: float) -> float:
        nearest_midi = None
        nearest_distance = None
        for note in notes:
            midi = self._note_midi(note)
            if midi is None:
                continue
            start = self._note_start(note)
            end = self._note_end(note)
            if start <= time_global < end:
                return midi_to_hz(midi)
            distance = abs(((start + end) / 2.0) - time_global)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_midi = midi
        return midi_to_hz(nearest_midi)
    def _mean_abs_semitone_error(self, target_f0, reference_f0, np) -> float:
        mask = (target_f0 > 0) & (reference_f0 > 0)
        if not bool(np.any(mask)):
            return 0.0
        return float(np.mean(np.abs(12.0 * np.log2(target_f0[mask] / reference_f0[mask]))))

    def _build_target_f0(self, wav, original_f0, frame_times, phrase: dict, phrase_report: dict, source_times, source_raw, source_refined, sample_rate: int, np):
        phrase_start = float(phrase_report.get("start", 0.0))
        phrase_duration = float(phrase_report.get("duration", len(wav) / sample_rate if sample_rate else 0.0))
        items = phrase.get("items", []) if isinstance(phrase.get("items", []), list) else []
        target_f0 = np.zeros_like(original_f0, dtype=np.float64)
        mapped_source_f0 = np.zeros_like(original_f0, dtype=np.float64)
        mapped_source_raw = np.zeros_like(original_f0, dtype=np.float64)
        midi_f0_curve = np.zeros_like(original_f0, dtype=np.float64)
        source_strength_curve = np.zeros_like(original_f0, dtype=np.float64)
        vowel_region_curve = np.zeros_like(original_f0, dtype=bool)
        item_reports = []
        any_item = False
        midi_anchor_strength = float(self._cfg("midi_anchor_strength", 0.35) or 0.35)
        flat_f0_blend_strength = float(self._cfg("flat_f0_blend_strength", 0.10) or 0.10)
        vowel_source_strength = float(self._cfg("vowel_source_strength", 0.90) or 0.90)
        consonant_source_strength = float(self._cfg("consonant_source_strength", 0.20) or 0.20)

        for item in items:
            slot = self._item_slot(item, phrase_start, phrase_duration)
            if slot is None:
                continue
            slot_start_global, slot_end_global, slot_start_local, slot_end_local, notes = slot
            slot_duration = slot_end_local - slot_start_local
            vowel_start, vowel_end = self._vowel_region(item, slot_start_global, slot_end_global)
            slot_indices = np.where((frame_times >= slot_start_local) & (frame_times < slot_end_local))[0]
            any_item = True
            for frame_index in slot_indices:
                original = float(original_f0[frame_index])
                if not self._valid_original_f0(original):
                    target_f0[frame_index] = 0.0
                    continue
                local_time = float(frame_times[frame_index])
                relative = (local_time - slot_start_local) / max(slot_duration, 1e-8)
                relative = min(max(relative, 0.0), 1.0)
                source_time_global = slot_start_global + relative * max(slot_end_global - slot_start_global, 1e-8)
                source_value = float(self._interp_curve(source_times, source_refined, np.asarray([source_time_global], dtype=np.float64), np)[0])
                source_raw_value = float(self._interp_curve(source_times, source_raw, np.asarray([source_time_global], dtype=np.float64), np)[0])
                midi_value = self._midi_f0_at_time(notes, source_time_global)
                in_vowel = vowel_start <= source_time_global <= vowel_end
                source_strength = vowel_source_strength if in_vowel else consonant_source_strength
                if source_value <= 0:
                    source_strength = 0.0
                    source_value = midi_value
                midi_weight = midi_anchor_strength if midi_value > 0 else 0.0
                flat_weight = flat_f0_blend_strength if original > 0 else 0.0
                weight_sum = source_strength + midi_weight + flat_weight
                if weight_sum <= 0:
                    target_f0[frame_index] = 0.0
                else:
                    target_f0[frame_index] = (source_value * source_strength + midi_value * midi_weight + original * flat_weight) / weight_sum
                mapped_source_f0[frame_index] = source_value
                mapped_source_raw[frame_index] = source_raw_value
                midi_f0_curve[frame_index] = midi_value
                source_strength_curve[frame_index] = source_strength
                vowel_region_curve[frame_index] = in_vowel
            item_reports.append({
                "unit": item.get("unit", ""),
                "word": item.get("word", ""),
                "slot_start": slot_start_global,
                "slot_end": slot_end_global,
                "slot_duration": slot_duration,
                "recommended_vowel_start": vowel_start,
                "recommended_vowel_end": vowel_end,
                "note_count": len(notes),
            })

        if not any_item:
            raise RuntimeError("Phrase has no usable alignment items with MIDI notes.")
        target_f0 = self._apply_portamento(target_f0, frame_times, items, phrase_start, np)
        target_f0 = self._limit_f0_jumps(target_f0, float(self._cfg("max_f0_jump_semitones", 10.0) or 10.0), np)
        target_f0 = self._smooth_voiced_f0(target_f0, int(self._cfg("f0_smooth_window", 5) or 5), np)
        target_f0 = self._limit_f0_jumps(target_f0, float(self._cfg("max_f0_jump_semitones", 10.0) or 10.0), np)
        stats = {
            "mean_midi_f0": float(np.mean(midi_f0_curve[midi_f0_curve > 0])) if bool(np.any(midi_f0_curve > 0)) else 0.0,
            "mean_source_f0": float(np.mean(mapped_source_f0[mapped_source_f0 > 0])) if bool(np.any(mapped_source_f0 > 0)) else 0.0,
            "mean_target_f0": float(np.mean(target_f0[target_f0 > 0])) if bool(np.any(target_f0 > 0)) else 0.0,
            "mean_semitone_error_to_source": self._mean_abs_semitone_error(target_f0, mapped_source_f0, np),
            "mean_semitone_error_to_midi": self._mean_abs_semitone_error(target_f0, midi_f0_curve, np),
        }
        debug = {
            "mapped_source_f0": mapped_source_f0,
            "mapped_source_raw": mapped_source_raw,
            "midi_f0": midi_f0_curve,
            "source_strength": source_strength_curve,
            "is_vowel_region": vowel_region_curve,
        }
        return target_f0, item_reports, stats, debug

    def _write_debug_csv(self, phrase_number: int, frame_times, original_f0, debug: dict, target_f0) -> str:
        if phrase_number > 5:
            return ""
        path = self._debug_dir() / f"phrase_{phrase_number:03d}_source_f0.csv"
        rows = [
            "time,original_flat_f0,midi_f0,source_f0_mapped,source_f0_refined,target_f0,is_vowel_region,source_strength,midi_anchor_strength,flat_f0_blend_strength"
        ]
        midi_anchor_strength = float(self._cfg("midi_anchor_strength", 0.35) or 0.35)
        flat_f0_blend_strength = float(self._cfg("flat_f0_blend_strength", 0.10) or 0.10)
        mapped_source = debug["mapped_source_f0"]
        midi_f0 = debug["midi_f0"]
        source_strength = debug["source_strength"]
        is_vowel = debug["is_vowel_region"]
        for values in zip(frame_times, original_f0, midi_f0, mapped_source, mapped_source, target_f0, is_vowel, source_strength):
            time, original, midi, mapped, refined, target, vowel, strength = values
            rows.append(
                f"{float(time):.6f},{float(original):.6f},{float(midi):.6f},{float(mapped):.6f},{float(refined):.6f},"
                f"{float(target):.6f},{int(bool(vowel))},{float(strength):.6f},{midi_anchor_strength:.6f},{flat_f0_blend_strength:.6f}"
            )
        write_text(path, "\n".join(rows) + "\n", self.config)
        return str(path)

    def _synthesize_phrase(self, aligned_path: Path, phrase: dict, phrase_report: dict, phrase_number: int, source_times, source_raw, source_refined, sample_rate: int, librosa, np, pw):
        wav, _ = librosa.load(aligned_path, sr=sample_rate, mono=True)
        target_len = len(wav)
        wav64 = np.asarray(wav, dtype=np.float64)
        f0, frame_times = pw.harvest(wav64, sample_rate, frame_period=self._frame_period())
        sp = pw.cheaptrick(wav64, f0, frame_times, sample_rate)
        ap = pw.d4c(wav64, f0, frame_times, sample_rate)
        target_f0, item_reports, stats, debug = self._build_target_f0(
            wav64,
            f0,
            frame_times,
            phrase,
            phrase_report,
            source_times,
            source_raw,
            source_refined,
            sample_rate,
            np,
        )
        synth = pw.synthesize(target_f0.astype(np.float64), sp, ap, sample_rate, frame_period=self._frame_period())
        synth = self._fit_length(synth, target_len, np)
        synth = self._postprocess_wave(synth, sample_rate, np)
        synth = self._fade(synth, sample_rate, np)
        debug_path = self._write_debug_csv(phrase_number, frame_times, f0, debug, target_f0)
        return synth, target_f0, item_reports, stats, debug_path
    def _settings_report(self) -> dict:
        return {
            "source_f0_detail_strength": self._cfg("source_f0_detail_strength", 0.65),
            "midi_anchor_strength": self._cfg("midi_anchor_strength", 0.35),
            "max_source_delta_semitones": self._cfg("max_source_delta_semitones", 2.5),
            "vowel_source_strength": self._cfg("vowel_source_strength", 0.90),
            "consonant_source_strength": self._cfg("consonant_source_strength", 0.20),
            "flat_f0_blend_strength": self._cfg("flat_f0_blend_strength", 0.10),
            "output_source_f0_as_default": self._cfg("output_source_f0_as_default", True),
        }

    def run(self) -> dict:
        if not bool(self._cfg("use_source_f0_guided", True)):
            return {
                "status": "skipped",
                "outputs": {},
                "message": "Source-F0-guided WORLD rendering is disabled by svs.use_source_f0_guided=false.",
            }

        alignment_path = path_from_config(self.config, "note_lyric_alignment")
        melody_path = path_from_config(self.config, "melody_notes")
        melotts_report_path = self._melotts_report_path()
        output_path = self._source_output_path()
        default_output_path = self._default_output_path()
        if not alignment_path.exists():
            raise FileNotFoundError("Missing note_lyric_alignment.json. Run first: python app.py --step alignment --target-language en")
        if not melody_path.exists():
            raise FileNotFoundError("Missing melody_notes.json. Run first: python app.py --step melody --target-language en")
        if not melotts_report_path.exists():
            raise FileNotFoundError("Missing melotts_render_report.json. Run first: python app.py --step svs --target-language en")

        librosa, np, pw, sf = self._load_dependencies()
        sample_rate = self._sample_rate()
        normalize_peak = float(self._cfg("normalize_peak", 0.95) or 0.95)
        melody_data = read_json(melody_path, {"phrases": []})
        alignment_data = read_json(alignment_path, {"phrases": []})
        melotts_report = read_json(melotts_report_path, {"phrases": []})
        alignment_phrases = alignment_data.get("phrases", [])
        report_phrases = melotts_report.get("phrases", [])
        if not alignment_phrases:
            raise RuntimeError("note_lyric_alignment.json contains no phrases.")
        if not report_phrases:
            raise RuntimeError("melotts_render_report.json contains no phrases.")

        source_times, source_raw, _source_midi, source_refined, source_csv, refined_csv = self._extract_and_refine_source_f0(melody_data)
        alignment_by_id = self._phrase_by_id(alignment_phrases)
        final_duration = max(float(item.get("start", 0.0)) + float(item.get("duration", 0.0)) for item in report_phrases)
        final = np.zeros(int((final_duration + 0.5) * sample_rate), dtype=np.float64)
        phrase_reports = []

        for index, phrase_report in enumerate(report_phrases):
            phrase_id = phrase_report.get("id", index + 1)
            phrase = self._matched_phrase(alignment_by_id, alignment_phrases, phrase_id, index)
            aligned_path = self._resolve_report_path(phrase_report.get("aligned_path") or "")
            error = ""
            fallback = False
            item_reports = []
            debug_path = ""
            stats = {
                "mean_midi_f0": 0.0,
                "mean_source_f0": 0.0,
                "mean_target_f0": 0.0,
                "mean_semitone_error_to_source": 0.0,
                "mean_semitone_error_to_midi": 0.0,
            }
            try:
                if not aligned_path.exists():
                    raise FileNotFoundError(f"Missing aligned phrase wav: {aligned_path}")
                try:
                    phrase_number = int(phrase_id)
                except (TypeError, ValueError):
                    phrase_number = index + 1
                phrase_wav, target_f0, item_reports, stats, debug_path = self._synthesize_phrase(
                    aligned_path,
                    phrase,
                    phrase_report,
                    phrase_number,
                    source_times,
                    source_raw,
                    source_refined,
                    sample_rate,
                    librosa,
                    np,
                    pw,
                )
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
            phrase_reports.append({
                "id": phrase_id,
                "text": phrase_report.get("text", phrase.get("text", "")),
                "start": phrase_report.get("start", 0.0),
                "duration": phrase_report.get("duration", 0.0),
                "aligned_path": str(aligned_path),
                "item_count": len(item_reports),
                "voiced_frame_count": int(np.sum(target_f0 > 0)) if len(target_f0) else 0,
                "mean_midi_f0": stats["mean_midi_f0"],
                "mean_source_f0": stats["mean_source_f0"],
                "mean_target_f0": stats["mean_target_f0"],
                "mean_semitone_error_to_source": stats["mean_semitone_error_to_source"],
                "mean_semitone_error_to_midi": stats["mean_semitone_error_to_midi"],
                "debug_source_f0_path": debug_path,
                "fallback": fallback,
                "error": error,
            })

        peak = float(np.max(np.abs(final))) if final.size else 0.0
        if peak > normalize_peak > 0:
            final = final / peak * normalize_peak
        ensure_parent(output_path)
        sf.write(str(output_path), final.astype(np.float32), sample_rate)
        used_as_default = bool(self._cfg("output_source_f0_as_default", True))
        if used_as_default:
            ensure_parent(default_output_path)
            sf.write(str(default_output_path), final.astype(np.float32), sample_rate)

        report_path = self._report_path()
        write_json(report_path, {
            "backend": "Source F0 Guided WORLD",
            "output": str(output_path),
            "default_output": str(default_output_path) if used_as_default else "",
            "used_as_default": used_as_default,
            "source_f0_csv": str(source_csv),
            "source_f0_refined_csv": str(refined_csv),
            "phrase_count": len(phrase_reports),
            "settings": self._settings_report(),
            "phrases": phrase_reports,
            "warning": self.WARNING,
        }, self.config)
        outputs = {
            "svs_vocal_source_f0": str(output_path),
            "source_f0_guided_report": str(report_path),
            "source_f0_csv": str(source_csv),
            "source_f0_refined_csv": str(refined_csv),
        }
        if used_as_default:
            outputs["svs_vocal"] = str(default_output_path)
        return {"status": "success", "outputs": outputs, "message": self.WARNING}