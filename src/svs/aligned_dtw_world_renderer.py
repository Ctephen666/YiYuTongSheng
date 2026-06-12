from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, path_from_config, project_root, resolve_path
from src.common.json_utils import read_json, write_json


def midi_to_hz(midi: int | float | None) -> float:
    if midi is None:
        return 0.0
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


class AlignedDTWWorldRenderer:
    """Use NoteLyricAligner syllable slots, local DTW, and WORLD F0 replacement."""

    WARNING = "This is syllable-slot DTW plus MIDI F0 replacement. It is still not full SVS."

    def __init__(self, config: dict):
        self.config = config

    def _svs_config(self) -> dict:
        return self.config.get("svs", {})

    def _sample_rate(self) -> int:
        return int(self.config.get("project", {}).get("sample_rate", 44100))

    def _frame_period(self) -> float:
        return float(self._svs_config().get("world_frame_period", 5.0) or 5.0)

    def _flat_vocal_path(self) -> Path:
        paths = self.config.get("paths", {})
        if paths.get("svs_vocal_flat"):
            return path_from_config(self.config, "svs_vocal_flat")
        if self._svs_config().get("flat_vocal"):
            return resolve_path(self.config, self._svs_config()["flat_vocal"])
        return project_root(self.config) / "data" / "svs" / "target_language_vocal_flat.wav"

    def _melotts_report_path(self) -> Path:
        return self._flat_vocal_path().parent / "melotts_render_report.json"

    def _dtw_alignment_report_path(self) -> Path:
        if self.config.get("paths", {}).get("dtw_syllable_alignment"):
            return path_from_config(self.config, "dtw_syllable_alignment")
        return project_root(self.config) / "data" / "alignment" / "dtw_syllable_alignment.json"

    def _world_report_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "aligned_dtw_world_report.json"

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
            return float(note.get("midi")) + float(self._svs_config().get("midi_transpose", 0) or 0)
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
        fade_ms = float(self._svs_config().get("phrase_fade_ms", 20) or 0)
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
        fps = float(self._svs_config().get("dtw_energy_frames_per_second", 200) or 200)
        target_frame_count = max(4, int(slot_duration * fps))

        if len(item_wav) < hop_length or slot_duration <= 0:
            return {
                "source_to_target_time": self._linear_mapping(1, max(slot_duration, 0.0), np),
                "source_frame_count": 1,
                "target_frame_count": target_frame_count,
                "dtw_cost": None,
                "used_fallback": True,
            }

        src_rms = librosa.feature.rms(y=item_wav, frame_length=1024, hop_length=hop_length)[0]
        src = self._normalize_feature(np.log1p(src_rms), np)
        source_frame_count = int(len(src))
        target_energy = self._target_energy_template(notes, slot_start_global, slot_duration, target_frame_count, np)
        target = self._normalize_feature(target_energy, np)

        if source_frame_count < 2 or target_frame_count < 2:
            return {
                "source_to_target_time": self._linear_mapping(max(source_frame_count, 1), slot_duration, np),
                "source_frame_count": max(source_frame_count, 1),
                "target_frame_count": target_frame_count,
                "dtw_cost": None,
                "used_fallback": True,
            }

        try:
            D, wp = librosa.sequence.dtw(
                X=src.reshape(1, -1),
                Y=target.reshape(1, -1),
                metric="euclidean",
                global_constraints=True,
                band_rad=float(self._svs_config().get("dtw_band_rad", 0.25) or 0.25),
            )
            wp = wp[::-1]
            grouped: dict[int, list[int]] = {}
            for source_index, target_index in wp:
                source_i = int(source_index)
                target_i = int(target_index)
                if 0 <= source_i < source_frame_count:
                    grouped.setdefault(source_i, []).append(target_i)

            source_to_target_time = self._linear_mapping(source_frame_count, slot_duration, np)
            for source_i, target_indices in grouped.items():
                target_mean = float(np.mean(target_indices))
                if target_frame_count > 1:
                    source_to_target_time[source_i] = target_mean / float(target_frame_count - 1) * slot_duration
                else:
                    source_to_target_time[source_i] = 0.0

            return {
                "source_to_target_time": source_to_target_time,
                "source_frame_count": source_frame_count,
                "target_frame_count": target_frame_count,
                "dtw_cost": float(D[-1, -1]),
                "used_fallback": False,
            }
        except Exception:
            return {
                "source_to_target_time": self._linear_mapping(source_frame_count, slot_duration, np),
                "source_frame_count": source_frame_count,
                "target_frame_count": target_frame_count,
                "dtw_cost": None,
                "used_fallback": True,
            }

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
            center = (start + end) / 2.0
            distance = abs(center - target_time_global)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_midi = midi

        return nearest_midi

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
        if not bool(self._svs_config().get("enable_vibrato", True)):
            return target_f0

        rate = float(self._svs_config().get("vibrato_rate", 5.5) or 5.5)
        depth = float(self._svs_config().get("vibrato_depth", 0.006) or 0.006)
        min_duration = float(self._svs_config().get("min_vibrato_note_duration", 0.45) or 0.45)

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

        slot_start_global = min(self._note_start(note) for note in notes)
        slot_end_global = max(self._note_end(note) for note in notes)
        slot_start_local = max(0.0, slot_start_global - phrase_start)
        slot_end_local = min(phrase_duration, slot_end_global - phrase_start)
        if slot_end_local <= slot_start_local:
            return None
        return slot_start_global, slot_end_global, slot_start_local, slot_end_local, notes

    def _build_target_f0(self, wav, f0, frame_times, phrase: dict, phrase_report: dict, sample_rate: int, librosa, np):
        phrase_start = float(phrase_report.get("start", 0.0))
        phrase_duration = float(phrase_report.get("duration", len(wav) / sample_rate if sample_rate else 0.0))
        items = phrase.get("items", []) if isinstance(phrase.get("items", []), list) else []
        target_f0 = np.zeros_like(f0, dtype=np.float64)
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
            item_wav = wav[start_sample:end_sample]
            any_slot = True

            dtw_result = self._run_energy_dtw(
                item_wav=item_wav,
                notes=notes,
                slot_start_global=slot_start_global,
                slot_duration=slot_duration,
                sample_rate=sample_rate,
                librosa=librosa,
                np=np,
            )
            source_to_target_time = dtw_result["source_to_target_time"]
            source_frame_count = int(dtw_result["source_frame_count"])
            rms_hop_seconds = 256.0 / float(sample_rate)
            slot_mask = (frame_times >= slot_start_local) & (frame_times < slot_end_local)
            slot_indices = np.where(slot_mask)[0]

            for frame_index in slot_indices:
                if f0[frame_index] <= 0:
                    target_f0[frame_index] = 0.0
                    continue

                source_frame_index = int(round((float(frame_times[frame_index]) - slot_start_local) / rms_hop_seconds))
                if 0 <= source_frame_index < source_frame_count:
                    target_time = float(source_to_target_time[source_frame_index])
                else:
                    target_time = (float(frame_times[frame_index]) - slot_start_local) / max(slot_duration, 1e-8) * slot_duration

                target_time = min(max(target_time, 0.0), slot_duration)
                target_time_global = slot_start_global + target_time
                target_midi = self._pitch_at_time(notes, target_time_global)
                target_f0[frame_index] = midi_to_hz(target_midi)

            item_reports.append({
                "unit": item.get("unit", ""),
                "word": item.get("word", ""),
                "slot_start": slot_start_global,
                "slot_end": slot_end_global,
                "slot_start_local": slot_start_local,
                "slot_end_local": slot_end_local,
                "slot_duration": slot_duration,
                "note_count": len(notes),
                "source_frame_count": dtw_result["source_frame_count"],
                "target_frame_count": dtw_result["target_frame_count"],
                "dtw_cost": dtw_result["dtw_cost"],
                "used_fallback": dtw_result["used_fallback"],
                "notes": notes,
            })

        phrase_notes = self._all_item_notes(items)
        if not any_slot or not phrase_notes:
            raise RuntimeError("Phrase has no usable alignment items with MIDI notes.")

        missing_voiced = (f0 > 0) & (target_f0 <= 0)
        if bool(np.any(missing_voiced)):
            for frame_index in np.where(missing_voiced)[0]:
                target_midi = self._pitch_at_time(phrase_notes, phrase_start + float(frame_times[frame_index]))
                target_f0[frame_index] = midi_to_hz(target_midi)

        target_f0 = self._smooth_voiced_f0(target_f0, int(self._svs_config().get("f0_smooth_window", 5) or 5), np)
        target_f0 = self._apply_vibrato(target_f0, items, phrase_start, frame_times, np)
        return target_f0, item_reports

    def _synthesize_phrase(self, aligned_path: Path, phrase: dict, phrase_report: dict, sample_rate: int, librosa, np, pw):
        wav, _ = librosa.load(aligned_path, sr=sample_rate, mono=True)
        target_len = len(wav)
        wav64 = np.asarray(wav, dtype=np.float64)
        frame_period = self._frame_period()

        f0, frame_times = pw.harvest(wav64, sample_rate, frame_period=frame_period)
        sp = pw.cheaptrick(wav64, f0, frame_times, sample_rate)
        ap = pw.d4c(wav64, f0, frame_times, sample_rate)

        target_f0, item_reports = self._build_target_f0(wav64, f0, frame_times, phrase, phrase_report, sample_rate, librosa, np)
        synth = pw.synthesize(target_f0.astype(np.float64), sp, ap, sample_rate, frame_period=frame_period)
        synth = self._fit_length(synth, target_len, np)
        synth = self._fade(synth, sample_rate, np)
        return synth, target_f0, item_reports

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
        normalize_peak = float(self._svs_config().get("normalize_peak", 0.98) or 0.98)

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

            try:
                if not aligned_path.exists():
                    raise FileNotFoundError(f"Missing aligned phrase wav: {aligned_path}")
                phrase_wav, target_f0, item_reports = self._synthesize_phrase(
                    aligned_path,
                    phrase,
                    phrase_report,
                    sample_rate,
                    librosa,
                    np,
                    pw,
                )
            except Exception as exc:
                error = str(exc)
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

            dtw_phrases.append({
                "id": phrase_id,
                "text": phrase_report.get("text", phrase.get("text", "")),
                "items": item_reports,
            })
            world_phrases.append({
                "id": phrase_id,
                "text": phrase_report.get("text", phrase.get("text", "")),
                "aligned_path": str(aligned_path),
                "item_count": len(item_reports),
                "voiced_frame_count": int(np.sum(target_f0 > 0)) if len(target_f0) else 0,
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
        write_json(
            world_report_path,
            {
                "backend": "MeloTTS + NoteLyricAligner + local DTW + WORLD",
                "input_flat_vocal": str(flat_vocal),
                "output": str(output_path),
                "phrase_count": len(world_phrases),
                "sample_rate": sample_rate,
                "frame_period": self._frame_period(),
                "phrases": world_phrases,
                "warning": self.WARNING,
            },
            self.config,
        )

        return {
            "status": "success",
            "outputs": {
                "svs_vocal": str(output_path),
                "dtw_syllable_alignment": str(dtw_report_path),
                "aligned_dtw_world_report": str(world_report_path),
            },
            "message": self.WARNING,
        }