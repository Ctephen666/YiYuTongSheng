from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, path_from_config, project_root, resolve_path
from src.common.json_utils import read_json, write_json


def midi_to_hz(midi: int | float | None) -> float:
    if midi is None:
        return 0.0
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


class MidiF0WorldRenderer:
    """Replace MeloTTS phrase F0 with an OpenCpop MIDI-derived F0 curve using WORLD."""

    WARNING = "This is WORLD-based F0 replacement using MIDI notes. It improves pitch contour but is still not full SVS."

    def __init__(self, config: dict):
        self.config = config

    def _svs_config(self) -> dict:
        return self.config.get("svs", {})

    def _sample_rate(self) -> int:
        return int(self.config.get("project", {}).get("sample_rate", 44100))

    def _flat_vocal_path(self) -> Path:
        paths = self.config.get("paths", {})
        if paths.get("svs_vocal_flat"):
            return path_from_config(self.config, "svs_vocal_flat")
        if self._svs_config().get("flat_vocal"):
            return resolve_path(self.config, self._svs_config()["flat_vocal"])
        return project_root(self.config) / "data" / "svs" / "target_language_vocal_flat.wav"

    def _report_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "midi_f0_world_report.json"

    def _melotts_report_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "melotts_render_report.json"

    def _load_dependencies(self):
        try:
            import librosa
            import numpy as np
            import pyworld as pw
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyworld/scipy dependencies are not installed. Install with: pip install pyworld scipy") from exc
        return librosa, np, pw, sf

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

    def _smooth_voiced_f0(self, f0, window_size: int, np):
        if window_size <= 1:
            return f0
        smoothed = f0.copy()
        voiced = f0 > 0
        start = None
        kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
        for idx, is_voiced in enumerate(voiced.tolist() + [False]):
            if is_voiced and start is None:
                start = idx
            elif not is_voiced and start is not None:
                end = idx
                if end - start >= window_size:
                    segment = f0[start:end]
                    smoothed[start:end] = np.convolve(segment, kernel, mode="same")
                start = None
        return smoothed

    def _phrase_by_id(self, melody_data: dict) -> dict:
        return {phrase.get("id"): phrase for phrase in melody_data.get("phrases", [])}

    def _note_f0_curve(self, phrase: dict, frame_times, original_f0, np):
        notes = phrase.get("notes", []) if isinstance(phrase, dict) else []
        curve = np.zeros_like(frame_times, dtype=np.float64)
        phrase_start = float(phrase.get("start", 0.0)) if isinstance(phrase, dict) else 0.0
        valid_note_f0 = []

        for note in notes:
            hz = midi_to_hz(note.get("midi"))
            if hz <= 0:
                continue
            local_start = float(note.get("start", 0.0)) - phrase_start
            local_end = float(note.get("end", 0.0)) - phrase_start
            if local_end <= local_start:
                local_end = local_start + float(note.get("duration", 0.0))
            mask = (frame_times >= local_start) & (frame_times < local_end)
            curve[mask] = hz
            valid_note_f0.append(hz)

        if valid_note_f0:
            nearest = float(np.mean(valid_note_f0))
        else:
            voiced_original = original_f0[original_f0 > 0]
            nearest = float(np.mean(voiced_original)) if len(voiced_original) else 0.0

        target_f0 = np.zeros_like(original_f0, dtype=np.float64)
        voiced = original_f0 > 0
        target_f0[voiced] = curve[voiced]
        if nearest > 0:
            target_f0[voiced & (target_f0 <= 0)] = nearest
        return target_f0

    def _apply_vibrato(self, target_f0, phrase: dict, frame_times, np):
        if not bool(self._svs_config().get("enable_vibrato", True)):
            return target_f0

        rate = float(self._svs_config().get("vibrato_rate", 5.5) or 5.5)
        depth = float(self._svs_config().get("vibrato_depth", 0.01) or 0.01)
        min_duration = float(self._svs_config().get("min_vibrato_note_duration", 0.4) or 0.4)
        phrase_start = float(phrase.get("start", 0.0))

        for note in phrase.get("notes", []):
            local_start = float(note.get("start", 0.0)) - phrase_start
            local_end = float(note.get("end", 0.0)) - phrase_start
            if local_end - local_start < min_duration:
                continue
            mask = (frame_times >= local_start) & (frame_times < local_end) & (target_f0 > 0)
            local_t = frame_times[mask] - local_start
            target_f0[mask] *= 1.0 + depth * np.sin(2.0 * np.pi * rate * local_t)
        return target_f0

    def _synthesize_phrase(self, aligned_path: Path, phrase: dict, sample_rate: int, librosa, np, pw):
        wav, _ = librosa.load(aligned_path, sr=sample_rate, mono=True)
        target_len = len(wav)
        wav64 = np.asarray(wav, dtype=np.float64)
        frame_period = float(self._svs_config().get("world_frame_period", 5.0) or 5.0)

        f0, frame_times = pw.harvest(wav64, sample_rate, frame_period=frame_period)
        sp = pw.cheaptrick(wav64, f0, frame_times, sample_rate)
        ap = pw.d4c(wav64, f0, frame_times, sample_rate)

        target_f0 = self._note_f0_curve(phrase, frame_times, f0, np)
        target_f0 = self._smooth_voiced_f0(target_f0, int(self._svs_config().get("f0_smooth_window", 5) or 5), np)
        target_f0 = self._apply_vibrato(target_f0, phrase, frame_times, np)

        synth = pw.synthesize(target_f0.astype(np.float64), sp, ap, sample_rate, frame_period=frame_period)
        synth = self._fit_length(synth, target_len, np)
        synth = self._fade(synth, sample_rate, np)
        return synth, target_f0, f0

    def run(self) -> dict:
        melody_path = path_from_config(self.config, "melody_notes")
        flat_vocal = self._flat_vocal_path()
        melotts_report = self._melotts_report_path()
        output_path = path_from_config(self.config, "svs_vocal")

        if not flat_vocal.exists():
            raise FileNotFoundError("Missing flat vocal. Run first: python app.py --step svs --target-language en")
        if not melotts_report.exists():
            raise FileNotFoundError("Missing melotts_render_report.json. Run first: python app.py --step svs --target-language en")
        if not melody_path.exists():
            raise FileNotFoundError("Missing melody_notes.json. Run first: python app.py --step melody --target-language en")

        librosa, np, pw, sf = self._load_dependencies()
        sample_rate = self._sample_rate()
        normalize_peak = float(self._svs_config().get("normalize_peak", 0.98) or 0.98)
        melody_data = read_json(melody_path, {"phrases": []})
        report = read_json(melotts_report, {"phrases": []})
        melody_by_id = self._phrase_by_id(melody_data)
        report_phrases = report.get("phrases", [])

        if not report_phrases:
            raise RuntimeError("melotts_render_report.json contains no phrases.")

        final_duration = max(float(item.get("start", 0.0)) + float(item.get("duration", 0.0)) for item in report_phrases)
        final = np.zeros(int((final_duration + 0.5) * sample_rate), dtype=np.float64)
        output_phrases = []

        for index, item in enumerate(report_phrases):
            phrase_id = item.get("id", index + 1)
            phrase = melody_by_id.get(phrase_id)
            if phrase is None and str(phrase_id).isdigit():
                phrase = melody_by_id.get(int(phrase_id))
            phrase = phrase or {"id": phrase_id, "start": item.get("start", 0.0), "duration": item.get("duration", 0.0), "notes": []}
            aligned_path = Path(item.get("aligned_path") or "")
            if not aligned_path.is_absolute():
                aligned_path = project_root(self.config) / aligned_path

            error = ""
            try:
                if not aligned_path.exists():
                    raise FileNotFoundError(f"Missing aligned phrase wav: {aligned_path}")
                if not phrase.get("notes"):
                    raise RuntimeError("Phrase has no MIDI notes; using aligned phrase fallback.")
                phrase_wav, target_f0, original_f0 = self._synthesize_phrase(aligned_path, phrase, sample_rate, librosa, np, pw)
            except Exception as exc:
                error = str(exc)
                if aligned_path.exists():
                    phrase_wav, _ = librosa.load(aligned_path, sr=sample_rate, mono=True)
                    phrase_wav = np.asarray(phrase_wav, dtype=np.float64)
                else:
                    fallback_len = max(1, int(float(item.get("duration", 0.0)) * sample_rate))
                    phrase_wav = np.zeros(fallback_len, dtype=np.float64)
                target_f0 = np.zeros(0, dtype=np.float64)
                original_f0 = np.zeros(0, dtype=np.float64)

            start_sample = max(0, int(float(item.get("start", 0.0)) * sample_rate))
            end_sample = min(len(final), start_sample + len(phrase_wav))
            if end_sample > start_sample:
                final[start_sample:end_sample] += phrase_wav[: end_sample - start_sample]

            notes = phrase.get("notes", [])
            midi_values = [int(note.get("midi")) for note in notes if note.get("midi") is not None]
            output_phrases.append({
                "id": phrase_id,
                "text": item.get("text", ""),
                "start": item.get("start", 0.0),
                "duration": item.get("duration", 0.0),
                "note_count": len(notes),
                "voiced_frame_count": int(np.sum(target_f0 > 0)) if len(target_f0) else 0,
                "midi_min": min(midi_values) if midi_values else None,
                "midi_max": max(midi_values) if midi_values else None,
                "aligned_path": str(aligned_path),
                "error": error,
            })

        peak = float(np.max(np.abs(final))) if final.size else 0.0
        if peak > normalize_peak > 0:
            final = final / peak * normalize_peak

        ensure_parent(output_path)
        sf.write(str(output_path), final.astype(np.float32), sample_rate)

        world_report = {
            "backend": "MeloTTS + MIDI F0 + WORLD",
            "input_flat_vocal": str(flat_vocal),
            "output": str(output_path),
            "sample_rate": sample_rate,
            "frame_period": float(self._svs_config().get("world_frame_period", 5.0) or 5.0),
            "phrase_count": len(output_phrases),
            "phrases": output_phrases,
            "warning": self.WARNING,
        }
        report_path = self._report_path()
        write_json(report_path, world_report, self.config)

        return {
            "status": "success",
            "outputs": {"svs_vocal": str(output_path), "midi_f0_world_report": str(report_path)},
            "message": self.WARNING,
        }
