from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src.common.io_utils import path_from_config, project_root
from src.common.json_utils import read_json, write_json


class SimpleFormantSynthesizer:
    """A basic MIDI-driven singing synthesizer.

    This is a signal-processing baseline for course design.

    Input:
        data/svs/openutau_export_plan.json

    Output:
        data/svs/target_language_vocal.wav

    Core idea:
        MIDI pitch -> F0
        note duration -> waveform duration
        ARPABET vowel -> formant profile
        harmonic source + formant weighting -> simple singing voice
        consonants -> short noise burst
    """

    DEFAULT_SAMPLE_RATE = 44100

    VOWELS = {
        "AA", "AE", "AH", "AO", "AW", "AY",
        "EH", "ER", "EY", "IH", "IY",
        "OW", "OY", "UH", "UW",
    }

    FRICATIVES = {
        "F", "V", "S", "Z", "SH", "ZH", "TH", "DH", "HH",
    }

    PLOSIVES = {
        "P", "B", "T", "D", "K", "G", "CH", "JH",
    }

    FORMANTS = {
        "IY": [(270, 60), (2290, 120), (3010, 180)],
        "IH": [(390, 70), (1990, 130), (2550, 180)],
        "EH": [(530, 80), (1840, 130), (2480, 180)],
        "AE": [(660, 90), (1720, 140), (2410, 180)],
        "AH": [(640, 90), (1190, 120), (2390, 180)],
        "AA": [(730, 100), (1090, 120), (2440, 180)],
        "AO": [(570, 90), (840, 100), (2410, 180)],
        "UH": [(440, 80), (1020, 120), (2240, 180)],
        "UW": [(300, 70), (870, 110), (2240, 180)],
        "ER": [(490, 80), (1350, 120), (1690, 160)],
        "EY": [(390, 70), (2300, 140), (3000, 180)],
        "OW": [(500, 80), (900, 110), (2400, 180)],
        "AY": [(660, 90), (1700, 140), (2500, 180)],
        "AW": [(650, 90), (1200, 130), (2400, 180)],
        "OY": [(570, 90), (1000, 120), (2400, 180)],
    }

    DEFAULT_FORMANTS = [(600, 100), (1200, 140), (2400, 200)]

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Paths and config
    # ------------------------------------------------------------------

    def _plan_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "openutau_export_plan.json"

    def _output_path(self) -> Path:
        return Path(path_from_config(self.config, "svs_vocal"))

    def _analysis_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "simple_synth_report.json"

    def _sample_rate(self) -> int:
        return int(
            self.config.get("audio", {}).get("sample_rate")
            or self.config.get("project", {}).get("sample_rate")
            or self.DEFAULT_SAMPLE_RATE
        )

    def _keep_absolute_timing(self) -> bool:
        return bool(
            self.config.get("svs", {}).get("keep_absolute_timing", True)
        )

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------

    def _float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _strip_stress(self, phone: str) -> str:
        return "".join(ch for ch in str(phone or "").upper() if not ch.isdigit())

    def _midi_to_hz(self, midi: int) -> float:
        return 440.0 * (2.0 ** ((int(midi) - 69) / 12.0))

    def _first_vowel(self, phonemes: list[str]) -> str:
        for phone in phonemes:
            clean = self._strip_stress(phone)
            if clean in self.VOWELS:
                return clean
        return "AH"

    def _has_fricative(self, phonemes: list[str]) -> bool:
        return any(self._strip_stress(p) in self.FRICATIVES for p in phonemes)

    def _has_plosive(self, phonemes: list[str]) -> bool:
        return any(self._strip_stress(p) in self.PLOSIVES for p in phonemes)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _adsr(self, n: int, sr: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        env = np.ones(n, dtype=np.float32)

        attack = min(n // 3, int(0.025 * sr))
        release = min(n // 3, int(0.04 * sr))

        if attack > 1:
            env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)

        if release > 1:
            env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)

        return env

    def _formant_gain(
        self,
        frequency: float,
        formants: list[tuple[float, float]],
    ) -> float:
        gain = 0.04

        for center, bandwidth in formants:
            distance = (frequency - center) / max(1.0, bandwidth)
            gain += math.exp(-0.5 * distance * distance)

        return gain

    def _harmonic_voice(
        self,
        f0: float,
        duration: float,
        phonemes: list[str],
        sr: int,
    ) -> np.ndarray:
        n = max(1, int(round(duration * sr)))
        t = np.arange(n, dtype=np.float32) / float(sr)

        vowel = self._first_vowel(phonemes)
        formants = self.FORMANTS.get(vowel, self.DEFAULT_FORMANTS)

        max_harmonic = int((sr * 0.5) // max(1.0, f0))
        max_harmonic = max(1, min(max_harmonic, 80))

        signal = np.zeros(n, dtype=np.float32)

        for harmonic in range(1, max_harmonic + 1):
            harmonic_freq = f0 * harmonic

            if harmonic_freq >= sr * 0.5:
                break

            spectral_gain = self._formant_gain(harmonic_freq, formants)
            amplitude = spectral_gain / float(harmonic)

            signal += amplitude * np.sin(
                2.0 * math.pi * harmonic_freq * t
            ).astype(np.float32)

        max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0

        if max_abs > 1e-6:
            signal = signal / max_abs

        signal *= self._adsr(n, sr)
        signal *= 0.28

        return signal.astype(np.float32)

    def _consonant_noise(
        self,
        duration: float,
        phonemes: list[str],
        sr: int,
    ) -> np.ndarray:
        n = max(1, int(round(duration * sr)))
        noise = np.zeros(n, dtype=np.float32)

        if not phonemes:
            return noise

        rng = np.random.default_rng(1234)

        if self._has_fricative(phonemes):
            noise_len = min(n, int(0.08 * sr))
            burst = rng.normal(0.0, 1.0, noise_len).astype(np.float32)
            burst *= np.linspace(1.0, 0.0, noise_len, dtype=np.float32)
            noise[:noise_len] += burst * 0.08

        if self._has_plosive(phonemes):
            burst_start = min(n - 1, int(0.015 * sr))
            burst_len = min(n - burst_start, int(0.025 * sr))

            if burst_len > 0:
                burst = rng.normal(0.0, 1.0, burst_len).astype(np.float32)
                burst *= np.linspace(1.0, 0.0, burst_len, dtype=np.float32)
                noise[burst_start:burst_start + burst_len] += burst * 0.12

        return noise

    def _synthesize_event(
        self,
        event: dict,
        previous_phonemes: list[str],
        sr: int,
    ) -> tuple[np.ndarray, list[str]]:
        midi = event.get("midi")

        if midi is None:
            return np.zeros(0, dtype=np.float32), previous_phonemes

        start = self._float(event.get("start", 0.0))
        end = self._float(event.get("end", 0.0))
        duration = max(0.03, end - start)

        phonemes = event.get("phonemes") or previous_phonemes or ["AH"]

        f0 = self._midi_to_hz(int(midi))

        voiced = self._harmonic_voice(
            f0=f0,
            duration=duration,
            phonemes=phonemes,
            sr=sr,
        )

        consonant = self._consonant_noise(
            duration=duration,
            phonemes=phonemes,
            sr=sr,
        )

        signal = voiced + consonant

        if event.get("phonemes"):
            previous_phonemes = event.get("phonemes") or previous_phonemes

        return signal.astype(np.float32), previous_phonemes

    # ------------------------------------------------------------------
    # WAV writing
    # ------------------------------------------------------------------

    def _write_wav(self, path: Path, audio: np.ndarray, sr: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        audio = np.asarray(audio, dtype=np.float32)

        max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0

        if max_abs > 1e-6:
            audio = audio / max_abs * 0.88

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(pcm.tobytes())

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> dict:
        plan_path = self._plan_path()

        if not plan_path.exists():
            raise RuntimeError(
                f"openutau_export_plan.json does not exist: {plan_path}. "
                "Run SVSInputExporter and OpenUtauExporter first."
            )

        plan = read_json(plan_path, {})
        note_events = plan.get("note_events", [])

        if not note_events:
            raise RuntimeError(
                f"openutau_export_plan.json contains no note_events: {plan_path}"
            )

        sr = self._sample_rate()

        starts = [
            self._float(event.get("start", 0.0))
            for event in note_events
        ]

        ends = [
            self._float(event.get("end", 0.0))
            for event in note_events
        ]

        first_start = min(starts)
        last_end = max(ends)

        offset = 0.0 if self._keep_absolute_timing() else first_start

        total_duration = max(1.0, last_end - offset + 1.0)
        total_samples = int(round(total_duration * sr))

        audio = np.zeros(total_samples, dtype=np.float32)

        previous_phonemes: list[str] = ["AH"]
        rendered_events = 0
        skipped_events = 0

        for event in note_events:
            start = self._float(event.get("start", 0.0)) - offset

            if start < 0:
                skipped_events += 1
                continue

            signal, previous_phonemes = self._synthesize_event(
                event=event,
                previous_phonemes=previous_phonemes,
                sr=sr,
            )

            if signal.size == 0:
                skipped_events += 1
                continue

            start_sample = int(round(start * sr))
            end_sample = min(total_samples, start_sample + signal.size)

            if start_sample >= total_samples or end_sample <= start_sample:
                skipped_events += 1
                continue

            chunk_len = end_sample - start_sample
            audio[start_sample:end_sample] += signal[:chunk_len]
            rendered_events += 1

        output_path = self._output_path()
        self._write_wav(output_path, audio, sr)

        report = {
            "source": "openutau_export_plan",
            "format": "simple_formant_synth_report_v1",
            "synthesis_type": "basic_source_filter_formant_synthesis",
            "sample_rate": sr,
            "keep_absolute_timing": self._keep_absolute_timing(),
            "note_event_count": len(note_events),
            "rendered_events": rendered_events,
            "skipped_events": skipped_events,
            "start": first_start,
            "end": last_end,
            "duration": total_duration,
            "output": str(output_path),
            "description": (
                "A basic course-design singing synthesis baseline. "
                "It uses MIDI pitch as F0, note duration as timing, "
                "ARPABET vowels as formant profiles, harmonic excitation as source, "
                "and short noise bursts for consonants."
            ),
        }

        write_json(self._analysis_path(), report, self.config)

        return {
            "status": "success",
            "outputs": {
                "svs_vocal": str(output_path),
                "simple_synth_report": str(self._analysis_path()),
            },
            "message": (
                "Generated target-language vocal with basic formant synthesis."
            ),
        }