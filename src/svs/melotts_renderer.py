from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, path_from_config, should_write
from src.common.json_utils import read_json, write_json


def get_phrase_text(phrase: dict) -> str:
    """Return a robust English text field from a lyric phrase."""
    candidates = [
        phrase.get("text"),
        phrase.get("selected") if isinstance(phrase.get("selected"), str) else None,
        phrase.get("selected", {}).get("text") if isinstance(phrase.get("selected"), dict) else None,
        phrase.get("en"),
        phrase.get("target"),
        phrase.get("translation"),
    ]

    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text

    return "la"


class MeloTTSRenderer:
    """Render phrase-level English vocals with MeloTTS and MIDI phrase timing."""

    WARNING = "This is a MeloTTS phrase-level timing-aligned baseline, not full note-level SVS."

    def __init__(self, config: dict):
        self.config = config

    def _sample_rate(self) -> int:
        return int(self.config.get("project", {}).get("sample_rate", 44100))

    def _svs_config(self) -> dict:
        return self.config.get("svs", {})

    def _import_audio_dependencies(self):
        try:
            import librosa
            import numpy as np
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Audio dependencies are not installed. Install them with: "
                "pip install pretty_midi librosa soundfile numpy"
            ) from exc
        return librosa, np, sf

    def _load_melotts(self):
        try:
            from melo.api import TTS
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MeloTTS is not installed. Install it with:\n"
                "pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
                "python -m unidic download"
            ) from exc
        return TTS

    def _device(self) -> str:
        device = str(self._svs_config().get("device", "auto") or "auto")
        if device != "auto":
            return device

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _build_model(self):
        TTS = self._load_melotts()
        language = str(self._svs_config().get("melotts_language", "EN") or "EN")
        return TTS(language=language, device=self._device())

    def _speaker_id(self, model: Any) -> int:
        speaker = str(self._svs_config().get("melotts_speaker", "EN-US") or "EN-US").strip()

        if speaker.isdigit():
            return int(speaker)

        # MeloTTS English exposes these stable speaker names. Resolve them
        # before touching version-dependent internals such as hps.data.spk2id.
        known_english_speakers = {
            "EN-US": 0,
            "EN-BR": 1,
            "EN_INDIA": 2,
            "EN-AU": 3,
            "EN-Default": 4,
        }
        if speaker in known_english_speakers:
            return known_english_speakers[speaker]

        speaker_sources = [
            getattr(model, "speaker_ids", None),
            getattr(model, "spk2id", None),
        ]

        hps_data = getattr(getattr(model, "hps", None), "data", None)
        if hps_data is not None:
            speaker_sources.extend([
                hps_data.get("spk2id") if hasattr(hps_data, "get") else None,
                getattr(hps_data, "spk2id", None),
            ])

        available = None
        for speaker_ids in speaker_sources:
            if speaker_ids is None:
                continue
            available = speaker_ids

            try:
                resolved = speaker_ids.get(speaker)
            except Exception:
                resolved = None
            if resolved is not None:
                return int(resolved)

            try:
                return int(speaker_ids[speaker])
            except Exception:
                pass

            try:
                items = list(speaker_ids.items())
            except Exception:
                items = []
            if items:
                fallback_name, fallback_id = items[0]
                return int(fallback_id)

        raise RuntimeError(
            "Could not resolve MeloTTS speaker name to numeric speaker_id. "
            f"requested={speaker!r}, available={available!r}. "
            "For English use one of: EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default, or set melotts_speaker to a numeric id."
        )
    def _phrase_by_id(self, lyric_phrases: list[dict]) -> dict:
        result = {}
        for phrase in lyric_phrases:
            if isinstance(phrase, dict) and phrase.get("id") is not None:
                result[phrase.get("id")] = phrase
        return result

    def _matched_lyric(self, melody_phrase: dict, lyric_phrases: list[dict], lyric_by_id: dict, index: int) -> dict:
        phrase_id = melody_phrase.get("id")
        if phrase_id in lyric_by_id:
            return lyric_by_id[phrase_id]
        if index < len(lyric_phrases):
            return lyric_phrases[index]
        return {}

    def _tts_to_file(self, model: Any, text: str, speaker_id: Any, output_path: Path, speed: float) -> None:
        ensure_parent(output_path)
        try:
            model.tts_to_file(
                text=text,
                speaker_id=speaker_id,
                output_path=str(output_path),
                speed=speed,
                quiet=True,
            )
        except TypeError:
            model.tts_to_file(
                text=text,
                speaker_id=speaker_id,
                output_path=str(output_path),
                speed=speed,
            )

    def _fit_length(self, wav, target_len: int, sample_rate: int, librosa, np):
        if target_len <= 0:
            raise RuntimeError("MIDI phrase duration produced a non-positive target length.")

        wav = np.asarray(wav, dtype=np.float32)
        wav = np.nan_to_num(wav)

        if wav.size == 0:
            raise RuntimeError("MeloTTS generated an empty phrase wav.")

        current_duration = len(wav) / sample_rate
        target_duration = target_len / sample_rate

        if current_duration > 0 and target_duration > 0:
            rate = current_duration / target_duration
            if 0.5 <= rate <= 2.0:
                wav = librosa.effects.time_stretch(wav, rate=rate)

        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))

        return np.asarray(wav, dtype=np.float32)

    def _fade(self, wav, sample_rate: int, np):
        fade_ms = float(self._svs_config().get("phrase_fade_ms", 20) or 0)
        fade_samples = int(sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(wav) // 2)

        if fade_samples <= 0:
            return wav

        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        wav[:fade_samples] *= fade_in
        wav[-fade_samples:] *= fade_out
        return wav

    def _required_input(self, path: Path, message: str) -> None:
        if not path.exists():
            raise FileNotFoundError(message)

    def run(self) -> dict:
        melody_path = path_from_config(self.config, "melody_notes")
        lyrics_path = path_from_config(self.config, "lyrics_singable")
        output_path = path_from_config(self.config, "svs_vocal")
        report_path = output_path.parent / "melotts_render_report.json"
        phrase_dir = output_path.parent / "melotts_phrases"

        self._required_input(
            melody_path,
            "Missing melody_notes.json. Run first: python app.py --step melody --target-language en",
        )
        self._required_input(
            lyrics_path,
            "Missing lyrics_singable.json. Run first: python app.py --step lyrics --target-language en",
        )

        librosa, np, sf = self._import_audio_dependencies()
        model = self._build_model()
        speaker_id = self._speaker_id(model)

        sample_rate = self._sample_rate()
        speed = float(self._svs_config().get("melotts_speed", 0.85) or 0.85)
        language = str(self._svs_config().get("melotts_language", "EN") or "EN")
        speaker = str(self._svs_config().get("melotts_speaker", "EN-US") or "EN-US")
        normalize_peak = float(self._svs_config().get("normalize_peak", 0.98) or 0.98)

        melody_data = read_json(melody_path, {"phrases": []})
        lyric_data = read_json(lyrics_path, {"phrases": []})
        melody_phrases = melody_data.get("phrases", [])
        lyric_phrases = lyric_data.get("phrases", [])

        if not melody_phrases:
            raise RuntimeError("melody_notes.json contains no phrases.")
        if not lyric_phrases:
            raise RuntimeError("lyrics_singable.json contains no phrases.")

        lyric_by_id = self._phrase_by_id(lyric_phrases)
        final_duration = max(
            float(phrase.get("start", 0.0)) + float(phrase.get("duration", 0.0))
            for phrase in melody_phrases
        )
        final = np.zeros(int((final_duration + 0.5) * sample_rate), dtype=np.float32)

        report_phrases = []

        for index, melody_phrase in enumerate(melody_phrases):
            lyric_phrase = self._matched_lyric(melody_phrase, lyric_phrases, lyric_by_id, index)
            text = get_phrase_text(lyric_phrase)
            phrase_id = melody_phrase.get("id", index + 1)
            start = float(melody_phrase.get("start", 0.0))
            duration = float(melody_phrase.get("duration", 0.0))
            target_samples = max(1, int(duration * sample_rate))

            try:
                phrase_file_id = int(phrase_id)
            except (TypeError, ValueError):
                phrase_file_id = index + 1

            raw_phrase_path = phrase_dir / f"phrase_{phrase_file_id:03d}_raw.wav"
            if should_write(raw_phrase_path, self.config):
                self._tts_to_file(model, text, speaker_id, raw_phrase_path, speed)

            wav, _ = librosa.load(raw_phrase_path, sr=sample_rate, mono=True)
            phrase_wav = self._fit_length(wav, target_samples, sample_rate, librosa, np)
            phrase_wav = self._fade(phrase_wav, sample_rate, np)

            start_sample = max(0, int(start * sample_rate))
            end_sample = min(len(final), start_sample + len(phrase_wav))
            if end_sample > start_sample:
                final[start_sample:end_sample] += phrase_wav[: end_sample - start_sample]

            report_phrases.append(
                {
                    "id": phrase_id,
                    "text": text,
                    "start": start,
                    "duration": duration,
                    "raw_path": str(raw_phrase_path),
                    "target_samples": target_samples,
                }
            )

        peak = float(np.max(np.abs(final))) if final.size else 0.0
        if peak > normalize_peak > 0:
            final = final / peak * normalize_peak

        ensure_parent(output_path)
        sf.write(str(output_path), final, sample_rate)

        report = {
            "backend": "melotts",
            "language": language,
            "speaker": speaker,
            "sample_rate": sample_rate,
            "phrase_count": len(report_phrases),
            "output": str(output_path),
            "phrases": report_phrases,
            "warning": self.WARNING,
        }
        write_json(report_path, report, self.config)

        return {
            "status": "success",
            "outputs": {
                "svs_vocal": str(output_path),
                "melotts_render_report": str(report_path),
            },
            "message": self.WARNING,
        }
