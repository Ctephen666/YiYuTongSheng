from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import path_from_config, project_root
from src.common.json_utils import read_json, write_json


class OpenUtauExporter:
    """Export svs_input.json into a real OpenUtau .ustx project.

    Input:
        data/svs/svs_input.json

    Outputs:
        data/svs/openutau_export_plan.json
        data/svs/target_language.ustx

    This module does not mock-render audio. It only creates a real OpenUtau
    project file that can be opened in OpenUtau.
    """

    TICKS_PER_BEAT = 480
    DEFAULT_BPM = 120.0

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _path_or_default(self, key: str, default: str) -> Path:
        try:
            return Path(path_from_config(self.config, key))
        except Exception:
            return project_root(self.config) / default

    def _svs_input_path(self) -> Path:
        return self._path_or_default(
            "svs_input",
            "data/svs/svs_input.json",
        )

    def _plan_output_path(self) -> Path:
        return project_root(self.config) / "data" / "svs" / "openutau_export_plan.json"

    def _ustx_output_path(self) -> Path:
        return self._path_or_default(
            "openutau_project",
            "data/svs/target_language.ustx",
        )

    # ------------------------------------------------------------------
    # Safe parsing
    # ------------------------------------------------------------------

    def _float(self, value: Any, default: float = 0.0) -> float:
        try:
            return round(float(value), 5)
        except Exception:
            return default

    def _int_or_none(self, value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _bpm(self) -> float:
        value = (
            self.config.get("music", {}).get("bpm")
            or self.config.get("project", {}).get("bpm")
            or self.config.get("svs", {}).get("bpm")
            or self.DEFAULT_BPM
        )
        return self._float(value, self.DEFAULT_BPM)

    def _seconds_to_ticks(self, seconds: float, bpm: float | None = None) -> int:
        """Convert absolute seconds into OpenUtau ticks.

        OpenUtau uses tick positions. We use a constant BPM approximation here
        because the current upstream melody extraction only provides absolute
        start/end times, not a tempo map.

        ticks_per_second = resolution * bpm / 60
        """

        if bpm is None:
            bpm = self._bpm()

        seconds = max(0.0, self._float(seconds))
        ticks_per_second = self.TICKS_PER_BEAT * bpm / 60.0
        return max(0, int(round(seconds * ticks_per_second)))

    def _duration_ticks(self, start: float, end: float, bpm: float | None = None) -> int:
        start_tick = self._seconds_to_ticks(start, bpm)
        end_tick = self._seconds_to_ticks(end, bpm)
        return max(1, end_tick - start_tick)

    def _safe_lyric(self, value: str) -> str:
        text = str(value or "").strip()

        if not text:
            return "la"

        # OpenUtau sustain lyric.
        if text == "-":
            return "-"

        # Avoid spaces in a note lyric. Use hyphens for merged syllable groups.
        text = text.replace(" ", "-")
        return text

    # ------------------------------------------------------------------
    # Build intermediate note events
    # ------------------------------------------------------------------

    def _build_note_events_from_unit(self, phrase: dict, unit: dict) -> list[dict]:
        """Convert one syllable group into note events.

        If one syllable spans multiple notes:
            first note lyric = syllable
            later note lyric = "-"
        """

        events = []
        notes = unit.get("notes", [])

        syllable = self._safe_lyric(unit.get("syllable", ""))
        word = str(unit.get("word", "") or "")
        phonemes = unit.get("phonemes", [])

        for note_index, note in enumerate(notes):
            is_first_note = note_index == 0

            event = {
                "phrase_id": phrase.get("id"),
                "unit_index": unit.get("unit_index"),
                "word": word,
                "syllable": syllable,
                "lyric": syllable if is_first_note else "-",
                "is_sustain": not is_first_note,
                "phonemes": phonemes if is_first_note else [],
                "pitch": note.get("pitch", ""),
                "midi": self._int_or_none(note.get("midi")),
                "start": self._float(note.get("start", 0.0)),
                "end": self._float(note.get("end", 0.0)),
                "duration": self._float(note.get("duration", 0.0)),
                "source_lyric": note.get("source_lyric", ""),
                "source_syllable": note.get("source_syllable", ""),
            }

            if event["midi"] is None:
                continue

            if event["end"] <= event["start"]:
                continue

            events.append(event)

        return events

    def _build_phrase_track(self, phrase: dict) -> dict:
        units = phrase.get("units", [])

        note_events = []

        for unit in units:
            note_events.extend(
                self._build_note_events_from_unit(
                    phrase=phrase,
                    unit=unit,
                )
            )

        return {
            "id": phrase.get("id"),
            "zh": phrase.get("zh", ""),
            "text": phrase.get("text", ""),
            "start": self._float(phrase.get("start", 0.0)),
            "end": self._float(phrase.get("end", 0.0)),
            "duration": self._float(phrase.get("duration", 0.0)),
            "unit_count": len(units),
            "note_count": len(note_events),
            "merge_info": phrase.get("merge_info", {}),
            "note_events": note_events,
        }

    def _flatten_note_events(self, phrase_tracks: list[dict]) -> list[dict]:
        events = []

        for phrase in phrase_tracks:
            events.extend(phrase.get("note_events", []))

        events.sort(
            key=lambda item: (
                self._float(item.get("start", 0.0)),
                self._float(item.get("end", 0.0)),
            )
        )

        return events

    # ------------------------------------------------------------------
    # Build USTX
    # ------------------------------------------------------------------

    def _build_expression_defaults(self) -> dict:
        """Minimal expression block commonly found in OpenUtau projects."""

        return {
            "dyn": {
                "name": "dynamics (curve)",
                "abbr": "dyn",
                "type": "Curve",
                "min": -240,
                "max": 120,
                "default_value": 0,
                "is_flag": False,
                "flag": "",
            },
            "pitd": {
                "name": "pitch deviation (curve)",
                "abbr": "pitd",
                "type": "Curve",
                "min": -1200,
                "max": 1200,
                "default_value": 0,
                "is_flag": False,
                "flag": "",
            },
            "clr": {
                "name": "voice color",
                "abbr": "clr",
                "type": "Options",
                "min": 0,
                "max": -1,
                "default_value": 0,
                "is_flag": False,
                "options": [],
            },
            "vel": {
                "name": "velocity",
                "abbr": "vel",
                "type": "Numerical",
                "min": 0,
                "max": 200,
                "default_value": 100,
                "is_flag": False,
                "flag": "",
            },
            "vol": {
                "name": "volume",
                "abbr": "vol",
                "type": "Numerical",
                "min": 0,
                "max": 200,
                "default_value": 100,
                "is_flag": False,
                "flag": "",
            },
            "atk": {
                "name": "attack",
                "abbr": "atk",
                "type": "Numerical",
                "min": 0,
                "max": 200,
                "default_value": 100,
                "is_flag": False,
                "flag": "",
            },
            "dec": {
                "name": "decay",
                "abbr": "dec",
                "type": "Numerical",
                "min": 0,
                "max": 100,
                "default_value": 0,
                "is_flag": False,
                "flag": "",
            },
            "gen": {
                "name": "gender",
                "abbr": "gen",
                "type": "Numerical",
                "min": -100,
                "max": 100,
                "default_value": 0,
                "is_flag": True,
                "flag": "g",
            },
            "bre": {
                "name": "breath",
                "abbr": "bre",
                "type": "Numerical",
                "min": 0,
                "max": 100,
                "default_value": 0,
                "is_flag": True,
                "flag": "B",
            },
            "lpf": {
                "name": "lowpass",
                "abbr": "lpf",
                "type": "Numerical",
                "min": 0,
                "max": 100,
                "default_value": 0,
                "is_flag": True,
                "flag": "H",
            },
            "mod": {
                "name": "modulation",
                "abbr": "mod",
                "type": "Numerical",
                "min": 0,
                "max": 100,
                "default_value": 0,
                "is_flag": False,
                "flag": "",
            },
        }

    def _track_phonemizer(self) -> str:
        return (
            self.config.get("svs", {}).get("openutau_phonemizer")
            or self.config.get("models", {}).get("openutau_phonemizer")
            or "OpenUtau.Core.DefaultPhonemizer"
        )

    def _build_ustx_notes(self, note_events: list[dict]) -> list[dict]:
        bpm = self._bpm()
        notes = []

        for event in note_events:
            start = self._float(event.get("start", 0.0))
            end = self._float(event.get("end", 0.0))

            position = self._seconds_to_ticks(start, bpm)
            duration = self._duration_ticks(start, end, bpm)

            note = {
                "position": position,
                "duration": duration,
                "tone": int(event["midi"]),
                "lyric": self._safe_lyric(event.get("lyric", "")),
                "pitch": {
                    "data": [
                        {
                            "x": -40,
                            "y": 0,
                            "shape": "io",
                        },
                        {
                            "x": 0,
                            "y": 0,
                            "shape": "io",
                        },
                    ],
                    "snap_first": True,
                },
                "vibrato": {
                    "length": 0,
                    "period": 175,
                    "depth": 25,
                    "in": 10,
                    "out": 10,
                    "shift": 0,
                    "drift": 0,
                    "vol_link": 0,
                },
                "phoneme_expressions": [],
                "phoneme_overrides": [],
            }

            notes.append(note)

        notes.sort(key=lambda item: (item["position"], item["tone"]))
        return notes
    
    def _build_ustx_project(self, note_events: list[dict]) -> dict:
        bpm = self._bpm()
        notes = self._build_ustx_notes(note_events)

        if not notes:
            raise RuntimeError("Cannot build .ustx because no notes were generated.")

        part_position = min(note["position"] for note in notes)
        part_end = max(note["position"] + note["duration"] for note in notes)
        part_duration = max(1, part_end - part_position)

        # Make note positions relative to the voice part.
        for note in notes:
            note["position"] = max(0, note["position"] - part_position)

        project = {
            "name": "YiYuTongSheng Target Language Vocal",
            "comment": "Generated automatically from svs_input.json.",
            "output_dir": "Vocal",
            "cache_dir": "UCache",
            "ustx_version": "0.7",
            "resolution": self.TICKS_PER_BEAT,
            "bpm": bpm,
            "beat_per_bar": 4,
            "beat_unit": 4,
            "expressions": self._build_expression_defaults(),
            "exp_selectors": [
                "dyn",
                "pitd",
                "clr",
                "vel",
                "vol",
                "atk",
                "dec",
                "gen",
                "bre",
                "lpf",
                "mod",
            ],
            "exp_primary": 0,
            "exp_secondary": 1,
            "key": 0,
            "time_signatures": [
                {
                    "bar_position": 0,
                    "beat_per_bar": 4,
                    "beat_unit": 4,
                }
            ],
            "tempos": [
                {
                    "position": 0,
                    "bpm": bpm,
                }
            ],
            "tracks": [
                {
                    "phonemizer": self._track_phonemizer(),
                    "renderer_settings": {},
                    "track_name": "English Vocal",
                    "track_color": "Blue",
                    "mute": False,
                    "solo": False,
                    "volume": 0,
                    "pan": 0,
                    "track_expressions": [],
                    "voice_color_names": [""],
                }
            ],
            "voice_parts": [
                {
                    "duration": part_duration,
                    "name": "Target Language Vocal",
                    "comment": "Generated by YiYuTongSheng",
                    "track_no": 0,
                    "position": part_position,
                    "notes": notes,
                    "curves": [],
                }
            ],
            "wave_parts": [],
        }

        return project

    def _write_yaml(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required to write a real .ustx file. "
                "Install it with: pip install pyyaml"
            ) from exc

        text = yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

        path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> dict:
        svs_input = self._svs_input_path()

        if not svs_input.exists():
            raise RuntimeError(
                f"SVS input file does not exist: {svs_input}. "
                "Run alignment and SVSInputExporter first."
            )

        svs_data = read_json(svs_input, {"phrases": []})
        phrases = svs_data.get("phrases", [])

        if not phrases:
            raise RuntimeError(
                f"svs_input.json contains no phrases: {svs_input}"
            )

        phrase_tracks = [
            self._build_phrase_track(phrase)
            for phrase in phrases
        ]

        note_events = self._flatten_note_events(phrase_tracks)

        if not note_events:
            raise RuntimeError(
                f"No note events generated from svs_input.json: {svs_input}"
            )

        plan_output = self._plan_output_path()
        ustx_output = self._ustx_output_path()

        ustx_project = self._build_ustx_project(note_events)
        self._write_yaml(ustx_output, ustx_project)

        plan = {
            "source": "svs_input",
            "format": "openutau_export_plan_v1",
            "language": svs_data.get("language", "en"),
            "unit_type": svs_data.get("unit_type", "syllable_group"),
            "description": (
                "OpenUtau export plan generated from svs_input.json. "
                "A real .ustx project is also generated."
            ),
            "inputs": {
                "svs_input": str(svs_input),
            },
            "outputs": {
                "openutau_export_plan": str(plan_output),
                "openutau_project": str(ustx_output),
            },
            "summary": {
                "phrase_count": len(phrase_tracks),
                "note_event_count": len(note_events),
                "start": note_events[0]["start"],
                "end": note_events[-1]["end"],
                "duration": round(note_events[-1]["end"] - note_events[0]["start"], 5),
                "bpm": self._bpm(),
                "resolution": self.TICKS_PER_BEAT,
            },
            "phrases": phrase_tracks,
            "note_events": note_events,
        }

        write_json(plan_output, plan, self.config)

        return {
            "status": "success",
            "outputs": {
                "openutau_export_plan": str(plan_output),
                "openutau_project": str(ustx_output),
            },
            "message": "Generated real OpenUtau .ustx project from svs_input.json.",
        }
