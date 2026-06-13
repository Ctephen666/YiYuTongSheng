from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent, resolve_path


MIDI_EXTS = (".midi", ".mid")
WAV_EXTS = (".wav", ".flac")
TEXTGRID_EXTS = (".TextGrid", ".textgrid")
LYRIC_EXTS = (".txt", ".lab", ".lyric", ".lyrics", ".json", ".csv", ".tsv")
META_EXTS = (".json", ".yaml", ".yml", ".csv", ".tsv")

DIR_CANDIDATES = {
    "midi": ("midis", "midi", "MIDI", "Midis"),
    "wav": ("wavs", "wav", "WAV", "audio", "audios"),
    "textgrid": ("TextGrid", "textgrids", "textgrid", "TextGrids"),
    "segments": ("segments", "segment", "labs", "labels"),
    "transcriptions": ("transcriptions", "transcription", "lyrics", "texts"),
    "metadata": ("metadata", "meta", "annotations", "annotation"),
}

REQUIRED_DIR_KEYS = {"midi", "wav", "textgrid"}

def _project_root(config: dict) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def _resolve_config_value(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _project_root(config) / path


def _config_value(config: dict, section: str, key: str, default: Any = None) -> Any:
    data = config.get(section, {})
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def _output_path(config: dict, key: str, default: str) -> Path:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs", {}), dict) else {}
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    value = outputs.get(key) or paths.get(key) or default
    return _resolve_config_value(config, value)


def _json_write(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalise_song_id(song_id: str) -> str:
    value = str(song_id).strip()
    for suffix in (".midi", ".mid", ".TextGrid", ".textgrid", ".wav", ".txt"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    if not value:
        raise ValueError("OpenCpop song_id cannot be empty.")
    return value


def _candidate_names(song_id: str, extensions: tuple[str, ...]) -> list[str]:
    bases = [song_id]
    if song_id.isdigit():
        bases.extend([song_id.zfill(4), song_id.zfill(5)])
    seen: set[str] = set()
    names: list[str] = []
    for base in bases:
        for ext in extensions:
            for candidate in (f"{base}{ext}", f"{base}{ext.lower()}", f"{base}{ext.upper()}"):
                if candidate not in seen:
                    seen.add(candidate)
                    names.append(candidate)
    return names


def _existing_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    if not root.is_dir():
        return None
    lower_map = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    for name in names:
        match = lower_map.get(name.lower())
        if match is not None:
            return match
    return None


def _find_file(directories: list[Path], song_id: str, extensions: tuple[str, ...]) -> Path | None:
    for directory in directories:
        if not directory or not directory.exists():
            continue
        for name in _candidate_names(song_id, extensions):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_lyrics(path: Path | None) -> tuple[str, list[dict], list[str]]:
    if path is None or not path.exists():
        return "", [], []

    warnings: list[str] = []
    if path.suffix.lower() == ".json":
        data = _read_json(path)
        if not isinstance(data, dict):
            return "", [], [f"Could not parse lyric JSON: {path}"]

        text = str(data.get("text") or data.get("lyrics") or data.get("lyric") or "")
        raw_phrases = data.get("phrases") or data.get("sentences") or []
        phrases: list[dict] = []
        if isinstance(raw_phrases, list):
            for index, phrase in enumerate(raw_phrases, start=1):
                if isinstance(phrase, dict):
                    phrase_text = phrase.get("zh") or phrase.get("text") or phrase.get("lyric") or ""
                    phrases.append({"id": phrase.get("id", index), "zh": str(phrase_text)})
                elif phrase:
                    phrases.append({"id": index, "zh": str(phrase)})
        if not text and phrases:
            text = "\n".join(str(item.get("zh", "")) for item in phrases if item.get("zh"))
        return text, phrases, warnings

    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return "", [], [f"Lyric file is empty: {path}"]
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    phrases = [{"id": index, "zh": line} for index, line in enumerate(lines, start=1)]
    return "\n".join(lines), phrases, warnings


def discover_opencpop_structure(dataset_root: str | Path) -> dict:
    """Discover the expected OpenCpop top-level directories without recursive scanning."""
    root = Path(dataset_root)
    warnings: list[str] = []
    if not root.exists():
        warnings.append(f"OpenCpop dataset root does not exist: {root}")
    elif not root.is_dir():
        warnings.append(f"OpenCpop dataset root is not a directory: {root}")

    dirs: dict[str, dict[str, Any]] = {}
    for key, names in DIR_CANDIDATES.items():
        found = _existing_dir(root, names)
        dirs[key] = {
            "path": str(found if found is not None else root / names[0]),
            "exists": bool(found and found.is_dir()),
            "candidates": list(names),
        }
        if found is None and key in REQUIRED_DIR_KEYS:
            warnings.append(f"OpenCpop {key} directory not found under {root}; candidates={list(names)}")

    return {
        "dataset": "opencpop",
        "root": str(root),
        "exists": root.is_dir(),
        "directories": dirs,
        "warnings": warnings,
    }


def load_opencpop_item(dataset_root: str | Path, song_id: str, strict: bool = True) -> dict:
    """Build one OpenCpop song descriptor from dataset files only."""
    root = Path(dataset_root)
    song_id = _normalise_song_id(song_id)
    manifest = discover_opencpop_structure(root)
    warnings = list(manifest.get("warnings", []))

    dirs = manifest.get("directories", {})
    midi_dirs = [Path(dirs.get("midi", {}).get("path", root / "midis")), root]
    wav_dirs = [Path(dirs.get("wav", {}).get("path", root / "wavs")), root]
    textgrid_dirs = [Path(dirs.get("textgrid", {}).get("path", root / "TextGrid")), root]
    lyric_dirs = [
        Path(dirs.get("transcriptions", {}).get("path", root / "transcriptions")),
        Path(dirs.get("segments", {}).get("path", root / "segments")),
    ]
    metadata_dirs = [Path(dirs.get("metadata", {}).get("path", root / "metadata")), root]

    midi_path = _find_file(midi_dirs, song_id, MIDI_EXTS)
    wav_path = _find_file(wav_dirs, song_id, WAV_EXTS)
    textgrid_path = _find_file(textgrid_dirs, song_id, TEXTGRID_EXTS)
    lyric_path = _find_file(lyric_dirs, song_id, LYRIC_EXTS)
    metadata_path = _find_file(metadata_dirs, song_id, META_EXTS)

    for label, path in (
        ("midi", midi_path),
        ("wav", wav_path),
        ("textgrid", textgrid_path),
    ):
        if path is None:
            warnings.append(f"OpenCpop item {song_id} missing {label} file in dataset root {root}.")
    if lyric_path is None and textgrid_path is None:
        warnings.append(f"OpenCpop item {song_id} missing lyric/transcription file and TextGrid in dataset root {root}.")

    text, phrases, lyric_warnings = _read_lyrics(lyric_path)
    warnings.extend(lyric_warnings)
    if strict and not text and not phrases and textgrid_path is None:
        warnings.append(
            "Strict dataset source is enabled; no legacy lyrics_zh.txt or phrase_map.json fallback was used."
        )

    if text or phrases:
        lyrics_source = "dataset"
    elif textgrid_path is not None:
        lyrics_source = "textgrid_available"
    else:
        lyrics_source = "dataset_missing"
    item = {
        "dataset": "opencpop",
        "song_id": song_id,
        "strict": bool(strict),
        "dataset_root": str(root),
        "midi_path": str(midi_path) if midi_path else "",
        "wav_path": str(wav_path) if wav_path else "",
        "textgrid_path": str(textgrid_path) if textgrid_path else "",
        "lyric_path": str(lyric_path) if lyric_path else "",
        "metadata_path": str(metadata_path) if metadata_path else "",
        "lyrics": {
            "source": lyrics_source,
            "text": text,
            "phrases": phrases,
        },
        "notes": {
            "source": "dataset_midi" if midi_path else "dataset_midi_missing",
            "items": [],
        },
        "phoneme_annotations": {
            "source": "dataset_if_available",
            "items": [],
        },
        "warnings": warnings,
    }
    return item


def save_opencpop_manifest(manifest: dict, output_path: str | Path) -> None:
    _json_write(Path(output_path), manifest)


class OpenCpopDatasetLoader:
    """Pipeline stage that writes OpenCpop dataset manifest files."""

    def __init__(self, config: dict):
        self.config = config

    def _dataset_root(self) -> Path:
        root = _config_value(self.config, "dataset", "root") or _config_value(
            self.config, "inputs", "opencpop_root", "/data/dataset/opencpop"
        )
        return _resolve_config_value(self.config, root)

    def _song_id(self) -> str:
        return _normalise_song_id(
            _config_value(self.config, "dataset", "default_song_id")
            or _config_value(self.config, "inputs", "opencpop_default_song_id")
            or _config_value(self.config, "opencpop", "sample_id")
            or "2001"
        )

    def _strict(self) -> bool:
        return bool(_config_value(self.config, "dataset", "strict_dataset_source", True))

    def run(self) -> dict:
        root = self._dataset_root()
        song_id = self._song_id()
        strict = self._strict()

        manifest = discover_opencpop_structure(root)
        item = load_opencpop_item(root, song_id, strict=strict)

        manifest_path = _output_path(
            self.config,
            "dataset_manifest",
            "data/dataset_manifest/opencpop_dataset_manifest.json",
        )
        item_path = _output_path(
            self.config,
            "opencpop_item",
            f"data/dataset_manifest/opencpop_item_{song_id}.json",
        )

        save_opencpop_manifest(manifest, manifest_path)
        save_opencpop_manifest(item, item_path)

        warning_count = len(manifest.get("warnings", [])) + len(item.get("warnings", []))
        status = "success" if warning_count == 0 else "warning"
        return {
            "status": status,
            "outputs": {
                "dataset_manifest": str(manifest_path),
                "opencpop_item": str(item_path),
            },
            "warnings": manifest.get("warnings", []) + item.get("warnings", []),
            "message": f"OpenCpop dataset descriptor prepared for song {song_id}; warnings={warning_count}.",
        }
