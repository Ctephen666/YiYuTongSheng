from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.alignment.zh_note_alignment import ZhNoteAligner
from src.common.io_utils import ensure_directories
from src.common.logger import setup_logger
from src.dataset.opencpop_loader import OpenCpopDatasetLoader
from src.lyrics.zh_phrase_builder import ZhPhraseBuilder
from src.melody.import_opencpop_midi import OpenCpopMidiImporter
from src.phoneme.zh_phonemizer import ZhPhonemizer
from src.svs.run_opencpop_svs import OpenCpopZhSvsRunner


VALID_STEPS = [
    "all",
    "melody",
    "lyrics",
    "phoneme",
    "alignment",
    "svs",
]

ALL_STEPS = ["melody", "svs"]
DEFAULT_OPENCPop_ROOT = "/data/dataset/opencpop"


def _minimal_yaml_load(path: Path) -> dict:
    """Load a simple indentation-based YAML subset when PyYAML is unavailable."""
    data: dict = {}
    stack: list[tuple[int, dict]] = [(-1, data)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not value:
            child: dict = {}
            current[key] = child
            stack.append((indent, child))
            continue

        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        current[key] = parsed

    return data


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _clean_song_id(sample_id: str) -> str:
    value = str(sample_id).strip()
    for suffix in (".midi", ".mid", ".TextGrid", ".textgrid", ".wav"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    if not value:
        raise ValueError("--opencpop-id cannot be empty.")
    return value


def _dataset_root(config: dict) -> str:
    return str(
        config.get("dataset", {}).get("root")
        or config.get("inputs", {}).get("opencpop_root")
        or DEFAULT_OPENCPop_ROOT
    )


def _apply_song_id(config: dict, sample_id: str) -> None:
    sample_id = _clean_song_id(sample_id)
    root = Path(_dataset_root(config))
    midi_dir = Path(str(config.get("inputs", {}).get("opencpop_midi_dir") or root / "midis"))
    textgrid_dir = Path(str(config.get("inputs", {}).get("opencpop_textgrid") or root / "TextGrid"))
    wav_dir = Path(str(config.get("inputs", {}).get("opencpop_wavs") or root / "wavs"))

    config.setdefault("dataset", {})["default_song_id"] = sample_id
    config.setdefault("inputs", {})["opencpop_default_song_id"] = sample_id
    config.setdefault("opencpop", {})["sample_id"] = sample_id

    item_path = f"data/dataset_manifest/opencpop_item_{sample_id}.json"
    config.setdefault("outputs", {})["opencpop_item"] = item_path
    config.setdefault("paths", {})["opencpop_item"] = item_path
    config.setdefault("paths", {})["opencpop_midi"] = str(midi_dir / f"{sample_id}.midi")
    config.setdefault("paths", {})["opencpop_textgrid"] = str(textgrid_dir / f"{sample_id}.TextGrid")
    config.setdefault("paths", {})["opencpop_wav"] = str(wav_dir / f"{sample_id}.wav")


def _apply_cli_overrides(
    config: dict,
    target_language: str | None = None,
    opencpop_id: str | None = None,
    opencpop_midi: str | None = None,
) -> dict:
    config.setdefault("project", {})["target_language"] = target_language or config.get("project", {}).get("target_language", "zh")

    if opencpop_id and opencpop_midi:
        raise ValueError("Use only one of --opencpop-id or --opencpop-midi.")

    if opencpop_id:
        _apply_song_id(config, opencpop_id)

    if opencpop_midi:
        config.setdefault("paths", {})["opencpop_midi"] = opencpop_midi

    return config


def load_config(
    config_path: Path,
    target_language: str | None = None,
    opencpop_id: str | None = None,
    opencpop_midi: str | None = None,
) -> dict:
    """Load project config and apply command-line overrides."""
    if yaml is not None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        config = _minimal_yaml_load(config_path)

    config = config or {}
    config["_project_root"] = str(Path(__file__).resolve().parent)
    return _apply_cli_overrides(config, target_language, opencpop_id, opencpop_midi)


def run_stage(name: str, factories: list[Callable[[dict], object]], config: dict, logger) -> list[dict]:
    """Run all modules in one named stage."""
    logger.info("Running stage: %s", name)
    results = []

    for factory in factories:
        module = factory(config)
        result = module.run()
        results.append(result)
        logger.info("%s -> %s | %s", module.__class__.__name__, result.get("status"), result.get("message"))

        for warning in result.get("warnings", []):
            logger.warning("  warning %s", warning)
        for key, value in result.get("outputs", {}).items():
            logger.info("  output %-24s %s", key + ":", value)

    return results


def build_pipeline(config: dict) -> dict[str, list[Callable[[dict], object]]]:
    """Build the OpenCpop Chinese SVS inference pipeline."""
    return {
        "melody": [OpenCpopDatasetLoader, OpenCpopMidiImporter],
        "lyrics": [ZhPhraseBuilder],
        "phoneme": [ZhPhonemizer],
        "alignment": [ZhNoteAligner],
        "svs": [OpenCpopZhSvsRunner],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YiYuTongSheng OpenCpop Chinese SVS inference pipeline.")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to project YAML config.")
    parser.add_argument("--step", default="all", choices=VALID_STEPS, help="Pipeline step to run.")
    parser.add_argument("--target-language", choices=["zh", "en"], default="zh", help="Override target language.")
    parser.add_argument(
        "--opencpop-id",
        default=None,
        help="OpenCpop song id under the configured dataset root, for example 2001.",
    )
    parser.add_argument(
        "--opencpop-midi",
        default=None,
        help="Explicit MIDI path. Used only when strict_dataset_source is false.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    config = load_config(
        config_path,
        target_language=args.target_language,
        opencpop_id=args.opencpop_id,
        opencpop_midi=args.opencpop_midi,
    )

    ensure_directories(config, extra_dirs=["data/dataset_manifest"])
    logger = setup_logger(log_file=Path(config["_project_root"]) / "outputs" / "logs" / "pipeline.log")
    logger.info("Project: %s", config.get("project", {}).get("name"))
    logger.info("Target language: %s", config.get("project", {}).get("target_language"))
    logger.info("OpenCpop root: %s", config.get("dataset", {}).get("root"))
    logger.info("OpenCpop song id: %s", config.get("dataset", {}).get("default_song_id"))
    logger.info("Strict dataset source: %s", config.get("dataset", {}).get("strict_dataset_source"))

    if config.get("project", {}).get("target_language") == "en":
        logger.warning(
            "English cross-language pipeline is disabled in this OpenCpop zh SVS branch; no MeloTTS, translation, or legacy English stage was run."
        )
        return

    pipeline = build_pipeline(config)
    selected_steps = ALL_STEPS if args.step == "all" else [args.step]

    for step in selected_steps:
        run_stage(step, pipeline[step], config, logger)

    logger.info("OpenCpop Chinese SVS inference finished.")


if __name__ == "__main__":
    main()
