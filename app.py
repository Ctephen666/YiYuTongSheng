from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.alignment.note_lyric_aligner import NoteLyricAligner
from src.common.io_utils import ensure_directories
from src.common.logger import setup_logger
from src.lyrics.adapt_singable_lyrics import SingableLyricAdapter
from src.lyrics.import_opencpop_textgrid import OpenCpopTextGridImporter
from src.lyrics.translate_lyrics import LyricTranslator
from src.melody.import_opencpop_midi import OpenCpopMidiImporter
from src.phoneme.phonemize_en import EnglishPhonemizer
from src.svs.melotts_renderer import MeloTTSRenderer
from src.svs.aligned_dtw_world_renderer import AlignedDTWWorldRenderer
from src.svs.source_f0_guided_world_renderer import SourceF0GuidedWorldRenderer


VALID_STEPS = [
    "all",
    "melody",
    "lyrics",
    "phoneme",
    "alignment",
    "svs",
]

ALL_STEPS = ["melody", "lyrics", "phoneme", "alignment", "svs"]
DEFAULT_OPENCPop_MIDI_DIR = "data/dataset/opencpop/midis"
DEFAULT_OPENCPop_TEXTGRID_DIR = "data/dataset/opencpop/textgrids"


def _minimal_yaml_load(path: Path) -> dict:
    """Load the simple project YAML format when PyYAML is unavailable."""
    data: dict[str, dict] = {}
    current_section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            data[current_section] = {}
            continue

        if current_section and ":" in line:
            key, value = line.strip().split(":", 1)
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

            data[current_section][key] = parsed

    return data


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _opencpop_midi_from_id(sample_id: str) -> str:
    sample_id = str(sample_id).strip()
    if not sample_id:
        raise ValueError("--opencpop-id cannot be empty.")

    suffix = ".midi" if not sample_id.lower().endswith((".mid", ".midi")) else ""
    return str(Path(DEFAULT_OPENCPop_MIDI_DIR) / f"{sample_id}{suffix}")



def _opencpop_textgrid_from_id(sample_id: str) -> str:
    sample_id = str(sample_id).strip()
    suffix = ".TextGrid" if not sample_id.lower().endswith(".textgrid") else ""
    return str(Path(DEFAULT_OPENCPop_TEXTGRID_DIR) / f"{sample_id}{suffix}")

def _apply_cli_overrides(
    config: dict,
    target_language: str | None = None,
    opencpop_id: str | None = None,
    opencpop_midi: str | None = None,
) -> dict:
    if target_language:
        config.setdefault("project", {})["target_language"] = target_language

    if opencpop_id and opencpop_midi:
        raise ValueError("Use only one of --opencpop-id or --opencpop-midi.")

    if opencpop_id:
        config.setdefault("paths", {})["opencpop_midi"] = _opencpop_midi_from_id(opencpop_id)
        config.setdefault("paths", {})["opencpop_textgrid"] = _opencpop_textgrid_from_id(opencpop_id)
        config.setdefault("opencpop", {})["sample_id"] = str(opencpop_id).strip().removesuffix(".midi").removesuffix(".mid")

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

        for key, value in result.get("outputs", {}).items():
            logger.info("  output %-24s %s", key + ":", value)

    return results


def build_pipeline(config: dict) -> dict[str, list[Callable[[dict], object]]]:
    """Build the current OpenCpop MIDI + MeloTTS baseline pipeline."""
    return {
        "melody": [OpenCpopTextGridImporter, OpenCpopMidiImporter],
        "lyrics": [OpenCpopTextGridImporter, LyricTranslator, SingableLyricAdapter],
        "phoneme": [EnglishPhonemizer],
        "alignment": [NoteLyricAligner],
        "svs": [MeloTTSRenderer, AlignedDTWWorldRenderer, SourceF0GuidedWorldRenderer],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YiYuTongSheng OpenCpop MIDI + MeloTTS baseline pipeline.")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to project YAML config.")
    parser.add_argument("--step", default="all", choices=VALID_STEPS, help="Pipeline step to run.")
    parser.add_argument("--target-language", choices=["en"], default=None, help="Override target language.")
    parser.add_argument(
        "--opencpop-id",
        default=None,
        help="OpenCpop sample id under data/dataset/opencpop/midis, for example 2001.",
    )
    parser.add_argument(
        "--opencpop-midi",
        default=None,
        help="Explicit OpenCpop MIDI path. Overrides paths.opencpop_midi in the config.",
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

    ensure_directories(config)
    logger = setup_logger(log_file=Path(config["_project_root"]) / "outputs" / "logs" / "pipeline.log")
    logger.info("Project: %s", config.get("project", {}).get("name"))
    logger.info("Target language: %s", config.get("project", {}).get("target_language"))
    logger.info("OpenCpop MIDI: %s", config.get("paths", {}).get("opencpop_midi"))
    logger.info("OpenCpop TextGrid: %s", config.get("paths", {}).get("opencpop_textgrid"))

    pipeline = build_pipeline(config)
    selected_steps = ALL_STEPS if args.step == "all" else [args.step]

    for step in selected_steps:
        run_stage(step, pipeline[step], config, logger)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
