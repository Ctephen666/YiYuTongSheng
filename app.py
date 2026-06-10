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
from src.evaluate.generate_report import ReportGenerator
from src.lyrics.adapt_singable_lyrics import SingableLyricAdapter
from src.lyrics.phrase_mapper import PhraseMapper
from src.lyrics.translate_lyrics import LyricTranslator
from src.melody.export_score import ScoreExporter
from src.melody.extract_f0 import F0Extractor
from src.melody.f0_to_note import F0ToNoteConverter
from src.mix.align_vocal import VocalAligner
from src.mix.mix_audio import AudioMixer
from src.phoneme.phonemize_en import EnglishPhonemizer
from src.phoneme.phonemize_ja import JapanesePhonemizer
from src.preprocess.clean_audio import AudioCleaner
from src.preprocess.normalize_audio import AudioNormalizer
from src.preprocess.separate_vocals import VocalSeparator
from src.svc.rvc_infer import RVCInferencer
from src.svs.export_openutau_project import OpenUtauExporter
from src.svs.render_singing import SingingRenderer


VALID_STEPS = [
    "all",
    "preprocess",
    "melody",
    "lyrics",
    "phoneme",
    "alignment",
    "svs",
    "svc",
    "mix",
    "evaluate",
]


def _minimal_yaml_load(path: Path) -> dict:
    """Load the simple project YAML format when PyYAML is unavailable.

    Input:
        path: Config file path.
    Output:
        Nested dict for the project scaffold config.
    TODO:
        Remove this fallback once PyYAML is guaranteed in the runtime environment.
    """
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
                    parsed = value
            data[current_section][key] = parsed
    return data


def load_config(config_path: Path, target_language: str | None = None) -> dict:
    """Load project config and apply command-line overrides.

    Input:
        config_path: YAML config path.
        target_language: Optional en/ja override.
    Output:
        Pipeline config dict.
    TODO:
        Add pydantic/dataclass validation for production runs.
    """
    if yaml is not None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        config = _minimal_yaml_load(config_path)

    config["_project_root"] = str(Path(__file__).resolve().parent)
    if target_language:
        config.setdefault("project", {})["target_language"] = target_language
    return config


def run_stage(name: str, factories: list[Callable[[dict], object]], config: dict, logger) -> list[dict]:
    """Run all modules in one named stage.

    Input:
        name: Stage name.
        factories: Classes or callables that accept config and expose run().
        config: Pipeline config.
        logger: Configured logger.
    Output:
        List of stage result dicts.
    TODO:
        Persist a run manifest with timing, versions, and input hashes.
    """
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
    """Build the stage-to-module mapping.

    Input:
        config: Pipeline config.
    Output:
        Dict mapping step names to module factories.
    TODO:
        Make this plugin-driven for swapping real model backends.
    """
    language = config.get("project", {}).get("target_language", "en")
    phonemizer = JapanesePhonemizer if language == "ja" else EnglishPhonemizer
    return {
        "preprocess": [VocalSeparator, AudioCleaner, AudioNormalizer],
        "melody": [F0Extractor, F0ToNoteConverter, ScoreExporter],
        "lyrics": [PhraseMapper, LyricTranslator, SingableLyricAdapter],
        "phoneme": [phonemizer],
        "alignment": [NoteLyricAligner],
        "svs": [OpenUtauExporter, SingingRenderer],
        "svc": [RVCInferencer],
        "mix": [VocalAligner, AudioMixer],
        "evaluate": [ReportGenerator],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Input:
        Command-line argv.
    Output:
        argparse.Namespace.
    TODO:
        Add run-id, dry-run, and strict/no-mock flags.
    """
    parser = argparse.ArgumentParser(description="YiyuTongsheng mock singing voice conversion pipeline.")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to project YAML config.")
    parser.add_argument("--step", default="all", choices=VALID_STEPS, help="Pipeline step to run.")
    parser.add_argument("--target-language", choices=["en", "ja"], default=None, help="Override target language.")
    return parser.parse_args()


def main() -> None:
    """Run the requested pipeline step.

    Input:
        CLI arguments: --config, --step, and --target-language.
    Output:
        Creates mock project artifacts and prints stage statuses.
    TODO:
        Add structured run manifests and resume support.
    """
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    config = load_config(config_path, args.target_language)
    ensure_directories(config)
    logger = setup_logger(log_file=Path(config["_project_root"]) / "outputs" / "logs" / "pipeline.log")
    logger.info("Project: %s", config.get("project", {}).get("name"))
    logger.info("Target language: %s", config.get("project", {}).get("target_language"))

    pipeline = build_pipeline(config)
    selected_steps = list(pipeline.keys()) if args.step == "all" else [args.step]
    for step in selected_steps:
        run_stage(step, pipeline[step], config, logger)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
