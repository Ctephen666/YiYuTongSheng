from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.alignment.zh_note_alignment import align_zh_notes, save_zh_alignment
from src.dataset.opencpop_loader import OpenCpopDatasetLoader
from src.lyrics.zh_phrase_builder import build_zh_phrases, save_zh_phrases
from src.melody.import_opencpop_midi import OpenCpopMidiImporter
from src.phoneme.zh_phonemizer import phonemize_zh, save_zh_phonemes
from src.svs.build_opencpop_svs_score import build_opencpop_svs_score, save_opencpop_svs_score
from src.svs.checkpoint_manager import check_svs_checkpoints, save_checkpoint_status
from src.svs.diffsinger_opencpop_exporter import build_diffsinger_export_plan, save_diffsinger_export_plan
from src.svs.opencpop_neural_svs_backend import build_neural_svs_render_plan, save_neural_svs_render_plan
from src.svs.rvc_voice_conversion import apply_rvc_voice_conversion


def _load_project_config() -> dict:
    path = PROJECT_ROOT / "configs" / "project.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to run web demo tasks.")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Project config is not a mapping: {path}")
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def _bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_song(config: dict, song_id: str) -> None:
    config.setdefault("dataset", {})["default_song_id"] = song_id
    config.setdefault("opencpop", {})["sample_id"] = song_id
    config.setdefault("inputs", {})["opencpop_default_song_id"] = song_id

    paths = config.setdefault("paths", {})
    outputs = config.setdefault("outputs", {})
    paths["opencpop_midi"] = f"data/dataset/opencpop/midis/{song_id}.midi"
    paths["opencpop_textgrid"] = f"data/dataset/opencpop/TextGrid/{song_id}.TextGrid"
    paths["opencpop_wav"] = f"data/dataset/opencpop/wavs/{song_id}.wav"
    paths["diffsinger_input"] = f"data/svs/diffsinger_opencpop/diffsinger_input_{song_id}.json"
    paths["diffsinger_phoneme_tsv"] = f"data/svs/diffsinger_opencpop/phonemes_{song_id}.tsv"
    paths["diffsinger_notes_csv"] = f"data/svs/diffsinger_opencpop/notes_{song_id}.csv"
    outputs["opencpop_item"] = f"data/dataset_manifest/opencpop_item_{song_id}.json"
    outputs["diffsinger_input"] = f"data/svs/diffsinger_opencpop/diffsinger_input_{song_id}.json"
    outputs["diffsinger_phoneme_tsv"] = f"data/svs/diffsinger_opencpop/phonemes_{song_id}.tsv"
    outputs["diffsinger_notes_csv"] = f"data/svs/diffsinger_opencpop/notes_{song_id}.csv"


def _apply_svs_args(config: dict, args: argparse.Namespace) -> None:
    svs = config.setdefault("svs", {})
    svs["start_phrase"] = int(args.start_phrase)
    svs["max_phrases"] = int(args.max_phrases) if args.max_phrases is not None else None
    svs["assembly_mode"] = args.assembly_mode
    svs["execute_model"] = True
    svs["dry_run"] = False
    diffsinger = svs.setdefault("diffsinger", {})
    diffsinger["infer_class"] = args.infer_class
    diffsinger["normalize_output"] = _bool_text(args.normalize)
    diffsinger.setdefault("f0_fallback", {})["naturalize"] = _bool_text(args.f0_naturalize)


def _apply_svc_args(config: dict, args: argparse.Namespace) -> None:
    svs = config.setdefault("svs", {})
    rvc = svs.setdefault("rvc", {})
    rvc["enabled"] = True
    if args.voice_model:
        rvc["model_name"] = Path(args.voice_model).name
    if args.index_file:
        rvc["index_path"] = args.index_file
    rvc["f0method"] = args.f0_method
    rvc["f0up_key"] = int(args.f0_up_key)
    rvc["index_rate"] = float(args.index_rate)
    rvc["protect"] = float(args.protect)
    rvc["filter_radius"] = int(args.filter_radius)
    rvc["resample_sr"] = int(args.resample_sr)


def _emit_result(name: str, payload: dict) -> None:
    print(json.dumps({"task": name, "result": payload}, ensure_ascii=False, indent=2))


def run_score(config: dict) -> dict:
    outputs: dict[str, str] = {}
    warnings: list[str] = []
    rvc = config.setdefault("svs", {}).setdefault("rvc", {})
    rvc_enabled = bool(rvc.get("enabled", False))
    rvc["enabled"] = False

    for result in [OpenCpopDatasetLoader(config).run(), OpenCpopMidiImporter(config).run()]:
        outputs.update({key: str(value) for key, value in result.get("outputs", {}).items()})
        warnings.extend(result.get("warnings", []))

    phrases = build_zh_phrases(config)
    outputs["lyrics_zh_phrases"] = str(save_zh_phrases(config, phrases))
    warnings.extend(phrases.get("warnings", []))

    phonemes = phonemize_zh(config)
    outputs["phonemes_zh"] = str(save_zh_phonemes(config, phonemes))
    warnings.extend(phonemes.get("warnings", []))

    alignment = align_zh_notes(config)
    outputs["alignment_zh"] = str(save_zh_alignment(config, alignment))
    warnings.extend(alignment.get("warnings", []))

    score = build_opencpop_svs_score(config)
    outputs["svs_score"] = str(save_opencpop_svs_score(config, score))
    warnings.extend(score.get("warnings", []))

    export_plan = build_diffsinger_export_plan(config)
    outputs["diffsinger_export_plan"] = str(save_diffsinger_export_plan(config, export_plan))
    warnings.extend(export_plan.get("warnings", []))

    checkpoint_status = check_svs_checkpoints(config)
    rvc["enabled"] = rvc_enabled
    outputs["checkpoint_status"] = str(save_checkpoint_status(config, checkpoint_status))
    warnings.extend(checkpoint_status.get("warnings", []))
    return {"status": "success" if checkpoint_status.get("ready") else "blocked_missing_checkpoints", "outputs": outputs, "warnings": warnings}


def run_svs(config: dict) -> dict:
    svs = config.setdefault("svs", {})
    svs.setdefault("rvc", {})["enabled"] = False
    score_result = run_score(config)
    if score_result["status"] != "success":
        return score_result
    render_plan = build_neural_svs_render_plan(config)
    render_plan_path = save_neural_svs_render_plan(config, render_plan)
    return {
        "status": render_plan.get("status"),
        "outputs": {
            "render_plan": str(render_plan_path),
            "svs_vocal": str(render_plan.get("output_audio", "")),
            "infer_report": str(render_plan.get("infer_report", "")),
        },
        "warnings": render_plan.get("warnings", []),
    }


def run_svc(config: dict) -> dict:
    base_svs = PROJECT_ROOT / "data" / "svs" / "target_language_vocal_diffsinger.wav"
    fallback_svs = PROJECT_ROOT / "data" / "svs" / "target_language_vocal.wav"
    input_audio = base_svs if base_svs.exists() else fallback_svs
    return apply_rvc_voice_conversion(config, input_audio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YiYuTongSheng web demo task runner.")
    parser.add_argument("task", choices=["score", "svs", "svc"])
    parser.add_argument("--song-id", default="2001")
    parser.add_argument("--start-phrase", type=int, default=1)
    parser.add_argument("--max-phrases", type=int, default=None)
    parser.add_argument("--assembly-mode", choices=["concat", "timeline"], default="timeline")
    parser.add_argument("--normalize", default="true")
    parser.add_argument("--infer-class", default="cascade")
    parser.add_argument("--f0-naturalize", default="true")
    parser.add_argument("--voice-model", default="")
    parser.add_argument("--index-file", default="")
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--f0-up-key", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.5)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_project_config()
    _apply_song(config, args.song_id)

    if args.task == "score":
        result = run_score(config)
    elif args.task == "svs":
        _apply_svs_args(config, args)
        result = run_svs(config)
    else:
        _apply_svc_args(config, args)
        result = run_svc(config)

    _emit_result(args.task, result)
    if result.get("status") not in {"success", "ready"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
