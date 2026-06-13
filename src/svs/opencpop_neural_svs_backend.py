from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.common.io_utils import ensure_parent


def _root(config: dict) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def _resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        text = str(value).replace("\\", "/")
        if text.startswith("/data/"):
            project_relative = _root(config) / text.lstrip("/")
            if project_relative.exists():
                return project_relative
        return path
    return _root(config) / path


def _config_path(config: dict, key: str, default: str) -> Path:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs", {}), dict) else {}
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    return _resolve(config, outputs.get(key) or paths.get(key) or default)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to stage DiffSinger config files. Install pyyaml.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"DiffSinger YAML config is not a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to stage DiffSinger config files. Install pyyaml.")
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def _pretrained(config: dict) -> dict:
    return config.get("svs", {}).get("pretrained", {}) if isinstance(config.get("svs", {}), dict) else {}


def _diffsinger_config(config: dict) -> dict:
    return config.get("svs", {}).get("diffsinger", {}) if isinstance(config.get("svs", {}), dict) else {}


def _diffsinger_root(config: dict) -> Path:
    return _resolve(config, _diffsinger_config(config).get("root", "external/DiffSinger"))


def _alias_file(source: Path, target: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Missing source file for DiffSinger staging: {source}")
    ensure_parent(target)
    if target.exists():
        try:
            if target.stat().st_size == source.stat().st_size:
                return "existing"
            target.unlink()
        except OSError:
            target.unlink(missing_ok=True)

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        pass

    try:
        os.symlink(source, target)
        return "symlink"
    except OSError:
        pass

    shutil.copy2(source, target)
    return "copy"


def _clear_runtime_checkpoints(directory: Path, keep: Path) -> None:
    for candidate in directory.glob("model_ckpt_steps_*.ckpt"):
        if candidate.resolve() == keep.resolve():
            continue
        candidate.unlink()


def _stage_configs_and_checkpoints(config: dict) -> dict:
    pretrained = _pretrained(config)
    ds_cfg = _diffsinger_config(config)
    root = _diffsinger_root(config)
    if not root.exists():
        raise FileNotFoundError(f"DiffSinger source root does not exist: {root}")

    configured_acoustic_exp = str(ds_cfg.get("acoustic_exp_name", "yiyutongsheng_acoustic"))
    configured_vocoder_exp = str(ds_cfg.get("vocoder_exp_name", "yiyutongsheng_vocoder"))
    acoustic_exp = f"{configured_acoustic_exp}_runtime"
    vocoder_exp = f"{configured_vocoder_exp}_runtime"
    acoustic_dir = root / "checkpoints" / acoustic_exp
    vocoder_dir = root / "checkpoints" / vocoder_exp
    acoustic_dir.mkdir(parents=True, exist_ok=True)
    vocoder_dir.mkdir(parents=True, exist_ok=True)

    acoustic_ckpt = _resolve(config, pretrained.get("acoustic_checkpoint", "checkpoints/diffsinger/acoustic.ckpt"))
    acoustic_config = _resolve(config, pretrained.get("acoustic_config", "checkpoints/diffsinger/acoustic.yaml"))
    vocoder_ckpt = _resolve(config, pretrained.get("vocoder_checkpoint", "checkpoints/diffsinger/vocoder.ckpt"))
    vocoder_config = _resolve(config, pretrained.get("vocoder_config", "checkpoints/diffsinger/vocoder.yaml"))

    acoustic_runtime_ckpt = acoustic_dir / "model_ckpt_steps_100000.ckpt"
    vocoder_runtime_ckpt = vocoder_dir / "model_ckpt_steps_100000.ckpt"
    _clear_runtime_checkpoints(acoustic_dir, acoustic_runtime_ckpt)
    _clear_runtime_checkpoints(vocoder_dir, vocoder_runtime_ckpt)
    acoustic_link_type = _alias_file(acoustic_ckpt, acoustic_runtime_ckpt)
    vocoder_link_type = _alias_file(vocoder_ckpt, vocoder_runtime_ckpt)

    acoustic_payload = _read_yaml(acoustic_config)
    overrides = ds_cfg.get("hparams_override", {}) if isinstance(ds_cfg.get("hparams_override", {}), dict) else {}
    acoustic_payload.update(overrides)
    acoustic_payload["vocoder_ckpt"] = f"checkpoints/{vocoder_exp}"
    acoustic_payload["pe_enable"] = bool(overrides.get("pe_enable", False))
    acoustic_payload["use_nsf"] = bool(overrides.get("use_nsf", False))
    if ds_cfg.get("pndm_speedup") is not None:
        acoustic_payload["pndm_speedup"] = int(ds_cfg["pndm_speedup"])
    _write_yaml(acoustic_dir / "config.yaml", acoustic_payload)

    vocoder_payload = _read_yaml(vocoder_config)
    _write_yaml(vocoder_dir / "config.yaml", vocoder_payload)

    return {
        "diffsinger_root": str(root),
        "configured_acoustic_exp_name": configured_acoustic_exp,
        "configured_vocoder_exp_name": configured_vocoder_exp,
        "acoustic_exp_name": acoustic_exp,
        "vocoder_exp_name": vocoder_exp,
        "acoustic_dir": str(acoustic_dir),
        "vocoder_dir": str(vocoder_dir),
        "acoustic_checkpoint_link": acoustic_link_type,
        "vocoder_checkpoint_link": vocoder_link_type,
        "acoustic_config": str(acoustic_dir / "config.yaml"),
        "vocoder_config": str(vocoder_dir / "config.yaml"),
    }


def _run_inference_subprocess(config: dict, staged: dict, input_json: Path, output_audio: Path) -> dict:
    ds_cfg = _diffsinger_config(config)
    runner = _root(config) / "tools" / "run_diffsinger_infer.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing DiffSinger runner script: {runner}")

    device = str(config.get("svs", {}).get("device", "auto"))
    if bool(ds_cfg.get("force_cpu", False)):
        device = "cpu"
    normalize = bool(ds_cfg.get("normalize_output", True))
    infer_class = str(ds_cfg.get("infer_class", "e2e"))
    timeout = int(config.get("svs", {}).get("inference_timeout_sec", 1800))
    max_phrases = config.get("svs", {}).get("max_phrases")
    start_phrase = int(config.get("svs", {}).get("start_phrase", 1) or 1)
    assembly_mode = str(config.get("svs", {}).get("assembly_mode", "timeline") or "timeline")

    root = Path(staged["diffsinger_root"])
    command = [
        sys.executable,
        str(runner),
        "--diffsinger-root",
        str(root),
        "--config",
        f"checkpoints/{staged['acoustic_exp_name']}/config.yaml",
        "--exp-name",
        staged["acoustic_exp_name"],
        "--input-json",
        str(input_json),
        "--output-wav",
        str(output_audio),
        "--infer-class",
        infer_class,
        "--device",
        device,
        "--segments-dir",
        str(output_audio.parent / "segments"),
        "--start-phrase",
        str(start_phrase),
        "--assembly-mode",
        assembly_mode,
    ]
    if max_phrases is not None:
        command.extend(["--max-phrases", str(int(max_phrases))])
    if normalize:
        command.append("--normalize")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        command,
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    runner_report = None
    for line in reversed(result.stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            runner_report = json.loads(text)
            break
        except json.JSONDecodeError:
            continue

    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timeout_sec": timeout,
        "runner_report": runner_report,
    }


def _has_diffsinger_input(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    inputs = payload.get("diffsinger_inputs")
    if isinstance(inputs, list):
        return any(isinstance(item, dict) and bool(item.get("ph_seq")) for item in inputs)
    return bool(payload.get("ph_seq"))


def build_neural_svs_render_plan(config: dict) -> dict:
    score_path = _config_path(config, "svs_score", "data/svs/opencpop_svs_score.json")
    input_json = _config_path(config, "diffsinger_input", "data/svs/diffsinger_opencpop/diffsinger_input_2001.json")
    checkpoint_status_path = _config_path(config, "checkpoint_status", "data/svs/checkpoint_status.json")
    output_audio = _config_path(config, "svs_vocal", "data/svs/target_language_vocal.wav")
    report_path = _config_path(config, "infer_report", "data/svs/diffsinger_infer_report.json")

    score = _read_json(score_path, {})
    checkpoint_status = _read_json(checkpoint_status_path, {})
    warnings: list[str] = []
    if isinstance(score, dict):
        warnings.extend(score.get("warnings", []))
    if isinstance(checkpoint_status, dict):
        warnings.extend(checkpoint_status.get("warnings", []))

    execute_model = bool(config.get("svs", {}).get("execute_model", False)) and not bool(config.get("svs", {}).get("dry_run", False))
    if not execute_model:
        return {
            "format": "neural_svs_render_plan.v2",
            "dry_run": True,
            "executed": False,
            "audio_generated": False,
            "status": "planned_not_executed",
            "score_input": str(score_path),
            "diffsinger_input": str(input_json),
            "planned_output_audio": str(output_audio),
            "warnings": list(dict.fromkeys(warnings)),
            "why_not_rendered": "svs.execute_model is false or svs.dry_run is true.",
        }

    if not checkpoint_status.get("ready"):
        return {
            "format": "neural_svs_render_plan.v2",
            "dry_run": False,
            "executed": False,
            "audio_generated": False,
            "status": "blocked_missing_checkpoints",
            "score_input": str(score_path),
            "diffsinger_input": str(input_json),
            "planned_output_audio": str(output_audio),
            "warnings": list(dict.fromkeys(warnings)),
            "why_not_rendered": "checkpoint_status.ready is false.",
        }

    if not input_json.exists():
        raise FileNotFoundError(f"Missing DiffSinger input JSON: {input_json}")
    payload = _read_json(input_json, {})
    if not _has_diffsinger_input(payload):
        raise RuntimeError(f"DiffSinger input JSON has no valid phrase inputs: {input_json}")

    staged = _stage_configs_and_checkpoints(config)
    run_result = _run_inference_subprocess(config, staged, input_json, output_audio)
    audio_generated = output_audio.exists() and output_audio.stat().st_size > 44
    status = "success" if run_result["returncode"] == 0 and audio_generated else "failed"

    report = {
        "format": "diffsinger_infer_report.v1",
        "status": status,
        "audio_generated": audio_generated,
        "output_audio": str(output_audio),
        "score_input": str(score_path),
        "diffsinger_input": str(input_json),
        "staged": staged,
        "run": run_result,
        "assembly_mode": run_result.get("runner_report", {}).get("assembly_mode") if isinstance(run_result.get("runner_report"), dict) else None,
        "warnings": list(dict.fromkeys(warnings)),
    }
    _write_json(report_path, report)

    if status != "success":
        raise RuntimeError(
            "DiffSinger inference failed. See report: "
            f"{report_path}. Return code={run_result['returncode']}"
        )

    return {
        "format": "neural_svs_render_plan.v2",
        "dry_run": False,
        "executed": True,
        "audio_generated": True,
        "status": "success",
        "backend": "diffsinger_opencpop",
        "language": "zh",
        "dataset": "opencpop",
        "song_id": score.get("song_id") if isinstance(score, dict) else None,
        "score_input": str(score_path),
        "diffsinger_input": str(input_json),
        "checkpoint_status": str(checkpoint_status_path),
        "output_audio": str(output_audio),
        "infer_report": str(report_path),
        "staged": staged,
        "warnings": list(dict.fromkeys(warnings)),
    }


def save_neural_svs_render_plan(config: dict, payload: dict) -> Path:
    output_path = _config_path(config, "render_plan", "data/svs/neural_svs_render_plan.json")
    _write_json(output_path, payload)
    return output_path


class OpenCpopNeuralSvsBackend:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = build_neural_svs_render_plan(self.config)
        output_path = save_neural_svs_render_plan(self.config, payload)
        outputs = {"render_plan": str(output_path)}
        if payload.get("output_audio"):
            outputs["svs_vocal"] = payload["output_audio"]
        if payload.get("infer_report"):
            outputs["infer_report"] = payload["infer_report"]
        return {
            "status": payload.get("status", "failed"),
            "outputs": outputs,
            "warnings": payload.get("warnings", []),
            "message": "DiffSinger inference completed." if payload.get("audio_generated") else payload.get("why_not_rendered"),
        }
