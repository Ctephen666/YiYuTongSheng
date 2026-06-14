from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.web.schemas import RunRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_CONFIG_PATH = PROJECT_ROOT / "configs" / "web_demo.yaml"
PROJECT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_read_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"File does not exist: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Cannot read JSON {path}: {exc}"


def load_web_config() -> dict:
    if not WEB_CONFIG_PATH.exists():
        return {}
    if yaml is None:
        return {}
    with WEB_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _cfg_path(config: dict, key: str, default: str) -> Path:
    value = config.get("paths", {}).get(key, default) if isinstance(config.get("paths", {}), dict) else default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def web_logs_dir() -> Path:
    path = _cfg_path(load_web_config(), "web_logs_dir", "data/web_logs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def web_runtime_config_dir() -> Path:
    path = PROJECT_ROOT / "data" / "web_configs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_project_config() -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to create runtime project configs.")
    with PROJECT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Project config is not a mapping: {PROJECT_CONFIG_PATH}")
    return data


def _apply_song_to_project_config(config: dict, song_id: str) -> None:
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


def _runtime_project_config(request: RunRequest, *, rvc_enabled: bool) -> Path:
    config = _load_project_config()
    _apply_song_to_project_config(config, request.song_id)
    svs = config.setdefault("svs", {})
    svs["execute_model"] = True
    svs["dry_run"] = False
    svs["start_phrase"] = int(request.start_phrase)
    svs["max_phrases"] = int(request.max_phrases) if request.max_phrases is not None else None
    svs["assembly_mode"] = request.assembly_mode

    diffsinger = svs.setdefault("diffsinger", {})
    diffsinger["infer_class"] = request.infer_class
    diffsinger["normalize_output"] = bool(request.normalize)
    diffsinger.setdefault("f0_fallback", {})["naturalize"] = bool(request.f0_naturalize)

    rvc = svs.setdefault("rvc", {})
    rvc["enabled"] = bool(rvc_enabled)
    if request.voice_model:
        rvc["model_name"] = Path(request.voice_model).name
    if request.index_file:
        rvc["index_path"] = request.index_file
    rvc["f0method"] = request.f0_method
    rvc["f0up_key"] = int(request.f0_up_key)
    rvc["index_rate"] = float(request.index_rate)
    rvc["protect"] = float(request.protect)
    rvc["filter_radius"] = int(request.filter_radius)
    rvc["resample_sr"] = int(request.resample_sr)

    path = web_runtime_config_dir() / f"project_{request.song_id}_{'full' if rvc_enabled else 'svs'}_{_now_id()}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return path


def scan_songs() -> list[dict]:
    config = load_web_config()
    root = _cfg_path(config, "opencpop_root", "data/dataset/opencpop")
    midi_dir = root / "midis"
    wav_dir = root / "wavs"
    textgrid_dirs = [root / "TextGrid", root / "textgrids", root / "textgrid"]
    song_ids: set[str] = set()
    for directory, patterns in [
        (midi_dir, ["*.mid", "*.midi"]),
        (wav_dir, ["*.wav"]),
        *[(directory, ["*.TextGrid", "*.textgrid"]) for directory in textgrid_dirs],
    ]:
        if directory.exists():
            for pattern in patterns:
                song_ids.update(path.stem for path in directory.glob(pattern))

    songs = []
    for song_id in sorted(song_ids):
        midi = next((midi_dir / f"{song_id}{suffix}" for suffix in [".mid", ".midi"] if (midi_dir / f"{song_id}{suffix}").exists()), midi_dir / f"{song_id}.midi")
        wav = wav_dir / f"{song_id}.wav"
        textgrid = next(
            (
                directory / f"{song_id}{suffix}"
                for directory in textgrid_dirs
                for suffix in [".TextGrid", ".textgrid"]
                if (directory / f"{song_id}{suffix}").exists()
            ),
            textgrid_dirs[0] / f"{song_id}.TextGrid",
        )
        songs.append(
            {
                "song_id": song_id,
                "label": f"歌曲 {song_id}",
                "midi_exists": midi.exists(),
                "wav_exists": wav.exists(),
                "textgrid_exists": textgrid.exists(),
                "midi_path": str(midi),
                "wav_path": str(wav),
                "textgrid_path": str(textgrid),
                "audio_url": f"/audio/opencpop/{song_id}.wav" if wav.exists() else None,
            }
        )
    return songs


def _scan_files(directories: list[Path], suffix: str) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob(f"*{suffix}"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def scan_voice_models() -> list[dict]:
    config = load_web_config()
    rvc_root = _cfg_path(config, "rvc_root", "external/rvc")
    roots = [PROJECT_ROOT / "external/rvc", PROJECT_ROOT / "external/RVC", rvc_root]
    weight_dirs = [
        root / "assets" / "weights" for root in roots
    ] + [root / "weights" for root in roots] + [root / "logs" for root in roots]
    index_dirs = [
        root / "assets" / "indices" for root in roots
    ] + [root / "indices" for root in roots] + [root / "logs" for root in roots]

    pth_files = _scan_files(weight_dirs, ".pth")
    index_files = _scan_files(index_dirs, ".index")
    models = []
    for pth in pth_files:
        stem = pth.stem.lower()
        matches = [path for path in index_files if stem in path.stem.lower()]
        index_path = matches[0] if matches else None
        maybe_f0 = not any(token in stem for token in ["nof0", "no_f0", "no-f0"])
        models.append(
            {
                "name": pth.name,
                "model_name": pth.name,
                "pth_path": str(pth),
                "index_path": str(index_path) if index_path else "",
                "has_index": index_path is not None,
                "maybe_f0_model": maybe_f0,
                "warning": "" if maybe_f0 else "该模型可能不是 F0 模型，唱歌音准可能不稳定。",
            }
        )
    models.sort(key=lambda item: (0 if "_f0" in item["name"].lower() or "f0" in item["name"].lower() else 1, item["name"].lower()))
    return models


def report_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "data/svs/checkpoint_status.json",
        PROJECT_ROOT / "data/svs/diffsinger_infer_report.json",
        PROJECT_ROOT / "data/svs/diffsinger_opencpop_export_plan.json",
        PROJECT_ROOT / "data/svs/neural_svs_render_plan.json",
        PROJECT_ROOT / "data/svs/opencpop_svs_score.json",
        PROJECT_ROOT / "data/svs/rvc_voice_conversion_report.json",
    ]


def latest_reports() -> dict:
    reports = {}
    warnings = []
    for path in report_paths():
        data, error = _safe_read_json(path)
        key = path.stem
        if error:
            warnings.append(error)
            reports[key] = {"path": str(path), "exists": False}
            continue
        reports[key] = {"path": str(path), "exists": True, "summary": summarize_report(key, data), "raw": data}
    return {"reports": reports, "warnings": warnings}


def summarize_report(name: str, data: Any) -> dict:
    if not isinstance(data, dict):
        return {"type": type(data).__name__}
    runner_report = data.get("run", {}).get("runner_report", {}) if isinstance(data.get("run"), dict) else {}
    segments = runner_report.get("segments", []) if isinstance(runner_report, dict) else data.get("segments", [])
    first_segment = segments[0] if isinstance(segments, list) and segments else {}
    song_id = data.get("song_id")
    phrase_count = data.get("phrase_count") or data.get("phrase_count_inferred")
    assembly_mode = data.get("assembly_mode")
    output_audio = data.get("output_audio")
    duration_sec = data.get("duration_sec")
    if isinstance(runner_report, dict):
        song_id = song_id or runner_report.get("song_id")
        phrase_count = phrase_count or runner_report.get("phrase_count_inferred")
        assembly_mode = assembly_mode or runner_report.get("assembly_mode")
        output_audio = output_audio or runner_report.get("output_wav")
        duration_sec = duration_sec or runner_report.get("duration_sec")
    return {
        "name": name,
        "status": data.get("status"),
        "checkpoint_ready": data.get("ready"),
        "dataset": data.get("dataset"),
        "song_id": song_id,
        "phrase_count": phrase_count,
        "selected_input_types": runner_report.get("selected_input_types") if isinstance(runner_report, dict) else None,
        "infer_class": runner_report.get("infer_class") if isinstance(runner_report, dict) else None,
        "assembly_mode": assembly_mode,
        "duration_sec": duration_sec,
        "output_audio": output_audio,
        "f0_source": first_segment.get("f0_source") if isinstance(first_segment, dict) else None,
        "f0_naturalized": first_segment.get("f0_naturalized") if isinstance(first_segment, dict) else None,
        "audio_generated": data.get("audio_generated") or data.get("converted"),
    }


def current_score() -> dict:
    path = PROJECT_ROOT / "data/svs/opencpop_svs_score.json"
    data, error = _safe_read_json(path)
    return {"path": str(path), "score": data, "error": error}


def phrase_segments() -> list[dict]:
    report = latest_reports().get("reports", {}).get("diffsinger_infer_report", {}).get("raw", {})
    runner_report = report.get("run", {}).get("runner_report", {}) if isinstance(report, dict) else {}
    segments = runner_report.get("segments", []) if isinstance(runner_report, dict) else []
    return segments if isinstance(segments, list) else []


def audio_list(song_id: str | None = None) -> list[dict]:
    selected_song = song_id or load_web_config().get("defaults", {}).get("song_id", "2001")
    reference = (
        "原始 OpenCpop 参考音频",
        PROJECT_ROOT / f"data/dataset/opencpop/wavs/{selected_song}.wav",
        f"/audio/opencpop/{selected_song}.wav",
    )
    full_candidates = [
        ("完整版生成结果", PROJECT_ROOT / "data/svc/final_mix.wav", "/audio/svc/final_mix.wav"),
        ("完整版生成结果", PROJECT_ROOT / "data/svc/converted_target_voice.wav", "/audio/svc/converted_target_voice.wav"),
        ("完整版生成结果", PROJECT_ROOT / "data/svs/target_language_vocal.wav", "/audio/svs/target_language_vocal.wav"),
        ("完整版生成结果", PROJECT_ROOT / "data/svs/target_language_vocal_diffsinger.wav", "/audio/svs/target_language_vocal_diffsinger.wav"),
    ]
    full = next((item for item in full_candidates if item[1].exists()), full_candidates[0])
    return [
        {"label": reference[0], "path": str(reference[1]), "exists": reference[1].exists(), "url": reference[2] if reference[1].exists() else None},
        {"label": full[0], "path": str(full[1]), "exists": full[1].exists(), "url": full[2] if full[1].exists() else None},
    ]


def latest_log() -> dict:
    logs = sorted(web_logs_dir().glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        return {"path": None, "content": ""}
    path = logs[0]
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"path": str(path), "content": text[-20000:]}


def latest_evaluate_report() -> dict:
    path = PROJECT_ROOT / "data" / "evaluate" / "evaluate_report.json"
    data, error = _safe_read_json(path)
    if error:
        return {"exists": False, "path": str(path), "report": None, "warning": "尚未评估或报告不存在。"}
    return {"exists": True, "path": str(path), "report": data, "warning": None}


def build_score_command(request: RunRequest) -> list[str]:
    return [sys.executable, "tools/run_web_task.py", "score", "--song-id", request.song_id]


def build_svs_command(request: RunRequest) -> list[str]:
    runtime_config = _runtime_project_config(request, rvc_enabled=False)
    return [
        sys.executable,
        "app.py",
        "--config",
        str(runtime_config),
        "--step",
        "svs",
        "--target-language",
        "zh",
        "--opencpop-id",
        request.song_id,
    ]


def build_svc_command(request: RunRequest) -> list[str]:
    command = [
        sys.executable,
        "tools/run_web_task.py",
        "svc",
        "--song-id",
        request.song_id,
        "--f0-method",
        request.f0_method,
        "--f0-up-key",
        str(request.f0_up_key),
        "--index-rate",
        str(request.index_rate),
        "--protect",
        str(request.protect),
        "--filter-radius",
        str(request.filter_radius),
        "--resample-sr",
        str(request.resample_sr),
    ]
    if request.voice_model:
        command.extend(["--voice-model", request.voice_model])
    if request.index_file:
        command.extend(["--index-file", request.index_file])
    return command


def build_evaluate_command(request: RunRequest) -> list[str]:
    return [
        sys.executable,
        "tools/run_evaluate.py",
        "--config",
        "configs/evaluate.yaml",
        "--song-id",
        request.song_id,
        "--svs-wav",
        "data/svs/target_language_vocal.wav",
        "--svc-wav",
        "data/svc/converted_target_voice.wav",
        "--out-dir",
        "data/evaluate",
    ]


def build_full_pipeline_commands(request: RunRequest) -> list[list[str]]:
    runtime_config = _runtime_project_config(request, rvc_enabled=True)
    return [[
        sys.executable,
        "app.py",
        "--config",
        str(runtime_config),
        "--step",
        "svs",
        "--target-language",
        "zh",
        "--opencpop-id",
        request.song_id,
    ]]


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _active_jobs(self) -> list[dict]:
        return [job for job in self._jobs.values() if job["status"] in {"pending", "running"}]

    def can_start(self, name: str) -> tuple[bool, str | None]:
        active = self._active_jobs()
        if any(job["name"] == "full" for job in active):
            return False, "完整流程正在运行，不能重复启动其他生成任务。"
        if name == "full" and active:
            return False, "已有生成任务正在运行，不能启动完整流程。"
        if any(job["name"] == name for job in active):
            return False, f"{name} 正在运行，不能重复启动。"
        return True, None

    def start(self, name: str, commands: list[list[str]]) -> tuple[dict | None, str | None]:
        with self._lock:
            allowed, warning = self.can_start(name)
            if not allowed:
                return None, warning
            job_id = uuid.uuid4().hex[:12]
            log_path = web_logs_dir() / f"{name}_{_now_id()}.log"
            job = {
                "job_id": job_id,
                "name": name,
                "status": "pending",
                "command": " && ".join(subprocess.list2cmdline(command) for command in commands),
                "start_time": None,
                "end_time": None,
                "returncode": None,
                "log_path": str(log_path),
            }
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job_id, commands, log_path), daemon=True)
        thread.start()
        return job, None

    def _run(self, job_id: str, commands: list[list[str]], log_path: Path) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["start_time"] = _iso_now()

        final_code = 0
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            for index, command in enumerate(commands, start=1):
                log.write(f"\n===== step {index}/{len(commands)} start: {subprocess.list2cmdline(command)} =====\n")
                log.flush()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(PROJECT_ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    code = process.wait()
                except Exception as exc:  # noqa: BLE001
                    code = 1
                    log.write(f"\n[web task error] {exc}\n")
                log.write(f"\n===== step {index}/{len(commands)} end: returncode={code} =====\n")
                log.flush()
                final_code = code
                if code != 0:
                    break

        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "success" if final_code == 0 else "failed"
            job["end_time"] = _iso_now()
            job["returncode"] = final_code

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.get("start_time") or "", reverse=True)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)


job_manager = JobManager()


def open_output_directory() -> dict:
    path = PROJECT_ROOT / "data/svs"
    path.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            opened = True
        else:
            opened = False
    except Exception:
        opened = False
    return {"path": str(path), "opened": opened}
