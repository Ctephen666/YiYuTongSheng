from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent


def _root(config: dict) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def _resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _root(config) / path


def _config_path(config: dict, key: str, default: str) -> Path:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs", {}), dict) else {}
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    return _resolve(config, outputs.get(key) or paths.get(key) or default)


def _write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _flatten_checkpoint_config(data: Any, prefix: str = "") -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_checkpoint_config(value, next_prefix))
    elif isinstance(data, str) and data.strip():
        result.append((prefix or "checkpoint", data.strip()))
    return result


def _auto_resolve_phoneme_dictionary(path: Path) -> tuple[Path, str | None]:
    if path.exists():
        return path, None

    alternates = []
    if path.suffix.lower() == ".txt":
        alternates.append(path.with_suffix(".json"))
    elif path.suffix.lower() == ".json":
        alternates.append(path.with_suffix(".txt"))
    else:
        alternates.extend([path.with_suffix(".json"), path.with_suffix(".txt")])

    for candidate in alternates:
        if candidate.exists():
            return candidate, f"Auto-adapted phoneme dictionary path from {path} to {candidate}."
    return path, None


def _load_phone_dict(path: Path) -> tuple[int | None, str | None]:
    if not path.exists():
        return None, None
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data), "json_list"
            if isinstance(data, dict):
                return len(data), "json_dict"
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return len(lines), "text_lines"
    except Exception:
        return None, "unreadable"
    return None, None


def _auto_rvc_index_path(rvc_root: Path, model_name: str) -> Path | None:
    stem = Path(model_name).stem.lower()
    indices_dir = rvc_root / "assets" / "indices"
    if not indices_dir.exists():
        return None
    candidates = [path for path in indices_dir.glob("*.index") if stem in path.name.lower()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.name), path.name.lower()))
    return candidates[0]


def check_svs_checkpoints(config: dict) -> dict:
    svs_config = config.get("svs", {}) if isinstance(config.get("svs", {}), dict) else {}
    pretrained = svs_config.get("pretrained", {}) if isinstance(svs_config.get("pretrained", {}), dict) else {}
    checks = []
    warnings = []

    for name, value in _flatten_checkpoint_config(pretrained):
        path = _resolve(config, value)
        adapted_from = None
        if name.endswith("phoneme_dictionary") or name == "phoneme_dictionary":
            resolved_path, adaptation = _auto_resolve_phoneme_dictionary(path)
            if adaptation:
                warnings.append(adaptation)
                adapted_from = str(path)
            path = resolved_path

        exists = path.exists()
        check = {"name": name, "path": str(path), "exists": exists}
        if adapted_from:
            check["adapted_from"] = adapted_from
        if name.endswith("phoneme_dictionary") or name == "phoneme_dictionary":
            phone_count, phone_format = _load_phone_dict(path)
            check["phone_count"] = phone_count
            check["phone_format"] = phone_format
        checks.append(check)
        if not exists:
            warnings.append(f"Missing SVS checkpoint/config asset: {name} -> {path}")

    diffsinger_root_value = svs_config.get("diffsinger", {}).get("root") if isinstance(svs_config.get("diffsinger", {}), dict) else None
    if diffsinger_root_value:
        diffsinger_root = _resolve(config, diffsinger_root_value)
        required_dirs = ["inference", "modules", "usr", "utils", "vocoders"]
        missing = [name for name in required_dirs if not (diffsinger_root / name).exists()]
        checks.append({
            "name": "diffsinger.root",
            "path": str(diffsinger_root),
            "exists": diffsinger_root.exists() and not missing,
            "missing_subdirs": missing,
        })
        if missing:
            warnings.append(f"DiffSinger source root is incomplete: {diffsinger_root}; missing={missing}")

    rvc_config = svs_config.get("rvc", {}) if isinstance(svs_config.get("rvc", {}), dict) else {}
    if bool(rvc_config.get("enabled", False)):
        rvc_root = _resolve(config, rvc_config.get("root", "external/rvc"))
        model_name = str(rvc_config.get("model_name", "bofan_voice.pth") or "bofan_voice.pth")
        model_path = rvc_root / "assets" / "weights" / model_name
        index_value = str(rvc_config.get("index_path", "") or "").strip()
        index_path = _resolve(config, index_value) if index_value else _auto_rvc_index_path(rvc_root, model_name)
        hubert_path = rvc_root / "assets" / "hubert" / "hubert_base.pt"
        rvc_checks = [
            ("rvc.root", rvc_root),
            ("rvc.model", model_path),
            ("rvc.index", index_path),
            ("rvc.hubert", hubert_path),
        ]
        if str(rvc_config.get("f0method", "")).lower().strip() == "rmvpe":
            rvc_checks.append(("rvc.rmvpe", rvc_root / "assets" / "rmvpe" / "rmvpe.pt"))
        for name, path in rvc_checks:
            exists = bool(path and path.exists())
            checks.append({"name": name, "path": str(path) if path else None, "exists": exists})
            if not exists:
                warnings.append(f"Missing RVC asset: {name} -> {path}")

    if not checks:
        warnings.append("No svs.pretrained checkpoint paths are configured.")

    ready = bool(checks) and all(item["exists"] for item in checks)
    return {
        "format": "checkpoint_status.v2",
        "dry_run": False,
        "downloaded": False,
        "ready": ready,
        "checks": checks,
        "warnings": warnings,
        "next_action": (
            "Ready for DiffSinger inference."
            if ready
            else "Fix missing checkpoint/config/source paths before running model inference."
        ),
    }


def save_checkpoint_status(config: dict, payload: dict) -> Path:
    output_path = _config_path(config, "checkpoint_status", "data/svs/checkpoint_status.json")
    _write_json(output_path, payload)
    return output_path


class CheckpointManager:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = check_svs_checkpoints(self.config)
        output_path = save_checkpoint_status(self.config, payload)
        status = "ready" if payload.get("ready") else "blocked_missing_checkpoints"
        return {
            "status": status,
            "outputs": {"checkpoint_status": str(output_path)},
            "warnings": payload.get("warnings", []),
            "message": payload.get("next_action"),
        }
