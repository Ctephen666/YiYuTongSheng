
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent


def _root(config: dict) -> Path:
    return Path(config.get('_project_root', '.')).resolve()


def _resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _root(config) / path


def _config_path(config: dict, key: str, default: str) -> Path:
    outputs = config.get('outputs', {}) if isinstance(config.get('outputs', {}), dict) else {}
    paths = config.get('paths', {}) if isinstance(config.get('paths', {}), dict) else {}
    return _resolve(config, outputs.get(key) or paths.get(key) or default)


def _write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _svs_config(config: dict) -> dict:
    value = config.get('svs', {})
    return value if isinstance(value, dict) else {}


def _rvc_config(config: dict) -> dict:
    value = _svs_config(config).get('rvc', {})
    return value if isinstance(value, dict) else {}


def _auto_index_path(rvc_root: Path, model_name: str) -> Path | None:
    stem = Path(model_name).stem.lower()
    indices_dir = rvc_root / 'assets' / 'indices'
    if not indices_dir.exists():
        return None
    candidates = [path for path in indices_dir.glob('*.index') if stem in path.name.lower()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.name), path.name.lower()))
    return candidates[0]


def _copy_input_for_rvc(input_audio: Path, source_audio: Path) -> None:
    if not input_audio.exists():
        raise FileNotFoundError(f'Missing SVS vocal for RVC conversion: {input_audio}')
    ensure_parent(source_audio)
    if input_audio.resolve() != source_audio.resolve():
        shutil.copy2(input_audio, source_audio)


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def apply_rvc_voice_conversion(config: dict, input_audio: str | Path | None = None) -> dict:
    rvc_cfg = _rvc_config(config)
    if not bool(rvc_cfg.get('enabled', False)):
        return {
            'status': 'skipped',
            'enabled': False,
            'message': 'RVC voice conversion is disabled.',
            'outputs': {},
            'warnings': [],
        }

    rvc_root = _resolve(config, rvc_cfg.get('root', 'external/rvc'))

    # IMPORTANT: keep the pure DiffSinger/SVS output separate from the RVC/SVC output.
    # The old implementation moved the RVC temp file back to svs_vocal and then copied it
    # to svc_vocal, making data/svs/target_language_vocal.wav and
    # data/svc/converted_target_voice.wav identical after conversion.
    svs_audio = _config_path(config, 'svs_vocal', 'data/svs/target_language_vocal.wav')
    source_audio = _config_path(config, 'svs_vocal_diffsinger', 'data/svs/target_language_vocal_diffsinger.wav')
    svc_audio = _config_path(config, 'svc_vocal', 'data/svc/converted_target_voice.wav')
    report_path = _config_path(config, 'rvc_report', 'data/svc/rvc_convert_report.json')
    legacy_report_path = _config_path(config, 'rvc_legacy_report', 'data/svs/rvc_voice_conversion_report.json')
    temp_output = svc_audio.with_name(f'{svc_audio.stem}_tmp{svc_audio.suffix}')

    input_path = _resolve(config, input_audio) if input_audio else (source_audio if source_audio.exists() else svs_audio)
    model_name = str(rvc_cfg.get('model_name', 'bofan_voice.pth') or 'bofan_voice.pth')
    model_path = rvc_root / 'assets' / 'weights' / model_name

    if not rvc_root.exists():
        raise FileNotFoundError(f'RVC root does not exist: {rvc_root}')
    if not model_path.exists():
        raise FileNotFoundError(f'RVC model does not exist: {model_path}')

    index_value = str(rvc_cfg.get('index_path', '') or '').strip()
    index_path = _resolve(config, index_value) if index_value else _auto_index_path(rvc_root, model_name)
    if index_path is None or not index_path.exists():
        raise FileNotFoundError(
            'RVC index file was not found. Set svs.rvc.index_path or put a matching .index file under external/rvc/assets/indices.'
        )

    _copy_input_for_rvc(input_path, source_audio)
    ensure_parent(svc_audio)
    if temp_output.exists():
        temp_output.unlink()

    python_exe = str(rvc_cfg.get('python', '') or sys.executable)
    device = str(rvc_cfg.get('device') or _svs_config(config).get('device') or 'auto')
    if device == 'auto':
        device = ''

    command = [
        python_exe,
        'tools/infer_cli.py',
        '--input_path',
        str(source_audio),
        '--opt_path',
        str(temp_output),
        '--model_name',
        model_name,
        '--index_path',
        str(index_path),
        '--f0method',
        str(rvc_cfg.get('f0method', 'harvest') or 'harvest'),
        '--f0up_key',
        str(int(rvc_cfg.get('f0up_key', 0) or 0)),
        '--index_rate',
        str(float(rvc_cfg.get('index_rate', 0.66) or 0.66)),
        '--filter_radius',
        str(int(rvc_cfg.get('filter_radius', 3) or 3)),
        '--resample_sr',
        str(int(rvc_cfg.get('resample_sr', 0) or 0)),
        '--rms_mix_rate',
        str(float(rvc_cfg.get('rms_mix_rate', 1.0) or 1.0)),
        '--protect',
        str(float(rvc_cfg.get('protect', 0.33) or 0.33)),
    ]
    if device:
        command.extend(['--device', device])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(rvc_root) + os.pathsep + env.get('PYTHONPATH', '')
    timeout = int(rvc_cfg.get('timeout_sec', _svs_config(config).get('inference_timeout_sec', 1800)) or 1800)
    result = subprocess.run(
        command,
        cwd=str(rvc_root),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    converted = temp_output.exists() and temp_output.stat().st_size > 44
    status = 'success' if result.returncode == 0 and converted else 'failed'
    warnings: list[str] = []
    source_hash = _sha256_file(source_audio)
    output_hash = _sha256_file(temp_output) if converted else None
    identical_to_source = converted and source_hash is not None and output_hash is not None and source_hash == output_hash

    if status == 'success':
        if identical_to_source:
            status = 'failed'
            warnings.append('RVC produced an output identical to the SVS source; treating conversion as failed.')
        else:
            if svc_audio.exists():
                svc_audio.unlink()
            shutil.move(str(temp_output), str(svc_audio))

    report = {
        'format': 'rvc_voice_conversion_report.v2',
        'status': status,
        'enabled': True,
        'converted': converted and status == 'success',
        'input_audio': str(source_audio),
        'svs_audio_preserved': str(svs_audio),
        'output_audio': str(svc_audio),
        'svc_audio': str(svc_audio),
        'temp_output_audio': str(temp_output),
        'rvc_root': str(rvc_root),
        'model_name': model_name,
        'model_path': str(model_path),
        'index_path': str(index_path),
        'source_sha256': source_hash,
        'temp_output_sha256': output_hash,
        'identical_to_source': bool(identical_to_source),
        'command': command,
        'returncode': result.returncode,
        'stdout_tail': result.stdout[-8000:],
        'stderr_tail': result.stderr[-8000:],
        'warnings': warnings,
    }
    _write_json(report_path, report)
    # Keep compatibility with older web report readers that still look under data/svs.
    _write_json(legacy_report_path, report)

    if status != 'success':
        raise RuntimeError(f'RVC voice conversion failed. See report: {report_path}. Return code={result.returncode}')

    return {
        'status': 'success',
        'enabled': True,
        'outputs': {
            'svs_vocal': str(svs_audio),
            'svs_vocal_diffsinger': str(source_audio),
            'svc_vocal': str(svc_audio),
            'rvc_report': str(report_path),
        },
        'warnings': warnings,
        'message': 'RVC voice conversion completed.',
        'report': report,
    }


class RvcVoiceConversionRunner:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        return apply_rvc_voice_conversion(self.config)
