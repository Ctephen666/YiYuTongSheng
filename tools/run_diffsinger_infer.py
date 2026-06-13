from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path


def _patch_numpy_aliases() -> None:
    try:
        import numpy as np
    except ImportError:
        return
    for name, value in {
        "int": int,
        "float": float,
        "bool": bool,
        "complex": complex,
        "object": object,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _patch_yaml_open_encoding() -> None:
    import builtins

    original_open = builtins.open

    def open_with_utf8_yaml(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if encoding is None and "b" not in mode:
            path_text = os.fspath(file) if isinstance(file, (str, os.PathLike)) else ""
            if path_text.lower().endswith((".yaml", ".yml")):
                encoding = "utf-8"
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    builtins.open = open_with_utf8_yaml


def _patch_scipy_signal_aliases() -> None:
    try:
        import scipy.signal as signal
        from scipy.signal import windows
    except ImportError:
        return
    if not hasattr(signal, "kaiser") and hasattr(windows, "kaiser"):
        signal.kaiser = windows.kaiser


def _install_optional_training_stubs() -> None:
    try:
        import webrtcvad  # noqa: F401
        return
    except ImportError:
        pass

    import types

    module = types.ModuleType("webrtcvad")

    class _Vad:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("webrtcvad is not installed; VAD trimming is unavailable during training/data preparation.")

    module.Vad = _Vad
    sys.modules["webrtcvad"] = module


def _install_pypinyin_stub() -> None:
    try:
        import pypinyin  # noqa: F401
        return
    except ImportError:
        pass

    import types

    module = types.ModuleType("pypinyin")

    class _Style:
        NORMAL = 0
        TONE3 = 8

    def _missing_pypinyin(*_args, **_kwargs):
        raise RuntimeError(
            "pypinyin is not installed. Word-level DiffSinger input is unavailable, "
            "but phoneme-level OpenCpop input does not require it."
        )

    module.Style = _Style
    module.lazy_pinyin = _missing_pypinyin
    module.pinyin = _missing_pypinyin
    sys.modules["pypinyin"] = module


def _install_diffsinger_task_stub() -> None:
    if "usr.diffsinger_task" in sys.modules:
        return

    import types
    from usr.diff.candidate_decoder import FFT
    from usr.diff.net import DiffNet

    module = types.ModuleType("usr.diffsinger_task")
    module.DIFF_DECODERS = {
        "wavenet": lambda hp: DiffNet(hp["audio_num_mel_bins"]),
        "fft": lambda hp: FFT(hp["hidden_size"], hp["dec_layers"], hp["dec_ffn_kernel_size"], hp["num_heads"]),
    }
    sys.modules["usr.diffsinger_task"] = module


def _load_infer_class(name: str):
    value = name.lower().strip()
    if value in {"e2e", "ds_e2e", "diffsinger_e2e"}:
        from inference.svs.ds_e2e import DiffSingerE2EInfer
        return DiffSingerE2EInfer
    if value in {"cascade", "ds_cascade", "diffsinger_cascade"}:
        from inference.svs.ds_cascade import DiffSingerCascadeInfer
        return DiffSingerCascadeInfer
    raise ValueError(f"Unsupported DiffSinger infer class: {name}")

def _resolve_infer_class(requested: str, hp: dict) -> tuple[str, bool, str]:
    """Resolve the infer class.

    Some old OpenCpop checkpoints ship cascade/opencs configs, but callers may
    still pass --infer-class e2e. That combination can run without crashing but
    often produces mechanical noise because the inference path does not match the
    checkpoint/config family. Prefer cascade when the loaded hparams clearly come
    from a cascade/opencs config chain.
    """
    requested_value = (requested or "e2e").lower().strip()
    base_config = hp.get("base_config", [])
    if isinstance(base_config, str):
        base_config_items = [base_config]
    else:
        base_config_items = [str(item) for item in base_config]
    config_text = " ".join(base_config_items).replace("\\", "/").lower()

    looks_cascade = ("cascade" in config_text) or ("/opencs/" in config_text)
    requested_e2e = requested_value in {"e2e", "ds_e2e", "diffsinger_e2e"}

    if requested_e2e and looks_cascade:
        reason = (
            f"requested infer_class={requested}, but hparams.base_config looks "
            f"cascade/opencs: {base_config_items}; using cascade instead"
        )
        return "cascade", True, reason

    return requested_value, False, ""




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiffSinger OpenCpop inference from a phoneme-level JSON input.")
    parser.add_argument("--diffsinger-root", required=True, help="Path to external DiffSinger source root.")
    parser.add_argument("--config", required=True, help="DiffSinger config YAML path, relative to DiffSinger root or absolute.")
    parser.add_argument("--exp-name", required=True, help="DiffSinger checkpoint experiment name under checkpoints/.")
    parser.add_argument("--input-json", required=True, help="Input JSON generated by src.svs.diffsinger_opencpop_exporter.")
    parser.add_argument("--output-wav", required=True, help="Destination wav path.")
    parser.add_argument("--infer-class", default="e2e", help="e2e or cascade.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--hparams", default="", help="Optional comma-separated DiffSinger hparams overrides.")
    parser.add_argument("--normalize", action="store_true", help="Peak-normalize saved wav.")
    parser.add_argument("--segments-dir", default="", help="Directory for phrase-level wav segments.")
    parser.add_argument("--max-phrases", type=int, default=None, help="Maximum number of phrases to infer.")
    parser.add_argument("--start-phrase", type=int, default=1, help="1-based phrase offset for low-memory tests/resume.")
    parser.add_argument("--assembly-mode", default="timeline", choices=["timeline", "concat"], help="How to assemble phrase wavs.")
    return parser.parse_args()


def _load_payloads(input_json: Path) -> tuple[list[dict], str]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("diffsinger_inputs"), list):
        return [item for item in payload["diffsinger_inputs"] if isinstance(item, dict)], "diffsinger_inputs"
    if isinstance(payload, dict):
        return [payload], "diffsinger_input"
    raise ValueError(f"Unsupported DiffSinger input JSON format: {input_json}")


def _validate_payload(payload: dict, index: int) -> None:
    required = ["text", "ph_seq", "note_seq", "note_dur_seq", "is_slur_seq", "input_type"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        name = payload.get("item_name") or f"phrase_{index}"
        raise ValueError(f"DiffSinger input {name} missing required fields: {missing}")


def _select_payloads(payloads: list[dict], start_phrase: int, max_phrases: int | None) -> list[dict]:
    start_index = max(start_phrase, 1) - 1
    selected = payloads[start_index:]
    if max_phrases is not None and max_phrases > 0:
        selected = selected[:max_phrases]
    return selected


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_wav(path: Path) -> tuple[object, int]:
    try:
        import soundfile as sf
        wav, sample_rate = sf.read(str(path), dtype="float32")
        return wav, int(sample_rate)
    except ImportError:
        from scipy.io import wavfile
        sample_rate, wav = wavfile.read(str(path))
        import numpy as np
        if wav.dtype.kind in {"i", "u"}:
            max_value = float(np.iinfo(wav.dtype).max)
            wav = wav.astype("float32") / max_value
        else:
            wav = wav.astype("float32")
        return wav, int(sample_rate)


def _to_mono(wav):
    if getattr(wav, "ndim", 1) == 2:
        return wav.mean(axis=1)
    return wav


def _assemble_concat(segment_paths: list[Path], output_wav: Path, sample_rate: int, normalize: bool) -> float:
    import numpy as np
    from utils.audio import save_wav

    parts = []
    for path in segment_paths:
        wav, sr = _read_wav(path)
        if sr != sample_rate:
            raise ValueError(f"Segment sample rate mismatch: {path} has {sr}, expected {sample_rate}")
        parts.append(_to_mono(wav))
    full = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
    save_wav(full, str(output_wav), sample_rate, norm=bool(normalize))
    return len(full) / float(sample_rate)


def _assemble_timeline(items: list[dict], segment_paths: list[Path], output_wav: Path, sample_rate: int, normalize: bool) -> float:
    import numpy as np
    from utils.audio import save_wav

    end_time = 0.0
    for item in items:
        try:
            end_time = max(end_time, float(item.get("end", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    if end_time <= 0.0:
        return _assemble_concat(segment_paths, output_wav, sample_rate, normalize)

    full = np.zeros(max(1, int(round(end_time * sample_rate))), dtype="float32")
    for item, path in zip(items, segment_paths):
        wav, sr = _read_wav(path)
        if sr != sample_rate:
            raise ValueError(f"Segment sample rate mismatch: {path} has {sr}, expected {sample_rate}")
        wav = _to_mono(wav)
        try:
            start_time = float(item.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            start_time = 0.0
        start = max(0, int(round(start_time * sample_rate)))
        end = start + len(wav)
        if end > len(full):
            full = np.pad(full, (0, end - len(full)))
        full[start:end] += wav

    peak = float(np.max(np.abs(full))) if len(full) else 0.0
    if peak > 1.0:
        full = full / peak
    save_wav(full, str(output_wav), sample_rate, norm=bool(normalize))
    return len(full) / float(sample_rate)


def main() -> None:
    args = parse_args()
    diffsinger_root = Path(args.diffsinger_root).resolve()
    if not diffsinger_root.exists():
        raise FileNotFoundError(f"DiffSinger source root does not exist: {diffsinger_root}")

    input_json = Path(args.input_json).resolve()
    output_wav = Path(args.output_wav).resolve()
    if not input_json.exists():
        raise FileNotFoundError(f"DiffSinger input JSON does not exist: {input_json}")

    _patch_numpy_aliases()
    _patch_yaml_open_encoding()
    _patch_scipy_signal_aliases()
    _install_optional_training_stubs()
    _install_pypinyin_stub()
    os.chdir(diffsinger_root)
    sys.path.insert(0, str(diffsinger_root))
    _install_diffsinger_task_stub()

    from utils.audio import save_wav
    from utils.hparams import hparams, set_hparams

    config_path = Path(args.config)
    if config_path.is_absolute():
        config_arg = str(config_path)
    else:
        config_arg = str(config_path).replace("\\", "/")

    set_hparams(
        config=config_arg,
        exp_name=args.exp_name,
        hparams_str=args.hparams,
        print_hparams=True,
    )

    payloads, input_format = _load_payloads(input_json)
    selected_payloads = _select_payloads(payloads, args.start_phrase, args.max_phrases)
    if not selected_payloads:
        raise ValueError(
            f"No DiffSinger inputs selected from {input_json}; "
            f"start_phrase={args.start_phrase}, max_phrases={args.max_phrases}."
        )
    for index, payload in enumerate(selected_payloads, start=1):
        _validate_payload(payload, index)

    infer_class_used, infer_class_auto_switched, infer_class_switch_reason = _resolve_infer_class(args.infer_class, hparams)
    if infer_class_auto_switched:
        print(f"| auto infer-class switch: {infer_class_switch_reason}")
    infer_cls = _load_infer_class(infer_class_used)
    device = None if args.device == "auto" else args.device
    infer = infer_cls(hparams, device=device)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    segment_dir = Path(args.segments_dir).resolve() if args.segments_dir else output_wav.parent / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    segment_reports = []
    segment_paths: list[Path] = []
    sample_rate = int(hparams["audio_sample_rate"])
    for index, payload in enumerate(selected_payloads, start=1):
        item_name = str(payload.get("item_name") or f"phrase_{index:03d}")
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in item_name)
        segment_wav = segment_dir / f"{safe_name}.wav"
        wav = infer.infer_once(payload)
        if wav is None:
            raise RuntimeError(f"DiffSinger inference returned None for {item_name}.")
        save_wav(wav, str(segment_wav), sample_rate, norm=False)
        segment_paths.append(segment_wav)
        segment_reports.append(
            {
                "item_name": item_name,
                "phrase_id": payload.get("phrase_id"),
                "start": payload.get("start"),
                "end": payload.get("end"),
                "output_wav": str(segment_wav),
                "phoneme_count": len(str(payload.get("ph_seq", "")).split()),
                "normalized": False,
            }
        )
        _cleanup_cuda()

    assembly_mode = args.assembly_mode
    if input_format == "diffsinger_input":
        assembly_mode = "concat"
    if assembly_mode == "timeline":
        duration_sec = _assemble_timeline(selected_payloads, segment_paths, output_wav, sample_rate, bool(args.normalize))
    else:
        duration_sec = _assemble_concat(segment_paths, output_wav, sample_rate, bool(args.normalize))

    report = {
        "status": "success",
        "output_wav": str(output_wav),
        "sample_rate": sample_rate,
        "infer_class_requested": args.infer_class,
        "infer_class": infer_class_used,
        "infer_class_auto_switched": infer_class_auto_switched,
        "infer_class_switch_reason": infer_class_switch_reason,
        "exp_name": args.exp_name,
        "input_json": str(input_json),
        "input_format": input_format,
        "phrase_count_total": len(payloads),
        "phrase_count_inferred": len(selected_payloads),
        "start_phrase": args.start_phrase,
        "max_phrases": args.max_phrases,
        "assembly_mode": assembly_mode,
        "normalize_output": bool(args.normalize),
        "duration_sec": duration_sec,
        "segments_dir": str(segment_dir),
        "segments": segment_reports,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
