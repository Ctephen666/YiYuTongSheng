from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.common.io_utils import ensure_parent


FALLBACK_PHONE_LIST = [
    "AP", "SP", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g",
    "h", "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "j", "k", "l", "m", "n", "o",
    "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v",
    "van", "ve", "vn", "w", "x", "y", "z", "zh",
]


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


def _load_phone_dict(config: dict) -> tuple[list[str], Path | None, list[str]]:
    warnings: list[str] = []
    pretrained = config.get("svs", {}).get("pretrained", {}) if isinstance(config.get("svs", {}), dict) else {}
    value = pretrained.get("phoneme_dictionary") or "checkpoints/diffsinger/zh_phoneme_dict.json"
    path = _resolve(config, value)
    if not path.exists() and path.suffix.lower() == ".txt" and path.with_suffix(".json").exists():
        warnings.append(f"Auto-adapted phoneme dictionary path from {path} to {path.with_suffix('.json')}.")
        path = path.with_suffix(".json")
    if not path.exists() and path.suffix.lower() == ".json" and path.with_suffix(".txt").exists():
        warnings.append(f"Auto-adapted phoneme dictionary path from {path} to {path.with_suffix('.txt')}.")
        path = path.with_suffix(".txt")

    if not path.exists():
        warnings.append(f"Phoneme dictionary not found; using built-in OpenCpop phone list: {path}")
        return FALLBACK_PHONE_LIST, None, warnings

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data], path, warnings
        if isinstance(data, dict):
            return [str(key) for key in data.keys()], path, warnings

    phones = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return phones, path, warnings


def _diffsinger_root(config: dict) -> Path:
    value = config.get("svs", {}).get("diffsinger", {}).get("root", "external/DiffSinger")
    return _resolve(config, value)


def _load_pinyin_map(config: dict) -> dict[str, str]:
    mapping = {"AP": "AP", "SP": "SP"}
    table_path = _diffsinger_root(config) / "inference" / "svs" / "opencpop" / "cpop_pinyin2ph.txt"
    if not table_path.exists():
        return mapping
    for line in table_path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def _normalise_pinyin(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[1-5]", "", text)
    text = text.replace("u:", "v").replace("ü", "v")
    if len(text) >= 2 and text[0] in "jqxy" and text[1] == "v":
        text = text[0] + "u" + text[2:]
    return text


def _pinyin_from_char(char: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return ""
    result = lazy_pinyin(char, style=Style.NORMAL, errors="ignore")
    return _normalise_pinyin(result[0]) if result else ""


def _phones_from_note(note: dict, pinyin_map: dict[str, str], phone_vocab: set[str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    raw = str(note.get("phoneme") or note.get("pinyin") or "").strip()
    char = str(note.get("lyric") or "").strip()

    candidates = [_normalise_pinyin(raw)]
    if char and char not in {"AP", "SP"}:
        candidates.append(_pinyin_from_char(char))
    if raw in {"AP", "SP"} or char in {"AP", "SP"}:
        candidates.insert(0, raw or char)

    for candidate in candidates:
        if not candidate:
            continue
        if candidate in pinyin_map:
            phones = pinyin_map[candidate].split()
            unknown = [phone for phone in phones if phone not in phone_vocab]
            if unknown:
                warnings.append(f"Phone(s) not in dictionary for pinyin {candidate}: {unknown}")
            return phones, warnings

        phones = candidate.split()
        if phones and all(phone in phone_vocab for phone in phones):
            return phones, warnings

    warnings.append(f"Could not map lyric={char!r}, phoneme={raw!r} to DiffSinger phones.")
    return [], warnings


def _midi_to_note_name(value: Any) -> str:
    try:
        midi = int(value)
    except (TypeError, ValueError):
        return "rest"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _note_pitch(note: dict) -> str:
    if note.get("is_rest"):
        return "rest"
    pitch = str(note.get("pitch") or "").strip()
    if pitch and pitch.lower() not in {"none", "null"}:
        return pitch
    return _midi_to_note_name(note.get("midi"))


def _duration_text(note: dict) -> str:
    try:
        duration = float(note.get("duration"))
    except (TypeError, ValueError):
        try:
            duration = float(note.get("end")) - float(note.get("start"))
        except (TypeError, ValueError):
            duration = 0.0
    return f"{max(duration, 0.001):.6f}"


def _build_diffsinger_input(score: dict, pinyin_map: dict[str, str], phone_vocab: set[str]) -> tuple[dict, list[dict], list[str]]:
    warnings: list[str] = []
    phone_seq: list[str] = []
    note_seq: list[str] = []
    dur_seq: list[str] = []
    slur_seq: list[str] = []
    rows: list[dict] = []
    text_chars: list[str] = []
    last_phones: list[str] = []

    for phrase in score.get("phrases", []) if isinstance(score, dict) else []:
        if not isinstance(phrase, dict):
            continue
        for note in phrase.get("notes", []):
            if not isinstance(note, dict):
                continue
            pitch = _note_pitch(note)
            duration = _duration_text(note)

            if note.get("is_rest") or pitch == "rest":
                phones = ["SP"]
                slur = "0"
            elif note.get("is_slur") and last_phones:
                phones = [last_phones[-1]]
                slur = "1"
            else:
                phones, phone_warnings = _phones_from_note(note, pinyin_map, phone_vocab)
                warnings.extend(phone_warnings)
                if not phones:
                    continue
                last_phones = phones
                slur = "0"
                lyric = str(note.get("lyric") or "")
                if lyric and lyric not in {"AP", "SP"}:
                    text_chars.append(lyric)

            for phone in phones:
                phone_seq.append(phone)
                note_seq.append(pitch)
                dur_seq.append(duration)
                slur_seq.append(slur)
                rows.append(
                    {
                        "phone": phone,
                        "note": pitch,
                        "duration": duration,
                        "is_slur": slur,
                        "lyric": note.get("lyric", ""),
                    }
                )

    if not phone_seq:
        warnings.append("No DiffSinger phoneme-level tokens were generated from opencpop_svs_score.json.")

    inp = {
        "item_name": f"opencpop_{score.get('song_id') or 'sample'}",
        "text": "".join(text_chars),
        "ph_seq": " ".join(phone_seq),
        "note_seq": " ".join(note_seq),
        "note_dur_seq": " ".join(dur_seq),
        "is_slur_seq": " ".join(slur_seq),
        "input_type": "phoneme",
    }

    lengths = {"phones": len(phone_seq), "notes": len(note_seq), "durations": len(dur_seq), "slurs": len(slur_seq)}
    if len(set(lengths.values())) != 1:
        warnings.append(f"DiffSinger input length mismatch: {lengths}")
    return inp, rows, warnings


def build_diffsinger_export_plan(config: dict) -> dict:
    score_path = _config_path(config, "svs_score", "data/svs/opencpop_svs_score.json")
    score = _read_json(score_path, {})
    diffsinger_config = config.get("svs", {}).get("diffsinger", {}) if isinstance(config.get("svs", {}), dict) else {}

    target_dir = diffsinger_config.get("export_dir") or "data/svs/diffsinger_opencpop"
    target_dir_path = _resolve(config, target_dir)
    input_path = _config_path(config, "diffsinger_input", f"data/svs/diffsinger_opencpop/diffsinger_input_{score.get('song_id', 'sample')}.json")
    phoneme_tsv_path = _config_path(config, "diffsinger_phoneme_tsv", f"data/svs/diffsinger_opencpop/phonemes_{score.get('song_id', 'sample')}.tsv")
    notes_csv_path = _config_path(config, "diffsinger_notes_csv", f"data/svs/diffsinger_opencpop/notes_{score.get('song_id', 'sample')}.csv")

    phones, phone_dict_path, dict_warnings = _load_phone_dict(config)
    pinyin_map = _load_pinyin_map(config)
    diffsinger_input, rows, input_warnings = _build_diffsinger_input(score, pinyin_map, set(phones))

    warnings = []
    if isinstance(score, dict):
        warnings.extend(score.get("warnings", []))
    warnings.extend(dict_warnings)
    warnings.extend(input_warnings)

    return {
        "format": "diffsinger_opencpop_export_plan.v2",
        "dry_run": False,
        "executed": True,
        "dataset": "opencpop",
        "song_id": score.get("song_id") if isinstance(score, dict) else None,
        "language": "zh",
        "score_input": str(score_path),
        "target_export_dir": str(target_dir_path),
        "diffsinger_input": diffsinger_input,
        "phoneme_rows": rows,
        "phone_dictionary": str(phone_dict_path) if phone_dict_path else "built_in",
        "phone_count": len(phones),
        "expected_files": {
            "input_json": str(input_path),
            "phoneme_tsv": str(phoneme_tsv_path),
            "notes_csv": str(notes_csv_path),
            "score_json": str(score_path),
        },
        "diffsinger_config": diffsinger_config,
        "warnings": list(dict.fromkeys(warnings)),
    }


def save_diffsinger_export_plan(config: dict, payload: dict) -> Path:
    input_path = Path(payload["expected_files"]["input_json"])
    phoneme_tsv_path = Path(payload["expected_files"]["phoneme_tsv"])
    notes_csv_path = Path(payload["expected_files"]["notes_csv"])

    _write_json(input_path, payload.get("diffsinger_input", {}))

    ensure_parent(phoneme_tsv_path)
    with phoneme_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("index\tphone\tnote\tduration\tis_slur\tlyric\n")
        for index, row in enumerate(payload.get("phoneme_rows", [])):
            handle.write(
                f"{index}\t{row.get('phone','')}\t{row.get('note','')}\t{row.get('duration','')}\t{row.get('is_slur','')}\t{row.get('lyric','')}\n"
            )

    ensure_parent(notes_csv_path)
    with notes_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "phone", "note", "duration", "is_slur", "lyric"])
        writer.writeheader()
        for index, row in enumerate(payload.get("phoneme_rows", [])):
            writer.writerow({"index": index, **row})

    output_path = _config_path(config, "diffsinger_export_plan", "data/svs/diffsinger_opencpop_export_plan.json")
    plan_payload = dict(payload)
    plan_payload.pop("phoneme_rows", None)
    _write_json(output_path, plan_payload)
    return output_path


class DiffSingerOpenCpopExporter:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> dict:
        payload = build_diffsinger_export_plan(self.config)
        output_path = save_diffsinger_export_plan(self.config, payload)
        generated = payload.get("expected_files", {})
        status = "success" if payload.get("diffsinger_input", {}).get("ph_seq") else "warning"
        return {
            "status": status,
            "outputs": {
                "diffsinger_export_plan": str(output_path),
                "diffsinger_input": generated.get("input_json"),
                "diffsinger_phoneme_tsv": generated.get("phoneme_tsv"),
                "diffsinger_notes_csv": generated.get("notes_csv"),
            },
            "warnings": payload.get("warnings", []),
            "message": "Prepared DiffSinger phoneme-level inference input.",
        }
