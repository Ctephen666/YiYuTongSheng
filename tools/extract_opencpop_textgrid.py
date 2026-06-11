import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from praatio import textgrid


REST_LABELS = {"SP", "AP", "sil", "silence", "rest", ""}

NOTE_BASE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def entry_to_tuple(entry: Any) -> Tuple[float, float, str]:
    """
    兼容不同 praatio 版本的 entry 表示。
    有些版本 entry 是 tuple，有些版本是 Interval 对象。
    """
    if hasattr(entry, "start") and hasattr(entry, "end") and hasattr(entry, "label"):
        return float(entry.start), float(entry.end), str(entry.label)

    start, end, label = entry
    return float(start), float(end), str(label)


def clean_label(label: str) -> str:
    return label.strip().replace(" ", "").replace("\n", "")


def is_rest(label: str) -> bool:
    return clean_label(label) in REST_LABELS


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def pitch_to_midi(pitch: str) -> Optional[int]:
    """
    将 D3、A#3、C4 这样的音高转成 MIDI number。
    rest 返回 None。
    """
    pitch = clean_label(pitch)

    if pitch.lower() == "rest" or pitch in REST_LABELS:
        return None

    match = re.match(r"^([A-G](?:#|b)?)(-?\d+)$", pitch)
    if not match:
        return None

    name = match.group(1)
    octave = int(match.group(2))

    if name not in NOTE_BASE:
        return None

    return (octave + 1) * 12 + NOTE_BASE[name]


def get_tier_entries(tg: textgrid.Textgrid, tier_name: str) -> List[Tuple[float, float, str]]:
    tier = tg.getTier(tier_name)
    return [entry_to_tuple(e) for e in tier.entries]


def find_textgrid_path(path: Path) -> Path:
    if path.exists():
        return path

    parent = path.parent
    stem = path.stem

    for suffix in [".TextGrid", ".textgrid", ".TEXTGRID"]:
        candidate = parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"TextGrid not found: {path}")


def extract_sentence_phrases(sentence_entries: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    phrases = []

    for start, end, label in sentence_entries:
        label = clean_label(label)

        if is_rest(label):
            continue

        if label.lower() == "silence":
            continue

        if not contains_chinese(label):
            continue

        phrases.append({
            "id": len(phrases) + 1,
            "zh": label,
            "start": round(float(start), 6),
            "end": round(float(end), 6),
            "note_start": 0,
            "note_end": 0,
        })

    return phrases


def build_notes(
    char_entries: List[Tuple[float, float, str]],
    syllable_entries: List[Tuple[float, float, str]],
    pitch_entries: List[Tuple[float, float, str]],
    duration_entries: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """
    从 汉字 / 音节 / 音高 / 音长 四个 tier 中构造 note。
    只保留非 SP/AP/rest 的真实演唱字。
    """
    n = min(
        len(char_entries),
        len(syllable_entries),
        len(pitch_entries),
        len(duration_entries),
    )

    notes = []

    for i in range(n):
        char_start, char_end, char_label = char_entries[i]
        _, _, syllable_label = syllable_entries[i]
        _, _, pitch_label = pitch_entries[i]
        _, _, duration_label = duration_entries[i]

        char_label = clean_label(char_label)
        syllable_label = clean_label(syllable_label)
        pitch_label = clean_label(pitch_label)

        if is_rest(char_label):
            continue

        if is_rest(syllable_label):
            continue

        if pitch_label.lower() == "rest":
            continue

        midi = pitch_to_midi(pitch_label)

        try:
            duration = float(duration_label)
        except ValueError:
            duration = float(char_end - char_start)

        notes.append({
            "index": len(notes),
            "lyric": char_label,
            "syllable": syllable_label,
            "pitch": pitch_label,
            "midi": midi,
            "start": round(float(char_start), 6),
            "end": round(float(char_end), 6),
            "duration": round(float(duration), 6),
        })

    return notes


def attach_notes_to_phrases(
    phrases: List[Dict[str, Any]],
    notes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    根据时间范围，把 notes 分配到对应句子。
    note_start / note_end 使用 Python 切片习惯：
    note_start inclusive, note_end exclusive。
    """
    for phrase in phrases:
        start = phrase["start"]
        end = phrase["end"]

        phrase_notes = [
            note for note in notes
            if note["start"] >= start - 1e-4 and note["end"] <= end + 1e-4
        ]

        if phrase_notes:
            phrase["note_start"] = phrase_notes[0]["index"]
            phrase["note_end"] = phrase_notes[-1]["index"] + 1
        else:
            phrase["note_start"] = 0
            phrase["note_end"] = 0

    return phrases


def build_melody_notes_by_phrase(
    phrases: List[Dict[str, Any]],
    notes: List[Dict[str, Any]],
    sample_id: str,
) -> Dict[str, Any]:
    melody_phrases = []

    for phrase in phrases:
        start_idx = phrase["note_start"]
        end_idx = phrase["note_end"]

        phrase_notes = [
            note for note in notes
            if start_idx <= note["index"] < end_idx
        ]

        melody_phrases.append({
            "id": phrase["id"],
            "zh": phrase["zh"],
            "start": phrase["start"],
            "end": phrase["end"],
            "notes": phrase_notes,
        })

    return {
        "source": "opencpop_textgrid",
        "sample_id": sample_id,
        "phrases": melody_phrases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--textgrid",
        type=str,
        required=True,
        help="Path to OpenCPOP TextGrid file.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample id, e.g. 2002. If not set, use TextGrid stem.",
    )
    parser.add_argument(
        "--out-lyrics",
        type=str,
        default="data/lyrics/lyrics_zh.txt",
    )
    parser.add_argument(
        "--out-phrase-map",
        type=str,
        default="data/lyrics/phrase_map.json",
    )
    parser.add_argument(
        "--out-melody-notes",
        type=str,
        default="data/score/melody_notes.json",
    )

    args = parser.parse_args()

    textgrid_path = find_textgrid_path(Path(args.textgrid))
    sample_id = args.sample_id or textgrid_path.stem

    out_lyrics = Path(args.out_lyrics)
    out_phrase_map = Path(args.out_phrase_map)
    out_melody_notes = Path(args.out_melody_notes)

    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)

    print("TextGrid:", textgrid_path)
    print("Tier names:", tg.tierNames)

    required_tiers = ["句子", "汉字", "音节", "音高", "音长"]
    for tier_name in required_tiers:
        if tier_name not in tg.tierNames:
            raise ValueError(f"Required tier not found: {tier_name}")

    sentence_entries = get_tier_entries(tg, "句子")
    char_entries = get_tier_entries(tg, "汉字")
    syllable_entries = get_tier_entries(tg, "音节")
    pitch_entries = get_tier_entries(tg, "音高")
    duration_entries = get_tier_entries(tg, "音长")

    phrases = extract_sentence_phrases(sentence_entries)
    notes = build_notes(
        char_entries=char_entries,
        syllable_entries=syllable_entries,
        pitch_entries=pitch_entries,
        duration_entries=duration_entries,
    )
    phrases = attach_notes_to_phrases(phrases, notes)

    phrase_map = {
        "source": "opencpop_textgrid",
        "sample_id": sample_id,
        "textgrid": str(textgrid_path),
        "note_index_rule": "note_start inclusive, note_end exclusive",
        "phrases": phrases,
    }

    melody_notes = build_melody_notes_by_phrase(
        phrases=phrases,
        notes=notes,
        sample_id=sample_id,
    )

    out_lyrics.parent.mkdir(parents=True, exist_ok=True)
    out_phrase_map.parent.mkdir(parents=True, exist_ok=True)
    out_melody_notes.parent.mkdir(parents=True, exist_ok=True)

    lyric_lines = [p["zh"] for p in phrases]
    out_lyrics.write_text("\n".join(lyric_lines) + "\n", encoding="utf-8")

    out_phrase_map.write_text(
        json.dumps(phrase_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_melody_notes.write_text(
        json.dumps(melody_notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print("lyrics_zh:", out_lyrics)
    print("phrase_map:", out_phrase_map)
    print("melody_notes:", out_melody_notes)
    print("\nPreview lyrics:")
    for line in lyric_lines[:10]:
        print(line)

    print(f"\nTotal phrases: {len(phrases)}")
    print(f"Total notes: {len(notes)}")


if __name__ == "__main__":
    main()