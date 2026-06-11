from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tools.extract_opencpop_textgrid import main as extract_main


def find_file(root: Path, sample_id: str, suffixes: list[str]) -> Path:
    for suffix in suffixes:
        matches = list(root.rglob(f"{sample_id}{suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find {sample_id} with suffixes {suffixes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out-root", default=".")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    textgrid = find_file(
        dataset_root,
        args.sample_id,
        [".TextGrid", ".textgrid", ".TEXTGRID"],
    )

    wav = find_file(
        dataset_root,
        args.sample_id,
        [".wav", ".flac"],
    )

    out_raw = out_root / "data" / "raw" / "original_song.wav"
    out_vocals = out_root / "data" / "stems" / "vocals.wav"

    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_vocals.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(wav, out_raw)
    shutil.copyfile(wav, out_vocals)

    # 这里直接调用原脚本最简单，也可以把 extract_opencpop_textgrid.py 的核心逻辑拆成函数。
    import subprocess
    subprocess.run(
        [
            "python",
            "tools/extract_opencpop_textgrid.py",
            "--textgrid",
            str(textgrid),
            "--sample-id",
            args.sample_id,
            "--out-lyrics",
            "data/lyrics/lyrics_zh.txt",
            "--out-phrase-map",
            "data/lyrics/phrase_map.json",
            "--out-melody-notes",
            "data/score/melody_notes.json",
        ],
        check=True,
    )

    print("Prepared sample:", args.sample_id)
    print("wav:", wav)
    print("textgrid:", textgrid)
    print("outputs:")
    print("  data/raw/original_song.wav")
    print("  data/stems/vocals.wav")
    print("  data/lyrics/lyrics_zh.txt")
    print("  data/lyrics/phrase_map.json")
    print("  data/score/melody_notes.json")


if __name__ == "__main__":
    main()