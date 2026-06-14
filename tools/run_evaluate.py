from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.runner import run_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YiYuTongSheng evaluation metrics.")
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--song-id", default=None)
    parser.add_argument("--reference-wav", default=None)
    parser.add_argument("--svs-wav", default=None)
    parser.add_argument("--svc-wav", default=None)
    parser.add_argument("--final-wav", default=None)
    parser.add_argument("--score-json", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict = {"inputs": {}, "output": {}}
    if args.song_id:
        overrides["inputs"]["song_id"] = args.song_id
        overrides["inputs"]["reference_wav"] = f"data/dataset/opencpop/wavs/{args.song_id}.wav"
    for attr, key in [
        ("reference_wav", "reference_wav"),
        ("svs_wav", "svs_wav"),
        ("svc_wav", "svc_wav"),
        ("final_wav", "final_wav"),
        ("score_json", "score_json"),
    ]:
        value = getattr(args, attr)
        if value:
            overrides["inputs"][key] = value
    if args.out_dir:
        out_dir = Path(args.out_dir)
        overrides["output"]["json_report"] = str(out_dir / "evaluate_report.json")
        overrides["output"]["markdown_report"] = str(out_dir / "evaluate_report.md")

    overrides = {key: value for key, value in overrides.items() if value}
    report = run_evaluate(args.config, overrides or None)
    output = report.get("overall", {})
    json_path = (overrides.get("output", {}) or {}).get("json_report", "data/evaluate/evaluate_report.json")
    md_path = (overrides.get("output", {}) or {}).get("markdown_report", "data/evaluate/evaluate_report.md")
    print("Evaluate completed.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Overall score: {output.get('score')}")


if __name__ == "__main__":
    main()
