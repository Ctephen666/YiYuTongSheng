# YiYuTongSheng

YiYuTongSheng is a course-project prototype for a Chinese-song-to-English-vocal pipeline.

The current baseline is intentionally narrow:

```text
OpenCpop MIDI + translated/singable English lyrics + MeloTTS + MIDI phrase timing alignment
```

OpenUtau, DiffSinger, RVC/SVC, mix, evaluate, and old F0 extraction modules have been removed from the main code path.

## Pipeline

`python app.py --step all --target-language en` runs only:

```text
melody -> lyrics -> phoneme -> alignment -> svs
```

Individual stages are also supported:

```bash
python app.py --step melody --target-language en
python app.py --step lyrics --target-language en
python app.py --step phoneme --target-language en
python app.py --step alignment --target-language en
python app.py --step svs --target-language en
```

## Setup

```bash
conda create -n yiyu_svc python=3.10 -y
conda activate yiyu_svc
pip install -r requirements.txt

# MeloTTS install
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

## MeloTTS + OpenCpop MIDI Baseline

Inputs:

```text
data/dataset/opencpop/midis/2001.midi
data/lyrics/lyrics_zh.txt
data/lyrics/phrase_map.json   # optional, can provide note_start/note_end
```

Run:

```bash
python app.py --step all --target-language en
```

Outputs:

```text
data/score/melody.mid
data/score/melody_notes.json
data/svs/target_language_vocal.wav
data/svs/melotts_render_report.json
```

## Stage Summary

- `melody`: imports the configured OpenCpop MIDI and writes phrase-level melody notes.
- `lyrics`: keeps the existing phrase mapping, translation, and singable lyric generation flow.
- `phoneme`: converts selected English lyrics to syllable-level phoneme units.
- `alignment`: aligns English syllable groups to MIDI phrase notes.
- `svs`: renders each phrase with MeloTTS, stretches it to the MIDI phrase duration, and places it on the final timeline.

Unused legacy Python modules for preprocess, old melody extraction, OpenUtau rendering, RVC/SVC, mix, and evaluate have been removed.

## Current Limits

- This is a MeloTTS phrase-level timing alignment baseline.
- It is not full note-level SVS.
- It does not do RVC voice conversion.
- It does not do note-level F0 replacement.
- A future version can replace this with WORLD F0 replacement or a real SVS backend.
