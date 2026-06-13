# YiYuTongSheng

YiYuTongSheng is currently configured for OpenCpop Chinese SVS inference with DiffSinger.

The default path is no longer the old English translation + MeloTTS mock/baseline path. The default target language is `zh`, and the main pipeline uses OpenCpop dataset files only when `dataset.strict_dataset_source: true`.

## Default Dataset

`configs/project.yaml` expects OpenCpop under:

```text
data/dataset/opencpop
```

Default song id:

```text
2001
```

Important config keys:

```yaml
dataset:
  name: opencpop
  root: data/dataset/opencpop
  strict_dataset_source: true
  default_song_id: "2001"
```

When strict mode is enabled, legacy files such as `data/lyrics/lyrics_zh.txt` and `data/lyrics/phrase_map.json` are not used as fallback. Missing OpenCpop files are reported as warnings and empty JSON structures, not mock lyrics.

## Current Flow

```text
opencpop_loader
-> melody extraction from OpenCpop MIDI
-> lyrics extraction from OpenCpop dataset/TextGrid
-> Chinese phoneme/pinyin preparation
-> Chinese note alignment
-> opencpop_svs_score.json
-> diffsinger phoneme-level input JSON
-> diffsinger_opencpop_export_plan.json
-> checkpoint_status.json
-> DiffSinger subprocess inference
-> neural_svs_render_plan.json
-> target_language_vocal.wav
```

The SVS stage executes local DiffSinger inference when `svs.execute_model: true` and `svs.dry_run: false`. It does not use the old English translation + MeloTTS path.

## Run

```bash
python app.py --step all --target-language zh --opencpop-id 2001
```

Useful individual stages:

```bash
python app.py --step melody --target-language zh --opencpop-id 2001
python app.py --step lyrics --target-language zh --opencpop-id 2001
python app.py --step phoneme --target-language zh --opencpop-id 2001
python app.py --step alignment --target-language zh --opencpop-id 2001
python app.py --step svs --target-language zh --opencpop-id 2001
```

`--target-language en` is intentionally disabled in this branch. It logs a warning and skips the old English/MeloTTS pipeline.

## Inference Outputs

```text
data/dataset_manifest/opencpop_dataset_manifest.json
data/dataset_manifest/opencpop_item_2001.json
data/score/melody_notes.json
data/lyrics/lyrics_zh_phrases.json
data/phoneme/phonemes_zh.json
data/alignment/zh_note_alignment.json
data/svs/opencpop_svs_score.json
data/svs/diffsinger_opencpop_export_plan.json
data/svs/diffsinger_opencpop/diffsinger_input_2001.json
data/svs/diffsinger_opencpop/phonemes_2001.tsv
data/svs/diffsinger_opencpop/notes_2001.csv
data/svs/checkpoint_status.json
data/svs/diffsinger_infer_report.json
data/svs/neural_svs_render_plan.json
data/svs/target_language_vocal.wav
```

## Model Assets

Place pretrained DiffSinger assets under `checkpoints/`:

```yaml
svs:
  pretrained:
    acoustic_checkpoint: checkpoints/diffsinger/acoustic.ckpt
    acoustic_config: checkpoints/diffsinger/acoustic.yaml
    variance_checkpoint: checkpoints/diffsinger/variance.ckpt
    variance_config: checkpoints/diffsinger/variance.yaml
    vocoder_checkpoint: checkpoints/diffsinger/vocoder.ckpt
    vocoder_config: checkpoints/diffsinger/vocoder.yaml
    phoneme_dictionary: checkpoints/diffsinger/zh_phoneme_dict.json
  diffsinger:
    root: external/DiffSinger
```

`zh_phoneme_dict.json` is supported directly. If the config points at `.txt` but only `.json` exists, the checkpoint checker and exporter auto-adapt to the JSON dictionary.