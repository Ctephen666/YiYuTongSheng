# Evaluate Metrics

`src/evaluate` provides lightweight, local metrics for checking the final YiYuTongSheng conversion result. It does not run DiffSinger, RVC, training, network APIs, or model downloads.

## Run

```cmd
D:\Anaconda_envs\envs\rvc\python.exe tools\run_evaluate.py --config configs/evaluate.yaml
```

With overrides:

```cmd
D:\Anaconda_envs\envs\rvc\python.exe tools\run_evaluate.py ^
  --song-id 2001 ^
  --svs-wav data/svs/target_language_vocal.wav ^
  --svc-wav data/svc/converted_target_voice.wav ^
  --out-dir data/evaluate
```

## Outputs

```text
data/evaluate/evaluate_report.json
data/evaluate/evaluate_report.md
```

## Audio Quality

Implemented in `src/evaluate/audio_metrics.py`.

Metrics:

```text
duration_sec
sample_rate
channels
peak
rms
clipping_ratio
silence_ratio
zero_crossing_rate
spectral_centroid
high_freq_ratio
low_freq_ratio
nan_or_inf
is_valid_audio
```

These metrics are lightweight and require no external model. `soundfile` is used when available; otherwise WAV files are read through Python's built-in `wave` module.

## Pitch Preservation

Implemented in `src/evaluate/pitch_metrics.py`.

Compares:

```text
data/svs/target_language_vocal.wav
vs
data/svc/converted_target_voice.wav
```

Metrics:

```text
f0_rmse_hz
f0_rmse_cents
f0_mae_hz
f0_mae_cents
f0_correlation
voiced_unvoiced_accuracy
```

Interpretation:

```text
f0_rmse_cents < 30: good pitch preservation
30-60: acceptable but slightly unstable
> 60: obvious pitch drift
```

F0 extraction tries local `pyworld`, then local `librosa`, then a simple autocorrelation fallback. No GPU, RMVPE, RVC, or network model is required.

## Intelligibility

Implemented in `src/evaluate/text_metrics.py`.

The module reads reference lyrics from `opencpop_svs_score.json` and provides pure Python:

```text
levenshtein_distance
char_error_rate
word_error_rate
```

ASR is optional and disabled by default. If no recognized text is supplied, CER/WER are skipped with a warning.

## Speaker Similarity

Implemented in `src/evaluate/speaker_metrics.py`.

No speaker embedding model is required by default. If embedding files exist, cosine similarity is computed:

```text
target_embedding.npy
source_embedding.npy
converted_embedding.npy
```

If embeddings are missing, the metric is skipped with a warning.

## Overall Score

Implemented in `src/evaluate/report.py`.

Weights:

```text
audio_quality: 40
pitch_preservation: 40
intelligibility: 5
speaker_similarity: 15
```

Unavailable categories are skipped and weights are normalized across available categories.

Levels:

```text
excellent: >= 90
good: >= 80
fair: >= 65
poor: < 65
```
