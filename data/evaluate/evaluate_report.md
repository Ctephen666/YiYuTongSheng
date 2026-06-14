# YiYuTongSheng Evaluate Report

## Summary
- Status: success
- Overall score: 93.254
- Level: excellent
- Available categories: audio_quality, pitch_preservation
- Skipped categories: intelligibility, speaker_similarity

## Inputs
- song_id: D:\YiYuTongSheng\2001
- reference_wav: D:\YiYuTongSheng\data\dataset\opencpop\wavs\2001.wav
- svs_wav: D:\YiYuTongSheng\data\svs\target_language_vocal.wav
- svc_wav: D:\YiYuTongSheng\data\svc\converted_target_voice.wav
- final_wav: D:\YiYuTongSheng\data\svc\final_mix.wav
- score_json: D:\YiYuTongSheng\data\svs\opencpop_svs_score.json
- svs_report_json: D:\YiYuTongSheng\data\svs\diffsinger_infer_report.json
- svc_report_json: D:\YiYuTongSheng\data\svc\rvc_convert_report.json

## Audio Quality
- reference: valid=True duration=235.2587755102041 rms=0.03569202125072479 clipping=0.0
- svs: valid=True duration=230.46 rms=0.12577015161514282 clipping=0.0
- svc: valid=True duration=230.46 rms=0.12577015161514282 clipping=0.0
- final: valid=False duration=None rms=None clipping=None

## Pitch Preservation
- f0_rmse_hz: 0.0
- f0_rmse_cents: 0.0
- f0_mae_hz: 0.0
- f0_mae_cents: 0.0
- f0_correlation: 1.0
- voiced_unvoiced_accuracy: 1.0

## Intelligibility
- CER: None
- WER: None
- ASR available: False

## Speaker Similarity
- Similarity to target: None
- Similarity to source: None
- Embedding available: False

## Overall Score
- audio_quality: 86.508
- pitch_preservation: 100.0
- intelligibility: None
- speaker_similarity: None

## Warnings
- Audio file not found: D:\YiYuTongSheng\data\svc\final_mix.wav
- ASR not available; intelligibility metric skipped.
- Speaker embedding model not available; speaker similarity skipped.
