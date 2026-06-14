# YiYuTongSheng Evaluate Report

## Summary
- Status: success
- Overall score: 68.135
- Level: fair
- Score confidence: partial
- Evaluation completeness: 0.75
- Available categories: audio_quality, conversion_integrity, pitch_preservation
- Skipped categories: intelligibility, speaker_similarity

## Inputs
- song_id: 2037
- reference_wav: D:\YiYuTongSheng\data\dataset\opencpop\wavs\2037.wav
- svs_wav: D:\YiYuTongSheng\data\svs\target_language_vocal.wav
- svc_wav: D:\YiYuTongSheng\data\svc\converted_target_voice.wav
- final_wav: D:\YiYuTongSheng\data\svc\final_mix.wav
- score_json: D:\YiYuTongSheng\data\svs\opencpop_svs_score.json
- svs_report_json: D:\YiYuTongSheng\data\svs\diffsinger_infer_report.json
- svc_report_json: D:\YiYuTongSheng\data\svc\rvc_convert_report.json

## Audio Quality
- reference: valid=True duration=240.3787755102041 rms=0.08210772275924683 clipping=0.0
- svs: valid=True duration=222.65658333333334 rms=0.14484429359436035 clipping=1.8713422276981853e-07
- svc: valid=True duration=222.62 rms=0.12813308835029602 clipping=0.0
- final: valid=False duration=None rms=None clipping=None

## Conversion Integrity
- likely_same_audio: False
- mean_abs_diff: 0.1248406657167898
- max_abs_diff: 1.444976806640625
- rms_diff: 0.20092913213228517
- waveform_correlation: -0.08003285053469333
- sha256_equal: False

## Pitch Preservation
- f0_rmse_hz: 83.56756286363326
- f0_rmse_cents: 251.8608837193672
- f0_mae_hz: 20.139694273471832
- f0_mae_cents: 67.39583901236725
- f0_correlation: 0.789237000310989
- voiced_unvoiced_accuracy: 0.928404599353216

## Intelligibility
- CER: None
- WER: None
- ASR available: False

## Speaker Similarity
- Similarity to target: None
- Similarity to source: None
- Embedding available: False

## Overall Score
- audio_quality: 86.224
- conversion_integrity: 100.0
- pitch_preservation: 31.818
- intelligibility: None
- speaker_similarity: None

## Warnings
- Audio file not found: D:\YiYuTongSheng\data\svc\final_mix.wav
- ASR not available; intelligibility metric skipped.
- Speaker embedding model not available; speaker similarity skipped.
