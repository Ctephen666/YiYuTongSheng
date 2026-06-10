# YiyuTongsheng Pipeline Report

## Artifact Summary

| Stage | Output | Exists | Current Mode | TODO |
| --- | --- | --- | --- | --- |
| preprocess | `data/stems/vocals.wav` | yes | mock | Connect Demucs/UVR separation. |
| preprocess | `data/stems/accompaniment.wav` | yes | mock | Validate stem quality. |
| melody | `data/score/f0.csv` | yes | mock | Replace mock F0 with RMVPE/librosa.pyin. |
| melody | `data/score/melody_notes.json` | yes | mock | Quantize and correct notes. |
| lyrics | `data/lyrics/lyrics_singable.json` | yes | mock | Use LLM and human review. |
| phoneme | `data/phoneme/phonemes_target.json` | yes | mock | Use real phonemizer backend. |
| alignment | `data/alignment/note_lyric_alignment.json` | yes | mock | Add dynamic programming alignment. |
| svs | `data/svs/target_language_vocal.wav` | yes | mock | Render with OpenUtau/DiffSinger. |
| svc | `data/rvc/converted_voice.wav` | yes | mock | Run RVC/SVC inference. |
| mix | `data/final/final_mix.wav` | yes | mock | Mix with pydub/librosa/ffmpeg. |

## Text Flowchart

original song
-> vocal/accompaniment separation
-> F0 extraction and melody reconstruction
-> Chinese phrase mapping
-> target-language lyric translation
-> singable lyric adaptation
-> target-language phonemization
-> syllable-note alignment
-> target-language SVS rendering
-> RVC/SVC timbre transfer
-> vocal/accompaniment mixing
-> final cross-lingual song output

## Notes

All generated media artifacts in the current scaffold are placeholders unless real tools are connected.
