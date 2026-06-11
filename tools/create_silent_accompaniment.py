from pathlib import Path
import soundfile as sf
import numpy as np

vocals_path = Path("data/stems/vocals.wav")
out_path = Path("data/stems/accompaniment.wav")

audio, sr = sf.read(vocals_path)
silence = np.zeros_like(audio)

out_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(out_path, silence, sr)

print("silent accompaniment saved to:", out_path)