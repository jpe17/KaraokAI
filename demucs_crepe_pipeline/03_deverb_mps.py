#!/usr/bin/env python3
import os, glob, torch, torchaudio
from openunmix import umxse

# Detect Apple MPS or fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
separator = umxse(pretrained=True, device=device)

in_dir = "03_data_preprocessing/voice"
for path in glob.glob(os.path.join(in_dir, "*.wav")):
    wav, sr = torchaudio.load(path, normalize=True)
    wav = wav.mean(0, keepdim=True).to(device)
    with torch.no_grad():
        enhanced = separator(wav)
    torchaudio.save(path, enhanced.cpu(), sr)
    print(f"De-reverbed: {os.path.basename(path)}")
