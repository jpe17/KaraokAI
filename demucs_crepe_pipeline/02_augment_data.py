#!/usr/bin/env python3
"""
augment_audio.py

Recursively reads .wav files from input folders (01_demucs/voice and 01_demucs/instrumental),
applies random augmentations (pitch shift ±2 semitones, time stretch 0.9–1.1, gain ±3 dB),
and writes 1000 augmented variants per source file into mirrored folders under 02_data_augment.
"""

import os
import random
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm

# Configuration
INPUT_ROOT = "01_demucs"
OUTPUT_ROOT = "02_data_augment"
AUG_PER_FILE = 30 # Changed from 1000 to 1

# Augmentation parameters
PITCH_SEMITONE_RANGE = (-2.0, 2.0)      # ±2 semitones
STRETCH_FACTOR_RANGE = (0.9, 1.1)       # 0.9–1.1 speed
GAIN_DB_RANGE = (-3.0, 3.0)             # ±3 dB

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def random_gain(audio, min_db, max_db):
    db = random.uniform(min_db, max_db)
    gain = 10**(db / 20)
    return audio * gain

def augment_and_save(src_path, dst_folder, basename, idx):
    # Load audio
    y, sr = librosa.load(src_path, sr=None, mono=False)  # preserve channels
    # 1) Pitch shift
    n_steps = random.uniform(*PITCH_SEMITONE_RANGE)
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    # 2) Time stretch
    rate = random.uniform(*STRETCH_FACTOR_RANGE)
    y_stretch = librosa.effects.time_stretch(y_shift, rate=rate)
    # 3) Gain
    y_aug = random_gain(y_stretch, *GAIN_DB_RANGE)
    # Normalize to avoid clipping
    max_amp = np.max(np.abs(y_aug))
    if max_amp > 1.0:
        y_aug = y_aug / max_amp
    # Write
    out_name = f"{basename}_{idx:04d}.wav"
    dst_path = os.path.join(dst_folder, out_name)
    sf.write(dst_path, y_aug.T, sr)  # transpose if multi-channel

def process_folder(subfolder):
    in_dir = os.path.join(INPUT_ROOT, subfolder)
    out_dir = os.path.join(OUTPUT_ROOT, subfolder)
    ensure_dir(out_dir)
    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(".wav"):
            continue
        src = os.path.join(in_dir, fname)
        name, _ = os.path.splitext(fname)
        # Generate AUG_PER_FILE variants
        for i in tqdm(range(AUG_PER_FILE), desc=f"Augmenting {subfolder}/{name}"):
            augment_and_save(src, out_dir, name, i)

def main():
    # Process voice and instrumental
    for part in ("voice", "instrumental"):
        process_folder(part)

if __name__ == "__main__":
    main()
