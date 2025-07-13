#!/usr/bin/env python3
"""
Convert MIDI files to WAV format for the karaoke app.
This script uses pretty_midi and soundfont to synthesize audio.
"""

import os
import pretty_midi
import numpy as np
from pathlib import Path
import soundfile as sf

def synthesize_midi_to_wav(midi_path, wav_path, sample_rate=44100):
    """
    Convert a MIDI file to WAV using pretty_midi's built-in synthesizer.
    """
    try:
        # Load the MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Synthesize the audio
        audio = pm.synthesize(fs=sample_rate)
        
        # Normalize audio to prevent clipping
        if len(audio) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Save as WAV
        sf.write(wav_path, audio, sample_rate)
        print(f"✅ Converted {midi_path} -> {wav_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {midi_path}: {e}")
        return False

def convert_all_midi_files():
    """Convert all MIDI files in the instrumental directory to WAV."""
    
    instrumental_dir = Path("03_data_preprocessing/instrumental")
    
    if not instrumental_dir.exists():
        print("❌ Instrumental directory not found. Run setup_karaoke_data.py first.")
        return
    
    midi_files = list(instrumental_dir.glob("*.mid"))
    
    if not midi_files:
        print("❌ No MIDI files found in instrumental directory.")
        return
    
    print(f"🎵 Found {len(midi_files)} MIDI files to convert...")
    
    success_count = 0
    for midi_file in midi_files:
        wav_file = midi_file.with_suffix('.wav')
        if synthesize_midi_to_wav(str(midi_file), str(wav_file)):
            success_count += 1
    
    print(f"\n🎉 Successfully converted {success_count}/{len(midi_files)} files!")

def main():
    """Main conversion function."""
    print("🎵 Converting MIDI files to WAV format...")
    convert_all_midi_files()

if __name__ == "__main__":
    main() 