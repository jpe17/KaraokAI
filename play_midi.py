#!/usr/bin/env python3
import pandas as pd, pygame, sys, time, io
import pretty_midi
from pathlib import Path

def play_notes_from_csv(csv_path: Path, instrument_name="Acoustic Grand Piano"):
    # Load CSV and ensure we have the right column names
    df = pd.read_csv(csv_path)
    
    # Handle different possible column names for backward compatibility
    if 'start_time' in df.columns:
        start_col = 'start_time'
    elif 'duration' in df.columns:
        start_col = 'duration'
    else:
        raise ValueError("CSV must contain either 'start_time' or 'duration' column")
    
    if 'duration' not in df.columns:
        raise ValueError("CSV must contain 'duration' column")
    
    # Rename columns to standard names for processing
    df = df.rename(columns={start_col: 'start_time', 'duration': 'duration'})
    df['end_time'] = df['start_time'] + df['duration']
    
    # Build MIDI in memory
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name))
    for _, r in df.iterrows():
        instr.notes.append(pretty_midi.Note(
            velocity=80, pitch=int(r['note']),
            start=r['start_time'], end=r['end_time']))
    pm.instruments.append(instr)
    midi_io = io.BytesIO()
    pm.write(midi_io)
    midi_io.seek(0)

    # Play via Pygame
    pygame.mixer.init()
    print(f"[▶] Playing {csv_path.name} ({len(instr.notes)} notes)…")
    try:
        pygame.mixer.music.load(midi_io)
        pygame.mixer.music.play()
        time.sleep(df['end_time'].max())
    except KeyboardInterrupt:
        print("\n[■] Stopped by user.")
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play_midi.py <notes.csv>")
        sys.exit(1)
    play_notes_from_csv(Path(sys.argv[1]))
