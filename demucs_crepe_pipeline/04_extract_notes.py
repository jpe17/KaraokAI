#!/usr/bin/env python3
"""
wav_to_csv_crepe.py - Fixed version with proper synchronization
"""
import os
import glob
import sys
import csv
import numpy as np
import librosa
import crepe
from tqdm import tqdm
import time

# CREPE model expects 16kHz sample rate for optimal performance.
CREPE_SAMPLE_RATE = 16000

def wav_to_events(wav_path, confidence_threshold=0.8, min_note_duration=0.1):
    """Extract pitch with progress indicators and proper timing"""
    
    print(f"Loading audio file: {wav_path}")
    start_time = time.time()
    
    # Load audio - get original sample rate for reference
    audio_original, sr_original = librosa.load(wav_path, sr=None)
    print(f"Original audio: {len(audio_original) / sr_original:.2f} seconds, sample rate: {sr_original}")
    
    # Resample to CREPE's expected rate
    audio, sr = librosa.load(wav_path, sr=CREPE_SAMPLE_RATE)
    duration = len(audio) / sr
    print(f"Resampled audio: {duration:.2f} seconds, sample rate: {sr}")
    
    # Run CREPE with only supported parameters
    print("Running CREPE pitch detection...")
    time_stamps, frequency, confidence, activation = crepe.predict(
        audio, 
        sr, 
        step_size=10,  # 10ms hop size
        verbose=1      # Show progress
    )
    
    processing_time = time.time() - start_time
    print(f"CREPE processing completed in {processing_time:.2f} seconds")
    
    # Filter by confidence
    print(f"Filtering by confidence threshold: {confidence_threshold}")
    confident_mask = confidence >= confidence_threshold
    time_stamps = time_stamps[confident_mask]
    frequency = frequency[confident_mask]
    confidence_filtered = confidence[confident_mask]
    
    print(f"Kept {len(time_stamps)}/{len(confidence)} frames ({len(time_stamps)/len(confidence)*100:.1f}%)")
    
    # Convert to MIDI notes
    print("Converting to MIDI notes...")
    midi_notes = librosa.hz_to_midi(frequency)
    midi_notes = np.round(midi_notes).astype(int)
    
    # Group into note events with proper timing
    print("Grouping into note events...")
    events = group_notes_fixed(time_stamps, midi_notes, min_note_duration)
    
    print(f"Generated {len(events)} note events")
    return events

def group_notes_fixed(time_stamps, midi_notes, min_note_duration):
    """Group consecutive similar notes into events with proper timing calculations"""
    events = []
    if len(midi_notes) == 0:
        return events
    
    current_note = midi_notes[0]
    start_time = time_stamps[0]
    last_time = time_stamps[0]
    
    # CREPE uses 10ms hop size
    hop_size_seconds = 0.01
    
    print("Grouping consecutive notes with fixed timing...")
    for i in range(1, len(midi_notes)):
        current_time = time_stamps[i]
        
        # Check if note changed (allow 1 semitone tolerance) OR if there's a significant gap
        note_changed = abs(midi_notes[i] - current_note) > 1
        # More reasonable gap threshold - allow for brief silences within a note
        time_gap = (current_time - last_time) > 0.2  # 200ms gap threshold
        
        if note_changed or time_gap:
            # End current note - use proper duration calculation
            # Add hop size to account for frame duration
            duration = last_time - start_time + hop_size_seconds
            
            if duration >= min_note_duration:
                events.append((start_time, duration, current_note))
            
            # Start new note
            current_note = midi_notes[i]
            start_time = current_time
        
        last_time = current_time
    
    # Add final note with proper duration
    duration = last_time - start_time + hop_size_seconds
    if duration >= min_note_duration:
        events.append((start_time, duration, current_note))
    
    return events

def write_csv(events, csv_path):
    """Write events to CSV file with high precision timestamps"""
    print(f"Writing {len(events)} events to {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_time_sec', 'duration_sec', 'midi_note'])
        for start_time, duration, note in events:
            # Use high precision for timing
            writer.writerow([f"{start_time:.6f}", f"{duration:.6f}", note])

def validate_timing(events, duration):
    """Validate that events don't exceed audio duration"""
    if not events:
        return True
    
    max_end_time = max(event[0] + event[1] for event in events)
    if max_end_time > duration * 1.1:  # Allow 10% tolerance
        print(f"⚠️  Warning: Some notes extend beyond audio duration ({max_end_time:.2f}s > {duration:.2f}s)")
        return False
    return True

def main():
    input_dir = '03_data_preprocessing/voice'
    output_dir = '04_extracted_notes'
    
    # Process only the specific file
    specific_file = 'voice_042000_takemehome_0000.wav'
    wav_path = os.path.join(input_dir, specific_file)

    # Parameters - made more conservative for better sync
    confidence_threshold = 0.85  # Slightly higher for better quality
    min_note_duration = 0.05     # Shorter minimum for better note capture

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the specific file exists
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing single file: {specific_file}")

    try:
        basename = os.path.basename(wav_path)
        filename_no_ext = os.path.splitext(basename)[0]
        output_csv_path = os.path.join(output_dir, f"{filename_no_ext}.csv")

        print(f"\nProcessing {wav_path} -> {output_csv_path}")

        events = wav_to_events(wav_path, confidence_threshold, min_note_duration)

        if events:
            # Get audio duration for validation
            audio_duration = librosa.get_duration(path=wav_path)
            
            if validate_timing(events, audio_duration):
                write_csv(events, output_csv_path)
                print(f"✓ Successfully extracted {len(events)} vocal notes to {output_csv_path}")
                
                # Print timing statistics
                total_note_time = sum(event[1] for event in events)
                print(f"  Total note time: {total_note_time:.2f}s / {audio_duration:.2f}s ({total_note_time/audio_duration*100:.1f}%)")
            else:
                print(f"⚠️  Timing validation failed for {wav_path}")
        else:
            print(f"No notes extracted from {wav_path} with current settings.")

    except Exception as e:
        print(f"Error processing {wav_path}: {e}", file=sys.stderr)

    print("\nProcessing complete.")


if __name__ == '__main__':
    main()
