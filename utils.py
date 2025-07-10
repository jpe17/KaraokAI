import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import librosa
import os
from typing import List, Tuple, Dict

def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"

def note_name_to_midi(note_name: str) -> int:
    """Convert note name to MIDI number"""
    notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
             'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    
    # Handle different formats
    if len(note_name) == 2:  # e.g., "C4"
        note, octave = note_name[0], int(note_name[1])
    elif len(note_name) == 3:  # e.g., "C#4"
        note, octave = note_name[:2], int(note_name[2])
    else:
        raise ValueError(f"Invalid note name: {note_name}")
    
    return notes[note] + (octave + 1) * 12

def frequency_to_midi(frequency: float) -> int:
    """Convert frequency in Hz to MIDI note number"""
    return int(69 + 12 * np.log2(frequency / 440.0))

def midi_to_frequency(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def calculate_note_accuracy(predicted_notes: List[Tuple], ground_truth_notes: List[Tuple], 
                           time_tolerance: float = 0.1, note_tolerance: int = 1) -> Dict:
    """
    Calculate accuracy metrics for note prediction
    
    Args:
        predicted_notes: List of (start_time, duration, note) tuples
        ground_truth_notes: List of (start_time, duration, note) tuples  
        time_tolerance: Maximum time difference for matching notes
        note_tolerance: Maximum semitone difference for matching notes
        
    Returns:
        Dictionary with accuracy metrics
    """
    if not ground_truth_notes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Match predicted notes with ground truth
    matched_pred = set()
    matched_gt = set()
    
    for i, (pred_start, pred_dur, pred_note) in enumerate(predicted_notes):
        for j, (gt_start, gt_dur, gt_note) in enumerate(ground_truth_notes):
            if j in matched_gt:
                continue
                
            # Check if notes match within tolerance
            time_match = abs(pred_start - gt_start) <= time_tolerance
            note_match = abs(pred_note - gt_note) <= note_tolerance
            
            if time_match and note_match:
                matched_pred.add(i)
                matched_gt.add(j)
                break
    
    # Calculate metrics
    precision = len(matched_pred) / len(predicted_notes) if predicted_notes else 0.0
    recall = len(matched_gt) / len(ground_truth_notes) if ground_truth_notes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched_predictions': len(matched_pred),
        'total_predictions': len(predicted_notes),
        'matched_ground_truth': len(matched_gt),
        'total_ground_truth': len(ground_truth_notes)
    }

def plot_note_comparison(predicted_notes: List[Tuple], ground_truth_notes: List[Tuple], 
                        output_path: str = None, figsize: Tuple = (15, 10)):
    """
    Plot comparison between predicted and ground truth notes
    
    Args:
        predicted_notes: List of (start_time, duration, note) tuples
        ground_truth_notes: List of (start_time, duration, note) tuples
        output_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Plot ground truth
    for start_time, duration, note in ground_truth_notes:
        ax1.barh(note, duration, left=start_time, height=0.8, alpha=0.7, color='green')
    ax1.set_title('Ground Truth Notes')
    ax1.set_ylabel('MIDI Note')
    ax1.grid(True, alpha=0.3)
    
    # Plot predictions
    for start_time, duration, note in predicted_notes:
        ax2.barh(note, duration, left=start_time, height=0.8, alpha=0.7, color='blue')
    ax2.set_title('Predicted Notes')
    ax2.set_ylabel('MIDI Note')
    ax2.grid(True, alpha=0.3)
    
    # Plot overlay
    for start_time, duration, note in ground_truth_notes:
        ax3.barh(note, duration, left=start_time, height=0.4, alpha=0.7, 
                color='green', label='Ground Truth' if start_time == ground_truth_notes[0][0] else "")
    for start_time, duration, note in predicted_notes:
        ax3.barh(note + 0.4, duration, left=start_time, height=0.4, alpha=0.7, 
                color='blue', label='Predicted' if start_time == predicted_notes[0][0] else "")
    
    ax3.set_title('Overlay Comparison')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('MIDI Note')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def export_to_midi(notes: List[Tuple], output_path: str, tempo: int = 120):
    """
    Export notes to MIDI file (requires mido library)
    
    Args:
        notes: List of (start_time, duration, note) tuples
        output_path: Path to save MIDI file
        tempo: MIDI tempo
    """
    try:
        import mido
    except ImportError:
        print("mido library not installed. Install with: pip install mido")
        return
    
    # Create MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    
    # Convert notes to MIDI events
    events = []
    for start_time, duration, note in notes:
        events.append((start_time, 'note_on', note))
        events.append((start_time + duration, 'note_off', note))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    # Convert to MIDI messages
    current_time = 0
    for time, event_type, note in events:
        delta_time = int((time - current_time) * 480)  # Convert to ticks
        
        if event_type == 'note_on':
            track.append(mido.Message('note_on', channel=0, note=int(note), 
                                    velocity=64, time=delta_time))
        else:
            track.append(mido.Message('note_off', channel=0, note=int(note), 
                                    velocity=64, time=delta_time))
        
        current_time = time
    
    # Save MIDI file
    mid.save(output_path)
    print(f"MIDI file saved to {output_path}")

def load_notes_from_csv(csv_path: str) -> List[Tuple]:
    """Load notes from CSV file"""
    df = pd.read_csv(csv_path)
    notes = []
    for _, row in df.iterrows():
        start_time = float(row['start_time'])
        duration = float(row['duration'])
        note = int(row['note'])
        notes.append((start_time, duration, note))
    return notes

def save_notes_to_csv(notes: List[Tuple], csv_path: str):
    """Save notes to CSV file"""
    df = pd.DataFrame(notes, columns=['start_time', 'duration', 'note'])
    df.to_csv(csv_path, index=False)
    print(f"Notes saved to {csv_path}")

def analyze_audio_properties(audio_path: str) -> Dict:
    """Analyze basic properties of audio file"""
    try:
        y, sr = librosa.load(audio_path)
        
        properties = {
            'duration': len(y) / sr,
            'sample_rate': sr,
            'channels': 1,  # librosa loads as mono by default
            'rms_energy': float(np.sqrt(np.mean(y**2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0])
        }
        
        return properties
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return {}

def print_model_summary(model):
    """Print model summary with parameter counts"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print layer-wise breakdown
    print("\nLayer-wise breakdown:")
    for name, module in model.named_children():
        layer_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {layer_params:,} parameters")

def create_training_summary(train_losses: List[float], val_losses: List[float], 
                          output_path: str = None):
    """Create training summary plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Example usage functions
def quick_test_dataloader():
    """Quick test of the dataloader"""
    from dataloader import create_dataloaders
    
    try:
        train_dl, val_dl = create_dataloaders(
            "processed_data/voices", 
            "processed_data/notes", 
            batch_size=2
        )
        print(f"✓ Dataloader test passed")
        print(f"  Train batches: {len(train_dl)}")
        print(f"  Val batches: {len(val_dl)}")
        return True
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        return False

def quick_test_model():
    """Quick test of the model"""
    from model import VoiceToNotesModel
    
    try:
        model = VoiceToNotesModel()
        print("✓ Model creation test passed")
        print_model_summary(model)
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running utility tests...")
    
    # Test MIDI conversion
    print("\nTesting MIDI utilities:")
    test_note = 60  # Middle C
    note_name = midi_to_note_name(test_note)
    converted_back = note_name_to_midi(note_name)
    print(f"MIDI {test_note} -> {note_name} -> {converted_back}")
    
    # Test frequency conversion
    freq = midi_to_frequency(test_note)
    midi_back = frequency_to_midi(freq)
    print(f"MIDI {test_note} -> {freq:.2f} Hz -> {midi_back}")
    
    # Test model and dataloader
    print("\nTesting components:")
    quick_test_model()
    quick_test_dataloader() 