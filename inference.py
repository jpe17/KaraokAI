import torch
import torchaudio
import numpy as np
import pandas as pd
import os
import argparse
from typing import List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from model import VoiceToNotesModel
from dataloader import AudioPreprocessor
from transformers import WhisperFeatureExtractor

class VoiceToNotesInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load checkpoint first to get model config
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with same config as training
        self.model = VoiceToNotesModel(
            whisper_model_name="openai/whisper-base",
            d_model=512,
            nhead=8,
            num_decoder_layers=6,
            max_notes=100  # Match training config
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_long_audio(self, audio_path, chunk_length=10.0, overlap=1.0):
        """
        Preprocess long audio file by splitting into overlapping chunks
        
        Args:
            audio_path: Path to audio file
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of preprocessed audio chunks and their time offsets
        """
        # Load full audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.preprocessor.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.preprocessor.target_sr)
            audio = resampler(audio)
            sr = self.preprocessor.target_sr
        
        # Calculate chunk parameters
        chunk_samples = int(chunk_length * sr)
        overlap_samples = int(overlap * sr)
        step_samples = chunk_samples - overlap_samples
        
        audio_chunks = []
        time_offsets = []
        
        print(f"Audio duration: {audio.shape[1] / sr:.2f}s, chunk_length: {chunk_length}s")
        print(f"Audio samples: {audio.shape[1]}, chunk_samples: {chunk_samples}")
        
        # Handle short audio files
        if audio.shape[1] <= chunk_samples:
            print("Audio is shorter than chunk length, processing as single chunk")
            # Pad short audio to chunk length
            padding = chunk_samples - audio.shape[1]
            audio_padded = torch.nn.functional.pad(audio, (0, padding))
            
            # Process single chunk
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                torchaudio.save(temp_file.name, audio_padded, sr)
                chunk_preprocessed = self.preprocessor.preprocess(temp_file.name, add_noise=False)
                os.unlink(temp_file.name)
            
            features = self.feature_extractor(
                chunk_preprocessed.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            )["input_features"]
            
            audio_chunks.append(features)
            time_offsets.append(0.0)
        else:
            # Split audio into chunks
            for start in range(0, audio.shape[1] - chunk_samples + 1, step_samples):
                end = start + chunk_samples
                chunk = audio[:, start:end]
            
            # Preprocess chunk (create temporary file for preprocessing)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                torchaudio.save(temp_file.name, chunk, sr)
                chunk_preprocessed = self.preprocessor.preprocess(temp_file.name, add_noise=False)
                os.unlink(temp_file.name)
            
            # Extract features
            features = self.feature_extractor(
                chunk_preprocessed.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            )["input_features"]
            
            audio_chunks.append(features)
            time_offsets.append(start / sr)
        
        # Handle remaining audio if any
        if audio.shape[1] % step_samples != 0:
            start = audio.shape[1] - chunk_samples
            if start >= 0:
                chunk = audio[:, start:]
                
                # Pad if necessary
                if chunk.shape[1] < chunk_samples:
                    padding = chunk_samples - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    torchaudio.save(temp_file.name, chunk, sr)
                    chunk_preprocessed = self.preprocessor.preprocess(temp_file.name, add_noise=False)
                    os.unlink(temp_file.name)
                features = self.feature_extractor(
                    chunk_preprocessed.squeeze().numpy(),
                    sampling_rate=sr,
                    return_tensors="pt"
                )["input_features"]
                
                audio_chunks.append(features)
                time_offsets.append(start / sr)
        
        return audio_chunks, time_offsets
    
    def predict_notes(self, audio_features, max_length=500):
        """
        Predict notes from audio features
        
        Args:
            audio_features: Preprocessed audio features
            max_length: Maximum number of notes to generate
            
        Returns:
            List of predicted notes (start_time, duration, note)
        """
        with torch.no_grad():
            audio_features = audio_features.to(self.device)
            
            # Add batch dimension if needed
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)
            
            # Generate note sequence
            print(f"Audio features shape: {audio_features.shape}")
            generated_tokens = self.model(audio_features, max_length=max_length)
            print(f"Generated tokens shape: {generated_tokens.shape}")
            print(f"Generated tokens sample: {generated_tokens[0][:10].tolist()}")
            
            # Convert tokens back to notes
            predicted_notes = []
            for token_sequence in generated_tokens:
                notes = self.model.detokenize_notes(token_sequence)
                print(f"Detokenized {len(notes)} notes from sequence")
                predicted_notes.extend(notes)
            
            return predicted_notes
    
    def merge_overlapping_predictions(self, chunk_predictions, time_offsets, overlap=1.0):
        """
        Merge predictions from overlapping chunks
        
        Args:
            chunk_predictions: List of note predictions for each chunk
            time_offsets: Time offsets for each chunk
            overlap: Overlap duration in seconds
            
        Returns:
            Merged list of notes
        """
        if not chunk_predictions:
            return []
        
        merged_notes = []
        
        for i, (notes, offset) in enumerate(zip(chunk_predictions, time_offsets)):
            for start_time, duration, note in notes:
                # Adjust time by chunk offset
                adjusted_start = start_time + offset
                
                # Skip notes in overlap region (except for first chunk)
                if i > 0 and start_time < overlap:
                    continue
                
                merged_notes.append((adjusted_start, duration, note))
        
        # Sort by start time
        merged_notes.sort(key=lambda x: x[0])
        
        return merged_notes
    
    def postprocess_notes(self, notes, min_duration=0.1, max_gap=0.5):
        """
        Post-process notes to remove noise and improve quality
        
        Args:
            notes: List of (start_time, duration, note) tuples
            min_duration: Minimum note duration
            max_gap: Maximum gap to fill between same notes
            
        Returns:
            Cleaned list of notes
        """
        if not notes:
            return notes
        
        cleaned_notes = []
        
        for start_time, duration, note in notes:
            # Filter out very short notes
            if duration < min_duration:
                continue
            
            # Try to merge with previous note if it's the same pitch and close in time
            if (cleaned_notes and 
                cleaned_notes[-1][2] == note and 
                start_time - (cleaned_notes[-1][0] + cleaned_notes[-1][1]) < max_gap):
                
                # Extend previous note
                prev_start, prev_duration, prev_note = cleaned_notes[-1]
                new_duration = start_time + duration - prev_start
                cleaned_notes[-1] = (prev_start, new_duration, prev_note)
            else:
                cleaned_notes.append((start_time, duration, note))
        
        return cleaned_notes
    
    def predict_from_file(self, audio_path, output_path=None, chunk_length=10.0, 
                         overlap=1.0, max_length=500):
        """
        Predict notes from audio file
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save output CSV (optional)
            chunk_length: Length of each processing chunk
            overlap: Overlap between chunks
            max_length: Maximum notes per chunk
            
        Returns:
            List of predicted notes
        """
        print(f"Processing audio file: {audio_path}")
        
        # Preprocess audio
        print("Preprocessing audio...")
        audio_chunks, time_offsets = self.preprocess_long_audio(
            audio_path, chunk_length, overlap
        )
        
        print(f"Split into {len(audio_chunks)} chunks")
        
        # Predict notes for each chunk
        print("Predicting notes...")
        chunk_predictions = []
        
        for i, chunk in enumerate(audio_chunks):
            print(f"Processing chunk {i+1}/{len(audio_chunks)}")
            print(f"Chunk shape: {chunk.shape}")
            notes = self.predict_notes(chunk, max_length)
            print(f"Predicted {len(notes)} notes for chunk {i+1}")
            chunk_predictions.append(notes)
        
        # Merge overlapping predictions
        print("Merging predictions...")
        merged_notes = self.merge_overlapping_predictions(
            chunk_predictions, time_offsets, overlap
        )
        
        # Post-process notes
        print("Post-processing...")
        final_notes = self.postprocess_notes(merged_notes)
        
        print(f"Generated {len(final_notes)} notes")
        
        # Save to CSV if output path specified
        if output_path:
            self.save_notes_to_csv(final_notes, output_path)
            print(f"Notes saved to {output_path}")
        
        return final_notes
    
    def save_notes_to_csv(self, notes, output_path):
        """Save notes to CSV file"""
        df = pd.DataFrame(notes, columns=['start_time', 'duration', 'note'])
        df.to_csv(output_path, index=False)
    
    def visualize_notes(self, notes, output_path=None, figsize=(15, 8)):
        """
        Visualize predicted notes as a piano roll
        
        Args:
            notes: List of (start_time, duration, note) tuples
            output_path: Path to save plot (optional)
            figsize: Figure size
        """
        if not notes:
            print("No notes to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Piano roll visualization
        for start_time, duration, note in notes:
            ax1.barh(note, duration, left=start_time, height=0.8, alpha=0.7)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('MIDI Note')
        ax1.set_title('Piano Roll Visualization')
        ax1.grid(True, alpha=0.3)
        
        # Note distribution
        note_counts = {}
        for _, _, note in notes:
            note_counts[note] = note_counts.get(note, 0) + 1
        
        notes_sorted = sorted(note_counts.items())
        ax2.bar([n[0] for n in notes_sorted], [n[1] for n in notes_sorted])
        ax2.set_xlabel('MIDI Note')
        ax2.set_ylabel('Count')
        ax2.set_title('Note Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.show()
    
    def analyze_predictions(self, notes):
        """Analyze and print statistics about predictions"""
        if not notes:
            print("No notes to analyze")
            return
        
        total_duration = max(start + duration for start, duration, _ in notes)
        note_range = (min(note for _, _, note in notes), max(note for _, _, note in notes))
        avg_duration = np.mean([duration for _, duration, _ in notes])
        
        print(f"\nPrediction Analysis:")
        print(f"Total notes: {len(notes)}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Note range: {note_range[0]} - {note_range[1]} (MIDI)")
        print(f"Average note duration: {avg_duration:.3f} seconds")
        print(f"Notes per second: {len(notes) / total_duration:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Voice-to-Notes Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--output', type=str,
                        help='Path to output CSV file')
    parser.add_argument('--chunk_length', type=float, default=10.0,
                        help='Length of processing chunks in seconds')
    parser.add_argument('--overlap', type=float, default=1.0,
                        help='Overlap between chunks in seconds')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum notes per chunk')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of predicted notes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Initialize inference
    inference = VoiceToNotesInference(args.checkpoint, args.device)
    
    # Generate output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.audio))[0]
        args.output = f"{base_name}_predicted_notes.csv"
    
    # Predict notes
    predicted_notes = inference.predict_from_file(
        args.audio,
        args.output,
        args.chunk_length,
        args.overlap,
        args.max_length
    )
    
    # Analyze predictions
    inference.analyze_predictions(predicted_notes)
    
    # Visualize if requested
    if args.visualize:
        viz_path = args.output.replace('.csv', '_visualization.png')
        inference.visualize_notes(predicted_notes, viz_path)

if __name__ == "__main__":
    main() 