import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
import os
import glob
import librosa
from transformers import WhisperFeatureExtractor
import random

class AudioPreprocessor:
    """Audio preprocessing pipeline optimized for Whisper input"""
    
    def __init__(self, 
                 target_sr=16000,
                 n_fft=400,
                 hop_length=160,
                 n_mels=80,
                 highpass_freq=80,
                 target_lufs=-20.0):
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.highpass_freq = highpass_freq
        self.target_lufs = target_lufs
        
        # Audio transforms
        self.resample = None
        self.highpass = None
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=target_sr // 2,
            power=2.0,
            normalized=False
        )
        
    def add_gaussian_noise(self, audio, noise_factor=0.005):
        """Add Gaussian noise to simulate recording conditions"""
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
        
    def normalize_loudness(self, audio, target_lufs=-20.0):
        """Normalize audio to target LUFS (simplified version)"""
        # Simple RMS-based normalization (approximation of LUFS)
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_lufs / 20.0)
            audio = audio * (target_rms / rms)
        return audio
        
    def apply_highpass_filter(self, audio, sr):
        """Apply highpass filter to remove low frequency noise"""
        # Simple highpass using frequency domain filtering
        if audio.shape[-1] < 1024:  # Skip very short audio
            return audio
        
        # FFT-based highpass filter
        fft = torch.fft.rfft(audio, dim=-1)
        freqs = torch.fft.rfftfreq(audio.shape[-1], 1/sr)
        
        # Create highpass mask
        mask = freqs >= self.highpass_freq
        fft = fft * mask.unsqueeze(0)
        
        # Convert back to time domain
        return torch.fft.irfft(fft, n=audio.shape[-1], dim=-1)
        
    def to_mono(self, audio):
        """Convert stereo to mono"""
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio
        
    def resample_audio(self, audio, orig_sr):
        """Resample audio to target sample rate"""
        if orig_sr != self.target_sr:
            if self.resample is None or self.resample.orig_freq != orig_sr:
                self.resample = T.Resample(orig_sr, self.target_sr)
            audio = self.resample(audio)
        return audio
        
    def simple_dereverb(self, audio, alpha=0.7):
        """Simple spectral subtraction for dereverberation"""
        # This is a simplified dereverb - you could use more sophisticated methods
        stft = torch.stft(audio.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length, 
                         return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Simple spectral subtraction
        noise_floor = torch.quantile(magnitude, 0.1, dim=-1, keepdim=True)
        magnitude_clean = magnitude - alpha * noise_floor
        magnitude_clean = torch.maximum(magnitude_clean, 0.1 * magnitude)
        
        # Reconstruct
        stft_clean = magnitude_clean * torch.exp(1j * phase)
        audio_clean = torch.istft(stft_clean, n_fft=self.n_fft, hop_length=self.hop_length)
        return audio_clean.unsqueeze(0)
        
    def preprocess(self, audio_path, add_noise=True):
        """Complete preprocessing pipeline"""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        audio = self.to_mono(audio)
        
        # Resample
        audio = self.resample_audio(audio, sr)
        
        # Apply highpass filter
        audio = self.apply_highpass_filter(audio, self.target_sr)
        
        # Normalize loudness
        audio = self.normalize_loudness(audio, self.target_lufs)
        
        # Simple dereverb
        audio = self.simple_dereverb(audio)
        
        # Add noise (for training robustness)
        if add_noise:
            audio = self.add_gaussian_noise(audio)
            
        return audio

class VoiceNotesDataset(Dataset):
    """Dataset for voice-to-notes training"""
    
    def __init__(self, 
                 voices_dir,
                 notes_dir,
                 whisper_model_name="openai/whisper-base",
                 max_audio_length=10.0,
                 max_notes=100,
                 add_noise=True,
                 max_samples=None,  # NEW: Limit number of samples for faster development
                 cache_features=False):  # NEW: Cache features for faster loading
        
        self.voices_dir = voices_dir
        self.notes_dir = notes_dir
        self.max_audio_length = max_audio_length
        self.max_notes = max_notes
        self.add_noise = add_noise
        self.max_samples = max_samples
        self.cache_features = cache_features
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        
        # Find all audio files and matching note files
        self.audio_files = []
        self.note_files = []
        
        self._find_matching_files()
        
        # NEW: Cache for faster loading
        self.feature_cache = {}
        
    def _find_matching_files(self):
        """Find all audio files with matching note files"""
        audio_files = glob.glob(os.path.join(self.voices_dir, "*.wav"))
        
        # NEW: Limit samples for faster development - RANDOM SAMPLING
        if self.max_samples:
            # Shuffle and take random samples from the full pool
            random.shuffle(audio_files)
            audio_files = audio_files[:self.max_samples]
            print(f"Randomly sampling {self.max_samples} samples from {len(glob.glob(os.path.join(self.voices_dir, '*.wav')))} total files for faster development")
        
        for audio_file in audio_files:
            # Extract base name without extension
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            # Look for corresponding CSV file
            # Try multiple patterns to handle naming mismatches
            possible_note_files = [
                os.path.join(self.notes_dir, base_name + ".csv"),
                os.path.join(self.notes_dir, base_name.replace("_voices", "_notes") + ".csv")
            ]
            
            # Also try to find any CSV file with similar pattern
            csv_files = glob.glob(os.path.join(self.notes_dir, "*.csv"))
            for csv_file in csv_files:
                csv_base = os.path.splitext(os.path.basename(csv_file))[0]
                # Simple matching - if they share significant common substring
                if len(set(base_name.split("_")) & set(csv_base.split("_"))) > 3:
                    possible_note_files.append(csv_file)
            
            # Use the first matching file found
            note_file = None
            for nf in possible_note_files:
                if os.path.exists(nf):
                    note_file = nf
                    break
            
            if note_file:
                self.audio_files.append(audio_file)
                self.note_files.append(note_file)
                
        print(f"Found {len(self.audio_files)} audio-note pairs")
        if len(self.audio_files) == 0:
            print("Warning: No matching audio-note pairs found!")
            print(f"Audio files sample: {audio_files[:3]}")
            print(f"Note files sample: {glob.glob(os.path.join(self.notes_dir, '*.csv'))[:3]}")
        
    def _load_notes(self, note_file):
        """Load notes from CSV file"""
        try:
            df = pd.read_csv(note_file)
            notes = []
            for _, row in df.iterrows():
                start_time = float(row['start_time'])
                duration = float(row['duration'])
                note = int(row['note'])
                notes.append((start_time, duration, note))
            return notes
        except Exception as e:
            print(f"Error loading notes from {note_file}: {e}")
            return []
    
    def _pad_or_trim_audio(self, audio):
        """Pad or trim audio to max length"""
        target_length = int(self.max_audio_length * self.preprocessor.target_sr)
        current_length = audio.shape[-1]
        
        if current_length > target_length:
            # Trim
            audio = audio[:, :target_length]
        elif current_length < target_length:
            # Pad
            padding = target_length - current_length
            audio = F.pad(audio, (0, padding))
            
        return audio
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        note_file = self.note_files[idx]
        
        # NEW: Check cache first
        cache_key = f"{audio_file}_{self.add_noise}"
        if self.cache_features and cache_key in self.feature_cache:
            audio_features = self.feature_cache[cache_key]
        else:
            # Load and preprocess audio
            audio = self.preprocessor.preprocess(audio_file, add_noise=self.add_noise)
            audio = self._pad_or_trim_audio(audio)
            
            # Extract features using Whisper feature extractor
            audio_features = self.feature_extractor(
                audio.squeeze().numpy(),
                sampling_rate=self.preprocessor.target_sr,
                return_tensors="pt"
            )["input_features"]
            
            # NEW: Cache features
            if self.cache_features:
                self.feature_cache[cache_key] = audio_features
        
        # Load notes
        notes = self._load_notes(note_file)
        
        # Limit number of notes
        if len(notes) > self.max_notes:
            notes = notes[:self.max_notes]
            
        return {
            'audio_features': audio_features.squeeze(0),
            'notes': notes,
            'audio_file': audio_file,
            'note_file': note_file
        }

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    audio_features = torch.stack([item['audio_features'] for item in batch])
    
    # Find all unique notes for tokenization
    all_notes = [item['notes'] for item in batch]
    
    # For now, return raw data - tokenization will be handled in training loop
    return {
        'audio_features': audio_features,
        'notes': all_notes,
        'audio_files': [item['audio_file'] for item in batch],
        'note_files': [item['note_file'] for item in batch]
    }

def create_dataloaders(voices_dir, 
                      notes_dir,
                      batch_size=8,
                      train_split=0.8,
                      whisper_model_name="openai/whisper-base",
                      max_audio_length=10.0,
                      max_notes=100,
                      num_workers=0,  # CHANGED: Reduced for faster development
                      max_samples=None,  # NEW: Limit samples
                      cache_features=False):  # NEW: Cache features
    """Create train and validation dataloaders"""
    
    # Create full dataset
    dataset = VoiceNotesDataset(
        voices_dir=voices_dir,
        notes_dir=notes_dir,
        whisper_model_name=whisper_model_name,
        max_audio_length=max_audio_length,
        max_notes=max_notes,
        add_noise=True,
        max_samples=max_samples,  # NEW: Pass sample limit
        cache_features=cache_features  # NEW: Pass cache option
    )
    
    if len(dataset) == 0:
        raise ValueError("No data found! Check your data paths.")
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create validation dataset with no noise
    val_dataset.dataset.add_noise = False
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,  # CHANGED: Reduced workers
        pin_memory=False  # CHANGED: Disabled for CPU
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,  # CHANGED: Reduced workers
        pin_memory=False  # CHANGED: Disabled for CPU
    )
    
    return train_dataloader, val_dataloader

# Test function
if __name__ == "__main__":
    # Test the dataloader
    voices_dir = "processed_data/voices"
    notes_dir = "processed_data/notes"
    
    if os.path.exists(voices_dir) and os.path.exists(notes_dir):
        # NEW: Test with limited samples for faster development
        train_dl, val_dl = create_dataloaders(
            voices_dir, notes_dir, 
            batch_size=2, 
            max_samples=10,  # NEW: Limit to 10 samples
            cache_features=True  # NEW: Enable caching
        )
        print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")
        
        # Test one batch
        batch = next(iter(train_dl))
        print(f"Audio features shape: {batch['audio_features'].shape}")
        print(f"Number of note sequences: {len(batch['notes'])}")
        print(f"Sample notes: {batch['notes'][0][:5] if batch['notes'][0] else 'No notes'}")
    else:
        print("Data directories not found!") 