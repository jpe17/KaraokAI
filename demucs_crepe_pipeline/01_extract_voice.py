import os
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def load_model():
    """Load the DEMUCS model"""
    try:
        from demucs.pretrained import get_model
        print("Loading htdemucs model...")
        model = get_model('htdemucs')
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def process_songs():
    """Process all MP3 files using DEMUCS Python API"""
    model = load_model()
    if model is None:
        return
    
    data_folder = Path("./00_data")
    mp3_files = list(data_folder.glob("*.mp3"))
    
    if not mp3_files:
        print("No MP3 files found in ./00_data folder")
        return
    
    print(f"Found {len(mp3_files)} MP3 files to process")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Process each file
    for i, mp3_file in enumerate(mp3_files, 1):
        print(f"\n[{i}/{len(mp3_files)}] Processing: {mp3_file.name}")
        
        try:
            # Load audio file using librosa
            print("  Loading audio...")
            waveform, sample_rate = librosa.load(str(mp3_file), sr=None, mono=False)
            
            # Ensure stereo format
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform], axis=0)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform).float()
            waveform = waveform.to(device)
            
            # Apply the model using apply_model function (THIS IS THE FIX)
            print("  Separating audio...")
            with torch.no_grad():
                from demucs.apply import apply_model  # Import apply_model
                sources = apply_model(model, waveform.unsqueeze(0), device=device)
                
                # Extract vocals and instrumental
                vocals = sources[0, 3]  # vocals track
                instrumental = sources[0, 0] + sources[0, 1] + sources[0, 2]  # drums + bass + other
            
            # Move back to CPU for saving
            vocals = vocals.cpu().numpy()
            instrumental = instrumental.cpu().numpy()
            
            # Save the separated files
            song_name = mp3_file.stem
            
            # Save vocal file
            vocal_path = Path("./01_demucs/voice") / f"voice_{song_name}.wav"
            sf.write(str(vocal_path), vocals.T, sample_rate)
            print(f"  ✅ Saved: voice_{song_name}.wav")
            
            # Save instrumental file
            instrumental_path = Path("./01_demucs/instrumental") / f"instrumental_{song_name}.wav"
            sf.write(str(instrumental_path), instrumental.T, sample_rate)
            print(f"  ✅ Saved: instrumental_{song_name}.wav")
            
        except Exception as e:
            print(f"  ❌ Error processing {mp3_file.name}: {e}")
            continue

def setup_directories():
    """Create the required directory structure"""
    os.makedirs("./01_demucs/voice", exist_ok=True)
    os.makedirs("./01_demucs/instrumental", exist_ok=True)
    print("✅ Directories created")

def main():
    """Main function"""
    setup_directories()
    process_songs()
    print("\n🎉 Processing complete!")

if __name__ == "__main__":
    main()
