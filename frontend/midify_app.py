#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import torch
import torchaudio
import numpy as np
import pandas as pd
import pretty_midi
import base64
import io
from datetime import datetime

# Import our model and inference classes
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path
from model import VoiceToNotesModel
from inference import VoiceToNotesInference
from dataloader import AudioPreprocessor

app = Flask(__name__)

# Global inference model - will be loaded on startup
inference_model = None
audio_preprocessor = None

# Directory to store temporary audio files for playback
# Use absolute path to ensure it works regardless of where Flask is run from
TEMP_AUDIO_DIR = Path(__file__).parent / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
print(f"Temp audio directory: {TEMP_AUDIO_DIR.absolute()}")

def cleanup_old_audio_files():
    """Remove audio files older than 1 hour to prevent disk space issues"""
    import time
    current_time = time.time()
    for audio_file in TEMP_AUDIO_DIR.glob("*.wav"):
        if current_time - audio_file.stat().st_mtime > 3600:  # 1 hour
            try:
                audio_file.unlink()
            except:
                pass

def init_model():
    """Initialize the trained model for inference"""
    global inference_model, audio_preprocessor
    
    # Get the project root directory (parent of frontend)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_epoch_16.pth")
    
    if not os.path.exists(checkpoint_path):
        # Try alternative checkpoint names
        checkpoint_path = os.path.join(project_root, "checkpoints", "checkpoint_epoch_16.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: No model checkpoint found at {checkpoint_path}. Model inference will not work.")
            return False
    
    try:
        print("Loading trained model...")
        inference_model = VoiceToNotesInference(checkpoint_path, device='cpu')
        audio_preprocessor = AudioPreprocessor()
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/')
def index():
    """Serves the main midify interface."""
    return render_template('midify.html')

@app.route('/api/convert_vocal', methods=['POST'])
def api_convert_vocal():
    """
    Converts recorded vocal audio to MIDI notes using the trained model.
    Expects audio data as base64 encoded WAV in the request.
    """
    try:
        if inference_model is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 500
        
        # Get audio data from request
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio data
        audio_data = data['audio_data']
        if audio_data.startswith('data:audio/webm;base64,'):
            audio_data = audio_data.split(',')[1]
        elif audio_data.startswith('data:audio/wav;base64,'):
            audio_data = audio_data.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_data)
        
        # Save the original audio for playback (like karaoke app)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"recording_{timestamp}_original.webm"
        original_file_path = TEMP_AUDIO_DIR / original_filename
        
        with open(original_file_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"Saved original file: {original_file_path.absolute()}")
        print(f"File exists: {original_file_path.exists()}")
        print(f"File size: {original_file_path.stat().st_size} bytes")
        
        # Convert to WAV for model processing only
        wav_filename = f"recording_{timestamp}.wav"
        wav_file_path = TEMP_AUDIO_DIR / wav_filename
        
        # Simple conversion using ffmpeg (like karaoke approach)
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', str(original_file_path), 
                '-ar', '44100', '-ac', '1', '-y',
                str(wav_file_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                temp_file_path = str(wav_file_path)
                print(f"Successfully converted to WAV for processing")
            else:
                print(f"Conversion failed: {result.stderr}")
                return jsonify({"error": "Audio conversion failed"}), 500
        except Exception as e:
            print(f"Error converting audio: {e}")
            return jsonify({"error": "Audio conversion failed"}), 500
        
        try:
            # Convert audio to notes using the trained model (optimized for speed)
            print(f"Converting audio file: {temp_file_path}")
            predicted_notes = inference_model.predict_from_file(
                temp_file_path,
                chunk_length=5.0,  # Reduced from 10.0 for faster processing
                overlap=0.5,       # Reduced from 1.0 for faster processing
                max_length=50      # Reduced from 100 for faster processing
            )
            
            print(f"Predicted {len(predicted_notes)} notes")
            
            # Convert to format expected by frontend
            notes_data = []
            for start_time, duration, midi_note in predicted_notes:
                notes_data.append({
                    'start_time': float(start_time),
                    'duration': float(duration),
                    'midi_note': int(midi_note),
                    'note_name': pretty_midi.note_number_to_name(int(midi_note))
                })
            
            # Get audio duration for playback
            try:
                audio_tensor, sample_rate = torchaudio.load(temp_file_path)
                audio_duration = audio_tensor.shape[1] / sample_rate
            except:
                audio_duration = max([note['start_time'] + note['duration'] for note in notes_data]) if notes_data else 5.0
            
            # Use the original WebM file for playback (browsers handle this well)
            response_data = {
                "song_name": f"Your Recording - {datetime.now().strftime('%H:%M:%S')}",
                "notes": notes_data,
                "audio_duration": float(audio_duration),
                "audio_url": f"/temp_audio/{original_filename}"  # Serve original WebM for playback
            }
            
            print(f"Saved files: {original_filename} (playback), {wav_filename} (processing)")
            print(f"Audio duration: {audio_duration} seconds")
            
            return jsonify(response_data)
            
        finally:
            # Clean up only the temporary processing file, keep the playback files
            try:
                if 'temp_file_path' in locals() and temp_file_path != str(wav_file_path):
                    os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error in vocal conversion: {e}")
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500

@app.route('/api/model_status')
def api_model_status():
    """Check if the model is loaded and ready"""
    return jsonify({
        "model_loaded": inference_model is not None,
        "status": "ready" if inference_model is not None else "model not found"
    })

@app.route('/temp_audio/<path:filename>')
def serve_temp_audio(filename):
    """Serves temporary audio files for playback"""
    print(f"Serving audio file: {filename} from {TEMP_AUDIO_DIR.absolute()}")
    file_path = TEMP_AUDIO_DIR / filename
    print(f"Full file path: {file_path.absolute()}")
    print(f"File exists: {file_path.exists()}")
    
    if not file_path.exists():
        # List all files in the directory for debugging
        print(f"Files in temp_audio directory:")
        for f in TEMP_AUDIO_DIR.iterdir():
            print(f"  - {f.name}")
        return f"Audio file not found: {filename}", 404
    
    try:
        response = send_from_directory(TEMP_AUDIO_DIR, filename)
        # Set proper content type based on file extension
        if filename.endswith('.webm'):
            response.headers['Content-Type'] = 'audio/webm'
        elif filename.endswith('.wav'):
            response.headers['Content-Type'] = 'audio/wav'
        else:
            response.headers['Content-Type'] = 'audio/webm'  # Default
        response.headers['Accept-Ranges'] = 'bytes'
        print(f"Successfully serving {filename}")
        return response
    except Exception as e:
        print(f"Error serving audio file {filename}: {e}")
        return f"Audio file not found: {filename}", 404

@app.route('/api/test_audio', methods=['POST'])
def api_test_audio():
    """
    Test endpoint that generates some dummy notes for testing the interface
    when the model isn't available
    """
    try:
        # Generate some test notes
        test_notes = []
        base_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
        
        for i, note in enumerate(base_notes):
            start_time = i * 0.5
            duration = 0.4
            test_notes.append({
                'start_time': start_time,
                'duration': duration,
                'midi_note': note,
                'note_name': pretty_midi.note_number_to_name(note)
            })
        
        response_data = {
            "song_name": "Test Pattern - C Major Scale",
            "notes": test_notes,
            "audio_duration": 4.0,
            "audio_url": None  # No audio for test mode
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"Test failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🎤 Starting Midify App...")
    
    # Clean up old audio files
    cleanup_old_audio_files()
    
    # Try to load the model
    model_loaded = init_model()
    if not model_loaded:
        print("⚠️  Running without model - only test mode will work")
    
    app.run(debug=True, host='0.0.0.0', port=5223) 