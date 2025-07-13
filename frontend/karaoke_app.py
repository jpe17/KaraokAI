#!/usr/bin/env python3
import os
import random
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import pretty_midi

app = Flask(__name__)

# Define the base directory for instrumentals (same directory as this script)
SCRIPT_DIR = Path(__file__).parent
INSTRUMENTAL_DIR = SCRIPT_DIR

@app.route('/')
def index():
    """Serves the main karaoke interface."""
    return render_template('karaoke.html')

@app.route('/api/random_song')
def api_random_song():
    """
    Finds a random song and returns its metadata and a URL to the instrumental audio.
    This endpoint is now extremely fast as it does no audio processing.
    """
    notes_dir = SCRIPT_DIR  # Frontend folder where script is located
    
    if not INSTRUMENTAL_DIR.exists():
        return jsonify({"error": "Data directories not found on the server."}), 500
        
    note_files = list(notes_dir.glob("voice_*.csv"))
    if not note_files:
        return jsonify({"error": "No note files found."}), 500

    # Loop to find a song that has a matching instrumental
    while note_files:
        selected_note_file = random.choice(note_files)
        song_id = selected_note_file.stem.replace("voice_", "")
        
        # Look for the corresponding instrumental file
        instrumental_file = INSTRUMENTAL_DIR / f"instrumental_{song_id}.wav"
        if instrumental_file.exists():
            # Found a match, prepare the data for the frontend
            df = pd.read_csv(selected_note_file)
            notes_data = [
                {
                    'start_time': float(row['start_time_sec']),
                    'duration': float(row['duration_sec']),
                    'midi_note': int(row['midi_note']),
                    'note_name': pretty_midi.note_number_to_name(int(row['midi_note']))
                } for _, row in df.iterrows()
            ]
            
            response_data = {
                "song_name": song_id.replace("_", " ").title(),
                "notes": notes_data,
                "instrumental_url": f"/instrumentals/{instrumental_file.name}"
            }
            return jsonify(response_data)
        else:
            # If no match, remove this file from the list and try another
            note_files.remove(selected_note_file)
            
    return jsonify({"error": "No songs with matching instrumentals could be found."}), 404

@app.route('/instrumentals/<path:filename>')
def serve_instrumental(filename):
    """Serves the instrumental audio files directly to the browser's <audio> tag."""
    return send_from_directory(INSTRUMENTAL_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5222) 