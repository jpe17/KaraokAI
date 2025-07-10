#!/usr/bin/env python3
"""
Universal GTSinger Data Preprocessor:
1. Copy WAV files to organized structure with unique identifiers (including language prefix)
2. Extract note information from JSON files to CSV format
3. Works for all languages in the gtsinger_data directory
"""

import os
import json
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import re

def create_unique_identifier(language, path_parts):
    """
    Create a unique identifier from the folder structure with language prefix
    Example: Spanish/ES-Bass-1/Breathy/Sabes/Breathy_Group/0000 -> ES_ES_Bass_1_Breathy_Sabes_Breathy_Group_0000
    """
    # Get language code from language name
    lang_codes = {
        'Russian': 'RU',
        'Spanish': 'ES', 
        'Italian': 'IT',
        'French': 'FR',
        'Japanese': 'JA',
        'Korean': 'KO'
    }
    
    lang_code = lang_codes.get(language, language[:2].upper())
    
    # Clean up parts for filesystem compatibility
    clean_parts = [lang_code]  # Start with language code
    for part in path_parts:
        # Replace hyphens and spaces with underscores, keep special characters
        clean_part = re.sub(r'[-\s]+', '_', part)
        clean_parts.append(clean_part)
    
    return "_".join(clean_parts)

def extract_notes_from_json(json_path):
    """
    Extract note information from JSON file
    Returns list of notes with start_time, duration, and note value
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        notes = []
        for item in data:
            # Skip silence markers
            if item.get('word') == '<AP>' or item.get('note', [0])[0] == 0:
                continue
            
            # Extract note information
            note_start = item.get('note_start', [])
            note_dur = item.get('note_dur', [])
            note_values = item.get('note', [])
            
            # Handle both single values and lists
            if not isinstance(note_start, list):
                note_start = [note_start]
            if not isinstance(note_dur, list):
                note_dur = [note_dur]
            if not isinstance(note_values, list):
                note_values = [note_values]
            
            # Create note entries
            for i in range(max(len(note_start), len(note_dur), len(note_values))):
                start = note_start[i] if i < len(note_start) else note_start[-1] if note_start else 0
                duration = note_dur[i] if i < len(note_dur) else note_dur[-1] if note_dur else 0
                note_val = note_values[i] if i < len(note_values) else note_values[-1] if note_values else 0
                
                if note_val > 0:  # Only include actual notes, not silence
                    notes.append({
                        'start_time': float(start),
                        'duration': float(duration),
                        'note': int(note_val)
                    })
        
        return notes
    
    except Exception as e:
        print(f"❌ Error processing JSON {json_path}: {e}")
        return []

def preprocess_all_gtsinger_data(input_dir="gtsinger_data", output_dir="preprocessed_gtsinger"):
    """
    Preprocess all GTSinger data from all languages
    """
    print("🎵 Universal GTSinger Data Preprocessor")
    print("=" * 50)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    voices_dir = output_path / "voices"
    notes_dir = output_path / "notes"
    
    voices_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input directory: {input_path.absolute()}")
    print(f"📁 Output directories:")
    print(f"  Voices: {voices_dir.absolute()}")
    print(f"  Notes: {notes_dir.absolute()}")
    
    # Find all language directories
    language_dirs = []
    for item in input_path.iterdir():
        if item.is_dir() and item.name not in ['.git', '__pycache__']:
            language_dirs.append(item)
    
    print(f"🌍 Found {len(language_dirs)} language directories: {[d.name for d in language_dirs]}")
    
    # Find all numbered audio/json pairs across all languages
    audio_json_pairs = []
    
    for lang_dir in language_dirs:
        language = lang_dir.name
        print(f"\n🔍 Scanning {language} data...")
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(lang_dir):
            root_path = Path(root)
            
            # Skip Paired_Speech_Group folders (contains speech, not singing)
            if "Paired_Speech_Group" in str(root_path):
                continue
            
            # Find numbered files (0000, 0001, etc.)
            numbered_files = {}
            for file in files:
                if file.endswith(('.wav', '.json')):
                    # Extract number (0000, 0001, etc.)
                    name_parts = file.split('.')
                    if len(name_parts) >= 2 and name_parts[0].isdigit():
                        number = name_parts[0]
                        extension = name_parts[1]
                        
                        if number not in numbered_files:
                            numbered_files[number] = {}
                        numbered_files[number][extension] = root_path / file
            
            # Create pairs for files that have both WAV and JSON
            for number, files_dict in numbered_files.items():
                if 'wav' in files_dict and 'json' in files_dict:
                    # Create path components for unique identifier
                    relative_path = root_path.relative_to(lang_dir)
                    path_parts = list(relative_path.parts) + [number]
                    
                    audio_json_pairs.append({
                        'language': language,
                        'number': number,
                        'wav_path': files_dict['wav'],
                        'json_path': files_dict['json'],
                        'path_parts': path_parts,
                        'unique_id': create_unique_identifier(language, path_parts)
                    })
        
        print(f"  Found {len([p for p in audio_json_pairs if p['language'] == language])} pairs in {language}")
    
    print(f"\n📊 Total audio-JSON pairs found: {len(audio_json_pairs)}")
    
    processed_count = 0
    failed_count = 0
    language_stats = {}
    
    # Process each pair
    for pair in tqdm(audio_json_pairs, desc="Processing files"):
        try:
            language = pair['language']
            unique_id = pair['unique_id']
            wav_path = pair['wav_path']
            json_path = pair['json_path']
            
            # Track language statistics
            if language not in language_stats:
                language_stats[language] = {'processed': 0, 'failed': 0}
            
            # Copy WAV file with unique identifier
            output_wav_path = voices_dir / f"{unique_id}.wav"
            shutil.copy2(wav_path, output_wav_path)
            
            # Extract notes from JSON
            notes = extract_notes_from_json(json_path)
            
            if notes:
                # Create DataFrame and save as CSV
                notes_df = pd.DataFrame(notes)
                # Sort by start time
                notes_df = notes_df.sort_values('start_time').reset_index(drop=True)
                
                output_csv_path = notes_dir / f"{unique_id}.csv"
                notes_df.to_csv(output_csv_path, index=False)
                
                processed_count += 1
                language_stats[language]['processed'] += 1
                
                # Show progress for first few files
                if processed_count <= 5:
                    print(f"\n✅ Processed: {unique_id}")
                    print(f"   Language: {language}")
                    print(f"   WAV: {wav_path.name} -> {output_wav_path.name}")
                    print(f"   Notes: {len(notes)} notes extracted -> {output_csv_path.name}")
                    if notes:
                        print(f"   Sample notes: {notes[:2] if len(notes) >= 2 else notes}")
            else:
                print(f"⚠️  No valid notes found in {json_path}")
                failed_count += 1
                language_stats[language]['failed'] += 1
                
        except Exception as e:
            print(f"❌ Error processing {pair.get('unique_id', 'unknown')}: {e}")
            failed_count += 1
            if pair.get('language') in language_stats:
                language_stats[pair['language']]['failed'] += 1
            continue
    
    print(f"\n🎉 Preprocessing completed!")
    print(f"📊 Overall Statistics:")
    print(f"  Total pairs found: {len(audio_json_pairs)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    
    print(f"\n📊 Statistics by Language:")
    for language, stats in language_stats.items():
        total = stats['processed'] + stats['failed']
        success_rate = (stats['processed'] / total * 100) if total > 0 else 0
        print(f"  {language}: {stats['processed']}/{total} processed ({success_rate:.1f}% success)")
    
    # Show sample of created files
    voice_files = list(voices_dir.glob('*.wav'))
    note_files = list(notes_dir.glob('*.csv'))
    
    print(f"\n📁 Created files:")
    print(f"  Voice files: {len(voice_files)}")
    print(f"  Notes files: {len(note_files)}")
    
    # Show samples by language
    if voice_files:
        print(f"\n📄 Sample voice files by language:")
        for lang_code in ['RU', 'ES', 'IT', 'FR', 'JA', 'KO']:
            lang_files = [f for f in voice_files if f.name.startswith(f"{lang_code}_")]
            if lang_files:
                print(f"  {lang_code}: {len(lang_files)} files")
                for f in lang_files[:2]:  # Show first 2 from each language
                    print(f"    {f.name}")
    
    if note_files:
        print(f"\n📄 Sample notes files:")
        for f in note_files[:3]:
            print(f"    {f.name}")
            try:
                df = pd.read_csv(f)
                print(f"      Shape: {df.shape}")
                print(f"      Columns: {list(df.columns)}")
                print(f"      Sample rows:")
                for idx, row in df.head(2).iterrows():
                    print(f"        {row.to_dict()}")
            except Exception as e:
                print(f"      Error reading CSV: {e}")
    
    return processed_count > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess all GTSinger data from all languages")
    parser.add_argument("--input_dir", default="gtsinger_data", 
                       help="Input directory containing language subdirectories")
    parser.add_argument("--output_dir", default="preprocessed_gtsinger", 
                       help="Output directory for preprocessed data")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"❌ Input directory '{args.input_dir}' does not exist")
        print("Make sure you have downloaded the GTSinger data first")
        exit(1)
    
    # Process the data
    success = preprocess_all_gtsinger_data(args.input_dir, args.output_dir)
    
    if success:
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"📁 Check the '{args.output_dir}' directory:")
        print(f"  - voices/ (contains .wav files with unique identifiers)")
        print(f"  - notes/ (contains .csv files with note data)")
        print(f"")
        print(f"🎵 File naming convention:")
        print(f"  [LANG]_[Singer]_[Style]_[Song]_[Group]_[Number]")
        print(f"  Example: ES_ES_Bass_1_Breathy_Sabes_Breathy_Group_0000")
    else:
        print(f"\n💥 Preprocessing failed") 