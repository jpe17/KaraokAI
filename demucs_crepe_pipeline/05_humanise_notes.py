import csv
import random
import numpy as np
from typing import List, Tuple
import math

class CoreVocalHumanizer:
    def __init__(self):
        # Most impactful parameters only
        self.timing_rules = {
            'micro_variance': 0.012,      # ±12ms - sweet spot for natural feel
            'phrase_end_drag': 0.035,     # 35ms drag at phrase ends
            'large_leap_hesitation': 0.025, # 25ms before big jumps
            'breath_gap': 0.08            # 80ms breathing gaps
        }
        
        self.duration_rules = {
            'min_vocal_duration': 0.18,   # 180ms minimum for clarity
            'high_note_boost': 1.25,      # 25% longer for notes >C5
            'phrase_end_boost': 1.4,      # 40% longer at phrase ends
            'stepwise_tightening': 0.92   # 8% shorter for stepwise motion
        }
        
        self.note_rules = {
            'comfort_range': (55, 75),    # G3 to D#5 - optimal vocal range
            'leap_threshold': 7,          # Semitones - when to add passing tones
            'octave_comfort': 12          # Prefer octave jumps over awkward intervals
        }
    
    def load_data(self, filepath: str) -> List[Tuple[float, float, int]]:
        """Load MIDI data efficiently"""
        data = []
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append((float(row['start_time_sec']), 
                           float(row['duration_sec']), 
                           int(row['midi_note'])))
        return data
    
    def detect_phrases(self, data: List[Tuple[float, float, int]]) -> List[bool]:
        """Fast phrase boundary detection"""
        is_phrase_end = [False] * len(data)
        
        for i in range(len(data) - 1):
            current_end = data[i][0] + data[i][1]
            next_start = data[i + 1][0]
            gap = next_start - current_end
            
            # Mark phrase end if gap > 300ms or large melodic leap
            if gap > 0.3 or abs(data[i+1][2] - data[i][2]) > 8:
                is_phrase_end[i] = True
        
        is_phrase_end[-1] = True  # Last note is always phrase end
        return is_phrase_end
    
    def optimize_timing(self, data: List[Tuple[float, float, int]], 
                       phrase_ends: List[bool]) -> List[float]:
        """Most effective timing adjustments"""
        new_times = []
        
        for i, (start_time, duration, note) in enumerate(data):
            adjusted_time = start_time
            
            # 1. Micro-timing variance (always applied)
            adjusted_time += random.uniform(-self.timing_rules['micro_variance'], 
                                          self.timing_rules['micro_variance'])
            
            # 2. Phrase-end drag (high impact)
            if phrase_ends[i]:
                adjusted_time += random.uniform(0, self.timing_rules['phrase_end_drag'])
            
            # 3. Large leap hesitation (natural vocal behavior)
            if i > 0:
                interval = abs(note - data[i-1][2])
                if interval > self.note_rules['leap_threshold']:
                    adjusted_time += random.uniform(0, self.timing_rules['large_leap_hesitation'])
            
            # 4. Breathing gaps (physiological necessity)
            if i > 0:
                prev_end = new_times[i-1] + data[i-1][1]  # Use previous adjusted time
                if adjusted_time - prev_end < self.timing_rules['breath_gap']:
                    adjusted_time = prev_end + self.timing_rules['breath_gap']
            
            new_times.append(max(0, adjusted_time))  # Prevent negative times
        
        return new_times
    
    def optimize_durations(self, data: List[Tuple[float, float, int]], 
                          phrase_ends: List[bool]) -> List[float]:
        """Most effective duration adjustments"""
        new_durations = []
        
        for i, (start_time, duration, note) in enumerate(data):
            adjusted_duration = duration
            
            # 1. Minimum duration enforcement (critical for vocal clarity)
            if adjusted_duration < self.duration_rules['min_vocal_duration']:
                adjusted_duration = self.duration_rules['min_vocal_duration']
            
            # 2. High note emphasis (singers naturally hold high notes)
            if note > 72:  # Above C5
                adjusted_duration *= self.duration_rules['high_note_boost']
            
            # 3. Phrase end elongation (natural musical phrasing)
            if phrase_ends[i]:
                adjusted_duration *= self.duration_rules['phrase_end_boost']
            
            # 4. Stepwise motion tightening (faster articulation)
            if i > 0 and abs(note - data[i-1][2]) <= 2:
                adjusted_duration *= self.duration_rules['stepwise_tightening']
            
            new_durations.append(adjusted_duration)
        
        return new_durations
    
    def optimize_notes(self, data: List[Tuple[float, float, int]]) -> List[int]:
        """Most effective note adjustments"""
        new_notes = []
        
        for i, (start_time, duration, note) in enumerate(data):
            adjusted_note = note
            
            # 1. Vocal range comfort (transpose octaves if needed)
            if note < self.note_rules['comfort_range'][0]:
                # Too low - move up octave
                adjusted_note = note + 12
            elif note > self.note_rules['comfort_range'][1]:
                # Too high - move down octave
                adjusted_note = note - 12
            
            # 2. Large leap mitigation (add passing tones strategically)
            if i > 0:
                prev_note = new_notes[i-1]
                interval = abs(adjusted_note - prev_note)
                
                if interval > self.note_rules['leap_threshold']:
                    # For very large leaps, consider intermediate note
                    if interval > 10 and random.random() < 0.3:  # 30% chance
                        # Move toward the middle of the interval
                        direction = 1 if adjusted_note > prev_note else -1
                        adjusted_note = prev_note + (interval // 2) * direction
            
            new_notes.append(adjusted_note)
        
        return new_notes
    
    def humanize_core(self, input_file: str, output_file: str):
        """Streamlined humanization process"""
        print(f"Loading data from {input_file}...")
        data = self.load_data(input_file)
        
        print("Analyzing phrase structure...")
        phrase_ends = self.detect_phrases(data)
        
        print("Optimizing timing...")
        new_times = self.optimize_timing(data, phrase_ends)
        
        print("Optimizing durations...")
        new_durations = self.optimize_durations(data, phrase_ends)
        
        print("Optimizing notes...")
        new_notes = self.optimize_notes(data)
        
        # Save results
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['start_time_sec', 'duration_sec', 'midi_note'])
            
            for i in range(len(data)):
                writer.writerow([
                    f"{new_times[i]:.6f}",
                    f"{new_durations[i]:.6f}",
                    new_notes[i]
                ])
        
        print(f"Results saved to {output_file}")
        
        # Show impact summary
        self._show_impact_summary(data, new_times, new_durations, new_notes)
    
    def _show_impact_summary(self, original_data, new_times, new_durations, new_notes):
        """Show the most impactful changes"""
        print("\n🎯 CORE OPTIMIZATION RESULTS")
        print("=" * 40)
        
        # Timing changes
        timing_changes = [abs(new_times[i] - original_data[i][0]) for i in range(len(original_data))]
        print(f"Timing adjustments: {np.mean(timing_changes):.3f}s average")
        
        # Duration changes
        duration_changes = [new_durations[i] / original_data[i][1] for i in range(len(original_data))]
        print(f"Duration scaling: {np.mean(duration_changes):.2f}x average")
        
        # Note changes
        note_changes = sum(1 for i in range(len(original_data)) if new_notes[i] != original_data[i][2])
        print(f"Notes adjusted: {note_changes}/{len(original_data)} ({note_changes/len(original_data)*100:.1f}%)")
        
        # Show most significant transformations
        print("\n🔥 Most Impactful Changes:")
        for i in range(min(3, len(original_data))):
            orig_time, orig_dur, orig_note = original_data[i]
            print(f"  Note {i+1}: {orig_time:.3f}→{new_times[i]:.3f}s, "
                  f"{orig_dur:.3f}→{new_durations[i]:.3f}s, "
                  f"MIDI {orig_note}→{new_notes[i]}")

# Usage - Maximum impact, minimum code
if __name__ == "__main__":
    
    # Apply core humanization
    humanizer = CoreVocalHumanizer()
    humanizer.humanize_core('04_extracted_notes/voice_005000_carpentersclosetoyou_0000.csv', '05_humanise_notes/voice_005000_carpentersclosetoyou_0000.csv')