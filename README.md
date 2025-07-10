# Voice-to-Notes AI System

A complete machine learning pipeline that converts singing voice recordings into musical note sequences using a Whisper encoder and transformer decoder architecture.

## 🎵 Overview

This system trains an encoder-decoder model to predict musical notes from audio recordings:
- **Encoder**: Whisper (pretrained, frozen) for audio feature extraction
- **Decoder**: Transformer-based architecture for note sequence generation
- **Training**: Autoregressive note prediction with attention mechanism
- **Inference**: Handles long audio files (3+ minutes) with overlapping chunks

## 📁 Project Structure

```
KaraokeAI/
├── model.py              # Main model architecture
├── dataloader.py         # Data loading and preprocessing
├── train.py              # Training script with wandb integration
├── inference.py          # Inference script for long audio files
├── utils.py              # Utility functions and evaluation metrics
├── requirements.txt      # Python dependencies
├── processed_data/       # Training data
│   ├── voices/          # Audio files (.wav)
│   └── notes/           # Note annotations (.csv)
├── checkpoints/         # Model checkpoints (created during training)
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd KaraokeAI

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for MIDI export (optional)
pip install mido
```

### 2. Data Format

Your data should be organized as:
- **Audio files**: `.wav` files in `processed_data/voices/`
- **Note annotations**: `.csv` files in `processed_data/notes/`

CSV format for notes:
```csv
start_time,duration,note
0.441,0.31579,60
0.75,0.94737,62
3.084,0.15789,61
```

Where:
- `start_time`: Note start time in seconds
- `duration`: Note duration in seconds  
- `note`: MIDI note number (60 = Middle C)

### 3. Training

```bash
# Basic training
python train.py

# Training with custom parameters
python train.py --batch_size 16 --lr 2e-4 --epochs 100

# Training without wandb logging
python train.py --no_wandb
```

### 4. Inference

```bash
# Predict notes from audio file
python inference.py --checkpoint checkpoints/best_checkpoint.pt --audio your_recording.wav

# With visualization
python inference.py --checkpoint checkpoints/best_checkpoint.pt --audio your_recording.wav --visualize

# Custom chunk processing
python inference.py --checkpoint checkpoints/best_checkpoint.pt --audio your_recording.wav --chunk_length 15.0 --overlap 2.0
```

## 🎯 Key Features

### Audio Preprocessing Pipeline
- **Gaussian noise addition** for training robustness
- **Resampling** to 16kHz for Whisper compatibility
- **Mono conversion** from stereo audio
- **High-pass filtering** (80Hz) to remove low-frequency noise
- **Loudness normalization** to -20 LUFS
- **Simple dereverberation** using spectral subtraction

### Model Architecture
- **Whisper encoder** (frozen pretrained weights)
- **Transformer decoder** with causal attention
- **Multi-head output** for note, timing, and duration prediction
- **Tokenization** with time and duration buckets
- **Positional encoding** for sequence modeling

### Training Features
- **Mixed precision training** for efficiency
- **Gradient clipping** for stability
- **Learning rate scheduling** (OneCycleLR)
- **Early stopping** to prevent overfitting
- **Wandb integration** for experiment tracking
- **Checkpoint saving** with best model selection

### Inference Capabilities
- **Long audio processing** with overlapping chunks
- **Automatic chunk merging** and overlap handling
- **Post-processing** to remove noise and merge similar notes
- **Visualization** with piano roll plots
- **CSV export** for further analysis
- **MIDI export** (optional, requires mido)

## 📊 Evaluation Metrics

The system includes several evaluation metrics:

- **Accuracy**: Token-level prediction accuracy
- **Perplexity**: Language model perplexity
- **Precision/Recall/F1**: Note-level matching metrics
- **Timing accuracy**: How well predicted timing matches ground truth

## 🔧 Configuration Options

### Training Parameters
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of training epochs (default: 50)
- `--voices_dir`: Directory with audio files
- `--notes_dir`: Directory with note annotations
- `--checkpoint_dir`: Where to save checkpoints

### Inference Parameters
- `--checkpoint`: Path to trained model checkpoint
- `--audio`: Input audio file path
- `--chunk_length`: Processing chunk length in seconds (default: 10.0)
- `--overlap`: Overlap between chunks in seconds (default: 1.0)
- `--max_length`: Maximum notes per chunk (default: 500)
- `--visualize`: Generate visualization plots

## 🎼 Audio Processing Details

### Preprocessing Steps
1. **Load audio** and convert to mono
2. **Resample** to 16kHz (Whisper's expected sample rate)
3. **Apply high-pass filter** to remove rumble and low-frequency noise
4. **Normalize loudness** to consistent level (-20 LUFS)
5. **Apply dereverberation** using spectral subtraction
6. **Add Gaussian noise** (training only) for robustness
7. **Generate spectrograms** using Whisper's feature extractor

### Note Tokenization
- Notes are tokenized using buckets for efficient processing
- Time buckets: 0-10 seconds mapped to 0-99 buckets
- Duration buckets: 0-10 seconds mapped to 0-99 buckets
- MIDI notes: 0-127 direct mapping
- Special tokens: [PAD], [START], [END]

## 🧪 Testing and Utilities

### Quick Tests
```bash
# Test model creation
python utils.py

# Test dataloader
python dataloader.py

# Test individual components
python -c "from utils import quick_test_model; quick_test_model()"
```

### Utility Functions
- **MIDI/frequency conversions**: Convert between MIDI numbers, note names, and frequencies
- **Evaluation metrics**: Calculate precision, recall, F1 for note prediction
- **Visualization**: Plot piano rolls and note comparisons
- **MIDI export**: Export predictions to MIDI files
- **Audio analysis**: Analyze audio properties and characteristics

## 📈 Training Tips

1. **Data Quality**: Ensure audio-note alignment is accurate
2. **Batch Size**: Start with smaller batch sizes (4-8) due to memory requirements
3. **Learning Rate**: Use lower learning rates (1e-4 to 1e-5) for stability
4. **Monitoring**: Use wandb to track training progress and metrics
5. **Checkpointing**: Save checkpoints frequently in case of interruption
6. **Validation**: Monitor validation loss to detect overfitting

## 🔍 Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or use gradient checkpointing
- **No matching audio-note pairs**: Check file naming and paths
- **Poor predictions**: Ensure audio quality and note alignment
- **Slow training**: Consider using mixed precision or smaller model

### Performance Optimization
- Use GPU acceleration when available
- Enable mixed precision training
- Optimize data loading with multiple workers
- Use appropriate chunk sizes for your hardware

## 🎨 Visualization Examples

The system can generate several types of visualizations:
- **Piano roll plots**: Show notes over time
- **Note distribution histograms**: Analyze note frequency
- **Training curves**: Monitor loss and metrics
- **Comparison plots**: Compare predictions vs ground truth

## 🚀 Advanced Usage

### Custom Model Configuration
```python
from model import VoiceToNotesModel

model = VoiceToNotesModel(
    whisper_model_name="openai/whisper-large",
    d_model=1024,
    nhead=16,
    num_decoder_layers=12,
    max_notes=1000
)
```

### Custom Preprocessing
```python
from dataloader import AudioPreprocessor

preprocessor = AudioPreprocessor(
    target_sr=16000,
    highpass_freq=100,
    target_lufs=-18.0
)
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📧 Contact

For questions or support, please open an issue on the GitHub repository. 