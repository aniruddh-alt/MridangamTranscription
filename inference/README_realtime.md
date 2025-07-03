# Real-Time Mridangam Stroke Detection

This directory contains a comprehensive real-time inference pipeline for detecting mridangam strokes using the CNN with Attention model.

## Features

- **Real-time audio capture** from microphone
- **Onset detection** optimized for mridangam percussion 
- **CNN with Attention model** inference
- **Per-mel-bin normalization** using training statistics
- **Configurable confidence thresholds**
- **Pattern analysis** and stroke sequence tracking
- **GPU acceleration** when available

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_realtime.txt
```

### 2. Basic Usage

```bash
# Using command line interface
python realtime_cnn_attention_inference.py --model ../model_checkpoints/best_model_20250622_010327.pth

# With custom confidence threshold
python realtime_cnn_attention_inference.py --model ../model_checkpoints/best_model_20250622_010327.pth --confidence 0.6

# With duration limit (30 seconds)
python realtime_cnn_attention_inference.py --model ../model_checkpoints/best_model_20250622_010327.pth --duration 30
```

### 3. Example Scripts

```bash
# Run interactive examples
python example_realtime_usage.py
```

## Command Line Options

- `--model`, `-m`: Path to model checkpoint (required)
- `--mel-stats`, `-s`: Path to mel statistics file (optional)
- `--confidence`, `-c`: Confidence threshold 0.0-1.0 (default: 0.5)
- `--duration`, `-d`: Detection duration in seconds (optional)
- `--classes`: Comma-separated list of class names (optional)

## Architecture

### CNN with Attention Model

The model uses the same architecture as your training code:

- **Convolutional layers** with batch normalization and dropout
- **Residual blocks** with self-attention mechanisms
- **Frequency attentive pooling** for feature aggregation
- **Fully connected classifier** with dropout regularization

### Preprocessing Pipeline

1. **Audio capture** at 22.05 kHz sample rate
2. **Onset detection** using spectral flux and energy-based methods
3. **Window extraction** around detected onsets
4. **Mel spectrogram** computation (128 mel bins)
5. **Per-mel-bin normalization** using training statistics
6. **Tensor formatting** for CNN input

### Real-time Processing

- **Circular audio buffer** for continuous audio capture
- **Onset detection** in 1-second sliding windows
- **Duplicate detection prevention** with 100ms minimum interval
- **Confidence-based filtering** to reduce false positives

## Programmatic Usage

```python
from realtime_cnn_attention_inference import RealTimeStrokeDetector

# Create detector
detector = RealTimeStrokeDetector(
    model_path="path/to/your/model.pth",
    confidence_threshold=0.5
)

# Run detection
for predicted_class, confidence, timestamp, all_probs in detector.run_detection():
    print(f"Detected: {predicted_class} (confidence: {confidence:.3f})")
    
    # Your custom processing here
    if confidence > 0.8:
        print("High confidence detection!")
```

## Model Compatibility

The system works with any model checkpoint saved during training:

- `best_model_*.pth` - Best performing models
- `latest_checkpoint_*.pth` - Latest training checkpoints
- `modelOutput.pth` - Final model output

The system automatically detects the number of classes from the model file.

## Audio Setup

### Microphone Requirements

- **Sample rate**: 22.05 kHz (automatically handled)
- **Channels**: Mono (single channel)
- **Format**: 16-bit PCM
- **Latency**: Low latency preferred for real-time use

### Troubleshooting Audio

If you encounter audio issues:

1. **Check microphone permissions** on your system
2. **Test with different audio devices** using system settings
3. **Reduce buffer size** for lower latency
4. **Close other audio applications** that might block access

## Performance Optimization

### For Better Real-time Performance

- **Use GPU** if available (CUDA)
- **Lower confidence threshold** to catch more strokes
- **Adjust onset detection sensitivity** by modifying delta parameters
- **Use smaller audio chunks** for lower latency

### For Higher Accuracy

- **Use mel statistics** from training for better normalization
- **Higher confidence threshold** to reduce false positives
- **Longer audio segments** for more context

## Example Output

```
🔧 Using device: cuda
📊 Model architecture: CNN with Attention (10 classes)
✅ Model loaded successfully
📋 Classes: ['Ta', 'Tha', 'Dha', 'Na', 'Thi', 'Ki', 'Dhi', 'Tam', 'Dhom', 'Kita']
🎯 Confidence threshold: 0.5

🚀 Starting real-time mridangam stroke detection...
📁 Model: best_model_20250622_010327.pth

🎤 Started recording at 22050Hz...
🎵 Listening for mridangam strokes... (Press Ctrl+C to stop)
🎯 Confidence threshold: 0.5
============================================================

🥁 [14:23:15] Ta     (confidence: 0.847)
    Top 3: Ta: 0.85 | Tha: 0.09 | Na: 0.04
🥁 [14:23:16] Dha    (confidence: 0.723)
🥁 [14:23:17] Na     (confidence: 0.691)
```

## Integration

You can easily integrate this system into larger applications:

- **MIDI output** for music production software
- **Pattern recognition** for rhythm training
- **Real-time visualization** for performance analysis
- **Recording and playback** for practice sessions

## Notes

- The system uses the **exact same preprocessing pipeline** as your training code
- **Mel statistics normalization** improves accuracy when available
- **Onset detection** is optimized for mridangam's percussive nature
- **Confidence thresholds** can be tuned based on your use case 