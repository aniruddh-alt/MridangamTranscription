# Mridangam Transcription Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A deep learning project for automatic transcription of mridangam strokes using Convolutional Neural Networks with Attention.

## Overview

The mridangam is a principal rhythmic accompaniment in Carnatic music. This project provides a robust pipeline for transcribing mridangam strokes from both live microphone input and audio files.

### Key Features
- **Clean Architecture**: Refactored into a modular Python package.
- **Improved Generalization**: Waveform-domain data augmentation and per-mel-bin normalization.
- **Real-Time Transcription**: Low-latency detection using onset-gating and CNN-based classification.
- **Flexible Output**: Support for raw stroke labels or mapped solkattu tokens (e.g., `cha` -> `c`).

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mridangam-transcription.git
   cd mridangam-transcription
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -e .
   ```
   *Note: For macOS users, ensure `portaudio` is installed for microphone support (`brew install portaudio`).*

---

## Usage

### 1. Training
To train a new model on the Mridangam Stroke dataset:
```bash
python scripts/train.py --data-dir dataset/raw_data/mridangam_stroke_1.0 --epochs 50
```
Checkpoints will be saved in `model_checkpoints/`.

### 2. Evaluation
Evaluate a trained model on the test set with a confusion matrix:
```bash
python scripts/eval_dataset.py --model-path model_checkpoints/YOUR_RUN/best.pth
```

### 3. Real-Time Transcription
Run transcription from your microphone:
```bash
python scripts/transcribe_realtime.py --model model_checkpoints/YOUR_RUN/best.pth --input mic
```

Or from a WAV file:
```bash
python scripts/transcribe_realtime.py --model model_checkpoints/YOUR_RUN/best.pth --input wav --wav test.wav
```

#### Solkattu Mapping
To output solkattu tokens, provide a mapping JSON:
```bash
python scripts/transcribe_realtime.py --model ... --mapping configs/label_mapping.example.json --output both
```

---

## Project Structure

- `src/mridangam_transcription/`: Core package.
  - `models/`: Neural network architectures (CNN + Attention).
  - `audio/`: Preprocessing and onset detection logic.
  - `data/`: Dataset loading and augmentation.
  - `utils/`: Checkpointing and visualization.
- `scripts/`: CLI entrypoints for training, evaluation, and transcription.
- `configs/`: Example mapping files.
- `dataset/raw_data/`: Place your audio datasets here.

---

## License

This project is licensed under the MIT License. The Mridangam Stroke 1.0 dataset used for training is available under CC BY-NC 3.0.

## Acknowledgments

- [akshaylaya](https://freesound.org/people/akshaylaya/) for the mridangam samples.
- The Carnatic music community for domain knowledge.
