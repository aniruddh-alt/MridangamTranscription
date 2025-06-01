# Mridangam Transcription Project

[![License: CC BY-NC](https://img.shields.io/badge/License-CC%20BY--NC-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/3.0/)

A deep learning project for automatic transcription of mridangam strokes using various neural network architectures.

## What is a Mridangam?

The mridangam is a percussion instrument from South India, originating from ancient Dravidian culture. It is the principal rhythmic accompaniment in a Carnatic music ensemble and plays a crucial role in South Indian classical music performances.

![Mridangam](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Mridangam.jpg/640px-Mridangam.jpg)

The mridangam is a double-sided drum made from a hollowed piece of jackfruit wood with goatskin heads on both sides. The right head (called the *valanthalai*) produces the main tonal sounds, while the left head (called the *thoppi*) produces bass sounds. Players use specific finger and palm techniques to create a variety of distinctive strokes and tones.

## Project Vision

This project aims to develop a real-time mridangam transcription system that can automatically identify and notate mridangam strokes from audio recordings or live performances. The system will:

1. **Recognize Different Strokes**: Identify various mridangam strokes (Cha, Nam, Dheem, Thom, etc.)
2. **Transcribe in Real-Time**: Convert audio input to notation as it's being played
3. **Support Multiple Models**: Implement and compare different neural network architectures:
   - Convolutional Neural Networks (CNN)
   - Combined CNN-LSTM (Long Short-Term Memory) networks
   - Temporal Convolutional Networks (TCN)

The end goal is to create an application that musicians, students, and researchers can use to:
- Transcribe and analyze mridangam performances
- Aid in teaching and learning mridangam techniques
- Preserve and document traditional performances
- Enhance music education and research in Carnatic music

## Dataset

The project uses the "Mridangam Stroke 1.0" dataset, which contains high-quality audio samples of individual mridangam strokes organized by tonal categories (B, C, C#, D, D#, E). This dataset was originally contributed by user "akshaylaya" on Freesound.org and is available under a Creative Commons Attribution Non-Commercial license.

## Technical Approach

### Data Processing Pipeline
- Audio preprocessing using mel-spectrograms
- On-the-fly data augmentation for robust training
- Efficient data loading with PyTorch DataLoaders

### Model Architectures

#### CNN Model
- Multiple convolutional layers for feature extraction
- Batch normalization and dropout for regularization
- Adaptive pooling for flexible input sizes

#### CNN-LSTM Model (Planned)
- CNN layers for spatial feature extraction
- LSTM layers to capture temporal dependencies in the audio
- Suitable for capturing the time-dependent nature of strokes

#### TCN Model (Planned)
- Temporal Convolutional Network with dilated convolutions
- Effective at capturing long-range dependencies in time series data
- Parallelizable architecture for efficient training and inference

### Real-Time Application (Planned)
- Low-latency audio processing
- User-friendly interface for visualization
- Export capabilities for notation and analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The mridangam audio samples are licensed under the [Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)](http://creativecommons.org/licenses/by-nc/3.0/) license.

## Acknowledgments

- Freesound.org and user "akshaylaya" for the mridangam samples
- The Carnatic music community for inspiration and domain knowledge
- PyTorch and the open-source deep learning community