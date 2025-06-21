# Mridangam Transcription Model on Kaggle
# This notebook runs the mridangam transcription model

import numpy as np
from typing import Tuple
from pathlib import Path


import librosa
from typing import Optional
import matplotlib.pyplot as plt

def get_audio(path: Path) -> Tuple[np.array, float]:
    """
    Load audio file.
    Args:
        path: Path to audio file
    Returns:
        Tuple of audio signal and sample rate
    """
    audio, sr = librosa.load(path, sr=22050)
    return audio, sr

def get_onset(audio: np.array, sr: float) -> Optional[float]:
    """
    Improved onset detection specifically for mridangam percussion.
    
    Args:
        audio: Audio signal
        sr: Sample rate
    Returns:
        Onset time in seconds
    """
    # Method 1: Spectral flux
    spectral_onsets = librosa.onset.onset_detect(
        y=audio, sr=sr, units='time',
        onset_envelope=librosa.onset.onset_strength(y=audio, sr=sr),
        pre_max=3, post_max=3, pre_avg=3, post_avg=3,
        delta=0.3, wait=5
    )
    
    # Method 2: Energy-based (fix: compute RMS separately)
    rms_features = librosa.feature.rms(y=audio)[0]
    # Convert RMS to onset strength manually
    rms_diff = np.diff(rms_features, prepend=rms_features[0])
    rms_diff = np.maximum(0, rms_diff)  # Only positive changes
    
    energy_onsets = librosa.onset.onset_detect(
        onset_envelope=rms_diff,
        sr=sr, units='time',
        pre_max=3, post_max=3, pre_avg=3, post_avg=3,
        delta=0.4, wait=5
    )
    
    # Combine all detected onsets
    all_onsets = np.concatenate([spectral_onsets, energy_onsets])
    
    # Remove duplicates within 50ms window
    if len(all_onsets) > 0:
        all_onsets = np.sort(all_onsets)
        unique_onsets = [all_onsets[0]]
        for onset in all_onsets[1:]:
            if onset - unique_onsets[-1] > 0.05:  # 50ms threshold
                unique_onsets.append(onset)
        onset_times = np.array(unique_onsets)
    else:
        onset_times = all_onsets
    
    return onset_times[0] if len(onset_times) > 0 else None

def get_window(onset: float, audio: np.array, sr: float, 
                       pre_onset: float = 0.05, post_onset: float = 0.15) -> np.array:
    """
    Get audio window around onset with adaptive timing based on audio energy.
    """
    duration: float = pre_onset + post_onset
    window_samples: int = int(duration * sr)
    if onset is None:
        if len(audio) >= window_samples:
            return audio[:window_samples]
        else:
            # Pad if audio is shorter than window
            return np.pad(audio, (0, window_samples - len(audio)), mode='constant')
    
    # Calculate window boundaries
    # Put onset at 25% of window (to capture pre-attack)
    pre_onset_duration = duration * 0.25
    duration * 0.75
    
    start_time = max(0, onset - pre_onset_duration)
    end_time = start_time + duration
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Handle edge cases
    if end_sample > len(audio):
        # If we exceed audio length, shift window back
        end_sample = len(audio)
        start_sample = max(0, end_sample - window_samples)
    
    # Extract window
    window = audio[start_sample:end_sample]
    
    # Ensure exact duration by padding if necessary
    if len(window) < window_samples:
        pad_amount = window_samples - len(window)
        window = np.pad(window, (0, pad_amount), mode='constant')
    elif len(window) > window_samples:
        window = window[:window_samples]
    
    return window

def get_mel_spectrogram(audio: np.array, sr: float) -> np.array:
    """
    Compute mel spectrogram.
    Args:
        audio: Audio signal
        sr: Sample rate
    Returns:
        Mel spectrogram
    """
    # Dynamic n_fft sizing based on audio length
    max_n_fft = min(512, len(audio))  # Use smaller of 512 or audio length
    n_fft = max(256, max_n_fft)  # Ensure minimum of 256
    hop_length = n_fft // 2  # Half of n_fft
    
    # Ensure audio is long enough for n_fft
    if len(audio) < n_fft:
        pad_amt = n_fft - len(audio)
        audio = np.pad(audio, (0, pad_amt), mode='constant')
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128
    )

    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def visualize_onset_detection(audio: np.array, sr: float, onset: float = None):
    """
    Visualize the onset detection results.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    times = np.linspace(0, len(audio)/sr, len(audio))
    plt.plot(times, audio)
    if onset is not None:
        plt.axvline(x=onset, color='r', linestyle='--', label=f'Onset: {onset:.3f}s')
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Plot onset strength
    plt.subplot(3, 1, 2)
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
    times_onset = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=sr)
    plt.plot(times_onset, onset_envelope)
    if onset is not None:
        plt.axvline(x=onset, color='r', linestyle='--', label=f'Onset: {onset:.3f}s')
    plt.title('Onset Strength')
    plt.xlabel('Time (s)')
    plt.ylabel('Strength')
    plt.legend()
    
    # Plot spectrogram
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
    if onset is not None:
        plt.axvline(x=onset, color='r', linestyle='--', label=f'Onset: {onset:.3f}s')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.legend()
    
    plt.tight_layout()
    plt.show()