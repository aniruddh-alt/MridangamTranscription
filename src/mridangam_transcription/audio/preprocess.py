import numpy as np
import librosa
from typing import Optional, Tuple

def get_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, float]:
    """Load audio file."""
    audio, loaded_sr = librosa.load(path, sr=sr)
    return audio, loaded_sr

def get_onset(audio: np.ndarray, sr: float) -> Optional[float]:
    """Improved onset detection for mridangam percussion."""
    # Spectral flux
    spectral_onsets = librosa.onset.onset_detect(
        y=audio, sr=sr, units='time',
        onset_envelope=librosa.onset.onset_strength(y=audio, sr=sr),
        pre_max=3, post_max=3, pre_avg=3, post_avg=3,
        delta=0.3, wait=5
    )
    
    # Energy-based
    rms_features = librosa.feature.rms(y=audio)[0]
    rms_diff = np.diff(rms_features, prepend=rms_features[0])
    rms_diff = np.maximum(0, rms_diff)
    
    energy_onsets = librosa.onset.onset_detect(
        onset_envelope=rms_diff,
        sr=sr, units='time',
        pre_max=3, post_max=3, pre_avg=3, post_avg=3,
        delta=0.4, wait=5
    )
    
    all_onsets = np.concatenate([spectral_onsets, energy_onsets])
    
    if len(all_onsets) > 0:
        all_onsets = np.sort(all_onsets)
        unique_onsets = [all_onsets[0]]
        for onset in all_onsets[1:]:
            if onset - unique_onsets[-1] > 0.05:
                unique_onsets.append(onset)
        return unique_onsets[0]
    return None

def get_window(onset: Optional[float], audio: np.ndarray, sr: float, 
               pre_onset: float = 0.05, post_onset: float = 0.15) -> np.ndarray:
    """Extract audio window around onset."""
    duration = pre_onset + post_onset
    window_samples = int(duration * sr)
    
    if onset is None:
        if len(audio) >= window_samples:
            return audio[:window_samples]
        return np.pad(audio, (0, window_samples - len(audio)), mode='constant')
    
    # Put onset at 25% of window
    pre_onset_duration = duration * 0.25
    start_time = max(0, onset - pre_onset_duration)
    start_sample = int(start_time * sr)
    end_sample = start_sample + window_samples
    
    if end_sample > len(audio):
        end_sample = len(audio)
        start_sample = max(0, end_sample - window_samples)
    
    window = audio[start_sample:end_sample]
    
    if len(window) < window_samples:
        window = np.pad(window, (0, window_samples - len(window)), mode='constant')
    elif len(window) > window_samples:
        window = window[:window_samples]
        
    return window

def get_mel_spectrogram(audio: np.ndarray, sr: float, n_mels: int = 128) -> np.ndarray:
    """Compute mel spectrogram."""
    max_n_fft = min(512, len(audio))
    n_fft = max(256, max_n_fft)
    hop_length = n_fft // 2
    
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel_spec, ref=np.max)

