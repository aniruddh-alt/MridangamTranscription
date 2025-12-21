import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def visualize_onset_detection(audio: np.ndarray, sr: float, onset: float = None):
    """Visualize the onset detection results."""
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

