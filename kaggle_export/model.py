
# Mridangam Transcription Model on Kaggle
# This notebook runs the mridangam transcription model

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple
from pathlib import Path

import sys

import librosa
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
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
    post_onset_duration = duration * 0.75
    
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
    n_fft = 512                  # or some value ≤ your shortest clip length
    hop_length = 256             # or half of n_fft
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

from typing import Optional, Dict, Any


class MridangamDataset(Dataset):
    """
    Memory-efficient dataset that processes audio on-the-fly.
    """
    def __init__(self, 
                 file_paths: List[Path], 
                 labels: List[str],
                 target_length: int = 128,
                 mel_stats: Optional[Dict[str, np.ndarray]] = None,
                 architecture: str = 'cnn',
                 augment: bool = False):
        """
        Initialize on-the-fly dataset.
        
        Args:
            file_paths: List of audio file paths
            labels: List of corresponding labels
            target_length: Fixed sequence length for padding/truncating
            mel_stats: Dictionary with 'mean' and 'std' for per-mel-bin normalization
            architecture: Target architecture ('cnn', 'cnn_rnn', 'cnn_lstm', 'tcn', 'rnn', 'lstm')
            augment: Whether to apply data augmentation
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        self.mel_stats = mel_stats
        self.architecture = architecture
        self.augment = augment
        self.num_augmentations = 3
        self.dataset_size = len(file_paths) * (self.num_augmentations + 1) if augment else len(file_paths)
        
        # Mapping for augmentation types
        self.augmentation_map = {
            0: "original",
            1: "pitch_shift",     # Pitch shifting
            2: "time_stretch",    # Time stretching
            3: "noise_injection"  # Noise injection
        }
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        if augment:
            print(f"Created dataset with {self.dataset_size} samples "
                  f"({len(file_paths)} original + {len(file_paths) * self.num_augmentations} augmented)")
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process audio on-the-fly.
        """
        base_size = len(self.file_paths)
        
        # Determine if this is an original or augmented sample
        if idx < base_size:
            file_idx = idx
            aug_type = 0  # Original (no augmentation)
        else:
            # Calculate which original sample and which augmentation type
            file_idx = (idx - base_size) % base_size
            aug_type = ((idx - base_size) // base_size) + 1
        
        file_path = self.file_paths[file_idx]
        label = self.encoded_labels[file_idx]
        
        try:
            # Load audio
            audio, sr = get_audio(file_path)
            
            # Apply audio-level augmentation if this is an augmented sample
            if aug_type > 0:
                audio = self._apply_audio_augmentation(audio, sr, aug_type)
            
            # Get onset and window
            onset = get_onset(audio, sr)
            audio_window = get_window(onset, audio, sr)
            
            # Get mel spectrogram
            mel_spec = get_mel_spectrogram(audio_window, sr)
            
            # Apply spectrogram-level augmentation if this is an augmented sample
            if aug_type > 0:
                mel_spec = self._apply_spec_augmentation(mel_spec, aug_type)
            # Apply standard augmentation if requested (for original samples)
            elif self.augment and aug_type == 0:
                mel_spec = self._apply_augmentation(mel_spec)
            
            # Normalize and format for architecture
            mel_spec = self._normalize_and_format(mel_spec)
            
            return mel_spec, torch.LongTensor([label])[0]
            
        except Exception as e:
            print(f"Error processing {file_path} (aug_type={self.augmentation_map[aug_type]}): {e}")
            # Return zero tensor with correct shape
            if self.architecture in ['cnn', 'cnn_rnn', 'cnn_lstm']:
                return torch.zeros(1, 128, self.target_length), torch.LongTensor([0])[0]
            elif self.architecture == 'tcn':
                return torch.zeros(128, self.target_length), torch.LongTensor([0])[0]
            else:  # rnn, lstm
                return torch.zeros(self.target_length, 128), torch.LongTensor([0])[0]
    
    def _apply_audio_augmentation(self, audio: np.ndarray, sr: float, aug_type: int) -> np.ndarray:
        """
        Apply audio-level augmentation based on augmentation type.
        """
        aug_type_name = self.augmentation_map.get(aug_type, "pitch_shift")
        
        if aug_type_name == "pitch_shift":
            # Shift pitch up or down by 0.5-2 semitones
            n_steps = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
        elif aug_type_name == "time_stretch":
            # Stretch time by factor of 0.9-1.1
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            # Ensure length matches original
            if len(audio) < len(audio):
                audio = np.pad(audio, (0, len(audio) - len(audio)), mode='constant')
            else:
                audio = audio[:len(audio)]
            return audio
            
        elif aug_type_name == "noise_injection":
            # Add background noise at varying SNR
            noise_factor = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_factor, audio.shape)
            return audio + noise
        
        return audio
    
    def _apply_spec_augmentation(self, mel_spec: np.ndarray, aug_type: int) -> np.ndarray:
        """
        Apply spectrogram-level augmentation based on augmentation type.
        """
        mel_spec = mel_spec.copy()
        n_mels, time_frames = mel_spec.shape
        
        aug_type_name = self.augmentation_map.get(aug_type, "pitch_shift")
        
        if aug_type_name == "pitch_shift":
            # For pitch-shifted audio, apply freq masking
            f_mask_param = min(20, n_mels // 3)
            f_start = np.random.randint(0, max(1, n_mels - f_mask_param))
            f_width = np.random.randint(10, f_mask_param)
            mel_spec[f_start:f_start + f_width, :] = mel_spec.min()
            
        elif aug_type_name == "time_stretch":
            # For time-stretched audio, apply time masking
            t_mask_param = min(30, time_frames // 3)
            t_start = np.random.randint(0, max(1, time_frames - t_mask_param))
            t_width = np.random.randint(10, t_mask_param)
            mel_spec[:, t_start:t_start + t_width] = mel_spec.min()
            
        elif aug_type_name == "noise_injection":
            # For noise-injected audio, apply random noise to spectrogram
            noise_factor = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_factor, mel_spec.shape)
            mel_spec += noise
        
        return mel_spec
    
    def _normalize_and_format(self, mel_spec: np.ndarray) -> torch.Tensor:
        """
        Normalize per mel-bin and format for specific architecture.
        """
        # Ensure consistent time dimension
        n_mels, time_frames = mel_spec.shape
        
        # Pad or truncate to target length
        if time_frames < self.target_length:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, self.target_length - time_frames)), mode='constant')
        else:
            mel_spec = mel_spec[:, :self.target_length]
        
        # Per-frequency normalization
        if self.mel_stats is not None:
            # Normalize each mel bin independently
            mel_spec = (mel_spec - self.mel_stats['mean'][:, np.newaxis]) / (self.mel_stats['std'][:, np.newaxis] + 1e-8)
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec)
        
        # Format for architecture
        return self._format_for_architecture(mel_tensor)
    
    def _format_for_architecture(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Format tensor based on target architecture.
        """
        if self.architecture in ['cnn', 'cnn_rnn', 'cnn_lstm']:
            # Add channel dimension: (1, n_mels, time_frames)
            return mel_tensor.unsqueeze(0)
        elif self.architecture == 'tcn':
            # TCN expects (n_mels, time_frames)
            return mel_tensor
        elif self.architecture in ['rnn', 'lstm']:
            # RNN/LSTM expects (time_frames, n_mels)
            return mel_tensor.transpose(0, 1)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _apply_augmentation(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment-style data augmentation.
        """
        if not self.augment:
            return mel_spec
        
        mel_spec = mel_spec.copy()
        n_mels, time_frames = mel_spec.shape
        
        # Time masking
        if np.random.random() < 0.5:
            t_mask_param = min(20, time_frames // 4)
            t_start = np.random.randint(0, max(1, time_frames - t_mask_param))
            mel_spec[:, t_start:t_start + t_mask_param] = mel_spec.min()
        
        # Frequency masking
        if np.random.random() < 0.5:
            f_mask_param = min(15, n_mels // 4)
            f_start = np.random.randint(0, max(1, n_mels - f_mask_param))
            mel_spec[f_start:f_start + f_mask_param, :] = mel_spec.min()
        
        # Add small amount of noise
        if np.random.random() < 0.3:
            noise_factor = 0.01
            noise = np.random.normal(0, noise_factor, mel_spec.shape)
            mel_spec += noise
        
        return mel_spec

def compute_mel_statistics(file_paths: List[Path], sample_ratio: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Compute per-mel-bin statistics for normalization.
    
    Args:
        file_paths: List of audio file paths
        sample_ratio: Fraction of files to use for statistics (to save time)
    
    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (n_mels,)
    """
    print("Computing mel-spectrogram statistics...")
    
    # Sample subset of files
    n_sample = max(1, int(len(file_paths) * sample_ratio))
    sampled_paths = np.random.choice(file_paths, n_sample, replace=False)
    
    all_mel_values = []  # List to collect mel values per frequency bin
    n_mels = 128  # From get_mel_spectrogram function
    
    # Initialize lists for each mel bin
    mel_bin_values = [[] for _ in range(n_mels)]
    
    for i, file_path in enumerate(sampled_paths):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(sampled_paths)} files for statistics...")
        
        try:
            audio, sr = get_audio(file_path)
            onset = get_onset(audio, sr)
            audio_window = get_window(onset, audio, sr)
            mel_spec = get_mel_spectrogram(audio_window, sr)
            
            # Collect values for each mel bin
            for mel_idx in range(min(n_mels, mel_spec.shape[0])):
                mel_bin_values[mel_idx].extend(mel_spec[mel_idx, :].flatten())
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Compute statistics per mel bin
    means = np.zeros(n_mels)
    stds = np.zeros(n_mels)
    
    for mel_idx in range(n_mels):
        if mel_bin_values[mel_idx]:
            means[mel_idx] = np.mean(mel_bin_values[mel_idx])
            stds[mel_idx] = np.std(mel_bin_values[mel_idx])
        else:
            means[mel_idx] = 0.0
            stds[mel_idx] = 1.0
    
    print(f"Computed statistics from {len(sampled_paths)} files")
    print(f"Mean range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"Std range: [{stds.min():.3f}, {stds.max():.3f}]")
    
    return {'mean': means, 'std': stds}

def create_file_based_dataset(directory: Path, 
                             test_size: float = 0.2,
                             target_length: int = 128,
                             architecture: str = 'cnn',
                             compute_stats: bool = True) -> Dict[str, Any]:
    """
    Create file-based datasets that process audio on-the-fly.
    
    Args:
        directory: Path to audio files directory
        test_size: Fraction for test split
        target_length: Fixed sequence length
        architecture: Target architecture
        compute_stats: Whether to compute normalization statistics
    
    Returns:
        Dictionary containing datasets and metadata
    """
    print("Creating file-based dataset...")
    
    # Collect file paths and labels
    file_paths = []
    labels = []
    
    for subdir in directory.iterdir():
        if subdir.is_dir():
            print(f"Scanning directory: {subdir.name}")
            for file in subdir.glob('*.wav'):
                try:
                    # Extract label from filename
                    label = file.stem.split('__')[2].split('-')[0]
                    file_paths.append(file)
                    labels.append(label)
                except IndexError:
                    print(f"Skipping file with unexpected name format: {file.name}")
                    continue
    
    print(f"Found {len(file_paths)} audio files")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Split files
    files_train, files_test, labels_train, labels_test = train_test_split(
        file_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Compute mel statistics from training set
    mel_stats = None
    if compute_stats:
        mel_stats = compute_mel_statistics(files_train, sample_ratio=0.1)
    
    # Create datasets
    train_dataset = MridangamDataset(
        files_train, labels_train, target_length, mel_stats, architecture, augment=True
    )
    
    test_dataset = MridangamDataset(
        files_test, labels_test, target_length, mel_stats, architecture, augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.label_encoder.classes_)}")
    print(f"Classes: {train_dataset.label_encoder.classes_}")
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'label_encoder': train_dataset.label_encoder,
        'mel_stats': mel_stats,
        'num_classes': len(train_dataset.label_encoder.classes_),
        'n_mels': 128,
        'time_steps': target_length,
        'architecture': architecture
    }

def create_efficient_dataloader(dataset: MridangamDataset, 
                              batch_size: int = 32, 
                              shuffle: bool = True,
                              num_workers: int = 0) -> DataLoader:
    """
    Create memory-efficient DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )


# Example usage for different architectures
print("Creating efficient datasets for different architectures...")

# CNN Architecture
cnn_data = create_file_based_dataset(
    Path('/kaggle/input/mridangam-ml/kaggle_export/data/mridangam_stroke_1.0'),
    test_size=0.2,
    target_length=128,
    architecture='cnn',
    compute_stats=True
)

cnn_train_loader = create_efficient_dataloader(cnn_data['train'], batch_size=32, shuffle=True)
cnn_test_loader = create_efficient_dataloader(cnn_data['test'], batch_size=32, shuffle=False)

print("\nCNN Dataset verification:")
sample_batch = next(iter(cnn_train_loader))
print(f"CNN Input shape: {sample_batch[0].shape}")
print("Expected for CNN: (batch_size, 1, n_mels, time_steps)")
print(f"Labels shape: {sample_batch[1].shape}")

import sys
import os
from pathlib import Path
from tqdm import tqdm
# Add code directory to path
sys.path.append('/kaggle/input/mridangam-ml/kaggle_export/code')

# Import functions from our modules
# from data_preparation import get_audio, get_mel_spectrogram, get_window, get_onset
# from dataset_creation import MridangamDataset, compute_mel_statistics, create_file_based_dataset, create_efficient_dataloader

# Set up paths for Kaggle
data_path = Path('/kaggle/input/mridangam-ml/kaggle_export/data/mridangam_stroke_1.0')

# Import and set up your model
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 50
architecture = 'cnn'

# Create datasets and dataloaders
print("Creating datasets...")
data = create_file_based_dataset(
    directory=data_path,
    test_size=0.2,
    target_length=128,
    architecture=architecture,
    compute_stats=True
)

# Create efficient dataloaders (optimized for Kaggle)
train_loader = create_efficient_dataloader(
    dataset=data['train'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=2  # Kaggle typically has 2-4 CPU cores
)

test_loader = create_efficient_dataloader(
    dataset=data['test'],
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

class MridangamCNN(nn.Module):
    def __init__(self, n_mels = 128, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(

            # input shape: (1, 128, 128)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (32, 128, 128)

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # output shape: (32, 64, 64)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (64, 64, 64)

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # output shape: (64, 32, 32)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (128, 32, 32)

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # output shape: (128, 16, 16)

            nn.AdaptiveAvgPool2d((1, None)) # output shape: (128, 1, 16)

        )

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),  # output shape: (64, 1, 16)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool1d(1),  # output shape: (64, 1, 1)
            nn.Flatten(),  # output shape: (64)
            nn.Linear(64, num_classes),  # output shape: (num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(2)
        x = self.classifier(x)
        return x
        
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data
    """
    model.eval()
    correct = 0
    total = 0
    losses = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Get predictions
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = sum(losses) / len(losses)
    return accuracy, avg_loss

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    """
    Train and validate the model with proper early stopping
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # ——— Training Phase ———
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * correct_train / total_train
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})

        # Calculate training metrics
        train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train

        # ——— Validation Phase ———
        val_acc, val_loss = evaluate_model(model, test_loader, criterion, device)
        val_acc_percent = val_acc * 100  # Convert to percentage

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc_percent)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc_percent:.2f}%"
        )

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("Training completed!")
    
    # Return training history for plotting
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

# Complete training script
def run_training():
    """
    Complete training pipeline
    """
    # Initialize model
    model = MridangamCNN(n_mels=128, num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train the model
    history = train_and_validate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50
    )
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    final_acc, final_loss = evaluate_model(model, test_loader, criterion, device)
    print(f'Final Test Accuracy: {final_acc*100:.2f}%')
    print(f'Final Test Loss: {final_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'mridangam_model_final.pth')
    print("Model saved as 'mridangam_model_final.pth'")
    
    return model, history

# Run the training
if __name__ == "__main__":
    model, training_history = run_training()