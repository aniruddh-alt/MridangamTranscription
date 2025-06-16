import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataset.data_processing.data_preparation import get_audio
from dataset.data_processing.data_preparation import get_mel_spectrogram
from dataset.data_processing.data_preparation import get_window
from dataset.data_processing.data_preparation import get_onset
from typing import Optional, Dict, Any

import librosa



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
        self.num_augmentations = 6
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
        
        try:
            if aug_type_name == "pitch_shift":
                # Shift pitch up or down by 0.5-2 semitones
                n_steps = np.random.uniform(-2, 2)
                return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
                
            elif aug_type_name == "time_stretch":
                # Stretch time by factor of 0.9-1.1
                rate = np.random.uniform(0.9, 1.1)
                try:
                    stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
                    # Ensure length consistency
                    original_length = len(audio)
                    if len(stretched_audio) < original_length:
                        # Pad if shorter
                        stretched_audio = np.pad(stretched_audio, (0, original_length - len(stretched_audio)), mode='constant')
                    elif len(stretched_audio) > original_length:
                        # Truncate if longer
                        stretched_audio = stretched_audio[:original_length]
                    return stretched_audio
                except Exception as e:
                    # If time stretch fails, return original audio
                    return audio
                
            elif aug_type_name == "noise_injection":
                # Add background noise at varying SNR
                noise_factor = np.random.uniform(0.005, 0.02)
                noise = np.random.normal(0, noise_factor, audio.shape)
                return audio + noise
        
        except Exception as e:
            # If any augmentation fails, return original audio
            pass
        
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
            if f_mask_param > 0 and n_mels > f_mask_param:
                f_start = np.random.randint(0, n_mels - f_mask_param)
                f_width = np.random.randint(1, f_mask_param + 1)
                f_end = min(f_start + f_width, n_mels)
                mel_spec[f_start:f_end, :] = mel_spec.min()
            
        elif aug_type_name == "time_stretch":
            try:
                t_mask_param = min(30, time_frames // 3)
                if t_mask_param > 0 and time_frames > t_mask_param:
                    t_start = np.random.randint(0, time_frames - t_mask_param)
                    t_width = np.random.randint(1, t_mask_param + 1)
                    t_end = min(t_start + t_width, time_frames)
                    mel_spec[:, t_start:t_end] = mel_spec.min()
            except Exception as e:
                return mel_spec  # If time stretch fails, return original spectrogram
            
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

        
        # Pitch shifting
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            mel_spec = librosa.effects.pitch_shift(mel_spec, sr=22050, n_steps=n_steps)
        
        # Frequency masking (SpecAugment)
        if np.random.random() < 0.5:
            freq_mask_param = 8
            num_freq_masks = np.random.randint(1, 3)
            for _ in range(num_freq_masks):
                f = np.random.uniform(0, freq_mask_param)
                f0 = np.random.uniform(0, mel_spec.shape[0] - f)
                mel_spec[int(f0):int(f0 + f), :] = 0
    
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
                             val_size: float = 0.2,
                             target_length: int = 128,
                             architecture: str = 'cnn',
                             compute_stats: bool = True) -> Dict[str, Any]:
    """
    Create file-based datasets that process audio on-the-fly.
    
    Args:
        directory: Path to audio files directory
        test_size: Fraction for test split
        val_size: Fraction for validation split (from remaining data after test split)
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
    
    # First split: separate test set
    files_temp, files_test, labels_temp, labels_test = train_test_split(
        file_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation from remaining data
    # Calculate validation size relative to the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    files_train, files_val, labels_train, labels_val = train_test_split(
        files_temp, labels_temp, test_size=val_size_adjusted, random_state=42, stratify=labels_temp
    )
    
    # Compute mel statistics from training set only
    mel_stats = None
    if compute_stats:
        mel_stats = compute_mel_statistics(files_train, sample_ratio=0.1)
    
    # Create datasets
    train_dataset = MridangamDataset(
        files_train, labels_train, target_length, mel_stats, architecture, augment=True
    )
    
    val_dataset = MridangamDataset(
        files_val, labels_val, target_length, mel_stats, architecture, augment=False
    )
    
    test_dataset = MridangamDataset(
        files_test, labels_test, target_length, mel_stats, architecture, augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.label_encoder.classes_)}")
    print(f"Classes: {train_dataset.label_encoder.classes_}")
    
    # Verify split proportions
    total_files = len(file_paths)
    print(f"Split proportions:")
    print(f"  Train: {len(files_train)/total_files:.1%} ({len(files_train)} files)")
    print(f"  Validation: {len(files_val)/total_files:.1%} ({len(files_val)} files)")
    print(f"  Test: {len(files_test)/total_files:.1%} ({len(files_test)} files)")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
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
