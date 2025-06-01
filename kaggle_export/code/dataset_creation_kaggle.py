import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple
from pathlib import Path

# Adjusted for Kaggle environment
import sys
sys.path.append('/kaggle/input/mridangam-transcription/code')

from data_preparation import get_audio
from data_preparation import get_mel_spectrogram
from data_preparation import get_window
from data_preparation import get_onset
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
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process audio on-the-fly.
        """
        file_path = self.file_paths[idx]
        label = self.encoded_labels[idx]
        
        try:
            # Load audio
            audio, sr = get_audio(file_path)
            
            # Get onset and window
            onset = get_onset(audio, sr)
            audio_window = get_window(onset, audio, sr)
            
            # Get mel spectrogram
            mel_spec = get_mel_spectrogram(audio_window, sr)
            
            # Apply augmentation if requested
            if self.augment:
                mel_spec = self._apply_augmentation(mel_spec)
            
            # Normalize and format for architecture
            mel_spec = self._normalize_and_format(mel_spec)
            
            return mel_spec, torch.LongTensor([label])[0]
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return zero tensor with correct shape
            if self.architecture in ['cnn', 'cnn_rnn', 'cnn_lstm']:
                return torch.zeros(1, 128, self.target_length), torch.LongTensor([0])[0]
            elif self.architecture == 'tcn':
                return torch.zeros(128, self.target_length), torch.LongTensor([0])[0]
            else:  # rnn, lstm
                return torch.zeros(self.target_length, 128), torch.LongTensor([0])[0]
    
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

if __name__ == "__main__":
    # Example usage for different architectures
    print("Creating efficient datasets for different architectures...")

    # CNN Architecture
    cnn_data = create_file_based_dataset(
        Path('/Users/aniachin/Projects/mridangam-transcription/dataset/raw_data/mridangam_stroke_1.0'),
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