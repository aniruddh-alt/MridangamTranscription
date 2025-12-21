import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import librosa

from mridangam_transcription.audio.preprocess import get_audio, get_onset, get_window, get_mel_spectrogram

class MridangamDataset(Dataset):
    """Memory-efficient dataset that processes audio on-the-fly."""
    def __init__(self, 
                 file_paths: List[Path], 
                 labels: List[str],
                 target_length: int = 128,
                 mel_stats: Optional[Dict[str, np.ndarray]] = None,
                 augment: bool = False):
        self.file_paths = file_paths
        self.target_length = target_length
        self.mel_stats = mel_stats
        self.augment = augment
        
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.classes = self.label_encoder.classes_.tolist()
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        label = self.encoded_labels[idx]
        
        try:
            audio, sr = get_audio(str(file_path))
            
            # On-the-fly waveform augmentation
            if self.augment:
                audio = self._augment_waveform(audio, sr)
                
            onset = get_onset(audio, sr)
            audio_window = get_window(onset, audio, sr)
            mel_spec = get_mel_spectrogram(audio_window, sr)
            
            # Normalization
            if self.mel_stats is not None:
                mel_spec = (mel_spec - self.mel_stats['mean'][:, np.newaxis]) / (self.mel_stats['std'][:, np.newaxis] + 1e-8)
            
            # Format for CNN: (1, n_mels, time)
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            # Pad/truncate time dimension
            if mel_tensor.shape[2] < self.target_length:
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, self.target_length - mel_tensor.shape[2]))
            else:
                mel_tensor = mel_tensor[:, :, :self.target_length]
                
            return mel_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return torch.zeros(1, 128, self.target_length), torch.tensor(0, dtype=torch.long)

    def _augment_waveform(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """Apply random waveform augmentations."""
        # Random gain
        if np.random.random() < 0.5:
            audio = audio * np.random.uniform(0.8, 1.2)
            
        # Additive noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, np.random.uniform(0.001, 0.005), audio.shape)
            audio = audio + noise
            
        # Time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            
        # Pitch shift
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-1, 1)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
        return audio

def compute_mel_statistics(file_paths: List[Path], sample_ratio: float = 0.1) -> Dict[str, np.ndarray]:
    """Compute per-mel-bin statistics for normalization."""
    print("Computing mel-spectrogram statistics...")
    n_sample = max(1, int(len(file_paths) * sample_ratio))
    sampled_paths = np.random.choice(file_paths, n_sample, replace=False)
    
    all_mel_values = []
    for path in sampled_paths:
        try:
            audio, sr = get_audio(str(path))
            onset = get_onset(audio, sr)
            audio_window = get_window(onset, audio, sr)
            mel_spec = get_mel_spectrogram(audio_window, sr)
            all_mel_values.append(mel_spec)
        except:
            continue
            
    if not all_mel_values:
        return {'mean': np.zeros(128), 'std': np.ones(128)}
        
    stacked_mels = np.concatenate(all_mel_values, axis=1)
    return {
        'mean': np.mean(stacked_mels, axis=1),
        'std': np.std(stacked_mels, axis=1)
    }

def create_datasets(directory: Path, test_size: float = 0.2, val_size: float = 0.1):
    """Scan directory and create train/val/test datasets."""
    file_paths = []
    labels = []
    
    # Iterate through tonal folders (B, C, C#, etc)
    for tonal_dir in directory.iterdir():
        if not tonal_dir.is_dir():
            continue
        for file in tonal_dir.glob('*.wav'):
            try:
                # Format: 224030__akshaylaya__bheem-b-001.wav -> bheem
                label = file.stem.split('__')[2].split('-')[0]
                file_paths.append(file)
                labels.append(label)
            except:
                continue
                
    # Initial split
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    # Second split for validation
    val_relative_size = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_relative_size, stratify=train_val_labels, random_state=42
    )
    
    # Compute stats on train set
    mel_stats = compute_mel_statistics(train_paths)
    
    return {
        'train': MridangamDataset(train_paths, train_labels, mel_stats=mel_stats, augment=True),
        'val': MridangamDataset(val_paths, val_labels, mel_stats=mel_stats, augment=False),
        'test': MridangamDataset(test_paths, test_labels, mel_stats=mel_stats, augment=False),
        'mel_stats': mel_stats
    }

