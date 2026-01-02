import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import librosa
from transformers import ASTFeatureExtractor

from mridangam_transcription.audio.preprocess import get_audio_16k, get_onset, get_window

class ASTMridangamDataset(Dataset):
    """AST-compatible dataset that processes audio on-the-fly."""
    def __init__(self, 
                 file_paths: List[Path], 
                 labels: List[str],
                 feature_extractor: ASTFeatureExtractor,
                 augment: bool = False):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor
        self.sample_rate = 16000  # AST expects 16kHz
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
            # Load audio at 16kHz for AST
            audio, sr = get_audio_16k(str(file_path))
            
            # Apply waveform augmentation before feature extraction
            if self.augment:
                audio = self._augment_waveform(audio, sr)
            
            # Extract window around onset (same as CNN approach)
            onset = get_onset(audio, sr)
            audio_window = get_window(onset, audio, sr, pre_onset=0.05, post_onset=0.15)
            
            # AST expects numpy array, not torch tensor
            # The feature extractor handles mel spectrogram computation and normalization
            inputs = self.feature_extractor(
                audio_window, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            # Extract the input_values tensor and squeeze batch dimension
            # Shape: (batch, time) -> (time,)
            input_values = inputs["input_values"].squeeze(0)
            
            return input_values, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return dummy tensor with correct shape for AST
            # AST expects ~10.24s at 16kHz = 163840 samples, but we use shorter windows
            # Return a zero tensor with reasonable length
            dummy_length = int(0.2 * self.sample_rate)  # 200ms
            return torch.zeros(dummy_length), torch.tensor(0, dtype=torch.long)

    def _augment_waveform(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """Apply random waveform augmentations with wider pitch shift for generalization."""
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
            
        # Wider pitch shift for tonal generalization (±4 semitones)
        if np.random.random() < 0.5:
            n_steps = np.random.uniform(-4, 4)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
        return audio

def create_ast_datasets(directory: Path, 
                       feature_extractor: ASTFeatureExtractor,
                       test_size: float = 0.2, 
                       val_size: float = 0.1):
    """Scan directory and create train/val/test datasets for AST."""
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
    
    return {
        'train': ASTMridangamDataset(train_paths, train_labels, feature_extractor, augment=True),
        'val': ASTMridangamDataset(val_paths, val_labels, feature_extractor, augment=False),
        'test': ASTMridangamDataset(test_paths, test_labels, feature_extractor, augment=False),
    }

