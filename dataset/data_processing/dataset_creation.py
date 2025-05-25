import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple
from pathlib import Path

from dataset.data_processing import get_audio
from dataset.data_processing import get_mel_spectrogram
from dataset.data_processing import visualize_onset_detection
from dataset.data_processing import get_window
from dataset.data_processing import get_onset

def create_dataset(directory: Path) -> List[Tuple[np.array, str]]:
    """
    Create a dataset of mel spectrograms and labels from audio files.
    
    Args:
        directory: Path to the directory containing audio files
    Returns:
        List of tuples containing mel spectrograms and labels
    """
    dataset = []
    
    for subdir in directory.iterdir():
        if subdir.is_dir():
            print(f"Parsing directory: {subdir.name}")
            for file in subdir.glob('*.wav'):
                audio, sr = get_audio(file)
                # get oneset time
                onset = get_onset(audio, sr)
                # get window around onset
                audio_window = get_window(onset, audio, sr)
                # get mel spectrogram
                mel_spectrogram = get_mel_spectrogram(audio_window, sr)
                
                # Extract label from filename
                label = file.stem.split('__')[2].split('-')[0]  # Extract stroke
                
                dataset.append((mel_spectrogram, label))
    return dataset


class MridangamDataset(Dataset):
    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset with preprocessed spectrograms and encoded labels.
        
        Args:
            spectrograms: Numpy array of mel spectrograms
            labels: Numpy array of encoded labels
        """
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectrograms[idx], self.labels[idx]

def create_torch_dataset(raw_dataset: List[Tuple[np.ndarray, str]], test_size: float = 0.2):
    """
    Create PyTorch datasets from the raw dataset.
    
    Args:
        raw_dataset: List of tuples containing mel spectrograms and labels
        test_size: Fraction of the dataset to use for testing (0.2 means 20% test, 80% train)
    Returns:
        Dictionary containing train and test datasets, label encoder, and scaler
    """
    # Separate features and labels
    X = np.array([item[0] for item in raw_dataset])
    y = np.array([item[1] for item in raw_dataset])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Unique labels: {np.unique(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Normalize spectrograms
    # Reshape for scaling: (n_samples, n_features)
    original_shape = X_train.shape
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(original_shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = MridangamDataset(X_train_scaled, y_train_encoded)
    test_dataset = MridangamDataset(X_test_scaled, y_test_encoded)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'num_classes': len(label_encoder.classes_)
    }

def create_dataloader(data_dict: dict, batch_size: int = 32):
    """
    Create DataLoaders for the datasets.
    
    Args:
        data_dict: Dictionary containing train and test datasets
        batch_size: Batch size for the DataLoaders
    Returns:
        Dictionary containing train and test DataLoaders
    """
    train_loader = DataLoader(
        data_dict['train'], 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        data_dict['test'], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test': test_loader
    }
