"""
Kaggle Dataset Validation Script

This script can be run on Kaggle to verify that the dataset is loaded correctly
and the model processes the data as expected.
"""

import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add code directory to path
sys.path.append('/kaggle/input/mridangam-transcription/code')

# Import our modules
from data_preparation import get_audio, get_mel_spectrogram, get_window, get_onset
from dataset_creation import MridangamDataset, create_file_based_dataset, create_efficient_dataloader
from CNN import MridangamCNN as CNNModel

# Print versions for debugging
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Set up paths for Kaggle
data_path = Path('/kaggle/input/mridangam-transcription/data/mridangam_stroke_1.0')
print(f"Dataset path exists: {data_path.exists()}")

# Count files in the dataset
total_files = 0
stroke_types = []
for subdir in data_path.iterdir():
    if subdir.is_dir():
        count = len(list(subdir.glob('*.wav')))
        total_files += count
        stroke_types.append(subdir.name)
        print(f"Found {count} files in stroke type {subdir.name}")

print(f"Total audio files: {total_files}")
print(f"Stroke types: {stroke_types}")

# Create datasets and dataloaders
print("\nCreating datasets...")
data = create_file_based_dataset(
    directory=data_path,
    test_size=0.2,
    target_length=128,
    architecture='cnn',
    compute_stats=True
)

# Create efficient dataloaders (optimized for Kaggle)
batch_size = 16  # Smaller batch size for testing
train_loader = create_efficient_dataloader(
    dataset=data['train'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = create_efficient_dataloader(
    dataset=data['test'],
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"Train dataset size: {len(data['train'])}")
print(f"Test dataset size: {len(data['test'])}")
print(f"Number of classes: {data['num_classes']}")
print(f"Class names: {data['train'].label_encoder.classes_}")

# Visualize a batch of data
batch = next(iter(train_loader))
inputs, labels = batch

print(f"\nInput tensor shape: {inputs.shape}")
print(f"Label tensor shape: {labels.shape}")

# Plot mel spectrograms from the batch
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
axes = axes.flatten()

for i in range(min(8, len(inputs))):
    # Get mel spectrogram (remove channel dimension)
    mel = inputs[i].squeeze(0).numpy()
    
    # Get label
    label_idx = labels[i].item()
    label_name = data['train'].label_encoder.classes_[label_idx]
    
    # Plot
    axes[i].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    axes[i].set_title(f"Class: {label_name}")
    axes[i].set_ylabel("Mel bin")
    axes[i].set_xlabel("Time frame")

plt.tight_layout()
plt.savefig('mel_spectrograms.png')
plt.close()

# Test model initialization
n_classes = data['num_classes']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(n_classes=n_classes).to(device)

print("\nModel initialized successfully")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Test a forward pass
test_input = inputs[:2].to(device)
with torch.no_grad():
    output = model(test_input)

print(f"Model output shape: {output.shape}")
print("Forward pass successful!")

print("\nDataset validation complete. The data and model are ready for training.")
