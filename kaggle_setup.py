import os
import shutil
import zipfile
from pathlib import Path
import pickle
import sys

# Set up paths
project_root = Path(__file__).parent
dataset_dir = project_root / "dataset"
kaggle_export_dir = project_root / "kaggle_export"
code_dir = kaggle_export_dir / "code"
data_dir = kaggle_export_dir / "data"

# Create export directories
os.makedirs(kaggle_export_dir, exist_ok=True)
os.makedirs(code_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Copy necessary code files
code_files = [
    dataset_dir / "data_processing" / "data_preparation.py",
    dataset_dir / "data_processing" / "dataset_creation.py",
    project_root / "model_architecture" / "CNN.py",
]

for file_path in code_files:
    if file_path.exists():
        shutil.copy2(file_path, code_dir)
        print(f"Copied {file_path.name} to code directory")
    else:
        print(f"Warning: {file_path} not found")

# Create a Kaggle-specific entry point
kaggle_notebook = """
# Mridangam Transcription Model on Kaggle
# This notebook runs the mridangam transcription model

import sys
import os
from pathlib import Path

# Add code directory to path
sys.path.append('/kaggle/input/mridangam-transcription/code')

# Import functions from our modules
from data_preparation import get_audio, get_mel_spectrogram, get_window, get_onset
from dataset_creation_kaggle import MridangamDataset, compute_mel_statistics, create_file_based_dataset, create_efficient_dataloader

# Set up paths for Kaggle
data_path = Path('/kaggle/input/mridangam-transcription/data/mridangam_stroke_1.0')

# Import and set up your model
import torch
import torch.nn as nn
import torch.optim as optim
from CNN import CNNModel  # Import your model architecture

# Hyperparameters
batch_size = 32
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

# Set up device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
n_classes = data['num_classes']
model = CNNModel(n_classes=n_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define a function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100.0 * correct / total
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_accuracy = 100.0 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_accuracy': test_accuracy,
            }, f'model_checkpoint_epoch_{epoch+1}.pth')

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_accuracy = 100.0 * correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy, test_loss/len(test_loader)

# Training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
evaluate_model(model, test_loader)

# Save final model
torch.save(model.state_dict(), 'mridangam_model_final.pth')
print("Training complete!")
"""

with open(code_dir / "kaggle_notebook.ipynb", "w") as f:
    f.write(kaggle_notebook)
    print("Created Kaggle notebook template")

# Make a modified version of dataset_creation.py for Kaggle
original_file = dataset_dir / "data_processing" / "dataset_creation.py"
kaggle_file = code_dir / "dataset_creation_kaggle.py"

with open(original_file, "r") as f:
    content = f.read()

# Modify imports and paths for Kaggle
kaggle_content = content.replace(
    "import sys\nproject_root = Path(__file__).parent.parent.parent\nsys.path.insert(0, str(project_root))",
    "# Adjusted for Kaggle environment\nimport sys\nsys.path.append('/kaggle/input/mridangam-transcription/code')"
)
kaggle_content = kaggle_content.replace(
    "from dataset.data_processing.data_preparation",
    "from data_preparation"
)

with open(kaggle_file, "w") as f:
    f.write(kaggle_content)
    print("Created Kaggle-adapted dataset_creation.py")

# Also update the original dataset_creation.py with the same changes for Kaggle
with open(code_dir / "dataset_creation.py", "w") as f:
    f.write(kaggle_content)
    print("Updated dataset_creation.py for Kaggle compatibility")

# Create a README for Kaggle
readme_content = """
# Mridangam Transcription Dataset

This dataset contains audio samples of mridangam strokes for classification and transcription tasks.

## Dataset Structure
- `/data/mridangam_stroke_1.0/`: Contains the audio files organized by stroke type
- `/code/`: Contains the Python modules for data processing and model architecture

## Usage
The main notebook provides a complete pipeline for:
1. Loading and processing the audio data
2. Creating datasets with on-the-fly processing
3. Training a CNN model for stroke classification
4. Evaluating model performance

## Model Architecture
The model uses a CNN architecture designed for audio classification tasks.
"""

with open(kaggle_export_dir / "README.md", "w") as f:
    f.write(readme_content)
    print("Created README for Kaggle")

# Prepare the dataset
mridangam_dataset_dir = dataset_dir / "raw_data" / "mridangam_stroke_1.0"
if mridangam_dataset_dir.exists():
    # Create a directory structure to preserve dataset organization
    kaggle_dataset_dir = data_dir / "mridangam_stroke_1.0"
    os.makedirs(kaggle_dataset_dir, exist_ok=True)
    
    # Copy folders with audio files
    for folder in mridangam_dataset_dir.iterdir():
        if folder.is_dir():
            # Replace '#' with 'sharp' in folder names
            folder_name = folder.name.replace('#', 'sharp')
            dest_folder = kaggle_dataset_dir / folder_name
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy all audio files
            for file in folder.glob("*.wav"):
                # Replace '#' with 'sharp' in filenames
                dest_filename = file.name.replace('#', 'sharp')
                shutil.copy2(file, dest_folder / dest_filename)
            
            print(f"Copied audio files from {folder.name} to dataset directory as {folder_name}")
    
    # Copy readme file if it exists
    readme_file = mridangam_dataset_dir / "_readme_and_license.txt"
    if readme_file.exists():
        shutil.copy2(readme_file, kaggle_dataset_dir)
        print("Copied readme and license file")
else:
    print(f"Warning: Dataset directory {mridangam_dataset_dir} not found")

# Create ZIP file for easy upload to Kaggle
def zipdir(path, ziph):
    # Zip the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            ziph.write(file_path, os.path.relpath(file_path, os.path.join(path, '..')))

zipf = zipfile.ZipFile(project_root / 'mridangam_transcription_kaggle.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir(kaggle_export_dir, zipf)
zipf.close()

print("\nSetup complete!")
print(f"Kaggle export files are in: {kaggle_export_dir}")
print(f"Upload the ZIP file 'mridangam_transcription_kaggle.zip' to Kaggle as a dataset")
print("Then, create a new notebook and use the provided kaggle_notebook.ipynb as a starting point")
