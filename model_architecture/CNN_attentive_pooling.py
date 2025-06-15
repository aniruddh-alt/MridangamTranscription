import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.data import DataLoader
import sys
import numpy as np

# Add the project root to Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from dataset.data_processing.dataset_creation import MridangamDataset
    from dataset.data_processing.dataset_creation import create_efficient_dataloader
    from dataset.data_processing.dataset_creation import create_file_based_dataset
except ImportError as e:
    print(f"Warning: Could not import dataset modules: {e}")
    print("Please make sure you're running from the project root directory")
    print("You can also run this as a module: python -m model_architecture.CNN_attentive_pooling")
    sys.exit(1)

from pathlib import Path

class FrequencyAttentivePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Apply frequency attentive pooling to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, freq, time).
        Returns:
            torch.Tensor: Pooled tensor of shape (batch, channels).
        """
        x_mean = x.mean(dim=-1) # (batch, channels, freq)
        attentive_pooling = self.attention(x_mean)  # (batch, 1, freq)
        att_weights = torch.softmax(attentive_pooling, dim=-1)  # Normalize weights
        x_weighted = x_mean * att_weights  # (batch, channels, freq)
        return x_weighted.sum(dim=-1)  # (batch, channels)    

class MridangamCNN(nn.Module):
    def __init__(self, n_mels=128, num_classes=10):
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
        )
        
        self.attention_pooling = FrequencyAttentivePooling(in_channels=128)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # (batch_size, 128, 16, 16)
        x = self.attention_pooling(x)  # (batch_size, 128)
        x = self.classifier(x)  # (batch_size, num_classes)
        return x

# Model setup
model = MridangamCNN(n_mels=128, num_classes=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.save_path = save_path
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model, epoch)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model, epoch)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Restored best weights from epoch {self.best_epoch}")
            return True
        return False
    
    def save_checkpoint(self, model, epoch):
        """Save the best model weights"""
        self.best_weights = model.state_dict().copy()
        
        # Also save to file if path is provided
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': self.best_loss,
            }, self.save_path)

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, train_acc, val_acc, filepath):
    """Save complete training checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, scheduler, filepath, device):
    """Load training checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Previous - Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"Previous - Train Acc: {checkpoint['train_acc']:.2f}%, Val Acc: {checkpoint['val_acc']:.2f}%")
        return start_epoch
    else:
        print(f"No checkpoint found at {filepath}")
        return 0

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test/validation set"""
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
            
            # Use argmax for multi-class classification
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total  # Return as fraction, not percentage
    avg_loss = sum(losses) / len(losses)
    return accuracy, avg_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=100, save_dir="checkpoints", resume_from_checkpoint=None):
    """
    Robust training function with checkpointing and early stopping
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Checkpoint paths
    best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")
    latest_checkpoint_path = os.path.join(save_dir, f"latest_checkpoint_{timestamp}.pth")
    
    # Initialize tracking variables
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch = load_checkpoint(model, optimizer, scheduler, resume_from_checkpoint, device)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=15, 
        min_delta=0.001, 
        restore_best_weights=True,
        save_path=best_model_path
    )
    
    print(f"Starting training from epoch {start_epoch + 1}")
    print(f"Best model will be saved to: {best_model_path}")
    print(f"Latest checkpoint will be saved to: {latest_checkpoint_path}")
    print("=" * 80)
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.2f}%'
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print epoch results
        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            train_loss, val_loss, train_acc*100, val_acc*100,
            latest_checkpoint_path
        )
        
        # Early stopping check
        if early_stopping(val_loss, model, epoch):
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch+1}")
            break
    
    print("=" * 80)
    print("Training completed!")
    print(f"Best model saved at: {best_model_path}")
    print(f"Latest checkpoint saved at: {latest_checkpoint_path}")
    
    # Return training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': early_stopping.best_loss,
        'best_epoch': early_stopping.best_epoch
    }
    
    return history

def test_model(model, test_loader, criterion, device, model_path=None):
    """Test the model on test set"""
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {model_path}")
    
    test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)
    
    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f'Test Accuracy: {test_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')
    print("=" * 50)
    
    return test_acc, test_loss

def get_model_summary(model, input_shape=(1, 1, 128, 128)):
    """Print model architecture summary"""
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    
    print("\nModel Architecture Summary:")
    print("=" * 50)
    
    # Test forward pass with intermediate outputs
    with torch.no_grad():
        print(f"Input shape: {x.shape}")
        
        # Features
        x_features = model.features(x)
        print(f"After features: {x_features.shape}")
        
        # Attention pooling
        x_pooled = model.attention_pooling(x_features)
        print(f"After attention pooling: {x_pooled.shape}")
        
        # Classifier
        output = model.classifier(x_pooled)
        print(f"Final output: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)

# Visualization and utility functions
def plot_training_history(history, save_path=None):
    """Plot training history"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_losses'], label='Train Loss', color='blue')
        ax1.plot(history['val_losses'], label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot([acc*100 for acc in history['train_accuracies']], label='Train Accuracy', color='blue')
        ax2.plot([acc*100 for acc in history['val_accuracies']], label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
        print("Training History Summary:")
        print(f"Final Train Loss: {history['train_losses'][-1]:.4f}")
        print(f"Final Val Loss: {history['val_losses'][-1]:.4f}")
        print(f"Final Train Acc: {history['train_accuracies'][-1]*100:.2f}%")
        print(f"Final Val Acc: {history['val_accuracies'][-1]*100:.2f}%")
        print(f"Best Val Loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch'] + 1}")

def save_training_summary(history, model_info, save_path):
    """Save training summary to text file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"Architecture: {model_info.get('architecture', 'CNN with Attentive Pooling')}\n")
        f.write(f"Total Parameters: {model_info.get('total_params', 'Unknown'):,}\n")
        f.write(f"Trainable Parameters: {model_info.get('trainable_params', 'Unknown'):,}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"Batch Size: {model_info.get('batch_size', 'Unknown')}\n")
        f.write(f"Learning Rate: {model_info.get('learning_rate', 'Unknown')}\n")
        f.write(f"Number of Epochs: {len(history['train_losses'])}\n")
        f.write(f"Device: {model_info.get('device', 'Unknown')}\n\n")
        
        f.write("Final Results:\n")
        f.write(f"Final Train Loss: {history['train_losses'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history['val_losses'][-1]:.4f}\n")
        f.write(f"Final Train Accuracy: {history['train_accuracies'][-1]*100:.2f}%\n")
        f.write(f"Final Val Accuracy: {history['val_accuracies'][-1]*100:.2f}%\n\n")
        
        f.write("Best Performance:\n")
        f.write(f"Best Val Loss: {history['best_val_loss']:.4f}\n")
        f.write(f"Best Epoch: {history['best_epoch'] + 1}\n")
        
        if 'test_acc' in model_info and 'test_loss' in model_info:
            f.write(f"\nTest Results:\n")
            f.write(f"Test Accuracy: {model_info['test_acc']*100:.2f}%\n")
            f.write(f"Test Loss: {model_info['test_loss']:.4f}\n")
    
    print(f"Training summary saved to: {save_path}")

def create_model_config(model, batch_size, learning_rate, device):
    """Create model configuration dictionary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    config = {
        'architecture': 'CNN with Attentive Pooling',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device)
    }
    
    return config

# Test the model
if __name__ == "__main__":
    # Print model summary
    print("Model Architecture Summary:")
    get_model_summary(model)
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    architecture = 'cnn'
    
    # Data path - use the correct local path
    data_path = Path(r'dataset\raw\mridangam_stroke_1.0')
    
    print(f"\nUsing data path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")
    
    if not data_path.exists():
        print("Error: Data path does not exist!")
        print("Please ensure the dataset is available at the specified path.")
        exit(1)
      # Create datasets and dataloaders
    print("Creating datasets...")
    try:
        data = create_file_based_dataset(
            directory=data_path,
            test_size=0.2,
            val_size=0.2,  # 20% of remaining data after test split
            target_length=128,
            architecture=architecture,
            compute_stats=True
        )
        
        print(f"Dataset created successfully!")
        print(f"Train samples: {len(data['train'])}")
        print(f"Validation samples: {len(data['val'])}")
        print(f"Test samples: {len(data['test'])}")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        exit(1)

    # Create efficient dataloaders
    train_loader = create_efficient_dataloader(
        dataset=data['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = create_efficient_dataloader(
        dataset=data['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = create_efficient_dataloader(
        dataset=data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print("\nDataset verification:")
    try:
        sample_batch = next(iter(train_loader))
        print(f"Input shape: {sample_batch[0].shape}")
        print(f"Labels shape: {sample_batch[1].shape}")
        print(f"Expected for CNN: (batch_size, 1, n_mels, time_steps)")
        print(f"Label range: {sample_batch[1].min().item()} to {sample_batch[1].max().item()}")
        
        # Verify the model works with sample data
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_batch[0].to(device))
            print(f"Model output shape: {sample_output.shape}")
            print(f"Expected output shape: (batch_size, {model.classifier[-1].out_features})")
        
    except Exception as e:
        print(f"Error during dataset verification: {e}")
        exit(1)
    
    # Setup model, criterion, optimizer, scheduler
    print(f"\nUsing device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Create save directory
    save_dir = "model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {save_dir}")
    print(f"Training configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Device: {device}")
    print("=" * 80)
    
    # Start training
    try:
        print("Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            save_dir=save_dir,
            resume_from_checkpoint=None  # Set to checkpoint path to resume
        )
        
        print("\nTraining completed successfully!")
        print(f"Best validation loss: {history['best_val_loss']:.4f}")
        print(f"Best epoch: {history['best_epoch'] + 1}")
        
        # Plot and save training history
        plot_path = os.path.join(save_dir, "training_history.png")
        plot_training_history(history, save_path=plot_path)
        
        # Save training summary
        summary_path = os.path.join(save_dir, "training_summary.txt")
        model_config = create_model_config(model, batch_size, learning_rate, device)
        save_training_summary(history, model_config, save_path=summary_path)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test the model
    try:
        print("\nTesting the model...")
        # Find the best model file
        import glob
        best_model_files = glob.glob(os.path.join(save_dir, "best_model_*.pth"))
        if best_model_files:
            best_model_path = best_model_files[-1]  # Get the latest one
            print(f"Loading best model from: {best_model_path}")
            
            test_acc, test_loss = test_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                model_path=best_model_path
            )
            
            print(f"\nFinal Test Results:")
            print(f"Test Accuracy: {test_acc*100:.2f}%")
            print(f"Test Loss: {test_loss:.4f}")
        else:
            print("No best model file found, testing current model...")
            test_acc, test_loss = test_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device
            )
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTraining pipeline completed!")
