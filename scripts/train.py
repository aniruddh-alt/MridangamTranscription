import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm

from mridangam_transcription.models.cnn_attention import MridangamCNN
from mridangam_transcription.data.stroke_dataset import create_datasets
from mridangam_transcription.utils.checkpointing import save_checkpoint

def train(args):
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"Loading data from {args.data_dir}...")
    dataset_dict = create_datasets(Path(args.data_dir), test_size=args.test_size, val_size=args.val_size)
    
    train_loader = DataLoader(dataset_dict['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset_dict['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    classes = dataset_dict['train'].classes
    num_classes = len(classes)
    mel_stats = dataset_dict['mel_stats']
    
    print(f"Classes: {classes}")
    
    # Initialize model
    model = MridangamCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    save_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata separately for easy access
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "classes": classes,
            "mel_stats": {k: v.tolist() for k, v in mel_stats.items()}
        }, f)
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save checkpoints
        train_metrics = {'loss': avg_train_loss, 'acc': avg_train_acc}
        val_metrics = {'loss': avg_val_loss, 'acc': avg_val_acc}
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, classes, mel_stats,
            train_metrics, val_metrics, save_dir / "latest.pth"
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, classes, mel_stats,
                train_metrics, val_metrics, save_dir / "best.pth"
            )
            print(f"New best model saved to {save_dir}/best.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mridangam Transcription Model")
    parser.add_argument("--data-dir", type=str, default="dataset/raw_data/mridangam_stroke_1.0", help="Path to raw stroke data")
    parser.add_argument("--output-dir", type=str, default="model_checkpoints", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    train(args)

