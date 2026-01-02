import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm
import numpy as np
from transformers import ASTFeatureExtractor

from mridangam_transcription.models.ast_classifier import ASTMridangamClassifier
from mridangam_transcription.data.ast_dataset import create_ast_datasets
from mridangam_transcription.utils.checkpointing import save_checkpoint

def train(args):
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Initialize AST feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    # Create datasets
    print(f"Loading data from {args.data_dir}...")
    dataset_dict = create_ast_datasets(
        Path(args.data_dir), 
        feature_extractor,
        test_size=args.test_size, 
        val_size=args.val_size
    )
    
    train_loader = DataLoader(
        dataset_dict['train'], 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn  # Custom collate to handle variable-length sequences
    )
    val_loader = DataLoader(
        dataset_dict['val'], 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    classes = dataset_dict['train'].classes
    num_classes = len(classes)
    
    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = ASTMridangamClassifier(num_classes=num_classes).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters (initial): {model.get_trainable_params():,}")
    
    # Freeze backbone initially
    model.freeze_backbone()
    print(f"Trainable parameters (after freeze): {model.get_trainable_params():,}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Train classifier head only
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.lr_head, 
        weight_decay=1e-4
    )
    
    # Cosine scheduler with warmup - Phase 1 only
    phase1_steps = len(train_loader) * args.freeze_epochs
    current_warmup_steps = int(0.1 * phase1_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_steps - current_warmup_steps
    )
    
    # Mixed precision training for CUDA
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    save_dir = Path(args.output_dir) / f"ast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "classes": classes,
            "model_type": "AST",
            "pretrained": "MIT/ast-finetuned-audioset-10-10-0.4593"
        }, f, indent=2)
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Phase 1 (frozen backbone): epochs 1-{args.freeze_epochs}")
    print(f"Phase 2 (full fine-tuning): epochs {args.freeze_epochs + 1}-{args.epochs}")
    
    for epoch in range(args.epochs):
        # Switch to full fine-tuning after freeze_epochs
        if epoch == args.freeze_epochs:
            print(f"\n{'='*60}")
            print(f"Unfreezing backbone at epoch {epoch + 1}")
            print(f"{'='*60}")
            model.unfreeze_backbone()
            print(f"Trainable parameters (after unfreeze): {model.get_trainable_params():,}")
            
            # Create new optimizer with lower learning rate
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_finetune,
                weight_decay=1e-4
            )
            # Reset scheduler for fine-tuning phase
            remaining_steps = len(train_loader) * (args.epochs - args.freeze_epochs)
            current_warmup_steps = int(0.1 * remaining_steps)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps - current_warmup_steps
            )
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Calculate current step for warmup
        if epoch < args.freeze_epochs:
            # Phase 1: steps from start of training
            phase_start_step = 0
        else:
            # Phase 2: steps from start of fine-tuning
            phase_start_step = args.freeze_epochs * len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Calculate global step within current phase
            global_step = epoch * len(train_loader) + batch_idx
            phase_step = global_step - phase_start_step
            
            # Warmup learning rate
            if epoch < args.freeze_epochs:
                # Phase 1: warmup for classifier head training
                if phase_step < current_warmup_steps:
                    warmup_lr = args.lr_head * (phase_step + 1) / current_warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                else:
                    scheduler.step()
            else:
                # Phase 2: warmup for fine-tuning
                if phase_step < current_warmup_steps:
                    warmup_lr = args.lr_finetune * (phase_step + 1) / current_warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                else:
                    scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': loss.item(), 
                'acc': 100. * train_correct / train_total,
                'lr': f'{current_lr:.2e}'
            })
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Save checkpoints
        train_metrics = {'loss': avg_train_loss, 'acc': avg_train_acc}
        val_metrics = {'loss': avg_val_loss, 'acc': avg_val_acc}
        
        # Use None for mel_stats since AST doesn't need it
        save_checkpoint(
            model, optimizer, scheduler, epoch, classes, None,
            train_metrics, val_metrics, save_dir / "latest.pth",
            model_type="AST"
        )
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, classes, None,
                train_metrics, val_metrics, save_dir / "best.pth",
                model_type="AST"
            )
            print(f"New best model saved (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_dir}")

def collate_fn(batch):
    """Custom collate function to pad variable-length sequences."""
    inputs, labels = zip(*batch)
    
    # Find max length in batch
    max_len = max(inp.shape[0] for inp in inputs)
    
    # Pad all sequences to max length
    padded_inputs = []
    for inp in inputs:
        if inp.shape[0] < max_len:
            padding = torch.zeros(max_len - inp.shape[0])
            padded = torch.cat([inp, padding])
        else:
            padded = inp
        padded_inputs.append(padded)
    
    return torch.stack(padded_inputs), torch.stack(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AST Model for Mridangam Transcription")
    parser.add_argument("--data-dir", type=str, default="dataset/raw_data/mridangam_stroke_1.0", help="Path to raw stroke data")
    parser.add_argument("--output-dir", type=str, default="model_checkpoints", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--freeze-epochs", type=int, default=10, help="Number of epochs with frozen backbone")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--lr-finetune", type=float, default=1e-5, help="Learning rate for full fine-tuning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience (0 to disable)")
    
    args = parser.parse_args()
    train(args)

