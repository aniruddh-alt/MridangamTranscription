import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, classes, mel_stats, train_metrics, val_metrics, filepath, model_type=None):
    """Save training checkpoint with metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'classes': classes,
        'mel_stats': mel_stats,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    if model_type is not None:
        checkpoint['model_type'] = model_type
    torch.save(checkpoint, filepath)

def load_checkpoint(model, filepath, device, optimizer=None, scheduler=None):
    """Load training checkpoint with metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint

