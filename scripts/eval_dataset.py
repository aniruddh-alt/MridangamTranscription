import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from mridangam_transcription.models.cnn_attention import MridangamCNN
from mridangam_transcription.data.stroke_dataset import create_datasets
from mridangam_transcription.utils.checkpointing import load_checkpoint

def evaluate(args):
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Load metadata/classes first if possible, or just create dataset to get them
    # For evaluation, we ideally want to use the EXACT classes and mel_stats from training
    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint['classes']
    mel_stats = checkpoint['mel_stats']
    
    print(f"Classes: {classes}")
    
    # Create datasets (only need test)
    dataset_dict = create_datasets(Path(args.data_dir), test_size=args.test_size, val_size=0.1)
    test_loader = DataLoader(dataset_dict['test'], batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = MridangamCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    output_path = Path(args.model_path).parent / "confusion_matrix.png"
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mridangam Transcription Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="dataset/raw_data/mridangam_stroke_1.0", help="Path to raw stroke data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    evaluate(args)

