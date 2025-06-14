"""
Simple Metrics Calculator for Mridangam Model Evaluation
=========================================================

This script provides simple functions to calculate essential metrics
for evaluating the mridangam transcription model.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

def get_predictions(model, test_loader, device):
    """
    Get predictions and true labels from the model
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run inference on
    
    Returns:
        tuple: (predictions, true_labels, probabilities)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Getting predictions...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def calculate_basic_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    return metrics

def calculate_confidence_metrics(probabilities, y_true, y_pred):
    """
    Calculate confidence-related metrics
    
    Args:
        probabilities: Prediction probabilities
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing confidence metrics
    """
    max_probs = np.max(probabilities, axis=1)
    correct_mask = y_pred == y_true
    
    confidence_metrics = {
        'avg_confidence_all': np.mean(max_probs),
        'avg_confidence_correct': np.mean(max_probs[correct_mask]) if np.any(correct_mask) else 0,
        'avg_confidence_incorrect': np.mean(max_probs[~correct_mask]) if np.any(~correct_mask) else 0,
        'low_confidence_count': np.sum(max_probs < 0.6),
        'low_confidence_percentage': np.sum(max_probs < 0.6) / len(max_probs) * 100
    }
    
    return confidence_metrics

def print_metrics_report(metrics, confidence_metrics, class_names=None):
    """
    Print a formatted metrics report
    
    Args:
        metrics: Dictionary of basic metrics
        confidence_metrics: Dictionary of confidence metrics
        class_names: List of class names (optional)
    """
    print("=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    
    # Basic metrics
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"Recall (Weighted):  {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    print("\n" + "-" * 40)
    print("CONFIDENCE METRICS")
    print("-" * 40)
    print(f"Average Confidence (All):       {confidence_metrics['avg_confidence_all']:.4f} ({confidence_metrics['avg_confidence_all']*100:.2f}%)")
    print(f"Average Confidence (Correct):   {confidence_metrics['avg_confidence_correct']:.4f} ({confidence_metrics['avg_confidence_correct']*100:.2f}%)")
    print(f"Average Confidence (Incorrect): {confidence_metrics['avg_confidence_incorrect']:.4f} ({confidence_metrics['avg_confidence_incorrect']*100:.2f}%)")
    print(f"Low Confidence Predictions:     {confidence_metrics['low_confidence_count']} ({confidence_metrics['low_confidence_percentage']:.1f}%)")
    
    # Per-class metrics
    if class_names is not None:
        print("\n" + "-" * 40)
        print("PER-CLASS METRICS")
        print("-" * 40)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 55)
        
        for i, class_name in enumerate(class_names):
            if i < len(metrics['precision_per_class']):
                precision = metrics['precision_per_class'][i]
                recall = metrics['recall_per_class'][i]
                f1 = metrics['f1_per_class'][i]
                print(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

def evaluate_model_simple(model, test_loader, device, label_encoder=None):
    """
    Simple one-function evaluation
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run inference on
        label_encoder: Label encoder (optional, for class names)
    
    Returns:
        dict: Complete evaluation results
    """
    # Get predictions
    predictions, true_labels, probabilities = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_basic_metrics(true_labels, predictions)
    confidence_metrics = calculate_confidence_metrics(probabilities, true_labels, predictions)
    
    # Get class names
    class_names = None
    if label_encoder is not None:
        class_names = label_encoder.classes_
    
    # Print report
    print_metrics_report(metrics, confidence_metrics, class_names)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Detailed classification report
    if class_names is not None:
        print("\n" + "-" * 40)
        print("DETAILED CLASSIFICATION REPORT")
        print("-" * 40)
        print(classification_report(true_labels, predictions, target_names=class_names))
    else:
        print("\n" + "-" * 40)
        print("DETAILED CLASSIFICATION REPORT")
        print("-" * 40)
        print(classification_report(true_labels, predictions))
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities,
        'metrics': metrics,
        'confidence_metrics': confidence_metrics,
        'confusion_matrix': cm,
        'class_names': class_names
    }

def quick_accuracy_check(model, test_loader, device):
    """
    Quick accuracy check without detailed metrics
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run inference on
    
    Returns:
        float: Accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Quick accuracy check"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Quick Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy

# Example usage:
if __name__ == "__main__":
    """
    Example usage of the metrics functions
    """
    
    # Assuming you have these variables from your training script:
    # model = your_trained_model
    # test_loader = your_test_dataloader
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # label_encoder = your_label_encoder
    
    print("Simple Metrics Calculator")
    print("=" * 40)
    print("To use this script:")
    print("1. Import the functions you need")
    print("2. Call evaluate_model_simple() with your model and data")
    print("\nExample:")
    print("from simple_metrics import evaluate_model_simple")
    print("results = evaluate_model_simple(model, test_loader, device, label_encoder)")
    print("\nFor quick accuracy:")
    print("from simple_metrics import quick_accuracy_check")
    print("accuracy = quick_accuracy_check(model, test_loader, device)")