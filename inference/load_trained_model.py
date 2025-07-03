#!/usr/bin/env python3
"""
Direct model loading script using the exact architecture from training
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path to import the original model
sys.path.append(str(Path(__file__).parent.parent))

def load_trained_model(model_path, device='cuda'):
    """Load the trained model using the exact architecture from training"""
    
    # Import the exact model from training code
    try:
        from model_architecture.CNN_with_attention import MridangamCNN, load_model_for_inference
        print("✅ Using original training model architecture")
        return load_model_for_inference(model_path, device)
    except ImportError:
        print("⚠️ Could not import original model, falling back to local implementation")
        
        # Fallback to local implementation
        from realtime_cnn_attention_inference import MridangamCNN
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Determine number of classes
        num_classes = 10
        for key, tensor in state_dict.items():
            if key.endswith('classifier.5.weight'):
                num_classes = tensor.shape[0]
                break
        
        # Create model
        model = MridangamCNN(n_mels=128, num_classes=num_classes, dropout_rate=0.5)
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        return model

if __name__ == "__main__":
    model_path = "../model_checkpoints/best_model_20250622_010327.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    
    try:
        model = load_trained_model(model_path, device)
        print("✅ Model loaded successfully!")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 128, 128).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful: {output.shape}")
        print(f"✅ Model ready for inference!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 