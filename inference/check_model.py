import torch

# Load the checkpoint
checkpoint = torch.load('../model_checkpoints/best_model_20250622_010327.pth', map_location='cpu')

print("Model keys in classifier:")
for key, tensor in checkpoint['model_state_dict'].items():
    if 'classifier' in key:
        print(f"{key}: {tensor.shape}")

print("\nAll model keys:")
for key in checkpoint['model_state_dict'].keys():
    print(key) 