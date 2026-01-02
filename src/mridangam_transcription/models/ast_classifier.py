import torch
import torch.nn as nn
from transformers import ASTForAudioClassification

class ASTMridangamClassifier(nn.Module):
    """AST model wrapper with freeze/unfreeze utilities for gradual fine-tuning."""
    
    def __init__(self, num_classes: int, pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        """
        Initialize AST model for mridangam stroke classification.
        
        Args:
            num_classes: Number of stroke classes
            pretrained: Hugging Face model identifier
        """
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.num_classes = num_classes
        
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AST model.
        
        Args:
            input_values: Audio input tensor of shape (batch, time) or (time,)
        
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Ensure batch dimension exists
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)
        
        outputs = self.ast(input_values)
        return outputs.logits
    
    def freeze_backbone(self):
        """Freeze the transformer backbone, keep only classifier trainable."""
        for param in self.ast.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        
        # Ensure classifier is trainable
        if hasattr(self.ast, 'classifier'):
            for param in self.ast.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.ast.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

