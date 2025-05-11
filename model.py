"""
Cardiac Scan Classifier Model Architecture
Uses ResNet50 as base with modifications for cardiac image analysis
Includes Grad-CAM implementation for explainability
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50
from typing import Tuple, Optional

class CardiacResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize cardiac scan classifier
        
        Args:
            num_classes: Number of output classes (healthy/abnormal + specific conditions)
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load base ResNet50 model
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Replace final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Grad-CAM setup
        self.gradients = None
        self.activations = None
        
        # Hook the layer we want to visualize (before final pooling)
        target_layer = self.base_model.layer4[-1]
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        return self.base_model(x)
    
    def forward_hook(self, module, input, output):
        """Store activations for Grad-CAM"""
        self.activations = output
    
    def backward_hook(self, module, grad_input, grad_output):
        """Store gradients for Grad-CAM"""
        self.gradients = grad_output[0]
    
    def get_gradcam(self, input_tensor: torch.Tensor, 
                   target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for visualization
        
        Args:
            input_tensor: Input image tensor
            target_class: Class to generate heatmap for (None uses predicted class)
            
        Returns:
            Grad-CAM heatmap tensor
        """
        # Forward pass
        self.zero_grad()
        output = self(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output)
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # Pool gradients and generate heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).detach().cpu()
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)  # Normalize
        
        return heatmap

def build_model(device: torch.device = torch.device('cpu'), 
                num_classes: int = 2) -> CardiacResNet:
    """
    Convenience function to build and prepare model
    
    Args:
        device: Target device (cpu/cuda)
        num_classes: Number of output classes
        
    Returns:
        Configured CardiacResNet model
    """
    model = CardiacResNet(num_classes=num_classes)
    model = model.to(device)
    return model

if __name__ == '__main__':
    # Test model initialization
    model = build_model()
    print("Model initialized successfully")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))