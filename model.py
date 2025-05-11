import torch
import torch.nn as nn
from typing import Optional
from torchvision.models.resnet import ResNet, Bottleneck  # tiny part of torchvision

class CardiacResNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Build ResNet50 from scratch (still minimal import from torchvision)
        self.base_model = ResNet(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes
        )

        self.gradients = None
        self.activations = None

        # Hook last Bottleneck layer
        target_layer = self.base_model.layer4[-1]
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_gradcam(self, input_tensor: torch.Tensor, 
                    target_class: Optional[int] = None) -> torch.Tensor:
        self.zero_grad()
        output = self(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output)

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).detach().cpu()
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)
        return heatmap

def build_model(device: torch.device = torch.device('cpu'), 
                num_classes: int = 2) -> CardiacResNet:
    model = CardiacResNet(num_classes=num_classes)
    return model.to(device)

if __name__ == '__main__':
    model = build_model()
    print("Model initialized.")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
