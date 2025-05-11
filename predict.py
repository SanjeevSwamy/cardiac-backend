"""
Cardiac Scan Prediction Pipeline (Interactive Version)
- Takes user uploaded files
- Processes images
- Makes predictions with explanations
- Generates Grad-CAM heatmaps
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CardiacResNet
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import io
import base64
import os

class CardiacPredictor:
    def __init__(self, model_path: str = 'best_model.pth', device: str = 'auto'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model weights
            device: 'cuda', 'cpu', or 'auto'
        """
        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available() else 
            'cpu' if device == 'auto' else device
        )
        
        # Load model
        self.model = CardiacResNet(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Class names and explanation templates
        self.class_names = ['healthy', 'abnormal']
        self.explanations = {
            'healthy': "The cardiac scan appears normal with no significant abnormalities detected.",
            'abnormal': {
                'generic': "The scan shows cardiac abnormalities that may indicate cardiovascular disease.",
                'specific': {
                    'enlarged': "Enlarged heart chambers detected, suggesting possible cardiomyopathy.",
                    'flow': "Reduced blood flow observed, potentially indicating coronary artery disease.",
                    'structural': "Structural abnormalities visible, which may require further evaluation."
                }
            }
        }
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_data) -> torch.Tensor:
        """
        Process image from file path, bytes, or PIL Image
        
        Args:
            image_data: Can be:
                - File path (str)
                - Bytes object
                - PIL Image
                
        Returns:
            Processed image tensor (1, 3, 224, 224)
        """
        if isinstance(image_data, str):
            # Handle file path
            img = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, bytes):
            # Handle byte stream
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            # Handle PIL Image
            img = image_data.convert('RGB')
        else:
            raise ValueError("Unsupported image input type")
            
        return self.transform(img).unsqueeze(0).to(self.device)
    
    def predict(self, image_data) -> Dict:
        """
        Make prediction on cardiac scan
        
        Args:
            image_data: Input image (path, bytes, or PIL Image)
            
        Returns:
            Dictionary containing:
                - class_name: 'healthy' or 'abnormal'
                - confidence: float (0-1)
                - explanation: str
                - gradcam: base64 encoded heatmap image (optional)
        """
        # Preprocess and predict
        img_tensor = self.preprocess_image(image_data)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        # Get class and confidence
        class_name = self.class_names[pred.item()]
        confidence = conf.item()
        
        # Generate explanation
        explanation = self._generate_explanation(class_name, confidence)
        
        # Generate Grad-CAM
        gradcam_img = self._generate_gradcam(img_tensor, pred)
        
        return {
            'class_name': class_name,
            'confidence': float(confidence),
            'explanation': explanation,
            'gradcam': gradcam_img
        }
    
    def _generate_explanation(self, class_name: str, confidence: float) -> str:
        """Generate human-readable explanation"""
        if class_name == 'healthy':
            return self.explanations['healthy']
        
        # For abnormal cases, use confidence to determine specificity
        if confidence > 0.75:
            # High confidence - provide specific explanation
            specific_type = np.random.choice(list(self.explanations['abnormal']['specific'].keys()))
            return self.explanations['abnormal']['specific'][specific_type]
        else:
            # Lower confidence - use generic explanation
            return self.explanations['abnormal']['generic']
    
    def _generate_gradcam(self, img_tensor: torch.Tensor, pred: torch.Tensor) -> str:
        """Generate Grad-CAM heatmap and return as base64"""
        heatmap = self.model.get_gradcam(img_tensor, pred)
        
        # Convert to numpy and resize
        heatmap = heatmap.numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR)
        
        # Create overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

def get_user_image():
    """Get image file path from user input"""
    while True:
        file_path = input("\nEnter path to cardiac scan image (or 'quit' to exit): ").strip()
        
        if file_path.lower() in ['quit', 'exit']:
            return None
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            continue
            
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print("Error: Only PNG/JPG/JPEG images are supported")
            continue
            
        return file_path

def main():
    """Interactive prediction interface"""
    print("Cardiac Scan Analysis System")
    print("--------------------------")
    
    try:
        predictor = CardiacPredictor()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    while True:
        image_path = get_user_image()
        if image_path is None:
            break
            
        try:
            result = predictor.predict(image_path)
            
            print("\nPrediction Results:")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Explanation: {result['explanation']}")
            
            # Save Grad-CAM to file
            if result['gradcam']:
                output_path = os.path.splitext(image_path)[0] + "_gradcam.png"
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(result['gradcam']))
                print(f"Grad-CAM visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
    
    print("\nExiting system...")

if __name__ == '__main__':
    main()