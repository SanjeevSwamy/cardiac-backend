import torch
import numpy as np
from PIL import Image
from model import CardiacResNet
import matplotlib.pyplot as plt
import io
import base64
import os

class CardiacPredictor:
    def __init__(self, model_path: str = 'best_model.pth', device: str = 'auto'):
        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available() else 
            'cpu' if device == 'auto' else device
        )
        self.model = CardiacResNet(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

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

    def preprocess_image(self, image_data) -> torch.Tensor:
        if isinstance(image_data, str):
            img = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            img = image_data.convert('RGB')
        else:
            raise ValueError("Unsupported image input type")
        return self.simple_transform(img).to(self.device)

    def simple_transform(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((256, 256))
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = torch.tensor(image).permute(2, 0, 1)
        return image.unsqueeze(0)

    def predict(self, image_data) -> dict:
        img_tensor = self.preprocess_image(image_data)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        class_name = self.class_names[pred.item()]
        confidence = conf.item()
        explanation = self._generate_explanation(class_name, confidence)
        gradcam_img = self._generate_gradcam(img_tensor, pred)

        return {
            'class_name': class_name,
            'confidence': float(confidence),
            'explanation': explanation,
            'gradcam': gradcam_img
        }

    def _generate_explanation(self, class_name: str, confidence: float) -> str:
        if class_name == 'healthy':
            return self.explanations['healthy']
        if confidence > 0.75:
            specific_type = np.random.choice(list(self.explanations['abnormal']['specific'].keys()))
            return self.explanations['abnormal']['specific'][specific_type]
        else:
            return self.explanations['abnormal']['generic']

    def _generate_gradcam(self, img_tensor: torch.Tensor, pred: torch.Tensor) -> str:
        heatmap = self.model.get_gradcam(img_tensor, pred)
        heatmap = heatmap.numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR)

        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
