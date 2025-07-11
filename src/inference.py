import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from pathlib import Path
import cv2
from unet import UNet

class DRSystem:
    def __init__(self):
        # Initialize device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Then load models
        self.class_model = self.load_classification_model()
        self.seg_model = self.load_segmentation_model()
        
        # Then other configurations
        self.img_size = 512
        self.class_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.seg_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
    def load_classification_model(self):
        # Create model architecture first
        model = models.resnet18(weights=None)  # Replace pretrained=False with weights=None
        model.fc = nn.Linear(model.fc.in_features, 5)
        
        # Then load weights
        model.load_state_dict(torch.load("../models/classification_model.pth", map_location=self.device))
        model.eval()
        return model.to(self.device)
   
    def load_segmentation_model(self):
        # Create UNet architecture first
        model = UNet()  
        model.load_state_dict(torch.load("../models/segmentation_model.pth", map_location=self.device))
        model.eval()
        return model.to(self.device)
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    def predict_classification(self, image):
        img_t = self.class_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.class_model(img_t)
        return torch.argmax(outputs).item()

    def predict_segmentation(self, image):
        img_t = self.seg_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.seg_model(img_t)
        return torch.sigmoid(outputs).cpu().numpy()[0]

    def analyze(self, image_path):
        image = self.preprocess_image(image_path)
        
        # Get predictions
        grade = self.predict_classification(image)
        seg_mask = self.predict_segmentation(image)
        
        # Convert mask to overlay
        overlay = self.create_overlay(np.array(image), seg_mask)
        
        return grade, overlay

    def create_overlay(self, image, seg_mask):
        # Convert image to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Create annotated overlay
        overlay = img.copy()
        
        # Define lesion colors and labels
        lesion_info = {
            0: {'color': (0, 0, 255), 'label': 'Microaneurysms'},     # Red
            1: {'color': (0, 255, 0), 'label': 'Haemorrhages'},       # Green
            2: {'color': (255, 0, 0), 'label': 'Hard Exudates'},      # Blue
            3: {'color': (0, 255, 255), 'label': 'Soft Exudates'},    # Yellow
            4: {'color': (255, 255, 0), 'label': 'Optic Disc'}        # Cyan
        }

        # Draw bounding circles and labels
        for channel in range(seg_mask.shape[0]):
            if channel >= len(lesion_info):
                continue
                
            mask = seg_mask[channel]
            contours, _ = cv2.findContours(
                (mask > 0.5).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            color = lesion_info[channel]['color']
            label = lesion_info[channel]['label']
            
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(overlay, center, radius, color, 2)
                
                # Add label
                cv2.putText(
                    overlay, label,
                    (center[0] + radius + 5, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1
                )

            # Add grade text
            cv2.putText(
                overlay, f"DR Grade: {grade}",
                (20, 40),  # Position
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2  # Green text
            )
            
            # Blend with original image
            return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

if __name__ == "__main__":
    system = DRSystem()
    grade, overlay = system.analyze("sample_image.jpg")
    print(f"DR Grade: {grade}")
    cv2.imwrite("output_overlay.jpg", overlay)