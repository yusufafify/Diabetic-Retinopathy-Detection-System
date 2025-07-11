from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from knowledge_base import KnowledgeBase
from datetime import datetime


# Load the pre-trained models
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.dec1 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec3 = self.upconv_block(128, 64)
        self.dec4 = self.upconv_block(64, 32)
        self.final = torch.nn.Conv2d(32, 5, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = self.dec1(e4) + e3
        d2 = self.dec2(d1) + e2
        d3 = self.dec3(d2) + e1
        d4 = self.dec4(d3)
        return self.final(d4)


# Configuration
SEG_MODEL_PATH = Path("models/segmentation_model.pth")
CLS_MODEL_PATH = Path("models/classification_model.pth")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models
seg_model = UNet()
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location='cpu'))
seg_model.eval()

cls_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
cls_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(cls_model.fc.in_features, 5)
)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location='cpu'))
cls_model.eval()

# Preprocessing functions
def preprocess_seg(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def preprocess_cls(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# AI model inference functions
def predict_classification(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
    
    pred_class = torch.argmax(probs).item()
    confidence = probs[0][pred_class].item()
    
    return pred_class, confidence

def predict_segmentation(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    with torch.no_grad():
        pred_mask = model(input_tensor).squeeze().numpy()
    
    class_mask = np.argmax(pred_mask, axis=0)
    
    return class_mask

# Initialize Knowledge Base
knowledge_base = KnowledgeBase()

# Define the grade_info dictionary (you had this missing)
grade_info = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Non-Proliferative DR',
    2: 'Moderate Non-Proliferative DR',
    3: 'Severe Non-Proliferative DR',
    4: 'Proliferative DR'
}

# Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    seg_model.eval()  # Ensure the model is in evaluation mode
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    try:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run AI models
        cls_pred, cls_confidence = predict_classification(cls_model, filepath)
        seg_mask = predict_segmentation(seg_model, filepath)



        # For segmentation, we can use the class index (assuming the model has 5 classes)
        seg_class = np.unique(seg_mask)[0]  # Taking the first class predicted in the mask

        # Generate recommendations
        recommended_action = knowledge_base.get_recommendation(cls_pred)
        features = knowledge_base.get_features(seg_class)

        # Convert segmentation mask from bool to int and then to a list for JSON serialization
        
        # Prepare the JSON response
        result = {
            'grade': int(cls_pred),
            'grade_label': grade_info.get(cls_pred, 'Unknown Grade'),
            'classification_confidence': float(cls_confidence),
            'segmentation_class': int(seg_class),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommendedAction': recommended_action,
            'features': features
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error during processing: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=5000)
