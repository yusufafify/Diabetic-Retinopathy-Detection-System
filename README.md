# 🩺 Diabetic Retinopathy Detection System

A comprehensive AI-powered system for automated detection and analysis of diabetic retinopathy in fundus images, combining deep learning models for classification and segmentation with an intuitive web interface.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)

## 🎯 Overview

Diabetic Retinopathy (DR) is a leading cause of blindness worldwide. This project provides an automated solution for early detection and grading of diabetic retinopathy using state-of-the-art computer vision techniques. The system performs both classification (grading severity) and segmentation (identifying lesions) on retinal fundus images.

### Key Capabilities:
- **Multi-class Classification**: Grades DR severity from 0 (No DR) to 4 (Proliferative DR)
- **Semantic Segmentation**: Identifies and localizes retinal lesions including:
  - Microaneurysms
  - Hemorrhages  
  - Hard Exudates
  - Cotton Wool Spots
  - Neovascularization
- **Clinical Recommendations**: Provides medical recommendations based on detected grade
- **Multi-Interface Support**: Web UI, Desktop GUI, and REST API

## ✨ Features

- 🔍 **Automated DR Detection**: AI-powered analysis of retinal images
- 📊 **Severity Grading**: 5-level classification system (0-4)
- 🎯 **Lesion Segmentation**: Precise identification of pathological features
- 💡 **Clinical Insights**: Evidence-based medical recommendations
- 🌐 **Modern Web Interface**: React-based responsive UI
- 🖥️ **Desktop Application**: PyQt5-based standalone GUI
- 📱 **REST API**: Integration-ready endpoints
- 📈 **Confidence Scoring**: Reliability metrics for predictions

## 🛠️ Technology Stack

### Machine Learning & Data Science
- **PyTorch**: Deep learning framework for model development
- **torchvision**: Computer vision utilities and pre-trained models
- **scikit-learn**: Machine learning utilities and metrics
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization and plotting
- **imgaug**: Image augmentation for data preprocessing
- **PIL (Pillow)**: Image loading and basic processing

### Backend & API
- **Flask**: Lightweight web framework for REST API
- **Flask-CORS**: Cross-origin resource sharing support

### Frontend & UI
- **React 19**: Modern frontend framework
- **TypeScript**: Type-safe JavaScript development
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **Lucide React**: Beautiful icon library
- **Axios**: HTTP client for API communication
- **React Router**: Client-side routing


### Development Tools
- **tqdm**: Progress bars for long-running operations
- **ESLint**: JavaScript/TypeScript linting
- **Git**: Version control with comprehensive .gitignore

## 📁 Project Structure

```
├── src/                          # Core application code
│   ├── app.py                    # Flask web server and API
│   ├── inference.py              # Main inference pipeline
│   ├── knowledge_base.py         # Medical knowledge and recommendations
│   ├── unet.py                   # UNet segmentation model architecture
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── train_classification.py   # Classification model training
│   ├── train_segmentation.py     # Segmentation model training
│   ├── data/                     # Dataset storage
│   │   ├── processed/            # Preprocessed data
│   │   │   ├── classification/   # Classification datasets
│   │   │   └── segmentation/     # Segmentation datasets
│   │   └── raw/                  # Original datasets
│   ├── models/                   # Trained model files
│   │   ├── classification_model.pth
│   │   └── segmentation_model.pth
│   └── uploads/                  # User-uploaded images
├── ui/                           # React frontend application
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # Application pages
│   │   ├── actions/              # API communication logic
│   │   └── assets/               # Static assets
│   ├── package.json              # Frontend dependencies
│   └── vite.config.ts            # Vite configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd diabetic-retinopathy-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare model files**
   - Ensure trained models are in `src/models/`:
     - `classification_model.pth`
     - `segmentation_model.pth`

### Frontend Setup

1. **Navigate to UI directory**
   ```bash
   cd ui
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

## 🎮 Usage

### Web Application

1. **Start the backend server**
   ```bash
   cd src
   python app.py
   ```
   Server runs on `http://localhost:5000`

2. **Start the frontend development server**
   ```bash
   cd ui
   npm run dev
   ```
   Frontend runs on `http://localhost:5173`

3. **Access the application**
   - Open your browser to `http://localhost:5173`
   - Upload a retinal fundus image
   - View classification results and segmentation overlay
   - Review medical recommendations



## 🧠 Model Architecture

### Classification Model
- **Base Architecture**: ResNet-18
- **Input Size**: 224×224×3
- **Output Classes**: 5 (DR grades 0-4)
- **Training**: Transfer learning with fine-tuning
- **Data Augmentation**: Rotation, flipping, brightness adjustment

### Segmentation Model  
- **Architecture**: U-Net
- **Input Size**: 512×512×3
- **Output Classes**: 5 lesion types
- **Features**: 
  - Encoder-decoder structure
  - Skip connections for feature preservation
  - Multi-class semantic segmentation

### Training Process
1. **Data Preprocessing**: Image normalization, resizing, augmentation
2. **Classification Training**: Multi-class cross-entropy loss
3. **Segmentation Training**: Pixel-wise cross-entropy loss
4. **Validation**: Held-out test set evaluation
5. **Model Selection**: Best performing checkpoint selection

## 📊 Performance Metrics

- **Classification Accuracy**: ~85% on test set
- **Segmentation IoU**: ~0.72 average across lesion classes
- **Inference Time**: ~2-3 seconds per image
- **Model Size**: 
  - Classification: ~45MB
  - Segmentation: ~95MB

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Development/production mode
- `MODEL_PATH`: Custom model directory path
- `UPLOAD_FOLDER`: Image upload directory

### Model Hyperparameters
- **Learning Rate**: 1e-4
- **Batch Size**: 16 (classification), 4 (segmentation)
- **Epochs**: 10-50 depending on model
- **Optimizer**: Adam with default parameters