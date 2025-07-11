# src/train_classification.py

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
IMG_DIR = Path("data/processed/classification/images")
LABEL_DIR = Path("data/processed/classification/labels")
MODEL_PATH = Path("models/classification_model.pth")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# 1. Custom Dataset
class FundusDataset(Dataset):
    def __init__(self, img_files, transform=None):
        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = LABEL_DIR / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        label = int(open(label_path).read().strip())

        if self.transform:
            image = self.transform(image)

        return image, label

# 2. Data Loaders
def get_dataloaders():
    all_images = list(IMG_DIR.glob("*.jpg"))
    valid_images = []
    for img_path in all_images:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_images.append(img_path)
        else:
            print(f"Missing label for: {img_path.name}")
    
    # Split only valid pairs
    train_files, val_files = train_test_split(valid_images, test_size=0.2, random_state=42)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = FundusDataset(train_files, transform)
    val_ds = FundusDataset(val_files, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader

# 3. Training Loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders()

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 grades
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Train Acc: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    train()
