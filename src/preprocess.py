from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Paths and hyperparameters (adjust as needed)
IMG_DIR = Path("data/processed/classification/images")
LABEL_DIR = Path("data/processed/classification/labels")
MODEL_PATH = Path("models/classification_model.pth")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

# Custom Dataset
class FundusDataset(Dataset):
    def __init__(self, img_files, transform=None):
        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        
        image = Image.open(img_path).convert("RGB")
        label = int(open(label_path).read().strip())
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# # Data Loaders
# def get_dataloaders():
#     all_images = list(IMG_DIR.glob("*.jpg"))
#     valid_images = []
    
#     # Filter valid image-label pairs
#     for img_path in all_images:
#         label_path = LABEL_DIR / f"{img_path.stem}.txt"
#         if label_path.exists():
#             valid_images.append(img_path)
#         else:
#             print(f"Missing label for: {img_path.name}")

#     # Split dataset
#     train_files, test_val_files = train_test_split(
#         valid_images, test_size=0.3, stratify=[int(open(LABEL_DIR/f"{img.stem}.txt").read()) for img in valid_images]
#     )
#     val_files, test_files = train_test_split(test_val_files, test_size=0.5)

#     # Augmentations
#     train_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.1, 0.1),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # Create datasets
#     train_ds = FundusDataset(train_files, train_transform)
#     val_ds = FundusDataset(val_files, val_transform)
#     test_ds = FundusDataset(test_files, val_transform)

#     # Create dataloaders
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, pin_memory=True)

#     return train_loader, val_loader, test_loader

def get_dataloaders():
    all_images = list(IMG_DIR.glob("*.jpg"))
    valid_images = []
    
    # Filter valid images
    for img_path in all_images:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_images.append(img_path)

    # Check minimum dataset size
    if len(valid_images) < 4:
        raise ValueError(f"Need at least 4 samples, got {len(valid_images)}")

    # Split dataset
    train_files, test_val_files = train_test_split(
        valid_images, 
        test_size=0.3, 
        stratify=[int(open(LABEL_DIR/f"{img.stem}.txt").read()) for img in valid_images]
    )
    val_files, test_files = train_test_split(test_val_files, test_size=0.5)

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_ds = FundusDataset(train_files, train_transform)
    val_ds = FundusDataset(val_files, val_transform)
    test_ds = FundusDataset(test_files, val_transform)

    # Create dataloaders with safety checks
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=min(BATCH_SIZE*2, len(val_ds)),  # Prevent single batches
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    

    return train_loader, val_loader, None  # Skip test loader for training

# Training Loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders()

    # Model setup with corrected dimensions
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Correct classifier dimensions
    model.fc = nn.Sequential(
        nn.Linear(512, 256),  # ResNet18 has 512 input features
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 5)
    ).to(device)


    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    early_stop_patience = 5
    no_improve = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += images.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += images.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Metrics
        train_loss = train_loss / total
        train_acc = correct / total
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Scheduler
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr:.2e}")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

# Run training
if __name__ == "__main__":
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train()