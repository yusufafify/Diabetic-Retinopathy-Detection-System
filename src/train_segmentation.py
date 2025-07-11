import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
NUM_CLASSES = 5  # Number of lesion types
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_files = sorted(img_dir.glob("*.jpg"))
        self.mask_files = sorted(mask_dir.glob("*.npy"))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]) if transform is None else transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(img_path).convert("RGB")
        mask = np.load(mask_path).astype(np.float32) / 255.0  # Convert to [0,1]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(mask).permute(2, 0, 1)  # Channels first

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)    # 512->256
        self.enc2 = self.conv_block(64, 128)  # 256->128
        self.enc3 = self.conv_block(128, 256) # 128->64
        self.enc4 = self.conv_block(256, 512) # 64->32
        
        # Decoder
        self.dec1 = self.upconv_block(512, 256)  # 32->64
        self.dec2 = self.upconv_block(256, 128)  # 64->128
        self.dec3 = self.upconv_block(128, 64)   # 128->256
        self.dec4 = self.upconv_block(64, 32)    # 256->512
        
        # Final convolution
        self.final = nn.Conv2d(32, NUM_CLASSES, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 512->256
        e2 = self.enc2(e1) # 256->128
        e3 = self.enc3(e2) # 128->64
        e4 = self.enc4(e3) # 64->32
        
        # Decoder with skip connections
        d1 = self.dec1(e4) + e3  # 32->64
        d2 = self.dec2(d1) + e2  # 64->128
        d3 = self.dec3(d2) + e1  # 128->256
        d4 = self.dec4(d3)       # 256->512
        
        return self.final(d4)  # 512×512 output
def train_segmentation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset and Loader
    img_dir = PROCESSED_DIR / "segmentation/images"
    mask_dir = PROCESSED_DIR / "segmentation/masks"
    
    dataset = SegmentationDataset(img_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model and Optimizer
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/segmentation_model.pth")
    print("Segmentation model saved!")

if __name__ == "__main__":
    train_segmentation()