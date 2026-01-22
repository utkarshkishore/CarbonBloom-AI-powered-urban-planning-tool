import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import os
import segmentation_models_pytorch as smp
import numpy as np
import albumentations as A

# --- CONFIG ---
DEVICE = 'cuda'
EPOCHS = 5           
BATCH_SIZE = 4       
LR = 0.0001
MODEL_FILE = "carbon_bloom_model.pth" # File to resume from

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3), 
])

class DeepGlobeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = np.transpose(img, (2, 0, 1)) / 255.0
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def train():
    print(f"🚀 Resuming Training on: {torch.cuda.get_device_name(0)}")
    
    dataset = DeepGlobeDataset("dataset/images", "dataset/masks", transform=transform)
    
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"🔥 Dataset Size: {len(dataset)} tiles.")

    # 1. Initialize the architecture
    model = smp.Unet(
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to(DEVICE)

    # 2. CHECK FOR EXISTING MODEL TO RESUME
    if os.path.exists(MODEL_FILE):
        print(f"✅ Found saved model: {MODEL_FILE}. Loading weights...")
        model.load_state_dict(torch.load(MODEL_FILE))
        print("🧠 Weights loaded! Starting with a smart brain.")
    else:
        print("⚠️ No saved model found. Starting from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    scaler = torch.cuda.amp.GradScaler() 

    # Set best_score to current validation loss (approx) so we only save if we get BETTER
    best_score = 0.3 # We know previous was ~0.29, so this is a safe start

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        total_loss = 0
        
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(imgs)
                loss = loss_fn(pred, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        print("Validating...")
        val_loss = 0
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs, v_masks = v_imgs.to(DEVICE), v_masks.to(DEVICE)
                v_pred = model(v_imgs)
                val_loss += loss_fn(v_pred, v_masks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"📉 Validation Loss: {avg_val_loss:.4f}")
        
        # Save ALWAYS if it's the last epoch, or if it improves
        if avg_val_loss < best_score:
            best_score = avg_val_loss
            torch.save(model.state_dict(), MODEL_FILE)
            print("✅ Model Saved (Improved)")
        elif epoch == EPOCHS - 1:
            torch.save(model.state_dict(), MODEL_FILE)
            print("✅ Final Model Saved")

if __name__ == "__main__":
    train()