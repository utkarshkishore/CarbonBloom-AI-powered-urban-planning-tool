import cv2
import numpy as np
import os
from glob import glob

# Config
INPUT_DIR = "raw_images"
OUTPUT_IMG_DIR = "dataset/images"
OUTPUT_MASK_DIR = "dataset/masks"
IMG_SIZE = 512  # Resize to 512x512 for your 3050 GPU

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def create_mask(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define "Green" range (Trees/Grass)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask: 1 = Green, 0 = Concrete
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert it: We want to detect CONCRETE (White=1), not Green
    mask_inv = cv2.bitwise_not(mask)
    
    return mask_inv

# --- FIXED LINE BELOW ---
# Now looks for .jpg, .png, AND .jpeg
images = glob(f"{INPUT_DIR}/*.jpg") + glob(f"{INPUT_DIR}/*.png") + glob(f"{INPUT_DIR}/*.jpeg")
print(f"Found {len(images)} images. Processing...")

for i, img_path in enumerate(images):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping broken file: {img_path}")
        continue
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Generate the "Ground Truth"
    mask = create_mask(img)
    
    # Save processed pairs
    cv2.imwrite(f"{OUTPUT_IMG_DIR}/image_{i}.jpg", img)
    cv2.imwrite(f"{OUTPUT_MASK_DIR}/image_{i}.png", mask)

print("✅ Data generation complete. Ready to train.")