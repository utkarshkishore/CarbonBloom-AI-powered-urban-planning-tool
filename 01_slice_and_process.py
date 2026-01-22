import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = "deepglobe_raw"   
OUTPUT_IMG_DIR = "dataset/images"
OUTPUT_MASK_DIR = "dataset/masks"
TILE_SIZE = 512
STRIDE = 512  

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def process_mask(mask_tile):
    # Create a blank black mask
    binary_mask = np.zeros((mask_tile.shape[0], mask_tile.shape[1]), dtype=np.uint8)
    
    # DeepGlobe "Urban" is Cyan (Blue-Green)
    # OpenCV reads as BGR, so Cyan is (255, 255, 0)
    # We define a range to catch it
    lower_cyan = np.array([200, 200, 0])
    upper_cyan = np.array([255, 255, 50])
    
    mask = cv2.inRange(mask_tile, lower_cyan, upper_cyan)
    
    # Set those pixels to 255 (White = Concrete)
    binary_mask[mask > 0] = 255
    return binary_mask

def slice_data():
    img_paths = glob(f"{INPUT_DIR}/*_sat.jpg")
    print(f"Found {len(img_paths)} large maps. Slicing into tiles...")

    count = 0
    for img_path in tqdm(img_paths):
        # Match the satellite image to its mask
        mask_path = img_path.replace("_sat.jpg", "_mask.png")
        
        if not os.path.exists(mask_path):
            continue 

        large_img = cv2.imread(img_path)
        large_mask = cv2.imread(mask_path)
        
        if large_img is None or large_mask is None:
            continue

        h, w, _ = large_img.shape
        for y in range(0, h - TILE_SIZE + 1, STRIDE):
            for x in range(0, w - TILE_SIZE + 1, STRIDE):
                img_tile = large_img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                mask_tile = large_mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

                binary_mask = process_mask(mask_tile)

                # Only save if the tile isn't empty 
                if img_tile.shape[0] == TILE_SIZE:
                    cv2.imwrite(f"{OUTPUT_IMG_DIR}/tile_{count}.jpg", img_tile)
                    cv2.imwrite(f"{OUTPUT_MASK_DIR}/tile_{count}.png", binary_mask)
                    count += 1

    print(f"✅ Created {count} training tiles! Ready for Deep Learning.")

if __name__ == "__main__":
    slice_data()