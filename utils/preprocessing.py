import os
import numpy as np
import cv2
from pathlib import Path
from skimage import exposure
import tifffile
from dotenv import load_dotenv

def preprocess_image(image_path, save_dir):
    # Fixed directories for saving files
    clean_dir = Path(save_dir) / "npy_clean"
    vis_dir = Path(save_dir) / "png_visual"
    clean_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(image_path)  # shape: (H, W, C) or (H, W)
    
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save .npy file
    clean_img = cv2.medianBlur(img, 3)  # remove noise
    clean_img = clean_img.astype(np.float32) / 255.0  # normalize
    np.save(clean_dir / f"{Path(image_path).stem}.npy", clean_img)

    # Save .png file
    if img.ndim == 3 and img.shape[2] >= 3:
        vis_img = img[:, :, :3].copy()
    else:
        vis_img = cv2.merge([img] * 3)

    lab = cv2.cvtColor(vis_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    vis_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    vis_img = exposure.adjust_gamma(vis_img, gamma=0.9)

    cv2.imwrite(
        str(vis_dir / f"{Path(image_path).stem}.png"),
        cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    )

    print(f"Saved clean .npy to {clean_dir}")
    print(f"Saved enhanced .png to {vis_dir}")


if __name__ == "__main__":
    load_dotenv()
    directory_path = os.getenv("INPUT_DIREC")

    if not directory_path:
        raise ValueError("Please set the INPUT_DIREC environment variable to the directory path.")

    if not Path(directory_path).exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Process all .tif files in the directory and its subdirectories
    for tif_file in Path(directory_path).rglob("*.tif"):
        print(f"Processing file: {tif_file}")
        preprocess_image(tif_file, save_dir="processed_patches")
