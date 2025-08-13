import os
import numpy as np
import cv2
from pathlib import Path
from skimage import exposure
import tifffile
from dotenv import load_dotenv

def preprocess_image(image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    clean_dir = Path(save_dir) / "npy_clean"
    vis_dir = Path(save_dir) / "png_visual"
    clean_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # --- Load .tif image ---
    img = tifffile.imread(image_path)  # shape: (H, W, C) or (H, W)
    
    # Ensure uint8 for processing
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- CNN Version (.npy) ---
    clean_img = cv2.medianBlur(img, 3)  # remove noise
    clean_img = clean_img.astype(np.float32) / 255.0  # normalize
    np.save(clean_dir / f"{Path(image_path).stem}.npy", clean_img)

    # --- Visualization (.png) ---
    if img.ndim == 3 and img.shape[2] >= 3:
        # Pick first three bands for visualization
        vis_img = img[:, :, :3].copy()
    else:
        # If grayscale, just duplicate channels
        vis_img = cv2.merge([img] * 3)

    # Convert to LAB for contrast enhancement
    lab = cv2.cvtColor(vis_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    vis_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Slight gamma correction
    vis_img = exposure.adjust_gamma(vis_img, gamma=0.9)

    # Save PNG
    cv2.imwrite(
        str(vis_dir / f"{Path(image_path).stem}.png"),
        cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    )

    print(f"✅ Saved clean .npy to {clean_dir}")
    print(f"✅ Saved enhanced .png to {vis_dir}")


if __name__ == "__main__":
    load_dotenv()
    tif_path = os.getenv("INPUT_TIF")

    if not tif_path:
        raise ValueError("❌ Please set the INPUT_TIF environment variable to the .tif file path.")

    if not Path(tif_path).exists():
        raise FileNotFoundError(f"❌ File not found: {tif_path}")

    preprocess_image(tif_path, save_dir="processed_patches")
