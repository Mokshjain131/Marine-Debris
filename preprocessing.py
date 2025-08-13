import os
import numpy as np
from PIL import Image
import tifffile as tiff
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get TIF path from environment variable
INPUT_TIF = os.getenv("INPUT_TIF")

# Output folder for CNN .npy
CNN_TILES_DIR = "cnn_tiles"
os.makedirs(CNN_TILES_DIR, exist_ok=True)

# Output file paths
OUTPUT_NPY = os.path.join(CNN_TILES_DIR, "output.npy")
OUTPUT_PNG = "output.png"

# Load the TIF image
image_array = tiff.imread(INPUT_TIF)
print(f"Loaded image shape: {image_array.shape}, dtype: {image_array.dtype}")

# Save as .npy for CNN training
np.save(OUTPUT_NPY, image_array)
print(f"Saved NumPy array as {OUTPUT_NPY}")

# ----- Create PNG preview -----
# If image has more than 3 bands, pick RGB bands (example: Sentinel-2 bands 4, 3, 2)
if image_array.ndim == 3 and image_array.shape[2] >= 3:
    rgb_array = image_array[:, :, [3, 2, 1]]  # Adjust indices if needed
else:
    rgb_array = image_array  # Single-band grayscale

# Normalize to 0-255 for visualization
image_min = np.min(rgb_array)
image_max = np.max(rgb_array)
if image_max > image_min:
    image_norm = ((rgb_array - image_min) / (image_max - image_min) * 255).astype(np.uint8)
else:
    image_norm = np.zeros_like(rgb_array, dtype=np.uint8)

# Save PNG
Image.fromarray(image_norm).save(OUTPUT_PNG)
print(f"Saved PNG preview as {OUTPUT_PNG}")
