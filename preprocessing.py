import os
import numpy as np
import tifffile as tiff
from dotenv import load_dotenv
from PIL import Image
from scipy.ndimage import median_filter

# Load .env
load_dotenv()
INPUT_TIF = os.getenv("INPUT_TIF")

# Read image
image_array = tiff.imread(INPUT_TIF).astype(np.float32)

# Keep only first 10 bands (adjust if needed)
image_array = image_array[:, :, :10]

# Clip extreme values to reduce sensor noise
p_low, p_high = 1, 99  # percentiles
for b in range(image_array.shape[2]):
    low_val = np.percentile(image_array[:, :, b], p_low)
    high_val = np.percentile(image_array[:, :, b], p_high)
    image_array[:, :, b] = np.clip(image_array[:, :, b], low_val, high_val)

# Normalize each band to 0–1
for b in range(image_array.shape[2]):
    band = image_array[:, :, b]
    band_min, band_max = band.min(), band.max()
    image_array[:, :, b] = (band - band_min) / (band_max - band_min)

# Optional: Apply median filter (remove speckle noise)
image_array = median_filter(image_array, size=(3, 3, 1))

# Save .npy in cnn_tiles
os.makedirs("cnn_tiles", exist_ok=True)
np.save("cnn_tiles/image.npy", image_array)

# Save PNG preview (only RGB bands)
rgb = (image_array[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(rgb).save("noise.png")

print("Preprocessing complete: clipped, normalized, and noise-filtered.")
