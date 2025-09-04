import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from dotenv import load_dotenv
from pathlib import Path
import os
import json

# Custom Dataset class
class MarineDebrisDataset(Dataset):
    def __init__(self, npy_dir, labels_dict, transform=None):
        self.npy_dir = Path(npy_dir)
        self.labels_dict = labels_dict
        self.files = list(labels_dict.keys())   # e.g., "S2_1-12-19_48MYU_0"
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx]  # already normalized stem

        # build npy path
        npy_path = self.npy_dir / f"{stem}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing file: {npy_path}")

        # Load image
        image = np.load(npy_path)  # (H, W, C)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Label
        label = torch.tensor(self.labels_dict[stem], dtype=torch.float32)

        return image, label

def load_labels(txt_path):
    with open(txt_path, "r") as f:
        text = f.read()

    # Ensure it's valid JSON (the MARIDA file sometimes ends with a trailing comma)
    text = text.strip().rstrip(",")  
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"

    data = json.loads(text)  # parse as dictionary

    labels_dict = {}
    for filename, labels in data.items():
        stem = Path(filename).stem  # remove ".tif"
        labels_dict[stem] = labels

    return labels_dict

def create_datasets(npy_dir, csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    labels_dict = load_labels(csv_path)

    available_stems = {
        Path(f).stem for f in os.listdir(npy_dir)
        if f.endswith(".npy") and not ("_cl" in f or "_conf" in f)
    }
    labels_dict = {stem: lbl for stem, lbl in labels_dict.items() if stem in available_stems}

    dataset = MarineDebrisDataset(npy_dir, labels_dict)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size   = int(val_ratio * total_size)
    test_size  = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Total usable samples: {total_size}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    npy_dir = "processed_patches/npy_clean"
    load_dotenv()
    csv_path = os.getenv("LABELS_CSV")
    print("Files in NPY_DIR:", os.listdir(npy_dir)[:10])  # show first 10
    print("CSV path:", csv_path)

    train_dataset, val_dataset, test_dataset = create_datasets(npy_dir, csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    for imgs, labels in train_loader:
        print("Batch imgs:", imgs.shape) # (N, C, H, W)
        print("Batch labels:", labels.shape) # (N, num_classes)
        break