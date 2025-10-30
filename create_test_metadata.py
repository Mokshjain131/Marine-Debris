#!/usr/bin/env python3
"""
Create metadata file for test_data.h5 with actual patch names

This script reconstructs the patch names by reading the dataset creation process
and saves them alongside the H5 file for use with generate_geolocation_map.py
"""

import json
import h5py
from pathlib import Path

def load_labels(txt_path):
    """Load labels from MARIDA JSON file"""
    with open(txt_path, "r") as f:
        text = f.read()
    
    # Ensure valid JSON
    text = text.strip().rstrip(",")
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"
    
    return json.loads(text)


def get_patch_names_from_splits(splits_file: str = "splits/test_X.txt") -> list:
    """
    Read patch names from splits file
    
    The splits files contain the actual patch names used during dataset creation
    """
    if not Path(splits_file).exists():
        print(f"[WARNING] Splits file not found: {splits_file}")
        return None
    
    with open(splits_file, 'r') as f:
        patch_names = [line.strip() for line in f if line.strip()]
    
    return patch_names


def create_metadata_file():
    """Create metadata JSON file with patch names"""
    
    print("="*70)
    print("Creating Metadata for test_data.h5")
    print("="*70)
    
    # Load test data to get size
    h5_path = 'test_data.h5'
    if not Path(h5_path).exists():
        print(f"[ERROR] File not found: {h5_path}")
        return
    
    with h5py.File(h5_path, 'r') as f:
        num_samples = f['images'].shape[0]
        print(f"\nTest dataset has {num_samples} samples")
    
    # Try to load from splits file
    patch_names = get_patch_names_from_splits("splits/test_X.txt")
    
    if patch_names is None:
        # Try to reconstruct from labels file
        print("\nTrying to reconstruct from labels.txt...")
        labels_path = "labels.txt"
        
        if Path(labels_path).exists():
            all_labels = load_labels(labels_path)
            all_patch_names = sorted(list(all_labels.keys()))
            
            # The test set is typically the last 20% or specified in splits
            # For now, we'll just use the sorted order and take last num_samples
            print(f"Found {len(all_patch_names)} total patches in labels.txt")
            print(f"Using last {num_samples} as test set (this is an approximation)")
            patch_names = all_patch_names[-num_samples:]
        else:
            print(f"[ERROR] Labels file not found: {labels_path}")
            print("\nGenerating placeholder names...")
            patch_names = [f"patch_{i:04d}" for i in range(num_samples)]
    
    if len(patch_names) != num_samples:
        print(f"\n[WARNING] Mismatch: {len(patch_names)} names vs {num_samples} samples")
        print("Adjusting to match...")
        if len(patch_names) > num_samples:
            patch_names = patch_names[:num_samples]
        else:
            # Pad with placeholders
            for i in range(len(patch_names), num_samples):
                patch_names.append(f"unknown_patch_{i}")
    
    # Create metadata
    metadata = {
        'patch_names': patch_names,
        'num_samples': num_samples,
        'description': 'Metadata for test_data.h5 containing patch geolocation identifiers'
    }
    
    # Save metadata
    metadata_path = 'test_data.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Metadata saved to: {metadata_path}")
    
    # Show sample
    print("\nSample patch names:")
    for i, name in enumerate(patch_names[:10]):
        print(f"  {i}: {name}")
    if len(patch_names) > 10:
        print(f"  ... and {len(patch_names)-10} more")
    
    # Extract unique geolocations
    geolocations = set()
    for name in patch_names:
        # Remove trailing _N (patch index)
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            geolocations.add(parts[0])
        else:
            geolocations.add(name)
    
    print(f"\nUnique geolocations: {len(geolocations)}")
    print("Sample geolocations:")
    for loc in sorted(geolocations)[:5]:
        print(f"  - {loc}")
    
    print("\n" + "="*70)
    print("Metadata creation complete!")
    print("="*70)
    print("\nYou can now run:")
    print("  python generate_geolocation_map.py --checkpoint checkpoints/resnet_best.pth")


if __name__ == '__main__':
    create_metadata_file()
