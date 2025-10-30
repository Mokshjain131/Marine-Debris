import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3
"""
Generate Geolocation Hashmap from Marine Debris Predictions

This script generates a hashmap with actual geolocations as keys:
  - Key: Patch identifier (e.g., "S2_1-12-19_48MYU" - includes date and UTM tile)
  - Value: True if debris detected, False if no debris
  
Also extracts coordinates from patch folders for map visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
from datetime import datetime
import sys
import re

# Import model
sys.path.append(str(Path(__file__).parent / 'models'))
from models.resnet18 import ResNetSentinel


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load ResNet18 model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    model = ResNetSentinel(num_bands=11, num_classes=15, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("[OK] Model loaded successfully")
    return model


def load_test_patches(splits_file: str = 'MARIDA_models/data/splits/test_X.txt') -> List[str]:
    """Load test patch names from splits file"""
    print(f"Loading test patch names from: {splits_file}")
    
    with open(splits_file, 'r') as f:
        patches = [line.strip() for line in f if line.strip()]
    
    print(f"[OK] Loaded {len(patches)} test patch names")
    return patches


def load_test_data(h5_path: str = 'test_data.h5') -> Tuple[np.ndarray, np.ndarray]:
    """Load test images and labels from H5 file"""
    print(f"Loading test data from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
    
    print(f"[OK] Loaded {len(images)} test samples")
    return images, labels


def extract_geolocation_info(patch_name: str) -> Dict:
    """
    Extract geolocation information from patch name
    
    Example: "S2_1-12-19_48MYU_0" or "12-12-20_16PCC_0"
    Returns: {
        'patch_id': 'S2_1-12-19_48MYU',
        'date': '1-12-19',
        'utm_tile': '48MYU',
        'tile_number': 0
    }
    """
    # Pattern: optional S2_, date (D-D-D or DD-DD-DD), utm_tile, tile_number
    match = re.match(r'(?:S2_)?(\d+-\d+-\d+)_([A-Z0-9]+)(?:_(\d+))?', patch_name)
    
    if match:
        date = match.group(1)
        utm_tile = match.group(2)
        tile_num = int(match.group(3)) if match.group(3) else 0
        
        # Create base patch identifier (without tile number for unique locations)
        patch_id = f"{date}_{utm_tile}"
        
        return {
            'patch_id': patch_id,
            'full_name': patch_name,
            'date': date,
            'utm_tile': utm_tile,
            'tile_number': tile_num
        }
    
    # Fallback: use the patch name as-is
    return {
        'patch_id': patch_name,
        'full_name': patch_name,
        'date': None,
        'utm_tile': None,
        'tile_number': 0
    }


def get_utm_tile_center_coords(utm_tile: str) -> Optional[Tuple[float, float]]:
    """
    Get approximate center coordinates for a UTM tile
    
    UTM tile format: <zone><latitude_band><square>
    Example: 48MYU = Zone 48, Band M, Square YU
    
    Returns (latitude, longitude) or None if cannot parse
    """
    if not utm_tile or len(utm_tile) < 3:
        return None
    
    try:
        # Extract zone number (first 1-2 digits)
        zone_match = re.match(r'(\d+)', utm_tile)
        if not zone_match:
            return None
        
        zone = int(zone_match.group(1))
        
        # Extract latitude band (letter after zone)
        band_match = re.search(r'\d+([A-Z])', utm_tile)
        if not band_match:
            return None
        
        band = band_match.group(1)
        
        # Approximate latitude from band (C=80S to X=84N)
        # Bands are roughly 8 degrees each, starting from -80 at C
        band_letters = 'CDEFGHJKLMNPQRSTUVWX'
        if band in band_letters:
            band_idx = band_letters.index(band)
            lat = -80 + (band_idx * 8) + 4  # Center of band
        else:
            lat = 0
        
        # Approximate longitude from zone (each zone is 6 degrees, starting at -180)
        lon = -180 + (zone - 1) * 6 + 3  # Center of zone
        
        return (lat, lon)
    
    except Exception:
        return None


def predict_debris(model: nn.Module, images: np.ndarray, threshold: float, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model predictions on images
    
    Returns:
        predictions: Binary predictions (True/False) for each sample
        confidences: Confidence scores (0-1) for each sample
    """
    print(f"Running predictions with threshold={threshold}...")
    
    images_tensor = torch.tensor(images, dtype=torch.float32).to(device)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 16
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i+batch_size]
            outputs = model(batch)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Marine Debris is class 0
            debris_probs = probs[:, 0].cpu().numpy()
            
            # Apply threshold
            batch_predictions = debris_probs >= threshold
            
            predictions.extend(batch_predictions)
            confidences.extend(debris_probs)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    debris_count = np.sum(predictions)
    print(f"[OK] Predictions complete: {debris_count}/{len(predictions)} samples have debris")
    
    return predictions, confidences


def create_geolocation_map(
    predictions: np.ndarray,
    confidences: np.ndarray,
    patch_names: List[str],
    format: str = 'simple'
) -> Dict:
    """
    Create geolocation hashmap from predictions
    
    Args:
        predictions: Binary predictions array
        confidences: Confidence scores array
        patch_names: List of patch names
        format: 'simple' (True/False) or 'detailed' (with confidence, coords, etc.)
    
    Returns:
        Dictionary mapping geolocations to debris presence
    """
    print("\nGenerating geolocation map...")
    
    # Group patches by location
    location_groups = {}
    
    # Match predictions to patch names (first N patches)
    num_predictions = len(predictions)
    
    for i in range(num_predictions):
        if i < len(patch_names):
            patch_name = patch_names[i]
        else:
            # Fallback if we run out of patch names
            patch_name = f"patch_{i:04d}"
        
        geo_info = extract_geolocation_info(patch_name)
        location_key = geo_info['patch_id']
        
        has_debris = bool(predictions[i])
        confidence = float(confidences[i])
        
        # Group by location
        if location_key not in location_groups:
            location_groups[location_key] = {
                'patches': [],
                'geo_info': geo_info
            }
        
        location_groups[location_key]['patches'].append({
            'index': i,
            'has_debris': has_debris,
            'confidence': confidence,
            'full_name': geo_info['full_name']
        })
    
    # Aggregate by location (if ANY patch has debris, location has debris)
    geolocation_map = {}
    metadata = {
        'total_patches': len(predictions),
        'patches_with_debris': int(np.sum(predictions)),
        'total_locations': len(location_groups),
        'patches': []
    }
    
    locations_with_debris = 0
    
    for location_key, location_data in location_groups.items():
        patches = location_data['patches']
        geo_info = location_data['geo_info']
        
        # Aggregate: location has debris if ANY patch has debris
        has_debris = any(p['has_debris'] for p in patches)
        # Use max confidence
        max_confidence = max(p['confidence'] for p in patches)
        # Count debris patches
        debris_count = sum(1 for p in patches if p['has_debris'])
        
        if has_debris:
            locations_with_debris += 1
        
        if format == 'simple':
            # Simple format: just True/False
            geolocation_map[location_key] = has_debris
        else:
            # Detailed format: include coordinates and metadata
            coords = get_utm_tile_center_coords(geo_info.get('utm_tile'))
            
            geolocation_map[location_key] = {
                'has_debris': has_debris,
                'confidence': round(max_confidence, 4),
                'num_patches': len(patches),
                'debris_patches': debris_count,
                'date': geo_info['date'],
                'utm_tile': geo_info['utm_tile'],
                'coordinates': {
                    'lat': round(coords[0], 4) if coords else None,
                    'lon': round(coords[1], 4) if coords else None
                } if coords else None
            }
        
        # Store for metadata
        for p in patches:
            metadata['patches'].append({
                'index': p['index'],
                'location': location_key,
                'has_debris': p['has_debris'],
                'confidence': round(p['confidence'], 4)
            })
    
    metadata['locations_with_debris'] = locations_with_debris
    
    return geolocation_map, metadata


def save_geolocation_map(geolocation_map: Dict, metadata: Dict, output_path: str):
    """Save geolocation map to JSON file"""
    print(f"\nSaving geolocation map to: {output_path}")
    
    output = {
        'geolocation_map': geolocation_map,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_locations': len(geolocation_map),
            'locations_with_debris': metadata.get('locations_with_debris', 0),
            'locations_without_debris': len(geolocation_map) - metadata.get('locations_with_debris', 0),
            'debris_percentage': round(100 * metadata.get('locations_with_debris', 0) / len(geolocation_map), 2) if len(geolocation_map) > 0 else 0,
            'total_patches': metadata['total_patches'],
            'patches_with_debris': metadata['patches_with_debris']
        },
        'samples': metadata['patches'][:20]  # Save first 20 as examples
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"[OK] Saved {len(geolocation_map)} locations")
    print(f"     - {metadata.get('locations_with_debris', 0)} locations with debris")
    print(f"     - {len(geolocation_map) - metadata.get('locations_with_debris', 0)} locations without debris")


def save_per_patch_map(
    predictions: np.ndarray,
    confidences: np.ndarray,
    patch_names: List[str],
    output_path: str
):
    """Save individual patch mapping to JSON file"""
    print(f"\nSaving per-patch geolocation map to: {output_path}")
    
    patch_map = {}
    
    num_predictions = len(predictions)
    debris_count = 0
    
    # Track patches per location for spatial distribution
    location_patch_counts = {}
    
    for i in range(num_predictions):
        if i < len(patch_names):
            patch_name = patch_names[i]
        else:
            patch_name = f"patch_{i:04d}"
        
        geo_info = extract_geolocation_info(patch_name)
        has_debris = bool(predictions[i])
        confidence = float(confidences[i])
        coords = get_utm_tile_center_coords(geo_info.get('utm_tile'))
        
        if has_debris:
            debris_count += 1
        
        # Add spatial offset based on tile number to spread patches out
        # This creates a grid pattern around the UTM center
        if coords:
            location_key = f"{geo_info['date']}_{geo_info['utm_tile']}"
            if location_key not in location_patch_counts:
                location_patch_counts[location_key] = 0
            
            patch_count = location_patch_counts[location_key]
            location_patch_counts[location_key] += 1
            
            # Create a grid offset (0.01 degrees ≈ 1.1 km)
            # Arrange patches in a circular pattern or grid
            tile_num = geo_info['tile_number']
            
            # Use tile number for deterministic offset
            # Create 6x6 grid (36 positions) repeating if more patches
            grid_size = 6
            grid_x = (tile_num % grid_size) - (grid_size / 2)
            grid_y = (tile_num // grid_size) - (grid_size / 2)
            
            # Offset by 0.01 degrees per grid position
            lat_offset = grid_y * 0.015
            lon_offset = grid_x * 0.015
            
            adjusted_lat = coords[0] + lat_offset
            adjusted_lon = coords[1] + lon_offset
        else:
            adjusted_lat = None
            adjusted_lon = None
        
        # Create entry for this specific patch
        patch_map[patch_name] = {
            'has_debris': has_debris,
            'confidence': round(confidence, 4),
            'patch_index': i,
            'location_id': geo_info['patch_id'],
            'date': geo_info['date'],
            'utm_tile': geo_info['utm_tile'],
            'tile_number': geo_info['tile_number'],
            'coordinates': {
                'lat': round(adjusted_lat, 4) if adjusted_lat else None,
                'lon': round(adjusted_lon, 4) if adjusted_lon else None
            } if adjusted_lat else None
        }
    
    output = {
        'patch_map': patch_map,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_patches': num_predictions,
            'patches_with_debris': debris_count,
            'patches_without_debris': num_predictions - debris_count,
            'debris_percentage': round(100 * debris_count / num_predictions, 2)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"[OK] Saved {len(patch_map)} individual patches")
    print(f"     - {debris_count} patches with debris")
    print(f"     - {num_predictions - debris_count} patches without debris")


def print_summary(geolocation_map: Dict, metadata: Dict):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("GEOLOCATION MAP SUMMARY")
    print("="*70)
    
    total = len(geolocation_map)
    with_debris = metadata.get('locations_with_debris', 0)
    without_debris = total - with_debris
    
    print(f"\nTotal locations: {total}")
    print(f"  Locations with debris: {with_debris} ({100*with_debris/total:.1f}%)")
    print(f"  Locations without debris: {without_debris} ({100*without_debris/total:.1f}%)")
    print(f"\nTotal patches analyzed: {metadata['total_patches']}")
    print(f"  Patches with debris: {metadata['patches_with_debris']} ({100*metadata['patches_with_debris']/metadata['total_patches']:.1f}%)")
    
    # Show samples sorted by confidence
    debris_samples = [(p['location'], p['confidence']) 
                     for p in metadata['patches'] if p['has_debris']]
    no_debris_samples = [(p['location'], p['confidence']) 
                        for p in metadata['patches'] if not p['has_debris']]
    
    if debris_samples:
        print(f"\nSample patches with debris (sorted by confidence):")
        for loc, conf in sorted(debris_samples, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  ✓ {loc}: True (confidence: {conf:.3f})")
        if len(debris_samples) > 5:
            print(f"  ... and {len(debris_samples)-5} more")
    
    if no_debris_samples:
        print(f"\nSample patches without debris (lowest confidence):")
        for loc, conf in sorted(no_debris_samples, key=lambda x: x[1])[:5]:
            print(f"  ✗ {loc}: False (confidence: {conf:.3f})")
        if len(no_debris_samples) > 5:
            print(f"  ... and {len(no_debris_samples)-5} more")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Generate Marine Debris Geolocation Map')
    parser.add_argument('--checkpoint', default='checkpoints/resnet_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', default='test_data.h5',
                       help='Path to test data H5 file')
    parser.add_argument('--splits_file', default='MARIDA_models/data/splits/test_X.txt',
                       help='Path to test splits file with patch names')
    parser.add_argument('--threshold', type=float, default=0.45,
                       help='Classification threshold for debris detection')
    parser.add_argument('--output', default='debris_geolocation_map.json',
                       help='Output JSON file path')
    parser.add_argument('--per_patch_output', default='test_patch_predictions.json',
                       help='Output JSON file for per-patch predictions')
    parser.add_argument('--format', choices=['simple', 'detailed'], default='detailed',
                       help='Output format: simple (True/False) or detailed (with metadata)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Print header
    print("="*70)
    print("Marine Debris Geolocation Mapper")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model checkpoint: {args.checkpoint}")
    print(f"  Test data: {args.test_data}")
    print(f"  Splits file: {args.splits_file}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output (aggregated): {args.output}")
    print(f"  Output (per-patch): {args.per_patch_output}")
    print(f"  Output format: {args.format.capitalize()}")
    print(f"  Device: {args.device}")
    print()
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load patch names
    patch_names = load_test_patches(args.splits_file)
    
    # Load test data
    images, labels = load_test_data(args.test_data)
    
    # Run predictions
    predictions, confidences = predict_debris(model, images, args.threshold, device)
    
    # Create aggregated geolocation map (by location)
    geolocation_map, metadata = create_geolocation_map(
        predictions, confidences, patch_names, format=args.format
    )
    
    # Save aggregated results
    save_geolocation_map(geolocation_map, metadata, args.output)
    
    # Save per-patch results
    save_per_patch_map(predictions, confidences, patch_names, args.per_patch_output)
    
    # Print summary
    print_summary(geolocation_map, metadata)
    
    print(f"\n[OK] Geolocation mapping complete!")
    print(f"\nGenerated files:")
    print(f"  1. Aggregated by location: {args.output}")
    print(f"  2. Individual patches: {args.per_patch_output}")
    print(f"\nUsage in Python:")
    print(f"  import json")
    print(f"  # Load aggregated map")
    print(f"  with open('{args.output}', 'r') as f:")
    print(f"      data = json.load(f)")
    print(f"  geolocation_map = data['geolocation_map']")
    print(f"  print(geolocation_map['1-12-19_48MYU'])  # Location data")
    print(f"")
    print(f"  # Load per-patch map")
    print(f"  with open('{args.per_patch_output}', 'r') as f:")
    print(f"      patch_data = json.load(f)")
    print(f"  patch_map = patch_data['patch_map']")
    print(f"  print(patch_map['12-12-20_16PCC_0'])  # Individual patch data")


if __name__ == '__main__':
    main()
