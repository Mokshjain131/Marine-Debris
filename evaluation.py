import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.shallow import ShallowCNN
from models.resnet18 import ResNetSentinel
from models.simple_cnn import SimpleCNN
import h5py
from torch.utils.data import DataLoader, TensorDataset
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional rasterio import for TIF support
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available. TIF file support disabled.")

# Define class names for marine debris classification
CLASS_NAMES = [
    "Marine Debris", "Dense Sargassum", "Sparse Sargassum", "Natural Organic Material",
    "Ship", "Clouds", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water",
    "Shallow Water", "Waves", "Cloud Shadows", "Wakes", "Mixed Water"
]

def load_model(checkpoint_path, model_type='shallow', num_bands=11, num_classes=15):
    """Load trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint first to inspect it
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Auto-detect model type from state_dict keys if needed
    first_key = list(state_dict.keys())[0]
    is_wrapped = first_key.startswith('model.')
    
    # Check what type of model this actually is based on keys
    if 'layer1.0.conv1.weight' in state_dict or 'model.layer1.0.conv1.weight' in state_dict:
        actual_model_type = 'resnet'
        print(f"Auto-detected ResNet model from checkpoint")
    elif 'conv1.weight' in state_dict or 'model.conv1.weight' in state_dict:
        if 'fc1.weight' in state_dict or 'model.fc1.weight' in state_dict:
            actual_model_type = 'simple_cnn'
            print(f"Auto-detected SimpleCNN model from checkpoint")
        else:
            actual_model_type = 'shallow'
            print(f"Auto-detected ShallowCNN model from checkpoint")
    else:
        # Use provided model_type
        actual_model_type = model_type
    
    # Initialize model with the correct type
    if actual_model_type == 'shallow':
        model = ShallowCNN(num_bands=num_bands, num_classes=num_classes)
    elif actual_model_type == 'resnet':
        model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=False)
    elif actual_model_type == 'simple_cnn':
        model = SimpleCNN(num_bands=num_bands, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {actual_model_type}")

    # Handle wrapped state dict (keys prefixed with 'model.')
    if is_wrapped:
        # Remove 'model.' prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Could not load state dict directly. Error: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    print(f"[OK] Loaded {actual_model_type} model from {checkpoint_path}")
    return model, device

def load_segmentation_model(model_type='unet', checkpoint_path=None, num_bands=11, num_classes=15):
    """Load segmentation model (U-Net or Random Forest)
    
    Args:
        model_type: 'unet' or 'random_forest'
        checkpoint_path: path to model checkpoint
        num_bands: number of input bands (may be overridden by checkpoint)
        num_classes: number of output classes (may be overridden by checkpoint)
    
    Returns:
        model, device, actual_num_classes (for U-Net) or model, None, num_classes (for Random Forest)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'unet':
        # Import U-Net from MARIDA models
        import sys
        sys.path.append('MARIDA_models/unet')
        from unet import UNet
        
        if checkpoint_path is None:
            # Try to find trained model
            checkpoint_path = 'MARIDA_models/unet/trained_models/u-net-44/model.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Auto-detect number of output classes from checkpoint
            # Check the output layer (outc.weight or outc.bias)
            if 'outc.weight' in state_dict:
                detected_classes = state_dict['outc.weight'].shape[0]
                detected_bands = state_dict['inc.0.weight'].shape[1]
                print(f"   Detected from checkpoint: {detected_bands} input bands, {detected_classes} output classes")
                num_classes = detected_classes
                num_bands = detected_bands
            
            # Initialize U-Net with correct dimensions
            model = UNet(input_bands=num_bands, output_classes=num_classes, hidden_channels=16)
            
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded U-Net model from {checkpoint_path}")
        else:
            print(f"[WARNING]  Checkpoint not found at {checkpoint_path}")
            print("   Using randomly initialized U-Net model")
            # Initialize with provided dimensions
            model = UNet(input_bands=num_bands, output_classes=num_classes, hidden_channels=16)
        
        model.to(device)
        model.eval()
        return model, device, num_classes
    
    elif model_type == 'random_forest':
        # Import Random Forest
        import joblib
        
        if checkpoint_path is None:
            checkpoint_path = 'MARIDA_models/random_forest/rf_classifier.joblib'
        
        if os.path.exists(checkpoint_path):
            model = joblib.load(checkpoint_path)
            print(f"[OK] Loaded Random Forest model from {checkpoint_path}")
        else:
            print(f"[ERROR] Error: Random Forest model not found at {checkpoint_path}")
            return None, None, num_classes
        
        return model, None, num_classes  # Random Forest doesn't need device
    
    else:
        raise ValueError(f"Unknown segmentation model type: {model_type}. Use 'unet' or 'random_forest'")

def load_test_data(h5_path='test_data.h5'):
    """Load test dataset from H5 file"""
    with h5py.File(h5_path, 'r') as f:
        images = torch.tensor(f['images'][:], dtype=torch.float32)
        labels = torch.tensor(f['labels'][:], dtype=torch.float32)

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Loaded test data: {images.shape[0]} samples")
    return dataloader

def predict_sample(model, image, device, threshold=0.45):
    """Make prediction for a single image
    
    Returns:
        predictions: full multi-label predictions (15 classes)
        probabilities: full probabilities (15 classes)
        marine_debris_pred: binary prediction for Marine Debris only (0 or 1)
        marine_debris_prob: probability for Marine Debris class
    """
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).cpu().numpy().squeeze()
        predictions = (probabilities > threshold).astype(int)
    
    # Extract Marine Debris (class 0) specifically
    marine_debris_pred = predictions[0]
    marine_debris_prob = probabilities[0]

    return predictions, probabilities, marine_debris_pred, marine_debris_prob

def visualize_rgb_bands(image_tensor, bands=[3, 2, 1]):
    """Create RGB visualization from multispectral image using specified bands"""
    # Select RGB bands (default: bands 4,3,2 which correspond to Red, Green, Blue for Sentinel-2)
    rgb_image = image_tensor[bands].permute(1, 2, 0).numpy()

    # Normalize for visualization (clip extreme values and scale to 0-1)
    rgb_image = np.clip(rgb_image, 0, np.percentile(rgb_image, 95))
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    return rgb_image

def plot_prediction_results(image, true_labels, pred_labels, probabilities, sample_idx=0):
    """Plot image with true and predicted labels - FOCUSED ON MARINE DEBRIS"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Display RGB image
    rgb_image = visualize_rgb_bands(image.squeeze())
    ax1.imshow(rgb_image)
    ax1.set_title(f'Sample {sample_idx} - RGB Visualization')
    ax1.axis('off')

    # Focus on Marine Debris classification
    true_marine_debris = true_labels[0]  # Class 0 = Marine Debris
    pred_marine_debris = pred_labels[0]
    marine_debris_prob = probabilities[0]

    # Display labels and probabilities
    ax2.axis('off')

    # Title
    ax2.text(0.5, 0.95, f'Sample {sample_idx} - Marine Debris Classification',
             ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes)

    # True label for Marine Debris
    ax2.text(0.02, 0.80, 'TRUE LABEL:', fontsize=12, fontweight='bold',
             color='green', transform=ax2.transAxes)
    true_text = "✓ Marine Debris Present" if true_marine_debris == 1 else "✗ No Marine Debris"
    true_color = 'green' if true_marine_debris == 1 else 'gray'
    ax2.text(0.02, 0.70, true_text, fontsize=11, color=true_color, transform=ax2.transAxes)

    # Predicted label for Marine Debris
    ax2.text(0.02, 0.55, 'PREDICTED LABEL:', fontsize=12, fontweight='bold',
             color='blue', transform=ax2.transAxes)
    pred_text = "✓ Marine Debris Present" if pred_marine_debris == 1 else "✗ No Marine Debris"
    pred_color = 'blue' if pred_marine_debris == 1 else 'gray'
    ax2.text(0.02, 0.45, pred_text, fontsize=11, color=pred_color, transform=ax2.transAxes)
    ax2.text(0.02, 0.38, f"   Confidence: {marine_debris_prob:.1%}", 
             fontsize=10, color=pred_color, transform=ax2.transAxes)

    # Prediction correctness
    is_correct = (true_marine_debris == pred_marine_debris)
    result_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    result_color = 'green' if is_correct else 'red'
    ax2.text(0.5, 0.25, result_text, ha='center', fontsize=14, fontweight='bold',
             color=result_color, transform=ax2.transAxes)

    # Additional info: other detected classes
    ax2.text(0.52, 0.80, 'OTHER DETECTED CLASSES:', fontsize=10, fontweight='bold',
             color='gray', transform=ax2.transAxes)
    other_classes = [CLASS_NAMES[i] for i in range(1, len(pred_labels)) if pred_labels[i] == 1]
    if other_classes:
        other_text = '\n'.join([f"• {cls}" for cls in other_classes[:4]])
        if len(other_classes) > 4:
            other_text += f"\n• ... and {len(other_classes)-4} more"
    else:
        other_text = "(none)"
    ax2.text(0.52, 0.70, other_text, fontsize=9, color='gray', transform=ax2.transAxes)

    plt.tight_layout()
    return fig

def evaluate_samples(model, dataloader, device, num_samples=5, threshold=0.45):
    """Evaluate and visualize multiple samples - FOCUSED ON MARINE DEBRIS ONLY"""
    print(f"\n{'='*60}")
    print(f"MARINE DEBRIS BINARY CLASSIFICATION EVALUATION")
    print(f"{'='*60}")
    print(f"Evaluating {num_samples} random samples...")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")

    sample_count = 0
    
    # Store Marine Debris predictions for metrics
    all_true_debris = []
    all_pred_debris = []
    all_debris_probs = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if sample_count >= num_samples:
            break

        # Get predictions
        predictions, probabilities, marine_debris_pred, marine_debris_prob = predict_sample(
            model, images, device, threshold
        )

        # Convert labels to numpy
        true_labels = labels.squeeze().numpy().astype(int)
        true_marine_debris = true_labels[0]  # Class 0 = Marine Debris

        # Store for metrics
        all_true_debris.append(true_marine_debris)
        all_pred_debris.append(marine_debris_pred)
        all_debris_probs.append(marine_debris_prob)

        # Create visualization
        fig = plot_prediction_results(images, true_labels, predictions, probabilities, sample_count + 1)

        # Save plot
        os.makedirs('evaluation_results', exist_ok=True)
        plt.savefig(f'evaluation_results/sample_{sample_count}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Print summary
        correct_symbol = "[OK]" if true_marine_debris == marine_debris_pred else "[ERROR]"
        print(f"{correct_symbol} Sample {sample_count + 1}: "
              f"True={'DEBRIS' if true_marine_debris else 'NO_DEBRIS':10s} | "
              f"Pred={'DEBRIS' if marine_debris_pred else 'NO_DEBRIS':10s} | "
              f"Confidence={marine_debris_prob:.1%}")

        sample_count += 1

    # Calculate Marine Debris-specific metrics
    all_true_debris = np.array(all_true_debris)
    all_pred_debris = np.array(all_pred_debris)
    all_debris_probs = np.array(all_debris_probs)

    print(f"\n{'='*60}")
    print("MARINE DEBRIS CLASSIFICATION METRICS")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(all_true_debris, all_pred_debris)
    precision = precision_score(all_true_debris, all_pred_debris, zero_division=0)
    recall = recall_score(all_true_debris, all_pred_debris, zero_division=0)
    f1 = f1_score(all_true_debris, all_pred_debris, zero_division=0)
    
    print(f"Accuracy:   {accuracy:.3f} ({accuracy:.1%})")
    print(f"Precision:  {precision:.3f} ({precision:.1%})")
    print(f"Recall:     {recall:.3f} ({recall:.1%})")
    print(f"F1 Score:   {f1:.3f} ({f1:.1%})")
    
    # Confusion matrix breakdown
    true_positives = np.sum((all_true_debris == 1) & (all_pred_debris == 1))
    true_negatives = np.sum((all_true_debris == 0) & (all_pred_debris == 0))
    false_positives = np.sum((all_true_debris == 0) & (all_pred_debris == 1))
    false_negatives = np.sum((all_true_debris == 1) & (all_pred_debris == 0))
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {true_positives} (Correctly detected debris)")
    print(f"  True Negatives:  {true_negatives} (Correctly detected no debris)")
    print(f"  False Positives: {false_positives} (False alarms)")
    print(f"  False Negatives: {false_negatives} (Missed debris)")
    
    print(f"\nDataset Distribution:")
    print(f"  Total samples with debris:    {np.sum(all_true_debris)} ({np.sum(all_true_debris)/len(all_true_debris):.1%})")
    print(f"  Total samples without debris: {len(all_true_debris) - np.sum(all_true_debris)} ({(len(all_true_debris) - np.sum(all_true_debris))/len(all_true_debris):.1%})")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to 'evaluation_results/' folder")
    print(f"{'='*60}\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def evaluate_full_dataset(model, dataloader, device, threshold=0.45):
    """Evaluate model on entire test dataset - MARINE DEBRIS BINARY CLASSIFICATION ONLY"""
    print(f"\n{'='*60}")
    print(f"MARINE DEBRIS BINARY CLASSIFICATION - FULL DATASET")
    print(f"{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")

    # Store all Marine Debris predictions
    all_true_debris = []
    all_pred_debris = []
    all_debris_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # Extract Marine Debris (class 0) for all samples in batch
            marine_debris_probs = probabilities[:, 0]
            marine_debris_preds = (marine_debris_probs > threshold).astype(int)
            true_marine_debris = labels[:, 0].astype(int)
            
            # Store
            all_true_debris.extend(true_marine_debris)
            all_pred_debris.extend(marine_debris_preds)
            all_debris_probs.extend(marine_debris_probs)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * len(images)} samples...")

    # Convert to numpy arrays
    all_true_debris = np.array(all_true_debris)
    all_pred_debris = np.array(all_pred_debris)
    all_debris_probs = np.array(all_debris_probs)

    # Calculate Marine Debris-specific metrics
    print(f"\n{'='*60}")
    print("MARINE DEBRIS CLASSIFICATION METRICS")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(all_true_debris, all_pred_debris)
    precision = precision_score(all_true_debris, all_pred_debris, zero_division=0)
    recall = recall_score(all_true_debris, all_pred_debris, zero_division=0)
    f1 = f1_score(all_true_debris, all_pred_debris, zero_division=0)
    
    print(f"Total samples evaluated: {len(all_true_debris)}")
    print(f"\nAccuracy:   {accuracy:.4f} ({accuracy:.1%})")
    print(f"Precision:  {precision:.4f} ({precision:.1%})")
    print(f"Recall:     {recall:.4f} ({recall:.1%})")
    print(f"F1 Score:   {f1:.4f} ({f1:.1%})")
    
    # Confusion matrix breakdown
    true_positives = np.sum((all_true_debris == 1) & (all_pred_debris == 1))
    true_negatives = np.sum((all_true_debris == 0) & (all_pred_debris == 0))
    false_positives = np.sum((all_true_debris == 0) & (all_pred_debris == 1))
    false_negatives = np.sum((all_true_debris == 1) & (all_pred_debris == 0))
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"  True Positives:  {true_positives:4d} (Correctly detected debris)")
    print(f"  True Negatives:  {true_negatives:4d} (Correctly detected no debris)")
    print(f"  False Positives: {false_positives:4d} (False alarms)")
    print(f"  False Negatives: {false_negatives:4d} (Missed debris)")
    
    print(f"\n{'='*60}")
    print("DATASET DISTRIBUTION")
    print(f"{'='*60}")
    num_with_debris = np.sum(all_true_debris)
    num_without_debris = len(all_true_debris) - num_with_debris
    print(f"  Samples WITH debris:    {num_with_debris:4d} ({num_with_debris/len(all_true_debris):.1%})")
    print(f"  Samples WITHOUT debris: {num_without_debris:4d} ({num_without_debris/len(all_true_debris):.1%})")
    
    # Additional statistics
    print(f"\n{'='*60}")
    print("CONFIDENCE STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean confidence (all):  {np.mean(all_debris_probs):.4f}")
    if num_with_debris > 0:
        print(f"  Mean confidence (true positives): {np.mean(all_debris_probs[all_true_debris == 1]):.4f}")
    if num_without_debris > 0:
        print(f"  Mean confidence (true negatives): {np.mean(all_debris_probs[all_true_debris == 0]):.4f}")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(true_positives),
        'true_negatives': int(true_negatives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'total_samples': len(all_true_debris),
        'samples_with_debris': int(num_with_debris),
        'samples_without_debris': int(num_without_debris)
    }

def load_tif_image(tif_path):
    """Load and preprocess a TIF image"""
    if not HAS_RASTERIO:
        print("Error: rasterio not available. Cannot load TIF files.")
        return None

    try:
        with rasterio.open(tif_path) as src:
            # Read all bands
            image = src.read()  # Shape: (bands, height, width)

            # Convert to torch tensor and ensure float32
            image = torch.tensor(image, dtype=torch.float32)

            # Add batch dimension
            image = image.unsqueeze(0)  # Shape: (1, bands, height, width)

        print(f"Loaded TIF image: {image.shape}")
        return image
    except Exception as e:
        print(f"Error loading TIF file: {e}")
        return None

def evaluate_tif_image(tif_path, checkpoint_path='checkpoints/resnet_best.pth', model_type='simple_cnn'):
    """Evaluate a single TIF image"""
    print(f"Evaluating TIF image: {tif_path}")

    # Check if file exists
    if not os.path.exists(tif_path):
        print(f"Error: TIF file '{tif_path}' not found!")
        return

    # Load model
    model, device = load_model(checkpoint_path, model_type)

    # Load TIF image
    image = load_tif_image(tif_path)
    if image is None:
        return

    # Make prediction
    predictions, probabilities = predict_sample(model, image, device)

    # Create visualization (no true labels for single image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Display RGB image
    rgb_image = visualize_rgb_bands(image.squeeze())
    ax1.imshow(rgb_image)
    ax1.set_title(f'Input Image: {Path(tif_path).name}')
    ax1.axis('off')

    # Display predictions
    ax2.axis('off')

    # Title
    ax2.text(0.5, 0.95, f'Predictions for {Path(tif_path).name}',
             ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes)

    # Predicted labels
    pred_classes = [CLASS_NAMES[i] for i, label in enumerate(predictions) if label == 1]
    ax2.text(0.02, 0.85, 'PREDICTED LABELS:', fontsize=12, fontweight='bold',
             color='blue', transform=ax2.transAxes)
    pred_text = '\n'.join([f"• {cls}" for cls in pred_classes]) if pred_classes else "No positive predictions"
    ax2.text(0.02, 0.75, pred_text, fontsize=10, color='blue', transform=ax2.transAxes)

    # Top probabilities
    ax2.text(0.02, 0.45, 'TOP PROBABILITIES:', fontsize=12, fontweight='bold',
             color='purple', transform=ax2.transAxes)

    # Sort probabilities and get top 5
    prob_indices = np.argsort(probabilities)[::-1][:5]
    prob_text = []
    for idx in prob_indices:
        prob_text.append(f"• {CLASS_NAMES[idx]}: {probabilities[idx]:.3f}")

    ax2.text(0.02, 0.35, '\n'.join(prob_text), fontsize=10, color='purple', transform=ax2.transAxes)

    plt.tight_layout()

    # Save plot
    os.makedirs('evaluation_results', exist_ok=True)
    output_name = f"tif_prediction_{Path(tif_path).stem}.png"
    plt.savefig(f'evaluation_results/{output_name}', dpi=150, bbox_inches='tight')
    plt.show()

    # Print results
    print(f"\n--- Results for {Path(tif_path).name} ---")
    print(f"Predicted classes: {pred_classes}")
    print(f"Top probability: {CLASS_NAMES[np.argmax(probabilities)]} ({np.max(probabilities):.3f})")
    print(f"Results saved as: evaluation_results/{output_name}")

# ============================================================================
# OPTION 1: SEGMENTATION-ONLY APPROACH
# Derive classification from segmentation model output
# ============================================================================

def segment_to_classify(segmentation_mask, debris_class_id=0, threshold_pixels=0, threshold_percentage=0.0):
    """
    Convert segmentation output to binary classification
    
    Args:
        segmentation_mask: 2D/3D numpy array of pixel labels (H, W) or (B, H, W)
        debris_class_id: label ID for marine debris class (default: 0)
        threshold_pixels: minimum number of debris pixels to classify as positive
        threshold_percentage: minimum percentage of debris pixels (0-100)
    
    Returns:
        bool or np.array: True if debris detected (single image) or array of bools (batch)
    """
    if len(segmentation_mask.shape) == 2:
        # Single image (H, W)
        debris_pixels = (segmentation_mask == debris_class_id).sum()
        total_pixels = segmentation_mask.size
        debris_percentage = (debris_pixels / total_pixels) * 100
        
        return debris_pixels > threshold_pixels or debris_percentage > threshold_percentage
    
    elif len(segmentation_mask.shape) == 3:
        # Batch of images (B, H, W)
        results = []
        for mask in segmentation_mask:
            debris_pixels = (mask == debris_class_id).sum()
            total_pixels = mask.size
            debris_percentage = (debris_pixels / total_pixels) * 100
            results.append(debris_pixels > threshold_pixels or debris_percentage > threshold_percentage)
        return np.array(results)
    
    else:
        raise ValueError(f"Invalid segmentation mask shape: {segmentation_mask.shape}")

def compute_iou(pred_mask, true_mask, class_id=0):
    """
    Compute Intersection over Union (IoU) for a specific class
    
    Args:
        pred_mask: predicted segmentation mask
        true_mask: ground truth segmentation mask
        class_id: class to compute IoU for
    
    Returns:
        float: IoU score
    """
    pred_binary = (pred_mask == class_id)
    true_binary = (true_mask == class_id)
    
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def evaluate_segmentation_as_classifier(segmentation_model, test_dataloader, device, 
                                       debris_class_id=0, threshold_percentage=0.1):
    """
    OPTION 1: Evaluate segmentation model for BOTH tasks:
    1. Pixel-level segmentation metrics (IoU, Dice/F1)
    2. Patch-level classification (derived from segmentation)
    
    Args:
        segmentation_model: trained segmentation model (U-Net, etc.)
        test_dataloader: DataLoader with test images and masks/labels
        device: torch device
        debris_class_id: class ID for marine debris in segmentation
        threshold_percentage: minimum % of debris pixels for positive classification
    
    Returns:
        dict: comprehensive metrics for both tasks
    """
    # Set to eval mode only for PyTorch models
    if hasattr(segmentation_model, 'eval'):
        segmentation_model.eval()
    
    # Storage for metrics
    pixel_ious = []
    pixel_dices = []
    
    patch_true_labels = []
    patch_pred_labels = []
    
    all_pred_masks = []
    all_true_masks = []
    
    print(f"\n{'='*60}")
    print("OPTION 1: SEGMENTATION-ONLY EVALUATION")
    print(f"{'='*60}")
    print(f"Debris class ID: {debris_class_id}")
    print(f"Classification threshold: {threshold_percentage}% debris pixels")
    
    # Check if this is a PyTorch model or scikit-learn model
    is_pytorch_model = hasattr(segmentation_model, 'eval')
    
    if is_pytorch_model:
        # PyTorch model (U-Net)
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_dataloader):
                images = images.to(device)
                
                # Get segmentation predictions
                outputs = segmentation_model(images)
                
                # Convert to class predictions (argmax for multi-class)
                if len(outputs.shape) == 4:  # (B, C, H, W)
                    pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                else:  # (B, H, W)
                    pred_masks = (outputs > 0.5).cpu().numpy().astype(int)
                
                # Handle targets (could be masks or labels)
                if len(targets.shape) == 3:  # Segmentation masks (B, H, W)
                    true_masks = targets.cpu().numpy()
                    
                    # Compute pixel-level metrics
                    for pred_mask, true_mask in zip(pred_masks, true_masks):
                        iou = compute_iou(pred_mask, true_mask, debris_class_id)
                        pixel_ious.append(iou)
                        
                        # Dice coefficient (F1 for segmentation)
                        dice = 2 * iou / (1 + iou) if iou < 1.0 else 1.0
                        pixel_dices.append(dice)
                    
                    # Derive patch-level classification from masks
                    true_labels = np.array([segment_to_classify(mask, debris_class_id, 
                                                               threshold_percentage=0) 
                                           for mask in true_masks])
                else:  # Classification labels (B,) or (B, num_classes)
                    if len(targets.shape) == 1:
                        true_labels = targets.cpu().numpy()
                    else:  # Multi-label (B, num_classes)
                        # Assume first class is marine debris
                        true_labels = targets[:, debris_class_id].cpu().numpy()
                
                # Derive patch-level classification from predicted masks
                pred_labels = np.array([segment_to_classify(mask, debris_class_id,
                                                           threshold_percentage=threshold_percentage)
                                       for mask in pred_masks])
                
                patch_true_labels.extend(true_labels)
                patch_pred_labels.extend(pred_labels)
                all_pred_masks.extend(pred_masks)
    else:
        # scikit-learn model (Random Forest)
        for batch_idx, (images, targets) in enumerate(test_dataloader):
            # Random Forest expects flattened features: (batch, height*width*channels)
            batch_size = images.shape[0]
            h, w = images.shape[2], images.shape[3]
            
            # Reshape: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
            images_np = images.numpy()
            images_reshaped = np.transpose(images_np, (0, 2, 3, 1))  # (B, H, W, C)
            images_flat = images_reshaped.reshape(-1, images_reshaped.shape[-1])  # (B*H*W, C)
            
            # Predict for all pixels
            pred_flat = segmentation_model.predict(images_flat)
            
            # Reshape back to masks: (B*H*W,) -> (B, H, W)
            pred_masks = pred_flat.reshape(batch_size, h, w)
            
            # Handle targets
            if len(targets.shape) == 3:  # Segmentation masks (B, H, W)
                true_masks = targets.numpy()
                
                # Compute pixel-level metrics
                for pred_mask, true_mask in zip(pred_masks, true_masks):
                    iou = compute_iou(pred_mask, true_mask, debris_class_id)
                    pixel_ious.append(iou)
                    
                    # Dice coefficient
                    dice = 2 * iou / (1 + iou) if iou < 1.0 else 1.0
                    pixel_dices.append(dice)
                
                # Derive patch-level classification from masks
                true_labels = np.array([segment_to_classify(mask, debris_class_id, 
                                                           threshold_percentage=0) 
                                       for mask in true_masks])
            else:  # Classification labels
                if len(targets.shape) == 1:
                    true_labels = targets.numpy()
                else:
                    true_labels = targets[:, debris_class_id].numpy()
            
            # Derive patch-level classification from predicted masks
            pred_labels = np.array([segment_to_classify(mask, debris_class_id,
                                                       threshold_percentage=threshold_percentage)
                                   for mask in pred_masks])
            
            patch_true_labels.extend(true_labels)
            patch_pred_labels.extend(pred_labels)
            all_pred_masks.extend(pred_masks)
    
    # Compute final metrics
    patch_true_labels = np.array(patch_true_labels)
    patch_pred_labels = np.array(patch_pred_labels)
    
    results = {
        'segmentation': {
            'mean_iou': np.mean(pixel_ious) if pixel_ious else None,
            'mean_dice': np.mean(pixel_dices) if pixel_dices else None,
        },
        'classification': {
            'accuracy': accuracy_score(patch_true_labels, patch_pred_labels),
            'precision': precision_score(patch_true_labels, patch_pred_labels, zero_division=0),
            'recall': recall_score(patch_true_labels, patch_pred_labels, zero_division=0),
            'f1': f1_score(patch_true_labels, patch_pred_labels, zero_division=0),
        }
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("SEGMENTATION METRICS (Pixel-level):")
    print(f"{'='*60}")
    if results['segmentation']['mean_iou'] is not None:
        print(f"Mean IoU:        {results['segmentation']['mean_iou']:.4f}")
        print(f"Mean Dice/F1:    {results['segmentation']['mean_dice']:.4f}")
    else:
        print("(No pixel-level ground truth available)")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION METRICS (Patch-level, derived):")
    print(f"{'='*60}")
    print(f"Accuracy:        {results['classification']['accuracy']:.4f}")
    print(f"Precision:       {results['classification']['precision']:.4f}")
    print(f"Recall:          {results['classification']['recall']:.4f}")
    print(f"F1 Score:        {results['classification']['f1']:.4f}")
    print(f"\nTotal patches:   {len(patch_true_labels)}")
    print(f"Positive (True): {patch_true_labels.sum()}")
    print(f"Positive (Pred): {patch_pred_labels.sum()}")
    
    return results, all_pred_masks

# ============================================================================
# OPTION 2: TWO-STAGE PIPELINE
# Fast classifier → Detailed segmentation on positives only
# ============================================================================

def two_stage_evaluation(classifier_model, segmentation_model, test_dataloader, device,
                        classifier_threshold=0.5, debris_class_id=0):
    """
    OPTION 2: Two-stage pipeline evaluation
    Stage 1: Fast ResNet classifier screens all patches
    Stage 2: U-Net segmentation only on predicted positive patches
    
    Args:
        classifier_model: fast classification model (ResNet, etc.)
        segmentation_model: detailed segmentation model (U-Net, etc.)
        test_dataloader: DataLoader with test data
        device: torch device
        classifier_threshold: probability threshold for stage 1
        debris_class_id: class ID for marine debris
    
    Returns:
        dict: metrics for both stages and efficiency statistics
    """
    classifier_model.eval()
    segmentation_model.eval()
    
    # Storage
    stage1_true_labels = []
    stage1_pred_labels = []
    stage1_probabilities = []
    
    stage2_processed = 0
    stage2_masks = []
    stage2_indices = []
    
    all_images = []
    all_true_labels = []
    
    print(f"\n{'='*60}")
    print("OPTION 2: TWO-STAGE PIPELINE EVALUATION")
    print(f"{'='*60}")
    print("Stage 1: ResNet Classifier (fast screening)")
    print("Stage 2: Segmentation Model (detailed analysis on positives)")
    
    with torch.no_grad():
        # STAGE 1: Fast classification on all patches
        print(f"\n{'='*60}")
        print("Stage 1: Running fast classifier on all patches...")
        print(f"{'='*60}")
        
        for batch_idx, (images, targets) in enumerate(test_dataloader):
            images_gpu = images.to(device)
            
            # Classification prediction
            outputs = classifier_model(images_gpu)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # Extract marine debris probability (assume first class)
            if len(probabilities.shape) == 2:
                debris_prob = probabilities[:, debris_class_id]
            else:
                debris_prob = probabilities
            
            predictions = (debris_prob > classifier_threshold).astype(int)
            
            # Store results
            all_images.append(images.cpu())
            stage1_pred_labels.extend(predictions)
            stage1_probabilities.extend(debris_prob)
            
            # Handle ground truth
            if len(targets.shape) == 1:
                true_labels = targets.cpu().numpy()
            elif len(targets.shape) == 2:
                true_labels = targets[:, debris_class_id].cpu().numpy()
            else:  # Segmentation masks
                true_labels = np.array([segment_to_classify(mask.numpy(), debris_class_id) 
                                       for mask in targets])
            
            all_true_labels.extend(true_labels)
            stage1_true_labels.extend(true_labels)
        
        # STAGE 2: Detailed segmentation on predicted positives
        print(f"\n{'='*60}")
        print("Stage 2: Running segmentation on predicted positive patches...")
        print(f"{'='*60}")
        
        stage1_pred_labels = np.array(stage1_pred_labels)
        positive_indices = np.where(stage1_pred_labels == 1)[0]
        
        print(f"Positives from Stage 1: {len(positive_indices)} / {len(stage1_pred_labels)}")
        print(f"Efficiency: Only {(len(positive_indices)/len(stage1_pred_labels)*100):.1f}% need segmentation")
        
        for idx in positive_indices:
            batch_idx = idx // test_dataloader.batch_size
            in_batch_idx = idx % test_dataloader.batch_size
            
            image = all_images[batch_idx][in_batch_idx:in_batch_idx+1].to(device)
            
            # Segmentation prediction
            seg_output = segmentation_model(image)
            
            if len(seg_output.shape) == 4:
                pred_mask = torch.argmax(seg_output, dim=1).cpu().numpy()[0]
            else:
                pred_mask = (seg_output > 0.5).cpu().numpy()[0]
            
            stage2_masks.append(pred_mask)
            stage2_indices.append(idx)
            stage2_processed += 1
    
    # Compute metrics
    stage1_true = np.array(stage1_true_labels)
    stage1_pred = np.array(stage1_pred_labels)
    
    results = {
        'stage1_classifier': {
            'accuracy': accuracy_score(stage1_true, stage1_pred),
            'precision': precision_score(stage1_true, stage1_pred, zero_division=0),
            'recall': recall_score(stage1_true, stage1_pred, zero_division=0),
            'f1': f1_score(stage1_true, stage1_pred, zero_division=0),
            'total_patches': len(stage1_true),
            'predicted_positive': stage1_pred.sum(),
        },
        'stage2_segmentation': {
            'patches_processed': stage2_processed,
            'masks_generated': len(stage2_masks),
        },
        'efficiency': {
            'percentage_segmented': (stage2_processed / len(stage1_true) * 100),
            'speedup_factor': len(stage1_true) / max(stage2_processed, 1),
        }
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("STAGE 1: CLASSIFIER RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:          {results['stage1_classifier']['accuracy']:.4f}")
    print(f"Precision:         {results['stage1_classifier']['precision']:.4f}")
    print(f"Recall:            {results['stage1_classifier']['recall']:.4f}")
    print(f"F1 Score:          {results['stage1_classifier']['f1']:.4f}")
    print(f"Total patches:     {results['stage1_classifier']['total_patches']}")
    print(f"Predicted positive: {results['stage1_classifier']['predicted_positive']}")
    
    print(f"\n{'='*60}")
    print("STAGE 2: SEGMENTATION RESULTS")
    print(f"{'='*60}")
    print(f"Patches processed:  {results['stage2_segmentation']['patches_processed']}")
    print(f"Masks generated:    {results['stage2_segmentation']['masks_generated']}")
    
    print(f"\n{'='*60}")
    print("EFFICIENCY STATISTICS")
    print(f"{'='*60}")
    print(f"% Requiring segmentation: {results['efficiency']['percentage_segmented']:.2f}%")
    print(f"Speedup factor:          {results['efficiency']['speedup_factor']:.2f}x")
    print(f"\n[INFO] By filtering with classifier first, we only need to run")
    print(f"   segmentation on {results['efficiency']['percentage_segmented']:.1f}% of patches!")
    
    return results, stage2_masks, stage2_indices

def visualize_segmentation_results(images, pred_masks, true_labels=None, 
                                   num_samples=5, debris_class_id=0):
    """
    Visualize segmentation masks overlaid on RGB images
    
    Args:
        images: list of image tensors
        pred_masks: list of predicted segmentation masks
        true_labels: optional ground truth labels
        num_samples: number of samples to visualize
        debris_class_id: class ID for marine debris
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        image = images[idx]
        mask = pred_masks[idx]
        
        # RGB visualization
        rgb_image = visualize_rgb_bands(image)
        
        # Plot RGB
        axes[idx, 0].imshow(rgb_image)
        axes[idx, 0].set_title(f'Sample {idx}: RGB')
        axes[idx, 0].axis('off')
        
        # Plot segmentation mask
        axes[idx, 1].imshow(mask, cmap='tab20')
        axes[idx, 1].set_title(f'Segmentation Mask')
        axes[idx, 1].axis('off')
        
        # Plot overlay
        axes[idx, 2].imshow(rgb_image)
        debris_pixels = (mask == debris_class_id)
        overlay = np.zeros((*mask.shape, 4))
        overlay[debris_pixels] = [1, 0, 0, 0.5]  # Red with 50% transparency
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Debris Overlay (Red)')
        axes[idx, 2].axis('off')
        
        # Add classification derived from mask
        has_debris = segment_to_classify(mask, debris_class_id)
        debris_percent = (debris_pixels.sum() / mask.size) * 100
        axes[idx, 2].text(0.5, -0.1, 
                         f'Classification: {"DEBRIS" if has_debris else "NO DEBRIS"} ({debris_percent:.2f}%)',
                         ha='center', transform=axes[idx, 2].transAxes,
                         fontsize=10, fontweight='bold',
                         color='red' if has_debris else 'green')
    
    plt.tight_layout()
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig('evaluation_results/segmentation_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSegmentation visualization saved to 'evaluation_results/segmentation_evaluation.png'")

def main():
    """Main evaluation function"""
    # Configuration
    checkpoint_path = 'checkpoints/resnet_best.pth'
    test_data_path = 'test_data.h5'
    model_type = 'auto'  # Will auto-detect from checkpoint

    # Check for INPUT_TIF environment variable first
    input_tif = os.getenv('INPUT_TIF')

    if input_tif:
        print(f"Found INPUT_TIF environment variable: {input_tif}")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
            return
        evaluate_tif_image(input_tif, checkpoint_path, model_type)
        return

    # If no INPUT_TIF, proceed with H5 test data evaluation
    print("No INPUT_TIF environment variable found. Using H5 test data...")

    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        print("Please ensure the model checkpoint exists.")
        return

    if not os.path.exists(test_data_path):
        print(f"Error: Test data file '{test_data_path}' not found!")
        print("Please ensure the H5 test dataset exists.")
        return

    try:
        # Load model
        model, device = load_model(checkpoint_path, model_type)

        # Load test data
        test_loader = load_test_data(test_data_path)

        # First, visualize a few samples
        print("\n" + "="*60)
        print("STEP 1: Visualizing sample predictions")
        print("="*60)
        evaluate_samples(model, test_loader, device, num_samples=5)
        
        # Then, evaluate on full dataset
        print("\n" + "="*60)
        print("STEP 2: Evaluating on full test dataset")
        print("="*60)
        
        # Reload data loader (since we consumed it in evaluate_samples)
        test_loader = load_test_data(test_data_path)
        results = evaluate_full_dataset(model, test_loader, device, threshold=0.45)
        
        # Save results to file
        results_file = 'evaluation_results/marine_debris_metrics.txt'
        os.makedirs('evaluation_results', exist_ok=True)
        with open(results_file, 'w') as f:
            f.write("MARINE DEBRIS BINARY CLASSIFICATION RESULTS\n")
            f.write(f"Classification Threshold: 0.45 (optimized)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples:     {results['total_samples']}\n")
            f.write(f"Samples with debris:    {results['samples_with_debris']} ({results['samples_with_debris']/results['total_samples']:.1%})\n")
            f.write(f"Samples without debris: {results['samples_without_debris']} ({results['samples_without_debris']/results['total_samples']:.1%})\n\n")
            f.write("METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:   {results['accuracy']:.4f} ({results['accuracy']:.1%})\n")
            f.write(f"Precision:  {results['precision']:.4f} ({results['precision']:.1%})\n")
            f.write(f"Recall:     {results['recall']:.4f} ({results['recall']:.1%})\n")
            f.write(f"F1 Score:   {results['f1']:.4f} ({results['f1']:.1%})\n\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 60 + "\n")
            f.write(f"True Positives:  {results['true_positives']}\n")
            f.write(f"True Negatives:  {results['true_negatives']}\n")
            f.write(f"False Positives: {results['false_positives']}\n")
            f.write(f"False Negatives: {results['false_negatives']}\n")
        
        print(f"\n[OK] Results saved to: {results_file}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check your model checkpoint and data files.")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Marine Debris Classification - Model Evaluation")
    print("=" * 50)
    
    # Check for evaluation mode from environment variable or ask user
    eval_mode = os.getenv('EVAL_MODE')
    
    if eval_mode is None:
        # Ask user to choose evaluation mode
        print("\nPlease select an evaluation mode:")
        print("=" * 50)
        print("1. Standard Classification Evaluation")
        print("   - Uses your trained classification model (ResNet/Shallow)")
        print("   - Evaluates on H5 test data or single TIF images")
        print()
        print("2. Segmentation-Only Evaluation")
        print("   - Uses segmentation model (U-Net/Random Forest)")
        print("   - Derives classification from segmentation masks")
        print("   - Provides both pixel-level and patch-level metrics")
        print()
        print("3. Two-Stage Pipeline Evaluation")
        print("   - Stage 1: Fast classifier screens all patches")
        print("   - Stage 2: Segmentation on predicted positives only")
        print("   - 3-10x speedup for large-scale processing")
        print()
        print("=" * 50)
        
        while True:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            if choice == '1':
                eval_mode = 'standard'
                break
            elif choice == '2':
                eval_mode = 'segmentation'
                break
            elif choice == '3':
                eval_mode = 'two-stage'
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\n" + "=" * 50)
    
    if eval_mode == 'segmentation':
        print("OPTION 2: Segmentation-Only Evaluation")
        print("=" * 50)
        print("Deriving classification from segmentation masks\n")
        
        # Ask user to choose segmentation model
        print("Select segmentation model:")
        print("-" * 50)
        print("1. U-Net (Deep Learning)")
        print("   - Best accuracy for complex scenes")
        print("   - Requires GPU for fast inference")
        print()
        print("2. Random Forest (Traditional ML)")
        print("   - Fast and lightweight")
        print("   - Works well on CPU")
        print("-" * 50)
        
        seg_model_choice = None
        while seg_model_choice not in ['1', '2']:
            seg_model_choice = input("\nEnter your choice (1 or 2): ").strip()
            if seg_model_choice not in ['1', '2']:
                print("Invalid choice. Please enter 1 or 2.")
        
        print()
        seg_model_type = 'unet' if seg_model_choice == '1' else 'random_forest'
        
        # Load test data
        test_data_path = 'test_data.h5'
        if not os.path.exists(test_data_path):
            print(f"[ERROR] Error: Test data file '{test_data_path}' not found!")
            print("Please ensure the H5 test dataset exists.")
        else:
            try:
                print(f"Loading {seg_model_type.upper()} model...")
                seg_model, device, actual_num_classes = load_segmentation_model(
                    model_type=seg_model_type,
                    num_bands=11,
                    num_classes=15
                )
                
                if seg_model is None:
                    print("[ERROR] Failed to load segmentation model.")
                else:
                    print("Loading test data...")
                    test_loader = load_test_data(test_data_path)
                    
                    print(f"\n{'='*60}")
                    print(f"Running segmentation-based evaluation...")
                    print(f"{'='*60}\n")
                    
                    # Run evaluation
                    results, pred_masks = evaluate_segmentation_as_classifier(
                        segmentation_model=seg_model,
                        test_dataloader=test_loader,
                        device=device if device else torch.device('cpu'),
                        debris_class_id=0,
                        threshold_percentage=0.1
                    )
                    
                    print(f"\n{'='*60}")
                    print("[OK] Evaluation Complete!")
                    print(f"{'='*60}")
                    
            except Exception as e:
                print(f"[ERROR] Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    elif eval_mode == 'two-stage':
        print("OPTION 3: Two-Stage Pipeline Evaluation")
        print("=" * 50)
        print("Fast classifier → Detailed segmentation on positives\n")
        
        # Ask user to choose segmentation model for stage 2
        print("Select segmentation model for Stage 2:")
        print("-" * 50)
        print("1. U-Net (Deep Learning)")
        print("2. Random Forest (Traditional ML)")
        print("-" * 50)
        
        seg_model_choice = None
        while seg_model_choice not in ['1', '2']:
            seg_model_choice = input("\nEnter your choice (1 or 2): ").strip()
            if seg_model_choice not in ['1', '2']:
                print("Invalid choice. Please enter 1 or 2.")
        
        print()
        seg_model_type = 'unet' if seg_model_choice == '1' else 'random_forest'
        
        # Load test data
        test_data_path = 'test_data.h5'
        checkpoint_path = 'checkpoints/resnet_best.pth'
        
        if not os.path.exists(test_data_path):
            print(f"[ERROR] Error: Test data file '{test_data_path}' not found!")
        elif not os.path.exists(checkpoint_path):
            print(f"[ERROR] Error: Classifier checkpoint '{checkpoint_path}' not found!")
        else:
            try:
                print("Loading Stage 1: Classification model...")
                classifier, device = load_model(checkpoint_path, model_type='auto')
                
                print(f"Loading Stage 2: {seg_model_type.upper()} model...")
                seg_model, seg_device, actual_num_classes = load_segmentation_model(
                    model_type=seg_model_type,
                    num_bands=11,
                    num_classes=15
                )
                
                if seg_model is None:
                    print("[ERROR] Failed to load segmentation model.")
                else:
                    print("Loading test data...")
                    test_loader = load_test_data(test_data_path)
                    
                    print(f"\n{'='*60}")
                    print(f"Running two-stage evaluation...")
                    print(f"{'='*60}\n")
                    
                    # Run two-stage evaluation
                    results, masks, indices = two_stage_evaluation(
                        classifier_model=classifier,
                        segmentation_model=seg_model,
                        test_dataloader=test_loader,
                        device=device,
                        classifier_threshold=0.5,
                        debris_class_id=0
                    )
                    
                    print(f"\n{'='*60}")
                    print("[OK] Two-Stage Evaluation Complete!")
                    print(f"{'='*60}")
                    
            except Exception as e:
                print(f"[ERROR] Error during evaluation: {e}")
                import traceback
                traceback.print_exc()

    
    else:
        # Standard evaluation (existing functionality)
        print("OPTION 1: Standard Classification Evaluation")
        print("=" * 50)
        main()