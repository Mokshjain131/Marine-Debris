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

    # Initialize model
    if model_type == 'shallow':
        model = ShallowCNN(num_bands=num_bands, num_classes=num_classes)
    elif model_type == 'resnet':
        model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=True)
    elif model_type == 'simple_cnn':
        model = SimpleCNN(num_bands=num_bands, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model, device

def load_test_data(h5_path='test_data.h5'):
    """Load test dataset from H5 file"""
    with h5py.File(h5_path, 'r') as f:
        images = torch.tensor(f['images'][:], dtype=torch.float32)
        labels = torch.tensor(f['labels'][:], dtype=torch.float32)

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Loaded test data: {images.shape[0]} samples")
    return dataloader

def predict_sample(model, image, device, threshold=0.5):
    """Make prediction for a single image"""
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).cpu().numpy().squeeze()
        predictions = (probabilities > threshold).astype(int)

    return predictions, probabilities

def visualize_rgb_bands(image_tensor, bands=[3, 2, 1]):
    """Create RGB visualization from multispectral image using specified bands"""
    # Select RGB bands (default: bands 4,3,2 which correspond to Red, Green, Blue for Sentinel-2)
    rgb_image = image_tensor[bands].permute(1, 2, 0).numpy()

    # Normalize for visualization (clip extreme values and scale to 0-1)
    rgb_image = np.clip(rgb_image, 0, np.percentile(rgb_image, 95))
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    return rgb_image

def plot_prediction_results(image, true_labels, pred_labels, probabilities, sample_idx=0):
    """Plot image with true and predicted labels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Display RGB image
    rgb_image = visualize_rgb_bands(image.squeeze())
    ax1.imshow(rgb_image)
    ax1.set_title(f'Sample {sample_idx} - RGB Visualization')
    ax1.axis('off')

    # Create labels comparison
    true_classes = [CLASS_NAMES[i] for i, label in enumerate(true_labels) if label == 1]
    pred_classes = [CLASS_NAMES[i] for i, label in enumerate(pred_labels) if label == 1]

    # Display labels and probabilities
    ax2.axis('off')

    # Title
    ax2.text(0.5, 0.95, f'Sample {sample_idx} - Label Comparison',
             ha='center', va='top', fontsize=16, fontweight='bold', transform=ax2.transAxes)

    # True labels
    ax2.text(0.02, 0.85, 'TRUE LABELS:', fontsize=12, fontweight='bold',
             color='green', transform=ax2.transAxes)
    true_text = '\n'.join([f"• {cls}" for cls in true_classes]) if true_classes else "No positive labels"
    ax2.text(0.02, 0.75, true_text, fontsize=10, color='green', transform=ax2.transAxes)

    # Predicted labels
    ax2.text(0.02, 0.55, 'PREDICTED LABELS:', fontsize=12, fontweight='bold',
             color='blue', transform=ax2.transAxes)
    pred_text = '\n'.join([f"• {cls}" for cls in pred_classes]) if pred_classes else "No positive predictions"
    ax2.text(0.02, 0.45, pred_text, fontsize=10, color='blue', transform=ax2.transAxes)

    # Top probabilities
    ax2.text(0.52, 0.85, 'TOP PROBABILITIES:', fontsize=12, fontweight='bold',
             color='purple', transform=ax2.transAxes)

    # Sort probabilities and get top 5
    prob_indices = np.argsort(probabilities)[::-1][:5]
    prob_text = []
    for idx in prob_indices:
        prob_text.append(f"• {CLASS_NAMES[idx]}: {probabilities[idx]:.3f}")

    ax2.text(0.52, 0.75, '\n'.join(prob_text), fontsize=10, color='purple', transform=ax2.transAxes)

    # Accuracy indicator
    match_score = np.sum(true_labels == pred_labels) / len(true_labels)
    ax2.text(0.52, 0.25, f'MATCH SCORE: {match_score:.3f}',
             fontsize=12, fontweight='bold',
             color='green' if match_score > 0.8 else 'orange' if match_score > 0.6 else 'red',
             transform=ax2.transAxes)

    plt.tight_layout()
    return fig

def evaluate_samples(model, dataloader, device, num_samples=5, threshold=0.5):
    """Evaluate and visualize multiple samples"""
    print(f"Evaluating {num_samples} random samples...")

    sample_count = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        if sample_count >= num_samples:
            break

        # Get predictions
        predictions, probabilities = predict_sample(model, images, device, threshold)

        # Convert labels to numpy
        true_labels = labels.squeeze().numpy().astype(int)

        # Create visualization
        fig = plot_prediction_results(images, true_labels, predictions, probabilities, sample_count)

        # Save plot
        os.makedirs('evaluation_results', exist_ok=True)
        plt.savefig(f'evaluation_results/sample_{sample_count}.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print summary
        print(f"\n--- Sample {sample_count} ---")
        print(f"True classes: {[CLASS_NAMES[i] for i, l in enumerate(true_labels) if l == 1]}")
        print(f"Predicted classes: {[CLASS_NAMES[i] for i, l in enumerate(predictions) if l == 1]}")
        print(f"Top probability: {CLASS_NAMES[np.argmax(probabilities)]} ({np.max(probabilities):.3f})")

        sample_count += 1

    print(f"\nEvaluation complete! Results saved to 'evaluation_results/' folder")

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

def main():
    """Main evaluation function"""
    # Configuration
    checkpoint_path = 'checkpoints/resnet_best.pth'  # Using the ResNet model checkpoint
    test_data_path = 'test_data.h5'
    model_type = 'simple_cnn'  # Using the SimpleCNN model (matches resnet_best.pth)

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

        # Evaluate samples
        evaluate_samples(model, test_loader, device, num_samples=5)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check your model checkpoint and data files.")

if __name__ == '__main__':
    print("Marine Debris Classification - Model Evaluation")
    print("=" * 50)
    main()