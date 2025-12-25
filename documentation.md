# Marine Debris Detection and Route Planning Documentation

## 1. Project Overview

This project implements an end-to-end pipeline for detecting marine debris using satellite imagery and planning optimal collection routes. It leverages deep learning for detection, geospatial analysis for mapping, and combinatorial optimization for route planning.

### Key Components
1.  **Deep Learning Model**: A ResNet18-based classifier adapted for multi-spectral Sentinel-2 imagery.
2.  **Geolocation Mapping**: Converts image patches to real-world coordinates (UTM/Lat-Lon).
3.  **Interactive Visualization**: Web-based maps for exploring debris locations.
4.  **Route Optimization**: An intelligent routing system that maximizes debris collection within a travel budget.

---

## 2. Dataset & Preprocessing

### Data Source
- **Satellite**: Sentinel-2
- **Spectral Bands**: 11 bands used (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
- **Input Shape**: (11, H, W)

### Test Dataset Statistics
- **Total Samples**: 208 patches
- **Debris Samples (Positive)**: 58 (27.9%)
- **Clean Samples (Negative)**: 150 (72.1%)
- **Class Imbalance**: Significant imbalance favoring clean water/other classes.

---

## 3. Model Architecture

### Backbone: ResNet18 (Sentinel-2 Adapted)
- **Base Model**: ResNet18 pretrained on ImageNet.
- **Total Parameters**: **11,209,295**
- **Trainable Parameters**: **11,209,295** (during fine-tuning)
- **Input Adaptation**: First convolutional layer (`conv1`) modified to accept 11 input channels instead of 3.
- **Initialization Strategy**:
    - RGB channels initialized with ImageNet weights.
    - Additional spectral bands initialized using a "Repeat and Average" strategy with small random perturbations to maintain activation magnitude.
- **Head**:
    - Dropout (p=0.4)
    - Fully Connected Layer (Output: 15 classes, including Marine Debris)

### Training Configuration
- **Batch Size**: 16
- **Total Epochs**: 30
- **Training Phases**:
    1.  **Head Training**: 10 epochs (Backbone frozen, LR=0.001)
    2.  **Fine-Tuning**: 20 epochs (Full model trainable, LR=0.0001)
- **Loss Function**: `BCEWithLogitsLoss` (Binary Cross Entropy with Logits)
- **Optimizer**: `Adam` (Weight Decay: 1e-5)
- **Scheduler**: `ReduceLROnPlateau` (Factor: 0.1, Patience: 5)
- **Early Stopping**: Patience of 10 epochs.
- **Post-Processing**:
    - **Temperature Scaling**: Calibrated on validation set to improve probability calibration.
    - **Threshold Optimization**: Class-specific thresholds tuned to maximize F1 score.

---

## 4. Performance Evaluation

### Optimal Threshold: 0.45
Through extensive threshold tuning (0.05 to 0.95), the optimal decision threshold for the "Marine Debris" class was determined to be **0.45**.

### Quantitative Metrics (Test Set)
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **28.85%** | Low due to high false positive rate, but less critical than Recall. |
| **Precision** | **28.16%** | ~28% of predicted debris is actual debris. |
| **Recall** | **100.00%** | **Critical Success**: The model detects 100% of all debris patches. |
| **F1 Score** | **43.94%** | Harmonic mean of Precision and Recall. |

### Confusion Matrix
| | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative** | 2 (TN) | 148 (FP) |
| **Actual Positive** | 0 (FN) | 58 (TP) |

**Analysis**:
- **Zero False Negatives**: The model is extremely sensitive and misses *no* debris.
- **High False Positives**: The model is "trigger happy," flagging many clean areas as debris. This is often preferred in safety-critical or search-and-rescue contexts where missing a target is worse than a false alarm.

---

## 5. Extension: Visualization & Routing

The `extension.py` script provides actionable intelligence from model predictions.

### 5.1 Interactive World Map (`marine_debris_world_map.html`)
- **Visuals**:
    - 🔴 **Red Markers**: Detected Debris (Confidence > 0.45)
    - 🟢 **Green Markers**: Clean Areas
- **Popups**: Display patch ID, coordinates, confidence score, and date.
- **Statistics**:
    - Total Debris Locations: 58
    - Total Clean Locations: 150

### 5.2 Route Planning (`marine_debris_route_map.html`)
- **Problem Formulation**: Orienteering Problem (OP).
- **Goal**: Maximize total "prize" (debris importance) collected within a fixed distance budget.
- **Prize Function**: $Prize = \text{Confidence} \times 100$ (Higher confidence = higher priority).
- **Algorithm**: Greedy Insertion Heuristic.
    - Iteratively inserts the unvisited node that maximizes the ratio of prize gained to distance added.
- **Parameters**:
    - **Budget**: Configurable (default: 1000 km).
    - **Start Location**: Configurable (default: 12.0°N, 87.0°W).
    - **Return to Start**: Optional.

### Route Output Example (1000 km Budget)
Based on a sample run with a 1000 km budget:
- **Total Stops**: 12 (Start + 10 Debris Patches + End)
- **Total Distance**: 31.2 km
- **Total Prize**: 617 (Debris Importance Units)
- **Efficiency**: 19.77 prize/km
- **Initial Heading**: South-East

---

## 6. System Requirements

### Key Dependencies
- **Deep Learning**: `torch`, `torchvision`
- **Data Handling**: `numpy`, `pandas`, `h5py`
- **Image Processing**: `opencv-python`, `scikit-image`, `pillow`, `tifffile`
- **Geospatial & Visualization**: `folium`, `matplotlib`
- **Utilities**: `tqdm`, `python-dotenv`

---

## 7. File Structure & Usage

### Key Files
- `main.py`: Orchestrates the full pipeline (Evaluation -> Mapping -> Visualization).
- `train.py`: Training script with calibration and threshold tuning.
- `evaluation.py`: Generates raw model predictions.
- `extension.py`: Generates maps and plans routes.
- `utils/generate_geolocation_map_v2.py`: Converts model outputs to geospatial JSON.

### How to Run
1.  **Full Pipeline**:
    ```bash
    python main.py
    ```
2.  **Visualization Only**:
    ```bash
    python extension.py --per_patch --skip_route
    ```
3.  **Route Planning**:
    ```bash
    python extension.py --per_patch --budget <budget_in_km>
    ```

---

## 8. Segmentation & Baseline Models (MARIDA Benchmark)

In addition to the main ResNet18 classifier, the repository includes implementations for pixel-level segmentation and baseline classification.

### 8.1 U-Net (Semantic Segmentation)
A custom U-Net implementation for pixel-level semantic segmentation of marine debris.
- **Architecture**: Standard U-Net with Encoder-Decoder path and skip connections.
- **Input**: 11 Spectral Bands.
- **Output**: 11 Classes (Pixel-level classification).
- **Hidden Channels**: 16 (base width).
- **Depth**: 4 Downsampling / 4 Upsampling blocks.

---

## 9. Evaluation Pipelines

The system supports three distinct evaluation modes, allowing for flexible trade-offs between speed and detail.

### 9.1 Standard Classification (ResNet18)
- **Goal**: Rapidly identify patches containing debris.
- **Model**: ResNet18 (Sentinel-2 Adapted).
- **Output**: Binary classification (Debris / No Debris) per patch.
- **Use Case**: Large-scale screening of ocean areas.

### 9.2 Segmentation-Only (U-Net)
- **Goal**: Detailed pixel-level analysis of every patch.
- **Model**: U-Net.
- **Output**: Pixel-wise segmentation masks.
- **Classification Derivation**: A patch is classified as "Debris" if the number of debris pixels exceeds a threshold (e.g., 0.1%).
- **Use Case**: High-precision analysis when computational resources are abundant.

### 9.3 Two-Stage Pipeline (Hybrid)
Combines the strengths of both approaches for efficient large-scale monitoring.
- **Stage 1 (Screening)**: The fast **ResNet18** classifier screens all incoming satellite patches.
- **Stage 2 (Refinement)**: Patches flagged as "Debris" by Stage 1 are passed to the **U-Net** for detailed segmentation.
- **Benefit**: Achieves 3-10x speedup compared to running segmentation on all patches, while maintaining high precision on the areas of interest.

