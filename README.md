# Marine Debris Detection and Route Planning

## Overview

This project provides an automated pipeline for detecting marine debris using Sentinel-2 satellite imagery and deep learning. Beyond detection, it translates model predictions into actionable insights by mapping debris locations and planning optimal collection routes for cleanup vessels.

The system leverages a ResNet18 classifier to identify debris patches, maps them to specific geolocations using UTM tile data, and visualizes them on interactive maps. It also includes a route planning module that optimizes collection paths based on a travel budget.

## Key Features

- **Deep Learning Detection**: Uses a ResNet18 model trained on Sentinel-2 imagery to classify marine debris.
- **Geolocation Mapping**: Maps image patches to real-world coordinates (approximate UTM centers) with spatial distribution to avoid marker overlap.
- **Interactive Visualization**: Generates HTML maps showing debris locations, confidence scores, and clean areas.
- **Route Optimization**: Plans efficient collection routes for ships, maximizing debris collection within a specified distance budget.
- **Comprehensive Evaluation**: Supports both classification and segmentation-based evaluation metrics.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Marine-Debris
    ```

2.  **Set up the environment:**
    Ensure you have Python installed. You can install the required dependencies using `pip` or `uv`.

    ```bash
    # Using pip
    pip install -r requirements.txt
    ```

    *Note: Key dependencies include `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `folium`, `geopandas`, and `shapely`.*

## Quick Start

To run the entire pipeline from evaluation to visualization, simply execute the `main.py` script:

```bash
python main.py
```

This script automates the following steps:
1.  **Evaluation**: Runs `evaluation.py` to generate predictions on the test set.
2.  **Mapping**: Runs `utils/generate_geolocation_map_v2.py` to create a geolocation JSON map from predictions.
3.  **Visualization**: Runs `extension.py` to generate the interactive world map.
4.  **Routing**: Runs `extension.py` again to generate the optimal route map.

**Output Files:**
- `marine_debris_world_map.html`: Interactive map showing all test patches.
- `marine_debris_route_map.html`: Interactive map showing the planned collection route.

## Project Structure

```
Marine-Debris/
├── docs/                           # Documentation and reports
├── models/                         # Model definitions (ResNet18, etc.)
├── utils/                          # Utility scripts (dataset, geolocation, etc.)
├── checkpoints/                    # Saved model checkpoints
├── evaluation.py                   # Script for model evaluation
├── extension.py                    # Visualization and route planning script
├── main.py                         # Main pipeline orchestration script
├── train.py                        # Training script
├── test_data.h5                    # Test dataset
├── test_patch_predictions.json     # Generated per-patch predictions
├── debris_geolocation_map.json     # Generated geolocation mapping
└── README.md                       # Project documentation
```

## Detailed Usage

### 1. Training
To train the model on your dataset:
```bash
python train.py
```
*Note: Ensure your training data (`train_data.h5`) is present.*

### 2. Evaluation
To evaluate the model and generate raw predictions:
```bash
python evaluation.py
```

### 3. Geolocation Mapping
To convert model predictions into a geolocated format:
```bash
python utils/generate_geolocation_map_v2.py --format detailed
```
This generates `debris_geolocation_map.json` and `test_patch_predictions.json`.

### 4. Visualization & Routing
To visualize the results and plan routes manually:
```bash
# Generate World Map (Visualization only)
python extension.py --per_patch --skip_route

# Generate Route Map (with 5000km budget)
python extension.py --per_patch --budget 5000
```

## Results

- **Detection**: The optimized model achieves an F1 score of **44.5%** and a recall of **91.4%** for marine debris detection.
- **Routing**: The system successfully plans realistic collection routes (e.g., ~31.2 km for the test set) connecting high-confidence debris patches.

For more details, please refer to the [Documentation](docs/documentation.md).

## License

[License Information]
