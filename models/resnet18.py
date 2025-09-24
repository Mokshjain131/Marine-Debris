import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import math

# Custom ResNet
class ResNetSentinel(nn.Module):
    def __init__(self, num_bands=11, num_classes=15, pretrained=False, freeze_backbone=False):
        super(ResNetSentinel, self).__init__()

        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Store original conv1 weights before modification
        if pretrained:
            original_conv1_weight = self.model.conv1.weight.data.clone()

        # Modify the first convolutional layer to accept num_bands instead of 3
        self.model.conv1 = nn.Conv2d(
            in_channels=num_bands,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False
        )

        # Initialize weights properly
        if pretrained and num_bands != 3:
            self._init_conv1_weights_improved(self.model.conv1, original_conv1_weight, num_bands)
        elif not pretrained:
            # Standard Kaiming initialization for training from scratch
            nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer with dropout
        dropout = 0.4
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def _init_conv1_weights_improved(self, new_conv, pretrained_weights, num_bands):
        """
        Improved weight initialization for multi-spectral adaptation of ImageNet pretrained weights
        """
        with torch.no_grad():
            # Get the pretrained RGB weights (64, 3, 7, 7)
            out_channels, in_channels_rgb, kh, kw = pretrained_weights.shape

            if num_bands <= 3:
                # If we have 3 or fewer bands, just use subset of pretrained weights
                new_conv.weight.data = pretrained_weights[:, :num_bands, :, :]
            else:
                # Method 1: Repeat and average strategy
                # This works better than random initialization for additional channels

                # Initialize all weights with pretrained RGB weights (repeated)
                for i in range(num_bands):
                    rgb_idx = i % 3  # Cycle through RGB channels
                    new_conv.weight.data[:, i, :, :] = pretrained_weights[:, rgb_idx, :, :]

                # Method 2: For bands beyond RGB, use spectral-aware initialization
                if num_bands > 3:
                    # For additional bands (like NIR, SWIR), use average of RGB with slight perturbation
                    rgb_avg = pretrained_weights.mean(dim=1, keepdim=True)  # Average across RGB channels

                    # Apply the average with small random perturbations for bands 4+
                    for i in range(3, num_bands):
                        # Use RGB average as base, add small noise for diversity
                        perturbation = torch.randn_like(rgb_avg) * 0.1 * pretrained_weights.std()
                        new_conv.weight.data[:, i, :, :] = (rgb_avg + perturbation).squeeze(1)

                # Normalize to maintain similar activation magnitude
                # Scale weights to preserve the magnitude similar to original
                original_std = pretrained_weights.std()
                new_std = new_conv.weight.data.std()
                if new_std > 0:
                    new_conv.weight.data *= (original_std / new_std)

        print(f"Initialized conv1 weights: {num_bands} bands using ImageNet RGB pretrained weights")
        print(f"Original RGB weight std: {pretrained_weights.std():.6f}, New weight std: {new_conv.weight.data.std():.6f}")

    def forward(self, x):
        return self.model(x)

# Instantiate the model
num_bands = 11 # Our dataset has 11 spectral bands
num_classes = 15 # Number of output classes

model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=True, freeze_backbone=True)

# Select a device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training Setup
criterion = nn.BCEWithLogitsLoss() # MARIDA is multi-label
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Forward Check
dummy = torch.randn(4, num_bands, 256, 256).to(device) # (batch_size, channels, height, width)
out = model(dummy)
print("Output shape: ", out.shape) # (4, num_classes)