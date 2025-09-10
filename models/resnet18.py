import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Custom ResNet
class ResNetSentinel(nn.Module):
    def __init__(self, num_bands=11, num_classes=15, pretrained=False):
        super(ResNetSentinel, self).__init__()

        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18.Weights.IMAGENET1K_V1 if pretrained else None)

        # Modify the first convolutional layer to accept 11 num_bands instead of 3
        self.model.conv1 = nn.Conv2d(
            in_channels=num_bands,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False
        )

        # Modify the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate the model
num_bands = 11 # Our dataset has 11 spectral bands
num_classes = 15 # Number of output classes

model = ResNetSentinel(num_bands=num_bands, num_classes=num_classes, pretrained=False)

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