import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture matching the resnet_best.pth checkpoint
    This is likely the original "ResNet" model used in training
    """
    def __init__(self, num_bands=11, num_classes=15):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_bands, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # Assuming input size after pooling: 256 channels * (reduced spatial dims)
        # This might need adjustment based on actual input image size
        self.fc1 = nn.Linear(256, 128)  # Will be adjusted based on checkpoint
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Global average pooling or adaptive pooling to handle variable input sizes
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x