import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN(nn.Module):
    def __init__(self, num_bands=11, num_classes=15):
        super(ShallowCNN, self).__init__()

        # More gradual feature extraction
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual-style blocks
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)

        # Less aggressive pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Progressive dropout
        self.dropout_light = nn.Dropout(0.2)
        self.dropout_medium = nn.Dropout(0.3)
        self.dropout_heavy = nn.Dropout(0.5)

        # Improved classifier
        self.fc1 = nn.Linear(256 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))

        # Block 1: Feature refinement
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.dropout_light(x)
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        # Block 2: Higher level features
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.dropout_medium(x)
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        # Adaptive pooling preserves more spatial info
        x = self.adaptive_pool(x)  # [batch, 256, 8, 8]

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_heavy(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_medium(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_light(x)
        x = self.classifier(x)

        return x
