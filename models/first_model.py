import torch.nn as nn
import torch.nn.functional as F


class FacialExpressionNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionNN, self).__init__()

        # First convolutional block - Reduced channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.3)  # Increased dropout

        # Second convolutional block
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )  # Reduced from 128 to 64
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)

        # Third convolutional block
        self.conv5 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )  # Reduced from 256 to 128
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.4)  # Increased dropout

        # Fourth convolutional block - Simplified
        self.conv7 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1
        )  # Reduced from 512 to 256
        self.bn7 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.4)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Simplified fully connected layers
        self.fc1 = nn.Linear(256, 128)  # Reduced from 512 to 128
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.6)  # Increased dropout

        self.fc2 = nn.Linear(
            128, num_classes
        )  # Direct to output, removed intermediate layer

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Fourth block
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout4(x)

        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        x = self.fc2(x)

        return x
