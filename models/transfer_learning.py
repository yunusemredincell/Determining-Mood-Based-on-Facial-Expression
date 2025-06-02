import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


class FacialExpressionEfficientNN(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, freeze_backbone=True):
        super(FacialExpressionEfficientNN, self).__init__()

        self.backbone = models.efficientnet_b1(pretrained=pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def optimizer(self, learning_rate):
        return optim.SGD(
            self.backbone.classifier.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )


class FacialExpressionResnetNN(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, freeze_backbone=True):
        super(FacialExpressionResnetNN, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def optimizer(self, learning_rate):
        return optim.SGD(
            self.backbone.fc.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
