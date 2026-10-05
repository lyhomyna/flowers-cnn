import torch
import torch.nn as nn
import torch.optim as optim

class FlowerCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerCNN, self).__init__()

        # feature extraction (convolutional part)
        self.features = nn.Sequential(
            # 1st conv block
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd Conv block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # the input size 32 * 32 * 32 assumes input images are resized to 128
            # 128 -> (MaxPool1) -> 64 -> (MaxPool2) -> 32
            # 32 channels * 32 height * 32 width
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            # output layer: 5 classes (softmax handled by loss function)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FlowerCNNDropout(nn.Module):
    """Базова архітектура + Dropout перед вихідним шаром (п.8 завдання)."""
    def __init__(self, num_classes=5, dropout_p=0.5):
        super(FlowerCNNDropout, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FlowerCNNBatchNorm(nn.Module):
    """Базова архітектура + Batch Normalization між Conv і ReLU (п.9 завдання)."""
    def __init__(self, num_classes=5):
        super(FlowerCNNBatchNorm, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

