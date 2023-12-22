import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, output_features: int = 7):
        super(SimpleCNN, self).__init__()
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(1),
            nn.Linear(1536, output_features)
        )

    def forward(self, x):
        return self.conv_stack(x)    


class ComplexCNN(nn.Module):
    def __init__(self, output_features: int = 7):
        super(ComplexCNN, self).__init__()
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(1),
            nn.Linear(512, output_features)
        )

    def forward(self, x):
        return self.conv_stack(x)


class ComplexCNNMeta(nn.Module):
    def __init__(self, output_features: int = 7):
        super(ComplexCNNMeta, self).__init__()
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(1),
            # nn.Linear(512, output_features)
        )
        
        self.meta_stack = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU()
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(640, output_features)
        )

    def forward(self, x, p):
        return self.linear_stack(torch.cat((self.conv_stack(x), self.meta_stack(p)), 1))         