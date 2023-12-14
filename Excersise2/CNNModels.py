import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, output_features: int = 7):
        super(SimpleCNN, self).__init__()
        
        self.stack = nn.Sequential(
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(1), # Check parameter
                nn.Linear(1536, output_features)
            )   

    def forward(self, x):
        return self.stack(x)
    
