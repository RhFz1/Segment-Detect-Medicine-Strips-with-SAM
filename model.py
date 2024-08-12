import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from typing import Tuple

class Model(nn.Module):
    
    def __init__(self, input_shape: Tuple = (224, 224)):

        super(Model, self).__init__()
        self.features = models.resnet18(pretrained=True)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.features(x))
