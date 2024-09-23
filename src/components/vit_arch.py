import torch
import torch.nn as nn
import warnings
from torchvision import models
from typing import Tuple

warnings.filterwarnings("ignore")

class Model(nn.Module):
    
    def __init__(self, input_shape: Tuple = (224, 224)):

        super(Model, self).__init__()
        self.features = models.vit_b_16(pretrained=True)
        num_ftrs = 768
        self.features.heads = nn.Sequential(nn.Linear(num_ftrs, 1))
    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.features(x))
