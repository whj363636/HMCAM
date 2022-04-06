import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mnistnet']


class mnistnet(nn.Module):
    def __init__(self, num_classes):
        super(mnistnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def mnistnet(num_classes):
    """
    Constructs a Wide Residual Networks.
    """
    model = mnistnet(num_classes=num_classes)
    return model