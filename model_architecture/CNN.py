import torch
import torch.nn as nn
import torch.optim as optim


class MridangamCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
            
        )
