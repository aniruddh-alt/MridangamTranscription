"""CNN-RNN model architecuter for mridangam transcription"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# using a GRU layer for RNN
