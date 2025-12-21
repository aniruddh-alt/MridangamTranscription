import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAttentivePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Apply frequency attentive pooling to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, freq, time).
        Returns:
            torch.Tensor: Pooled tensor of shape (batch, channels).
        """
        x_mean = x.mean(dim=-1) # (batch, channels, freq)
        attentive_pooling = self.attention(x_mean)  # (batch, 1, freq)
        att_weights = torch.softmax(attentive_pooling, dim=-1)  # Normalize weights
        x_weighted = x_mean * att_weights  # (batch, channels, freq)
        return x_weighted.sum(dim=-1)  # (batch, channels)    

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W)  # B x C' x N
        proj_key = self.key(x).view(B, -1, H * W)       # B x C' x N
        proj_value = self.value(x).view(B, -1, H * W)   # B x C x N

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # B x N x N
        attention = F.softmax(energy / (proj_query.size(1) ** 0.5), dim=-1)  # Scaled dot-product

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(B, C, H, W)
        out = self.gamma * out + x  # Residual connection
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attn = SelfAttention2D(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attn(out)
        out += residual
        return self.relu(out)

class MridangamCNN(nn.Module):
    def __init__(self, n_mels=128, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.6),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.6),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.8),
            
            ResidualBlock(channels=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.8),
        )
        
        self.attention_pooling = FrequencyAttentivePooling(in_channels=128)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention_pooling(x)
        x = self.classifier(x)
        return x

