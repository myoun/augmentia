import torch
import torch.nn as nn
import torch.nn.functional as F

SPATIAL_CHANNELS = 64
GLOBAL_FEATURES = 128
ACTION_SPACE_SIZE = 32768

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class AugmentNet(nn.Module):
    def __init__(self, num_blocks=6, hidden_dim=128):
        super().__init__()
        self.in_conv = nn.Conv2d(SPATIAL_CHANNELS, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(hidden_dim)
        
        self.global_proj = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SPACE_SIZE)
        
        # Value Head
        self.value_conv = nn.Conv2d(hidden_dim, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, spatial, global_feat, mask=None):
        # spatial: (B, 64, 8, 8)
        # global_feat: (B, 128)
        B = spatial.shape[0]
        x = F.relu(self.in_bn(self.in_conv(spatial)))
        
        # Global feature fusion across spatial channels
        g = self.global_proj(global_feat).view(B, -1, 1, 1)
        x = x + g
        
        for block in self.blocks:
            x = block(x)
            
        # Policy output
        p = F.relu(self.policy_bn(self.policy_conv(x))).view(B, -1)
        logits = self.policy_fc(p)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)
        policy = F.softmax(logits, dim=-1)
        
        # Value output
        v = F.relu(self.value_bn(self.value_conv(x))).view(B, -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value, logits

