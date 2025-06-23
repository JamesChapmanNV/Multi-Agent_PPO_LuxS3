
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
from typing import List, Tuple, Dict
from MAPPO_obs import (
    discovery_ch_idx,
    reward_ch_idx,
    astroid_ch_idx,
    tile_energy_ch_idx,
    friendly_ch_idx,
    friendly_energy_ch_idx,
    enemy_ch_idx,
    enemy_energy_ch_idx,
    relic_ch_idx,
)


class MAPPO_model(nn.Module):
    """
    uses the LuxConvNet backbone
      - use_coord_channels: If True, three normalized coordinate channels are added.
      - use_center_region_extraction_branch: If True, a branch extracts a central 5x5 region from the input.
      
    """
    def __init__(self, input_channels, output_dim, init_orthogonal=True, 
                 use_coord_channels=False, use_center_region_extraction_branch=False, 
                 use_nearby_features=False):
        # If coordinate channels are used, increase the input_channels by 3.
        if use_coord_channels:
            input_channels += 3
        super(MAPPO_model, self).__init__()
        self.use_coord_channels = use_coord_channels
        self.use_center_region_extraction_branch = use_center_region_extraction_branch
        self.use_nearby_features = use_nearby_features
        self.center_feature_dim = 0
        self.nearby_features_dim = 0
        # convolutions
        self.convnet = LuxConvNet(input_channels=input_channels, init_orthogonal=init_orthogonal)
        # center region (channels,5x5) 
        if self.use_center_region_extraction_branch:
            self.center_branch_conv = nn.Conv2d(input_channels - 3, 128, kernel_size=3, padding=1)
            self.center_branch_relu = nn.ReLU(inplace=True)
            self.center_branch_pool = nn.AdaptiveAvgPool2d((2, 2)) 
            self.center_branch_fc   = nn.Linear(512, 128)
            self.center_feature_dim = 128 
        # center left down right up features 5 channels + 1 (ship energy) 
        if self.use_nearby_features:
            self.nearby_features_dim = 26
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear((512 + self.center_feature_dim + self.nearby_features_dim), 256) # 512 + 128 + 26 = 666
        self.fc4 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)
        #
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # weights
        if init_orthogonal:
            self.apply(init_module_weights)
    
    def forward(self, state, additional_features=None):
        # If coordinate channels are enabled, add them.
        if self.use_coord_channels:
            state = add_coord_channels(state)
        # center 5x5 region
        if self.use_center_region_extraction_branch:
            # indices 15:20 yield a 5x5 crop
            center_region = state[:, :, 15:20, 15:20]
            channels = list(range(center_region.shape[1]))
            channels = [ch for ch in channels if ch not in (tile_energy_ch_idx, discovery_ch_idx, friendly_energy_ch_idx)]
            center_region = center_region[:, channels, :, :]
            center_features = self.center_branch_conv(center_region) 
            center_features = self.center_branch_relu(center_features)
            center_features = self.center_branch_pool(center_features)
            center_features = torch.flatten(center_features, 1)
            center_features = self.center_branch_fc(center_features)
            # Use center branch output as additional features.
            if additional_features is None:
                additional_features = center_features
            else:
                additional_features = torch.cat([additional_features, center_features], dim=1)
        # (center, left, down, right, up) of select channels + 1 ship energy = 26 
        if self.use_nearby_features:
            nearby_features = extract_nearby_features(state)
            if additional_features is None:
                additional_features = nearby_features
            else:
                additional_features = torch.cat([additional_features, nearby_features], dim=1)
        # entire state through the main convnet
        main_features = self.convnet(state)     # Expected shape: (batch, 256*3*3) = (batch, 2304)
        x = self.relu(self.fc1(main_features))  # (batch, 1024)
        x = self.tanh(self.fc2(x))              # (batch, 512)
        if additional_features is not None:     # (512 + 128 + 26) = (batch, 666)
            x = torch.cat([x, additional_features], dim=1)
        x = self.tanh(self.fc3(x))              # (batch, 256)
        x = self.tanh(self.fc4(x))              # (batch, 128)
        x = self.out(x)                         # (batch, num_actions)
        return x

class LuxConvNet(nn.Module):
    """
    Convolutional network backbone.
    """
    def __init__(self, input_channels=9, init_orthogonal=True):
        super(LuxConvNet, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        self.initial_gn = nn.GroupNorm(num_groups=8, num_channels=64)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResidualBlock(64, 128, stride=1, init_orthogonal=init_orthogonal)
        self.block2 = ResidualBlock(128, 128, stride=2, init_orthogonal=init_orthogonal)
        self.block3 = ResidualBlock(128, 256, stride=1, init_orthogonal=init_orthogonal)
        
        self.global_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        if init_orthogonal:
            self.apply(init_module_weights)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_gn(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

class ResidualBlock(nn.Module):
    """
    A basic residual block 
    """
    def __init__(self, in_channels, out_channels, stride=1, init_orthogonal=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut to match dimensions if needed.
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        if init_orthogonal:
            self.apply(init_module_weights)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += identity
        out = self.relu(out)
        return out


# TODO:dropout   

    
def init_module_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
def add_coord_channels(x):
    """
    generates and add 3 coordinate channels:
      - X coordinates (normalized 0 to 1)
      - Y coordinates (normalized 0 to 1)
      - Radial distance from center (normalized)
    """
    batch_size, _, H, W = x.shape
    device = x.device
    # Create coordinate grids.
    x_coords = torch.linspace(0, 1, steps=W, device=device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
    y_coords = torch.linspace(0, 1, steps=H, device=device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
    # Compute radial distances from the center (0.5, 0.5)
    radial = torch.sqrt((x_coords - 0.5) ** 2 + (y_coords - 0.5) ** 2)
    coord_channels = torch.cat([x_coords, y_coords, radial], dim=1)
    return torch.cat([x, coord_channels], dim=1)

def extract_nearby_features(state):
    """
    Extracts features from five channels at the center and adjacent tiles (left, down, right, up)
    and appends the energy value from ship

    Returns:
        torch.Tensor: shape [batch, 26] 
    """
    batch_size, _, H, W = state.shape
    center_h = H // 2
    center_w = W // 2

    # Define the five offsets: center, left, down, right, up.
    offsets = torch.tensor([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], device=state.device)
    # Define the channels from which to extract features.
    channel_indices = torch.tensor([reward_ch_idx, astroid_ch_idx, friendly_ch_idx, enemy_ch_idx, enemy_energy_ch_idx],
                                   device=state.device)
    # Compute the row and column indices for each offset.
    row_indices = center_h + offsets[:, 0]  # Shape: [5]
    col_indices = center_w + offsets[:, 1]  # Shape: [5]

    # index tensors 
    batch_idx = torch.arange(batch_size, device=state.device).view(-1, 1, 1)   # Shape: [batch, 1, 1]
    channel_idx = channel_indices.view(1, -1, 1)                               # Shape: [1, 5, 1]
    row_idx = row_indices.view(1, 1, -1)                                       # Shape: [1, 1, 5]
    col_idx = col_indices.view(1, 1, -1)                                       # Shape: [1, 1, 5]
    # indexing to extract features in one go 
    features_tensor = state[batch_idx, channel_idx, row_idx, col_idx]
    features_tensor = features_tensor.view(batch_size, -1)
    # Extract the center value from channel index 8.
    last_channel_feature = state[:, 8, center_h, center_w].unsqueeze(1)
    return torch.cat([features_tensor, last_channel_feature], dim=1)
