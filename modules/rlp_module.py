"""
Rain Location Prior (RLP) Module
Implicitly learns rain streak location information through recurrent residual learning.
Based on: "Learning Rain Location Prior for Nighttime Deraining" (ICCV 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentResidualBlock(nn.Module):
    """Recurrent residual block for iteratively refining rain location prior."""
    
    def __init__(self, in_channels=64, hidden_channels=64):
        super(RecurrentResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels + hidden_channels, hidden_channels, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, h):
        """
        Args:
            x: Current feature map [B, C, H, W]
            h: Hidden state from previous iteration [B, C, H, W]
        Returns:
            Updated hidden state
        """
        combined = torch.cat([x, h], dim=1)
        out = self.relu(self.conv1(combined))
        out = self.conv2(out)
        h_new = h + out  # Residual connection
        return h_new


class RainLocationPriorModule(nn.Module):
    """
    Rain Location Prior (RLP) Module
    Learns to identify rain streak locations through recurrent processing.
    """
    
    def __init__(self, in_channels=3, feature_channels=64, num_iterations=3):
        super(RainLocationPriorModule, self).__init__()
        
        self.num_iterations = num_iterations
        
        # Initial feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Recurrent residual blocks
        self.recurrent_block = RecurrentResidualBlock(feature_channels, feature_channels)
        
        # Rain location map generation
        self.rlp_head = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()  # Output probability map [0, 1]
        )
        
    def forward(self, x):
        """
        Args:
            x: Input rainy image [B, 3, H, W]
        Returns:
            rlp_map: Rain location prior map [B, 1, H, W], values in [0, 1]
            features: Intermediate features [B, C, H, W]
        """
        # Extract initial features
        features = self.feature_extractor(x)
        
        # Initialize hidden state
        h = torch.zeros_like(features)
        
        # Recurrent refinement
        for _ in range(self.num_iterations):
            h = self.recurrent_block(features, h)
        
        # Generate rain location prior map
        rlp_map = self.rlp_head(h)
        
        return rlp_map, h


class RainLocationPriorLoss(nn.Module):
    """
    Auxiliary loss for training RLP module.
    Encourages the model to correctly identify rain locations.
    """
    
    def __init__(self, weight=0.1):
        super(RainLocationPriorLoss, self).__init__()
        self.weight = weight
        self.bce = nn.BCELoss()
        
    def forward(self, rlp_map, rain_mask=None, input_img=None, gt_img=None):
        """
        Args:
            rlp_map: Predicted rain location prior [B, 1, H, W]
            rain_mask: Ground truth rain mask if available [B, 1, H, W]
            input_img: Rainy image [B, 3, H, W]
            gt_img: Clean image [B, 3, H, W]
        Returns:
            loss: RLP auxiliary loss
        """
        if rain_mask is not None:
            # Supervised: use ground truth rain mask
            loss = self.bce(rlp_map, rain_mask)
        else:
            # Unsupervised: use residual between rainy and clean as pseudo label
            with torch.no_grad():
                residual = torch.abs(input_img - gt_img).mean(dim=1, keepdim=True)
                # Normalize to [0, 1]
                residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            loss = self.bce(rlp_map, residual)
        
        return self.weight * loss


if __name__ == "__main__":
    # Test RLP module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RainLocationPriorModule(in_channels=3, feature_channels=64, num_iterations=3).to(device)
    
    # Create dummy input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    # Forward pass
    rlp_map, features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"RLP map shape: {rlp_map.shape}")
    print(f"Features shape: {features.shape}")
    print(f"RLP map range: [{rlp_map.min().item():.4f}, {rlp_map.max().item():.4f}]")
    
    # Test loss
    gt_img = torch.randn(2, 3, 256, 256).to(device)
    loss_fn = RainLocationPriorLoss(weight=0.1)
    loss = loss_fn(rlp_map, input_img=x, gt_img=gt_img)
    print(f"RLP loss: {loss.item():.4f}")
