"""
Rain Prior Injection Module (RPIM)
Increases the importance of features within rain streak areas indicated by RLP.
Based on: "Learning Rain Location Prior for Nighttime Deraining" (ICCV 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature recalibration."""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature recalibration."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class RainPriorInjectionModule(nn.Module):
    """
    Rain Prior Injection Module (RPIM)
    Enhances features in rain streak regions using RLP guidance.
    """
    
    def __init__(self, feature_channels=64, reduction=16):
        super(RainPriorInjectionModule, self).__init__()
        
        # Feature enhancement guided by RLP
        self.rlp_conv = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        )
        
        # Channel attention for adaptive feature recalibration
        self.channel_attention = ChannelAttention(feature_channels, reduction)
        
        # Spatial attention for region-specific enhancement
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, rlp_map):
        """
        Args:
            features: Input features from encoder [B, C, H, W]
            rlp_map: Rain location prior map [B, 1, H, W]
        Returns:
            Enhanced features [B, C, H, W]
        """
        # Ensure RLP map matches feature spatial dimensions
        if rlp_map.shape[-2:] != features.shape[-2:]:
            rlp_map = F.interpolate(rlp_map, size=features.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        
        # Transform RLP map to feature space
        rlp_features = self.rlp_conv(rlp_map)
        
        # Apply channel attention
        ca_weight = self.channel_attention(features)
        features_ca = features * ca_weight
        
        # Apply spatial attention weighted by RLP
        sa_weight = self.spatial_attention(features)
        rlp_weighted_sa = sa_weight * rlp_map  # Focus on rain regions
        features_sa = features * rlp_weighted_sa
        
        # Fuse original and attention-enhanced features
        fused = torch.cat([features_ca, features_sa], dim=1)
        fused = self.fusion_conv(fused)
        
        # Combine with RLP features using gating
        gate_weight = self.gate(rlp_features)
        enhanced = fused + gate_weight * rlp_features
        
        # Residual connection
        output = features + enhanced
        
        return output


class MultiScaleRPIM(nn.Module):
    """
    Multi-scale Rain Prior Injection Module
    Applies RPIM at multiple feature scales for hierarchical enhancement.
    """
    
    def __init__(self, channels_list=[64, 128, 256], reduction=16):
        super(MultiScaleRPIM, self).__init__()
        
        self.rpim_modules = nn.ModuleList([
            RainPriorInjectionModule(channels, reduction)
            for channels in channels_list
        ])
        
    def forward(self, feature_list, rlp_map):
        """
        Args:
            feature_list: List of features at different scales [(B,C1,H1,W1), ...]
            rlp_map: Rain location prior map [B, 1, H, W]
        Returns:
            List of enhanced features
        """
        enhanced_features = []
        for features, rpim in zip(feature_list, self.rpim_modules):
            enhanced = rpim(features, rlp_map)
            enhanced_features.append(enhanced)
        
        return enhanced_features


if __name__ == "__main__":
    # Test RPIM module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Single-scale test
    rpim = RainPriorInjectionModule(feature_channels=64, reduction=16).to(device)
    features = torch.randn(2, 64, 64, 64).to(device)
    rlp_map = torch.rand(2, 1, 256, 256).to(device)  # Different size
    
    enhanced = rpim(features, rlp_map)
    print(f"Input features shape: {features.shape}")
    print(f"RLP map shape: {rlp_map.shape}")
    print(f"Enhanced features shape: {enhanced.shape}")
    
    # Multi-scale test
    multi_rpim = MultiScaleRPIM(channels_list=[64, 128, 256], reduction=16).to(device)
    feature_list = [
        torch.randn(2, 64, 128, 128).to(device),
        torch.randn(2, 128, 64, 64).to(device),
        torch.randn(2, 256, 32, 32).to(device)
    ]
    
    enhanced_list = multi_rpim(feature_list, rlp_map)
    print("\nMulti-scale test:")
    for i, (orig, enh) in enumerate(zip(feature_list, enhanced_list)):
        print(f"Scale {i}: {orig.shape} -> {enh.shape}")
