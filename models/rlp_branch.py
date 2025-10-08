"""
RLP Branch - Nighttime Deraining Branch
Combines RLP module and RPIM for effective nighttime rain removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rlp_module import RainLocationPriorModule
from modules.rpim import MultiScaleRPIM


class EncoderBlock(nn.Module):
    """Encoder block for feature extraction."""
    
    def __init__(self, in_channels, out_channels, stride=2):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block for feature reconstruction."""
    
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', 
                                    align_corners=False)
        # After concatenation, channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Ensure spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class RLPBranch(nn.Module):
    """
    RLP Branch for nighttime deraining.
    Uses Rain Location Prior and RPIM for focused rain removal.
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super(RLPBranch, self).__init__()
        
        # Rain Location Prior Module
        self.rlp_module = RainLocationPriorModule(
            in_channels=in_channels,
            feature_channels=base_channels,
            num_iterations=3
        )
        
        # Multi-scale encoder
        self.enc1 = EncoderBlock(in_channels, base_channels, stride=1)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale RPIM
        self.rpim = MultiScaleRPIM(
            channels_list=[base_channels, base_channels * 2, base_channels * 4],
            reduction=16
        )
        
        # Decoder with proper channel dimensions
        # dec3: bottleneck (256) + e3_enhanced (256) -> 128
        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2, scale_factor=2)
        # dec2: d3 (128) + e2_enhanced (128) -> 64
        self.dec2 = DecoderBlock(base_channels * 2, base_channels * 2, base_channels, scale_factor=2)
        # dec1: d2 (64) + e1_enhanced (64) -> 3
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input rainy image [B, 3, H, W]
        Returns:
            output: Derained image [B, 3, H, W]
            rlp_map: Rain location prior [B, 1, H, W]
            features: Multi-scale features for fusion
        """
        original_size = x.shape[2:]
        
        # Get rain location prior
        rlp_map, rlp_features = self.rlp_module(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        bottleneck = self.bottleneck(e3)
        
        # Apply RPIM to encoder features
        enhanced_features = self.rpim([e1, e2, e3], rlp_map)
        e1_enhanced, e2_enhanced, e3_enhanced = enhanced_features
        
        # Decoder with skip connections
        d3 = self.dec3(bottleneck, e3_enhanced)
        d2 = self.dec2(d3, e2_enhanced)
        
        # Ensure spatial dimensions match before concatenation for dec1
        if d2.shape[2:] != e1_enhanced.shape[2:]:
            d2 = F.interpolate(d2, size=e1_enhanced.shape[2:], mode='bilinear', align_corners=False)
        
        d1 = self.dec1(torch.cat([d2, e1_enhanced], dim=1))
        
        # Ensure output matches input size
        if d1.shape[2:] != original_size:
            d1 = F.interpolate(d1, size=original_size, mode='bilinear', align_corners=False)
        
        # Residual learning: predict rain, subtract from input
        rain = d1
        output = x - rain
        
        # Return features at different scales for fusion
        multi_scale_features = {
            'enc1': e1_enhanced,
            'enc2': e2_enhanced,
            'enc3': e3_enhanced,
            'bottleneck': bottleneck
        }
        
        return output, rlp_map, multi_scale_features


if __name__ == "__main__":
    # Test RLP Branch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RLPBranch(in_channels=3, base_channels=64).to(device)
    
    # Create dummy input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    # Forward pass
    output, rlp_map, features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"RLP map shape: {rlp_map.shape}")
    print("\nMulti-scale features:")
    for key, feat in features.items():
        print(f"  {key}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
