"""
NeRD-Rain Branch - Daytime Deraining Branch
Bidirectional Multiscale Transformer with Implicit Neural Representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.transformer import BidirectionalMultiscaleTransformer
from modules.inr import ImplicitNeuralRepresentation, IntraScaleSharedEncoder


class NeRDRainBranch(nn.Module):
    """
    NeRD-Rain Branch for daytime deraining.
    Uses multiscale Transformer and INR for complex rain pattern modeling.
    """
    
    def __init__(self, in_channels=3, base_dim=64, scales=[1, 2, 4]):
        super(NeRDRainBranch, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Initial feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Bidirectional Multiscale Transformer (memory-efficient configuration)
        self.transformer = BidirectionalMultiscaleTransformer(
            scales=scales,
            base_dim=base_dim,
            depths=[2, 2, 2],  # Reduced from [4,4,4] for memory efficiency
            num_heads=[4, 8, 16],
            mlp_ratio=4.,
            patch_size=8  # Patch-based attention for large images
        )
        
        # Implicit Neural Representations for each scale
        self.inr_modules = nn.ModuleList([
            ImplicitNeuralRepresentation(feature_dim=base_dim * (2 ** i))
            for i in range(self.num_scales)
        ])
        
        # Intra-scale shared encoders (closed-loop)
        # Each encoder connects scale i (lower) to scale i+1 (upper)
        self.shared_encoders = nn.ModuleList([
            IntraScaleSharedEncoder(
                lower_dim=base_dim * (2 ** i),
                upper_dim=base_dim * (2 ** (i + 1))
            )
            for i in range(self.num_scales - 1)
        ])
        
        # Feature reconstruction to image space
        self.reconstruction_heads = nn.ModuleList()
        for i in range(self.num_scales):
            dim = base_dim * (2 ** i)
            self.reconstruction_heads.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim // 2, in_channels, kernel_size=3, padding=1)
                )
            )
        
        # Multi-scale fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * self.num_scales, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input rainy image [B, 3, H, W]
        Returns:
            output: Derained image [B, 3, H, W]
            scale_outputs: List of outputs at different scales
            features: Multi-scale features for fusion
        """
        B, C, H, W = x.shape
        
        # Extract initial features
        features = self.feature_extractor(x)
        
        # Bidirectional multiscale transformer
        transformer_features = self.transformer(features)
        
        # Apply INR to each scale
        inr_features = []
        for i, (feat, inr) in enumerate(zip(transformer_features, self.inr_modules)):
            inr_feat = inr(feat)
            inr_features.append(inr_feat)
        
        # Closed-loop framework with intra-scale shared encoders
        enhanced_features = [inr_features[0]]
        for i in range(self.num_scales - 1):
            enc_lower, enh_upper = self.shared_encoders[i](
                enhanced_features[-1], 
                inr_features[i + 1]
            )
            enhanced_features[-1] = enc_lower  # Update lower scale
            enhanced_features.append(enh_upper)  # Add enhanced upper scale
        
        # Reconstruct images at each scale
        scale_outputs = []
        for i, (feat, head) in enumerate(zip(enhanced_features, self.reconstruction_heads)):
            out = head(feat)
            # Upsample to original resolution
            if out.shape[-2:] != (H, W):
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            scale_outputs.append(out)
        
        # Fuse multi-scale outputs
        fused = torch.cat(scale_outputs, dim=1)
        rain_residual = self.fusion_conv(fused)
        
        # Residual learning
        output = x - rain_residual
        
        # Prepare features for fusion with other branches
        multi_scale_features = {
            f'scale{i}': feat for i, feat in enumerate(enhanced_features)
        }
        
        return output, scale_outputs, multi_scale_features


if __name__ == "__main__":
    # Test NeRD-Rain Branch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NeRDRainBranch(in_channels=3, base_dim=64, scales=[1, 2, 4]).to(device)
    
    # Create dummy input
    x = torch.randn(2, 3, 128, 128).to(device)  # Smaller size for testing
    
    # Forward pass
    output, scale_outputs, features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nScale outputs:")
    for i, out in enumerate(scale_outputs):
        print(f"  Scale {i}: {out.shape}")
    print(f"\nMulti-scale features:")
    for key, feat in features.items():
        print(f"  {key}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
