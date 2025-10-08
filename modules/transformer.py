"""
Bidirectional Multiscale Transformer
Processes features at multiple scales for complex rain pattern modeling.
Based on: NeRD-Rain (arXiv 2404.01547)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 attn_drop=0., proj_drop=0., dropout=0.):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ScaleSpecificTransformer(nn.Module):
    """
    Scale-specific Transformer branch with patch-based attention for memory efficiency.
    Processes image in patches to avoid huge attention matrices.
    """
    
    def __init__(self, dim, depth=4, num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, attn_drop=0., proj_drop=0., dropout=0., patch_size=8):
        super(ScaleSpecificTransformer, self).__init__()
        
        self.patch_size = patch_size
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, 
                           attn_drop, proj_drop, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: Input features [B, N, C] where N = H*W
        Returns:
            Output features [B, N, C]
        """
        # MEMORY OPTIMIZATION: Process in patches if N is too large
        B, N, C = x.shape
        
        # If sequence is too long (>16K tokens), use patch processing
        if N > 16384:  # 128×128
            # Infer H, W from N (assume square or use stored dimensions)
            H = W = int(math.sqrt(N))
            
            # Reshape to [B, H, W, C]
            x_spatial = x.view(B, H, W, C)
            
            # Process in non-overlapping patches
            P = self.patch_size
            pH = (H + P - 1) // P
            pW = (W + P - 1) // P
            
            # Pad if needed
            pad_h = pH * P - H
            pad_w = pW * P - W
            if pad_h > 0 or pad_w > 0:
                x_spatial = F.pad(x_spatial, (0, 0, 0, pad_w, 0, pad_h))
            
            # Split into patches: [B, pH, P, pW, P, C] -> [B*pH*pW, P*P, C]
            patches = x_spatial.view(B, pH, P, pW, P, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
            patches = patches.view(B * pH * pW, P * P, C)
            
            # Process each patch
            for block in self.blocks:
                patches = block(patches)
            patches = self.norm(patches)
            
            # Reconstruct: [B*pH*pW, P*P, C] -> [B, H, W, C]
            patches = patches.view(B, pH, pW, P, P, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
            x_spatial = patches.view(B, pH * P, pW * P, C)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                x_spatial = x_spatial[:, :H, :W, :]
            
            x = x_spatial.view(B, N, C)
        else:
            # Normal processing for small sequences
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
        
        return x


class BidirectionalMultiscaleTransformer(nn.Module):
    """
    Bidirectional Multiscale Transformer with unequal branches.
    Each branch processes features at a different scale.
    Memory-efficient with patch-based attention for large images.
    """
    
    def __init__(self, scales=[1, 2, 4], base_dim=64, depths=[2, 2, 2], 
                 num_heads=[4, 8, 16], mlp_ratio=4., patch_size=8):
        super(BidirectionalMultiscaleTransformer, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Scale-specific transformers with different capacities
        self.transformers = nn.ModuleList()
        self.dims = []
        
        for i, (scale, depth, heads) in enumerate(zip(scales, depths, num_heads)):
            dim = base_dim * (2 ** i)  # Increase capacity for smaller scales
            self.dims.append(dim)
            
            # Use larger patches for finer scales (more tokens)
            scale_patch_size = patch_size * (2 ** (self.num_scales - 1 - i))
            
            transformer = ScaleSpecificTransformer(
                dim=dim, 
                depth=depth, 
                num_heads=heads, 
                mlp_ratio=mlp_ratio,
                patch_size=scale_patch_size
            )
            self.transformers.append(transformer)
        
        # Downsampling and upsampling for scale conversion
        self.downsample_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        
        for i in range(self.num_scales - 1):
            # Downsample: increase channels, decrease spatial size
            self.downsample_convs.append(
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, 
                         stride=2, padding=1)
            )
            # Upsample: decrease channels, increase spatial size
            self.upsample_convs.append(
                nn.ConvTranspose2d(self.dims[i+1], self.dims[i], kernel_size=4,
                                  stride=2, padding=1)
            )
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Multi-scale features: List of [B, C_i, H_i, W_i]
        """
        B, C, H, W = x.shape
        
        # Create multi-scale inputs
        scale_inputs = [x]
        for i in range(1, self.num_scales):
            scale_factor = self.scales[i] / self.scales[i-1]
            downsampled = self.downsample_convs[i-1](scale_inputs[-1])
            scale_inputs.append(downsampled)
        
        # Forward pass through scale-specific transformers
        scale_features = []
        for i, (scale_input, transformer) in enumerate(zip(scale_inputs, self.transformers)):
            B, C_i, H_i, W_i = scale_input.shape
            
            # Reshape to [B, N, C] for transformer
            tokens = scale_input.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Process with transformer
            transformed = transformer(tokens)
            
            # Reshape back to [B, C, H, W]
            feature_map = transformed.transpose(1, 2).reshape(B, C_i, H_i, W_i)
            scale_features.append(feature_map)
        
        # Backward pass: information flows from fine to coarse
        for i in range(self.num_scales - 2, -1, -1):
            # Upsample from finer scale and add to current scale
            upsampled = self.upsample_convs[i](scale_features[i+1])
            scale_features[i] = scale_features[i] + upsampled
        
        return scale_features


if __name__ == "__main__":
    # Test Bidirectional Multiscale Transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BidirectionalMultiscaleTransformer(
        scales=[1, 2, 4],
        base_dim=64,
        depths=[2, 2, 2],  # Reduced for testing
        num_heads=[4, 8, 16],
        mlp_ratio=4.
    ).to(device)
    
    # Create dummy input
    x = torch.randn(2, 64, 128, 128).to(device)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    scale_features = model(x)
    
    print("\nMulti-scale features:")
    for i, feat in enumerate(scale_features):
        print(f"Scale {i}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
