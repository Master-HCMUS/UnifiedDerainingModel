"""
Adaptive Fusion Module
Dynamically combines RLP and NeRD-Rain branch features based on illumination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross-attention between two feature sets."""
    
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N1, C]
            key_value: [B, N2, C]
        Returns:
            Output: [B, N1, C]
        """
        B, N1, C = query.shape
        N2 = key_value.shape[1]
        
        q = self.q_proj(query).reshape(B, N1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        out = self.out_proj(out)
        
        return out


class FeatureAlignmentModule(nn.Module):
    """Aligns features from different branches to the same dimension."""
    
    def __init__(self, in_channels_list, out_channels):
        super(FeatureAlignmentModule, self).__init__()
        
        self.alignment_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])
        
    def forward(self, feature_list):
        """
        Args:
            feature_list: List of features with different channels
        Returns:
            Aligned features with same channels
        """
        aligned = [conv(feat) for feat, conv in zip(feature_list, self.alignment_convs)]
        return aligned


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive Fusion Module that combines RLP and NeRD-Rain outputs.
    Uses illumination-aware gating and cross-attention for optimal blending.
    """
    
    def __init__(self, in_channels=3, feature_dim=64):
        super(AdaptiveFusionModule, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Feature extraction from branch outputs
        self.rlp_feat_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )
        
        self.nerd_feat_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )
        
        # Cross-attention for feature interaction
        self.cross_attn_rlp2nerd = CrossAttention(feature_dim, num_heads=8)
        self.cross_attn_nerd2rlp = CrossAttention(feature_dim, num_heads=8)
        
        # Illumination-conditioned gating
        self.gate_generator = nn.Sequential(
            nn.Linear(2, 64),  # Input: [night_weight, day_weight]
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim * 2),  # Output: gates for both branches
            nn.Sigmoid()
        )
        
        # Spatial gating (pixel-wise)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, kernel_size=1),  # 2 channels: [rlp_gate, nerd_gate]
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, rlp_output, nerd_output, illumination_weights):
        """
        Args:
            rlp_output: Output from RLP branch [B, 3, H, W]
            nerd_output: Output from NeRD-Rain branch [B, 3, H, W]
            illumination_weights: Dict with 'night_weight', 'day_weight' [B, 1]
        Returns:
            Fused output [B, 3, H, W]
        """
        B, C, H, W = rlp_output.shape
        
        # Extract features from both outputs
        rlp_features = self.rlp_feat_extractor(rlp_output)  # [B, D, H, W]
        nerd_features = self.nerd_feat_extractor(nerd_output)  # [B, D, H, W]
        
        # Reshape for cross-attention [B, N, D] where N = H*W
        rlp_tokens = rlp_features.flatten(2).transpose(1, 2)
        nerd_tokens = nerd_features.flatten(2).transpose(1, 2)
        
        # Apply cross-attention
        rlp_enhanced = self.cross_attn_rlp2nerd(rlp_tokens, nerd_tokens)
        nerd_enhanced = self.cross_attn_nerd2rlp(nerd_tokens, rlp_tokens)
        
        # Reshape back to [B, D, H, W]
        rlp_enhanced = rlp_enhanced.transpose(1, 2).reshape(B, self.feature_dim, H, W)
        nerd_enhanced = nerd_enhanced.transpose(1, 2).reshape(B, self.feature_dim, H, W)
        
        # Generate channel-wise gates from illumination weights
        night_weight = illumination_weights['night_weight']  # [B, 1]
        day_weight = illumination_weights['day_weight']  # [B, 1]
        illum_vector = torch.cat([night_weight, day_weight], dim=1)  # [B, 2]
        
        channel_gates = self.gate_generator(illum_vector)  # [B, D*2]
        rlp_channel_gate = channel_gates[:, :self.feature_dim].unsqueeze(-1).unsqueeze(-1)
        nerd_channel_gate = channel_gates[:, self.feature_dim:].unsqueeze(-1).unsqueeze(-1)
        
        # Apply channel-wise gating
        rlp_gated = rlp_enhanced * rlp_channel_gate
        nerd_gated = nerd_enhanced * nerd_channel_gate
        
        # Generate spatial gates
        combined = torch.cat([rlp_gated, nerd_gated], dim=1)
        spatial_gates = self.spatial_gate(combined)  # [B, 2, H, W]
        rlp_spatial_gate = spatial_gates[:, 0:1, :, :]
        nerd_spatial_gate = spatial_gates[:, 1:2, :, :]
        
        # Apply spatial gating
        rlp_final = rlp_gated * rlp_spatial_gate
        nerd_final = nerd_gated * nerd_spatial_gate
        
        # Fuse features
        fused_features = torch.cat([rlp_final, nerd_final], dim=1)
        fused = self.fusion(fused_features)
        
        # Reconstruct output
        output = self.reconstruction(fused)
        
        # Adaptive combination of original outputs
        alpha = night_weight.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        beta = day_weight.unsqueeze(-1).unsqueeze(-1)
        
        # Combine: reconstructed output + weighted average of inputs
        combined_input = alpha * rlp_output + beta * nerd_output
        final_output = output + combined_input
        
        return final_output


if __name__ == "__main__":
    # Test Adaptive Fusion Module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AdaptiveFusionModule(in_channels=3, feature_dim=64).to(device)
    
    # Create dummy inputs
    rlp_out = torch.randn(2, 3, 256, 256).to(device)
    nerd_out = torch.randn(2, 3, 256, 256).to(device)
    
    # Test with night scene (high night_weight)
    night_weights = {
        'night_weight': torch.tensor([[0.8], [0.7]]).to(device),
        'day_weight': torch.tensor([[0.2], [0.3]]).to(device)
    }
    
    output = model(rlp_out, nerd_out, night_weights)
    print(f"Night scene fusion:")
    print(f"RLP output: {rlp_out.shape}")
    print(f"NeRD output: {nerd_out.shape}")
    print(f"Fused output: {output.shape}")
    
    # Test with day scene (high day_weight)
    day_weights = {
        'night_weight': torch.tensor([[0.2], [0.3]]).to(device),
        'day_weight': torch.tensor([[0.8], [0.7]]).to(device)
    }
    
    output = model(rlp_out, nerd_out, day_weights)
    print(f"\nDay scene fusion:")
    print(f"Fused output: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
