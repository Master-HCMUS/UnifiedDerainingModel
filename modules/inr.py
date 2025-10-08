"""
Implicit Neural Representation (INR) Module
Encodes images as continuous functions for robust rain degradation modeling.
Based on: NeRD-Rain (arXiv 2404.01547)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SineActivation(nn.Module):
    """Sine activation for periodic implicit representations."""
    
    def __init__(self, omega_0=30.0):
        super(SineActivation, self).__init__()
        self.omega_0 = omega_0
        
    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class CoordinateEmbedding(nn.Module):
    """
    Positional encoding for coordinates using Fourier features.
    Helps the network learn high-frequency details.
    """
    
    def __init__(self, dim=2, num_freq=10, include_input=True):
        super(CoordinateEmbedding, self).__init__()
        self.dim = dim
        self.num_freq = num_freq
        self.include_input = include_input
        
        # Frequency bands
        freq_bands = 2.0 ** torch.linspace(0, num_freq-1, num_freq)
        self.register_buffer('freq_bands', freq_bands)
        
        # Output dimension
        self.out_dim = dim * (2 * num_freq + (1 if include_input else 0))
        
    def forward(self, coords):
        """
        Args:
            coords: Coordinates [B, N, dim] or [B, H, W, dim]
        Returns:
            Embedded coordinates with dimension out_dim
        """
        shape = coords.shape
        coords_flat = coords.reshape(-1, self.dim)
        
        # Fourier features: [sin(2^k * pi * x), cos(2^k * pi * x)]
        embedded = []
        if self.include_input:
            embedded.append(coords_flat)
        
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                embedded.append(func(2.0 * math.pi * freq * coords_flat))
        
        embedded = torch.cat(embedded, dim=-1)
        embedded = embedded.reshape(*shape[:-1], -1)
        
        return embedded


class CoordinateMLP(nn.Module):
    """
    Coordinate-based MLP for implicit neural representation.
    Maps coordinates to feature values.
    """
    
    def __init__(self, coord_dim=2, hidden_dim=256, out_dim=64, 
                 num_layers=4, use_sine=True, omega_0=30.0):
        super(CoordinateMLP, self).__init__()
        
        self.coord_embedding = CoordinateEmbedding(dim=coord_dim, num_freq=10)
        
        layers = []
        in_dim = self.coord_embedding.out_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_sine and i == 0:
                layers.append(SineActivation(omega_0))
            else:
                layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights for sine activation
        if use_sine:
            self._initialize_weights(omega_0)
    
    def _initialize_weights(self, omega_0):
        """Initialize weights for SIREN-style network."""
        with torch.no_grad():
            for i, layer in enumerate(self.mlp):
                if isinstance(layer, nn.Linear):
                    num_input = layer.weight.size(-1)
                    if i == 0:
                        layer.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        layer.weight.uniform_(-math.sqrt(6 / num_input) / omega_0,
                                             math.sqrt(6 / num_input) / omega_0)
    
    def forward(self, coords):
        """
        Args:
            coords: Normalized coordinates [B, H, W, 2] or [B, N, 2]
        Returns:
            Features at coordinates [B, H, W, C] or [B, N, C]
        """
        embedded = self.coord_embedding(coords)
        features = self.mlp(embedded)
        return features


class ImplicitNeuralRepresentation(nn.Module):
    """
    Implicit Neural Representation (INR) Module with coarse and fine grids.
    Learns common rain degradation representations from diverse inputs.
    """
    
    def __init__(self, feature_dim=64, coarse_hidden=256, fine_hidden=128,
                 coarse_layers=4, fine_layers=3):
        super(ImplicitNeuralRepresentation, self).__init__()
        
        # Coarse-level MLP for global structure
        self.coarse_mlp = CoordinateMLP(
            coord_dim=2,
            hidden_dim=coarse_hidden,
            out_dim=feature_dim,
            num_layers=coarse_layers,
            use_sine=True,
            omega_0=30.0
        )
        
        # Fine-level MLP for local details
        self.fine_mlp = CoordinateMLP(
            coord_dim=2,
            hidden_dim=fine_hidden,
            out_dim=feature_dim,
            num_layers=fine_layers,
            use_sine=True,
            omega_0=60.0  # Higher frequency for fine details
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
    def create_coordinate_grid(self, height, width, device):
        """
        Create normalized coordinate grid in range [-1, 1].
        
        Args:
            height, width: Spatial dimensions
            device: torch device
        Returns:
            Coordinate grid [1, H, W, 2]
        """
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0)  # [1, H, W, 2]
        
        return coords
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C_in, H, W]
        Returns:
            INR-enhanced features [B, C_out, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create coordinate grid
        coords = self.create_coordinate_grid(H, W, device)
        coords = coords.repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # Generate coarse and fine features
        coarse_features = self.coarse_mlp(coords)  # [B, H, W, C]
        fine_features = self.fine_mlp(coords)      # [B, H, W, C]
        
        # Permute to [B, C, H, W]
        coarse_features = coarse_features.permute(0, 3, 1, 2)
        fine_features = fine_features.permute(0, 3, 1, 2)
        
        # Fuse coarse and fine
        combined = torch.cat([coarse_features, fine_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Add residual connection with input
        if C == fused_features.shape[1]:
            output = x + fused_features
        else:
            output = fused_features
        
        return output


class IntraScaleSharedEncoder(nn.Module):
    """
    Intra-scale shared encoder for closed-loop framework.
    Shares information between adjacent scales via INR.
    """
    
    def __init__(self, lower_dim=64, upper_dim=128):
        super(IntraScaleSharedEncoder, self).__init__()
        
        # Encoder for lower scale features
        self.encoder = nn.Sequential(
            nn.Conv2d(lower_dim, lower_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(lower_dim * 2, lower_dim, kernel_size=3, padding=1)
        )
        
        # Channel projection from lower to upper dimension
        self.channel_proj = nn.Conv2d(lower_dim, upper_dim, kernel_size=1)
        
        # INR works with upper dimension
        self.inr = ImplicitNeuralRepresentation(feature_dim=upper_dim)
        
    def forward(self, feat_lower, feat_upper):
        """
        Args:
            feat_lower: Features from lower scale [B, C_lower, H_l, W_l]
            feat_upper: Features from upper scale [B, C_upper, H_u, W_u]
        Returns:
            Enhanced features for both scales
        """
        # Encode lower scale
        encoded_lower = self.encoder(feat_lower)
        
        # Project to upper dimension and upsample to upper scale resolution
        projected = self.channel_proj(encoded_lower)
        upsampled = F.interpolate(projected, size=feat_upper.shape[-2:],
                                 mode='bilinear', align_corners=False)
        
        # Apply INR
        inr_features = self.inr(upsampled)
        
        # Enhance upper scale features
        enhanced_upper = feat_upper + inr_features
        
        return encoded_lower, enhanced_upper


if __name__ == "__main__":
    # Test INR module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test basic INR
    inr = ImplicitNeuralRepresentation(feature_dim=64).to(device)
    x = torch.randn(2, 64, 64, 64).to(device)
    
    output = inr(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test intra-scale shared encoder
    encoder = IntraScaleSharedEncoder(lower_dim=64, upper_dim=128).to(device)
    feat_lower = torch.randn(2, 64, 32, 32).to(device)
    feat_upper = torch.randn(2, 128, 64, 64).to(device)
    
    enc_lower, enh_upper = encoder(feat_lower, feat_upper)
    print(f"\nIntra-scale shared encoder:")
    print(f"Lower input: {feat_lower.shape} -> Encoded: {enc_lower.shape}")
    print(f"Upper input: {feat_upper.shape} -> Enhanced: {enh_upper.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in inr.parameters())
    print(f"\nINR parameters: {total_params:,}")
