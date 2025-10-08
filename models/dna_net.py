"""
DNA-Net: Day-Night Adaptive Deraining Network
Unified model combining RLP (nighttime) and NeRD-Rain (daytime) approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.illumination_estimator import IlluminationEstimator
from models.rlp_branch import RLPBranch
from models.nerd_rain_branch import NeRDRainBranch
from models.fusion_module import AdaptiveFusionModule


class DNANet(nn.Module):
    """
    Day-Night Adaptive Deraining Network (DNA-Net)
    
    A unified deraining model that adaptively combines:
    - RLP Branch: Specializes in nighttime deraining with rain location prior
    - NeRD-Rain Branch: Excels at daytime deraining with multiscale transformers
    
    The model automatically detects illumination and weights each branch accordingly.
    """
    
    def __init__(self, in_channels=3, base_channels=64, scales=[1, 2, 4]):
        super(DNANet, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Illumination-aware feature extraction and weight generation
        self.illumination_estimator = IlluminationEstimator(in_channels=in_channels)
        
        # RLP Branch (nighttime-focused)
        self.rlp_branch = RLPBranch(
            in_channels=in_channels,
            base_channels=base_channels
        )
        
        # NeRD-Rain Branch (daytime-focused)
        self.nerd_branch = NeRDRainBranch(
            in_channels=in_channels,
            base_dim=base_channels,
            scales=scales
        )
        
        # Adaptive fusion module
        self.fusion_module = AdaptiveFusionModule(
            in_channels=in_channels,
            feature_dim=base_channels
        )
        
        # Refinement module for final output polish
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, return_intermediates=False):
        """
        Args:
            x: Input rainy image [B, 3, H, W]
            return_intermediates: If True, return intermediate outputs for visualization
        
        Returns:
            output: Final derained image [B, 3, H, W]
            intermediates (optional): Dict containing intermediate results
        """
        # Estimate illumination and generate adaptive weights
        illum_features, illum_stats, branch_weights = self.illumination_estimator(x)
        
        # Process with RLP branch (nighttime-focused)
        rlp_output, rlp_map, rlp_features = self.rlp_branch(x)
        
        # Process with NeRD-Rain branch (daytime-focused)
        nerd_output, nerd_scale_outputs, nerd_features = self.nerd_branch(x)
        
        # Adaptively fuse outputs
        fused_output = self.fusion_module(rlp_output, nerd_output, branch_weights)
        
        # Apply refinement
        residual = self.refinement(fused_output)
        final_output = fused_output + residual
        
        # Clamp to valid range
        final_output = torch.clamp(final_output, 0, 1)
        
        if return_intermediates:
            intermediates = {
                'rlp_output': rlp_output,
                'nerd_output': nerd_output,
                'fused_output': fused_output,
                'rlp_map': rlp_map,
                'nerd_scale_outputs': nerd_scale_outputs,
                'illumination_stats': illum_stats,
                'branch_weights': branch_weights,
                'night_weight': branch_weights['night_weight'],
                'day_weight': branch_weights['day_weight']
            }
            return final_output, intermediates
        
        return final_output
    
    def inference(self, x):
        """
        Simplified inference method without intermediate outputs.
        
        Args:
            x: Input rainy image [B, 3, H, W]
        Returns:
            Derained image [B, 3, H, W]
        """
        with torch.no_grad():
            return self.forward(x, return_intermediates=False)


class DNANetLoss(nn.Module):
    """
    Comprehensive loss function for DNA-Net training.
    Combines reconstruction, perceptual, and auxiliary losses.
    """
    
    def __init__(self, l1_weight=1.0, l2_weight=1.0, rlp_weight=0.1, 
                 perceptual_weight=0.1, use_perceptual=False):
        super(DNANetLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.rlp_weight = rlp_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Perceptual loss (VGG features) - optional
        if use_perceptual:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features[:16]  # Up to relu3_3
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg.eval()
        
    def compute_rlp_loss(self, rlp_map, input_img, gt_img):
        """
        Auxiliary loss for Rain Location Prior.
        Uses residual as pseudo ground truth.
        """
        with torch.no_grad():
            residual = torch.abs(input_img - gt_img).mean(dim=1, keepdim=True)
            residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        
        rlp_loss = self.l2_loss(rlp_map, residual)
        return rlp_loss
    
    def compute_perceptual_loss(self, pred, target):
        """Perceptual loss using VGG features."""
        if not self.use_perceptual:
            return torch.tensor(0.0, device=pred.device)
        
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        loss = self.l2_loss(pred_features, target_features)
        return loss
    
    def forward(self, output, target, intermediates=None):
        """
        Args:
            output: Final derained image [B, 3, H, W]
            target: Ground truth clean image [B, 3, H, W]
            intermediates: Dict with intermediate outputs from model
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Main reconstruction losses
        l1_loss = self.l1_loss(output, target)
        l2_loss = self.l2_loss(output, target)
        
        # Perceptual loss
        perceptual_loss = self.compute_perceptual_loss(output, target)
        
        # Initialize total loss
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        total_loss += self.perceptual_weight * perceptual_loss
        
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'l2_loss': l2_loss.item(),
            'perceptual_loss': perceptual_loss.item() if self.use_perceptual else 0.0
        }
        
        # Add auxiliary losses if intermediates provided
        if intermediates is not None:
            # RLP auxiliary loss
            if 'rlp_map' in intermediates and 'input' in intermediates:
                rlp_loss = self.compute_rlp_loss(
                    intermediates['rlp_map'],
                    intermediates['input'],
                    target
                )
                total_loss += self.rlp_weight * rlp_loss
                loss_dict['rlp_loss'] = rlp_loss.item()
            
            # Branch-specific losses
            if 'rlp_output' in intermediates:
                rlp_l1 = self.l1_loss(intermediates['rlp_output'], target)
                total_loss += 0.5 * rlp_l1
                loss_dict['rlp_branch_loss'] = rlp_l1.item()
            
            if 'nerd_output' in intermediates:
                nerd_l1 = self.l1_loss(intermediates['nerd_output'], target)
                total_loss += 0.5 * nerd_l1
                loss_dict['nerd_branch_loss'] = nerd_l1.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test DNA-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing DNA-Net...")
    model = DNANet(in_channels=3, base_channels=64, scales=[1, 2, 4]).to(device)
    
    # Create dummy input
    x = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass with intermediates
    print("\nForward pass with intermediates...")
    output, intermediates = model(x, return_intermediates=True)
    
    print(f"Output shape: {output.shape}")
    print(f"\nIntermediate outputs:")
    print(f"  RLP output: {intermediates['rlp_output'].shape}")
    print(f"  NeRD output: {intermediates['nerd_output'].shape}")
    print(f"  Fused output: {intermediates['fused_output'].shape}")
    print(f"  RLP map: {intermediates['rlp_map'].shape}")
    print(f"\nBranch weights:")
    print(f"  Night weight: {intermediates['night_weight'].mean().item():.4f}")
    print(f"  Day weight: {intermediates['day_weight'].mean().item():.4f}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    intermediates['input'] = x
    loss_fn = DNANetLoss(l1_weight=1.0, l2_weight=1.0, rlp_weight=0.1)
    total_loss, loss_dict = loss_fn(output, target, intermediates)
    
    print(f"Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
