"""
Illumination Estimator Module
Automatically detects day/night conditions and generates adaptive weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationEstimator(nn.Module):
    """
    Estimates scene illumination and generates adaptive weights for day/night branches.
    """
    
    def __init__(self, in_channels=3):
        super(IlluminationEstimator, self).__init__()
        
        # Multi-scale illumination feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling and illumination prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Predict illumination statistics
        self.illumination_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # [brightness, contrast, illumination_score]
        )
        
        # Adaptive weight generation
        self.weight_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),  # [night_weight, day_weight]
            nn.Softmax(dim=1)
        )
        
    def compute_brightness(self, x):
        """
        Compute image brightness using luminance.
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            Brightness values [B, 1]
        """
        # Convert RGB to grayscale using standard luminance weights
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        brightness = gray.mean(dim=[2, 3])
        return brightness
    
    def compute_contrast(self, x):
        """
        Compute image contrast.
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            Contrast values [B, 1]
        """
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        contrast = gray.std(dim=[2, 3])
        return contrast
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            illumination_features: [B, 128, H/8, W/8]
            illumination_stats: Dict with 'brightness', 'contrast', 'score'
            weights: Dict with 'night_weight', 'day_weight' [B, 1]
        """
        # Extract multi-scale features
        feat1 = self.conv1(x)  # [B, 32, H/2, W/2]
        feat2 = self.conv2(feat1)  # [B, 64, H/4, W/4]
        feat3 = self.conv3(feat2)  # [B, 128, H/8, W/8]
        
        # Global statistics
        pooled = self.global_pool(feat3).flatten(1)  # [B, 128]
        
        # Predict illumination statistics
        illum_stats = self.illumination_head(pooled)  # [B, 3]
        
        # Also compute simple image statistics
        brightness = self.compute_brightness(x)
        contrast = self.compute_contrast(x)
        
        # Combine learned and computed statistics
        combined_stats = torch.cat([brightness, contrast, illum_stats[:, 2:3]], dim=1)
        
        # Generate adaptive weights
        weights = self.weight_head(combined_stats)  # [B, 2]
        
        # Prepare output
        illumination_stats = {
            'brightness': illum_stats[:, 0:1],
            'contrast': illum_stats[:, 1:2],
            'score': illum_stats[:, 2:3],
            'simple_brightness': brightness,
            'simple_contrast': contrast
        }
        
        branch_weights = {
            'night_weight': weights[:, 0:1],  # Higher for dark scenes
            'day_weight': weights[:, 1:2]     # Higher for bright scenes
        }
        
        return feat3, illumination_stats, branch_weights


if __name__ == "__main__":
    # Test illumination estimator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IlluminationEstimator(in_channels=3).to(device)
    
    # Create test images
    dark_img = torch.rand(2, 3, 256, 256).to(device) * 0.3  # Dark image
    bright_img = torch.rand(2, 3, 256, 256).to(device) * 0.7 + 0.3  # Bright image
    
    print("Testing dark image:")
    feat, stats, weights = model(dark_img)
    print(f"Features shape: {feat.shape}")
    print(f"Brightness: {stats['simple_brightness'].mean().item():.4f}")
    print(f"Contrast: {stats['simple_contrast'].mean().item():.4f}")
    print(f"Night weight: {weights['night_weight'].mean().item():.4f}")
    print(f"Day weight: {weights['day_weight'].mean().item():.4f}")
    
    print("\nTesting bright image:")
    feat, stats, weights = model(bright_img)
    print(f"Features shape: {feat.shape}")
    print(f"Brightness: {stats['simple_brightness'].mean().item():.4f}")
    print(f"Contrast: {stats['simple_contrast'].mean().item():.4f}")
    print(f"Night weight: {weights['night_weight'].mean().item():.4f}")
    print(f"Day weight: {weights['day_weight'].mean().item():.4f}")
