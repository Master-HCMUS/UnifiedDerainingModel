"""
Evaluation metrics for image deraining.
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import exp


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1, img2: Images in range [0, max_val]
        max_val: Maximum pixel value
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def gaussian(window_size, sigma):
    """Create Gaussian window."""
    gauss = torch.tensor([
        exp(-(x - window_size//2)**2 / float(2*sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        img1, img2: Images [B, C, H, W]
        window_size: Size of Gaussian window
        size_average: Whether to average over batch
    
    Returns:
        SSIM value
    """
    channel = img1.shape[1]
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_mae(img1, img2):
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        img1, img2: Images
    
    Returns:
        MAE value
    """
    mae = torch.mean(torch.abs(img1 - img2))
    return mae.item()


def calculate_rmse(img1, img2):
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        img1, img2: Images
    
    Returns:
        RMSE value
    """
    mse = torch.mean((img1 - img2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


class MetricTracker:
    """Track and compute average metrics."""
    
    def __init__(self, metrics=['psnr', 'ssim', 'mae']):
        self.metrics = metrics
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.values = {metric: [] for metric in self.metrics}
    
    def update(self, pred, target):
        """
        Update metrics with new prediction and target.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
        """
        # Ensure tensors are on CPU for some calculations
        pred = pred.detach()
        target = target.detach()
        
        # Calculate metrics for each image in batch
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_img = pred[i:i+1]
            target_img = target[i:i+1]
            
            if 'psnr' in self.metrics:
                psnr = calculate_psnr(pred_img, target_img)
                self.values['psnr'].append(psnr)
            
            if 'ssim' in self.metrics:
                ssim = calculate_ssim(pred_img, target_img)
                self.values['ssim'].append(ssim)
            
            if 'mae' in self.metrics:
                mae = calculate_mae(pred_img, target_img)
                self.values['mae'].append(mae)
            
            if 'rmse' in self.metrics:
                rmse = calculate_rmse(pred_img, target_img)
                self.values['rmse'].append(rmse)
    
    def get_average(self):
        """
        Get average values of all metrics.
        
        Returns:
            Dictionary with average metric values
        """
        avg_metrics = {}
        for metric in self.metrics:
            if len(self.values[metric]) > 0:
                avg_metrics[metric] = np.mean(self.values[metric])
            else:
                avg_metrics[metric] = 0.0
        
        return avg_metrics
    
    def get_summary(self):
        """
        Get summary statistics of all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        for metric in self.metrics:
            if len(self.values[metric]) > 0:
                values = np.array(self.values[metric])
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                summary[metric] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                }
        
        return summary


if __name__ == "__main__":
    # Test metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test images
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    
    print("Testing individual metrics:")
    
    psnr = calculate_psnr(pred, target)
    print(f"PSNR: {psnr:.4f} dB")
    
    ssim = calculate_ssim(pred, target)
    print(f"SSIM: {ssim:.4f}")
    
    mae = calculate_mae(pred, target)
    print(f"MAE: {mae:.4f}")
    
    rmse = calculate_rmse(pred, target)
    print(f"RMSE: {rmse:.4f}")
    
    # Test metric tracker
    print("\nTesting metric tracker:")
    tracker = MetricTracker(metrics=['psnr', 'ssim', 'mae', 'rmse'])
    
    # Update with multiple batches
    for _ in range(3):
        pred = torch.rand(2, 3, 256, 256).to(device)
        target = torch.rand(2, 3, 256, 256).to(device)
        tracker.update(pred, target)
    
    avg_metrics = tracker.get_average()
    print(f"\nAverage metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    summary = tracker.get_summary()
    print(f"\nMetric summary:")
    for metric, stats in summary.items():
        print(f"  {metric.upper()}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std:  {stats['std']:.4f}")
        print(f"    Min:  {stats['min']:.4f}")
        print(f"    Max:  {stats['max']:.4f}")
