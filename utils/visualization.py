"""
Visualization utilities for DNA-Net results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def visualize_rlp_map(rlp_map, save_path=None):
    """
    Visualize Rain Location Prior map.
    
    Args:
        rlp_map: RLP tensor [1, H, W] or [H, W]
        save_path: Path to save visualization
    """
    if isinstance(rlp_map, torch.Tensor):
        rlp_map = rlp_map.cpu().numpy()
    
    if rlp_map.ndim == 3:
        rlp_map = rlp_map.squeeze()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rlp_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Rain Probability')
    plt.title('Rain Location Prior Map')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"RLP map saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_branch_weights(night_weights, day_weights, save_path=None):
    """
    Visualize distribution of branch weights.
    
    Args:
        night_weights: List or array of night branch weights
        day_weights: List or array of day branch weights
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Night weights histogram
    axes[0].hist(night_weights, bins=50, alpha=0.7, color='navy', edgecolor='black')
    axes[0].set_xlabel('Night Branch Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Night Branch Weights')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(np.mean(night_weights), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(night_weights):.3f}')
    axes[0].legend()
    
    # Day weights histogram
    axes[1].hist(day_weights, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1].set_xlabel('Day Branch Weight')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Day Branch Weights')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(np.mean(day_weights), color='red', linestyle='--',
                    label=f'Mean: {np.mean(day_weights):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Weight distribution saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(rainy, rlp_out, nerd_out, fused_out, clean, save_path=None):
    """
    Create comprehensive comparison visualization.
    
    Args:
        rainy, rlp_out, nerd_out, fused_out, clean: Image tensors [C, H, W]
        save_path: Path to save visualization
    """
    def tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach()
        img = tensor.numpy()
        if img.shape[0] == 3:  # CHW to HWC
            img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0, 1)
    
    images = [
        tensor_to_numpy(rainy),
        tensor_to_numpy(rlp_out),
        tensor_to_numpy(nerd_out),
        tensor_to_numpy(fused_out),
        tensor_to_numpy(clean)
    ]
    
    titles = ['Rainy Input', 'RLP Output', 'NeRD-Rain Output', 'Fused Output', 'Ground Truth']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Comparison saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiscale_outputs(scale_outputs, save_path=None):
    """
    Visualize outputs at different scales.
    
    Args:
        scale_outputs: List of tensors at different scales
        save_path: Path to save visualization
    """
    num_scales = len(scale_outputs)
    
    fig, axes = plt.subplots(1, num_scales, figsize=(5*num_scales, 5))
    
    if num_scales == 1:
        axes = [axes]
    
    for i, (ax, output) in enumerate(zip(axes, scale_outputs)):
        img = output[0].cpu().detach().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'Scale {i} ({img.shape[1]}×{img.shape[0]})', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Multiscale outputs saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training log
        save_path: Path to save plot
    """
    # This is a placeholder - implement based on your logging format
    # Example: Parse tensorboard logs or custom logs
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Example plots (populate with actual data)
    epochs = range(100)
    
    # Loss curves
    axes[0, 0].plot(epochs, np.random.rand(100), label='Train Loss')
    axes[0, 0].plot(epochs, np.random.rand(100), label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR curves
    axes[0, 1].plot(epochs, 20 + np.random.rand(100) * 5, label='Train PSNR')
    axes[0, 1].plot(epochs, 22 + np.random.rand(100) * 4, label='Val PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM curves
    axes[1, 0].plot(epochs, 0.7 + np.random.rand(100) * 0.2, label='Train SSIM')
    axes[1, 0].plot(epochs, 0.75 + np.random.rand(100) * 0.15, label='Val SSIM')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('SSIM Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(epochs, 0.0002 * np.exp(-0.05 * np.array(epochs)))
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_paper_figure(rainy, output, clean, rlp_map, weights, save_path):
    """
    Create publication-quality figure showing all components.
    
    Args:
        rainy: Input rainy image
        output: Final derained output
        clean: Ground truth
        rlp_map: Rain location prior map
        weights: Dict with 'night_weight' and 'day_weight'
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach()
        img = tensor.numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0, 1)
    
    # Top row: Main results
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(to_numpy(rainy))
    ax1.set_title('(a) Rainy Input', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(to_numpy(output))
    ax2.set_title('(b) DNA-Net Output', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(to_numpy(clean))
    ax3.set_title('(c) Ground Truth', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # RLP map
    ax4 = fig.add_subplot(gs[0, 3])
    rlp_np = to_numpy(rlp_map).squeeze()
    im = ax4.imshow(rlp_np, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('(d) Rain Location Prior', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # Bottom row: Analysis
    # Difference maps
    ax5 = fig.add_subplot(gs[1, 0])
    diff_input = np.abs(to_numpy(rainy) - to_numpy(clean))
    ax5.imshow(diff_input)
    ax5.set_title('(e) Input Error', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    diff_output = np.abs(to_numpy(output) - to_numpy(clean))
    ax6.imshow(diff_output)
    ax6.set_title('(f) Output Error', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Weights visualization
    ax7 = fig.add_subplot(gs[1, 2])
    night_w = weights['night_weight'].item()
    day_w = weights['day_weight'].item()
    
    bars = ax7.bar(['Night Branch', 'Day Branch'], [night_w, day_w],
                   color=['navy', 'gold'], edgecolor='black', linewidth=2)
    ax7.set_ylabel('Weight', fontsize=12)
    ax7.set_title('(g) Branch Weights', fontsize=14, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    
    # Metrics comparison
    ax8 = fig.add_subplot(gs[1, 3])
    metrics = ['PSNR', 'SSIM', 'MAE']
    # These would be computed from actual data
    values = [28.5, 0.89, 0.045]  # Example values
    
    ax8.barh(metrics, values, color='steelblue', edgecolor='black', linewidth=2)
    ax8.set_xlabel('Metric Value', fontsize=12)
    ax8.set_title('(h) Quality Metrics', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(values):
        ax8.text(v, i, f' {v:.3f}', va='center', fontsize=11)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Paper figure saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities for DNA-Net")
    print("\nAvailable functions:")
    print("  - visualize_rlp_map()")
    print("  - visualize_branch_weights()")
    print("  - visualize_comparison()")
    print("  - visualize_multiscale_outputs()")
    print("  - plot_training_curves()")
    print("  - create_paper_figure()")
    print("\nExample usage in test.py or custom scripts.")
