"""
Testing script for DNA-Net
"""

import torch
import yaml
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms

from models.dna_net import DNANet
from utils.metrics import MetricTracker, calculate_psnr, calculate_ssim
from utils.dataset import DerainingDataset


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint."""
    model = DNANet(
        in_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels'],
        scales=config['model']['scales']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    
    return model


def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    
    return Image.fromarray(img)


@torch.no_grad()
def test_model(model, test_loader, device, config, save_images=True):
    """Test the model on test dataset."""
    model.eval()
    
    metric_tracker = MetricTracker(metrics=config['metrics'])
    output_dir = config['test']['output_dir']
    
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'derained'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    pbar = tqdm(test_loader, desc='Testing')
    
    for batch_idx, batch in enumerate(pbar):
        rainy = batch['rainy'].to(device)
        clean = batch['clean'].to(device)
        filename = batch['filename'][0]
        
        # Forward pass
        output, intermediates = model(rainy, return_intermediates=True)
        
        # Update metrics
        metric_tracker.update(output, clean)
        
        # Compute individual metrics for display
        psnr = calculate_psnr(output, clean)
        ssim = calculate_ssim(output, clean)
        pbar.set_postfix({'PSNR': f'{psnr:.2f}', 'SSIM': f'{ssim:.4f}'})
        
        # Save images
        if save_images:
            # Save derained image
            output_img = tensor_to_image(output)
            output_img.save(os.path.join(output_dir, 'derained', filename))
            
            # Create comparison image
            rainy_img = tensor_to_image(rainy)
            clean_img = tensor_to_image(clean)
            
            # Concatenate horizontally: rainy | output | clean
            comparison = Image.new('RGB', 
                                 (rainy_img.width * 3, rainy_img.height))
            comparison.paste(rainy_img, (0, 0))
            comparison.paste(output_img, (rainy_img.width, 0))
            comparison.paste(clean_img, (rainy_img.width * 2, 0))
            
            comparison.save(os.path.join(output_dir, 'comparisons', filename))
            
            # Save intermediate results for first few images
            if batch_idx < 10:
                rlp_output = tensor_to_image(intermediates['rlp_output'])
                nerd_output = tensor_to_image(intermediates['nerd_output'])
                
                # Create 5-way comparison
                full_comparison = Image.new('RGB',
                                          (rainy_img.width * 5, rainy_img.height))
                full_comparison.paste(rainy_img, (0, 0))
                full_comparison.paste(rlp_output, (rainy_img.width, 0))
                full_comparison.paste(nerd_output, (rainy_img.width * 2, 0))
                full_comparison.paste(output_img, (rainy_img.width * 3, 0))
                full_comparison.paste(clean_img, (rainy_img.width * 4, 0))
                
                comp_path = os.path.join(output_dir, 'comparisons', 
                                        f'full_{filename}')
                full_comparison.save(comp_path)
                
                # Save RLP map
                rlp_map = intermediates['rlp_map'][0].cpu().numpy()
                rlp_map = (rlp_map.squeeze() * 255).astype(np.uint8)
                rlp_map_img = Image.fromarray(rlp_map, mode='L')
                rlp_map_path = os.path.join(output_dir, 'comparisons',
                                           f'rlp_map_{filename}')
                rlp_map_img.save(rlp_map_path)
    
    # Get final metrics
    avg_metrics = metric_tracker.get_average()
    summary = metric_tracker.get_summary()
    
    return avg_metrics, summary


@torch.no_grad()
def inference_single_image(model, image_path, device, output_path=None):
    """Run inference on a single image."""
    model.eval()
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # Forward pass
    output = model.inference(img_tensor)
    
    # Convert to image
    output_img = tensor_to_image(output)
    
    # Save if output path provided
    if output_path:
        output_img.save(output_path)
        print(f"Saved derained image to: {output_path}")
    
    return output_img


def main():
    parser = argparse.ArgumentParser(description='Test DNA-Net')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'inference'],
                       help='Test mode: test (with GT) or inference (single image)')
    parser.add_argument('--input', type=str, default='',
                       help='Input image path for inference mode')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Output image path for inference mode')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_gpu']
                         else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)
    
    if args.mode == 'test':
        # Test on dataset
        print("Creating test dataset...")
        test_dataset = DerainingDataset(
            rainy_dir=config['test']['test_rainy_dir'],
            clean_dir=config['test']['test_clean_dir'],
            patch_size=-1,  # No cropping
            augment=False,
            mode='test'
        )
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Run testing
        print("\nRunning evaluation...")
        avg_metrics, summary = test_model(
            model, test_loader, device, config,
            save_images=config['test']['save_images']
        )
        
        # Print results
        print("\n" + "="*50)
        print("Test Results")
        print("="*50)
        
        print("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nDetailed Statistics:")
        for metric, stats in summary.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
        
        # Save results to file
        output_dir = config['test']['output_dir']
        results_path = os.path.join(output_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            f.write("Test Results\n")
            f.write("="*50 + "\n\n")
            f.write("Average Metrics:\n")
            for metric, value in avg_metrics.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            
            f.write("\nDetailed Statistics:\n")
            for metric, stats in summary.items():
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std:  {stats['std']:.4f}\n")
                f.write(f"  Min:  {stats['min']:.4f}\n")
                f.write(f"  Max:  {stats['max']:.4f}\n")
        
        print(f"\nResults saved to: {results_path}")
        
    elif args.mode == 'inference':
        # Single image inference
        if not args.input:
            raise ValueError("Please provide --input for inference mode")
        
        print(f"Processing image: {args.input}")
        output_img = inference_single_image(model, args.input, device, args.output)
        print("Done!")


if __name__ == '__main__':
    main()
