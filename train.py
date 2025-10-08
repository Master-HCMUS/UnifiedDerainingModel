"""
Training script for DNA-Net
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.dna_net import DNANet, DNANetLoss
from utils.dataset import create_dataloaders
from utils.metrics import MetricTracker
from utils.losses import CombinedLoss


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model, config):
    """Create optimizer from config."""
    optimizer_type = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    betas = tuple(config['training']['betas'])
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, 
                              weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=weight_decay, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config."""
    scheduler_type = config['training']['lr_scheduler'].lower()
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['min_lr']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['training']['lr_decay_epochs'],
            gamma=config['training']['lr_decay_rate']
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['training']['lr_decay_rate'],
            patience=10
        )
    else:
        scheduler = None
    
    return scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, best_metric


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, writer, global_step):
    """Train for one epoch."""
    model.train()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    log_freq = config['logging']['log_freq']
    clip_grad = config['training']['clip_grad_norm']
    
    epoch_losses = []
    
    for batch_idx, batch in enumerate(pbar):
        rainy = batch['rainy'].to(device)
        clean = batch['clean'].to(device)
        
        # Forward pass
        output, intermediates = model(rainy, return_intermediates=True)
        
        # Add input to intermediates for loss computation
        intermediates['input'] = rainy
        
        # Compute loss
        loss, loss_dict = criterion(output, clean, intermediates)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Update progress bar
        epoch_losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Log to tensorboard
        if writer is not None and batch_idx % log_freq == 0:
            for key, value in loss_dict.items():
                writer.add_scalar(f'Train/{key}', value, global_step)
            
            writer.add_scalar('Train/learning_rate', 
                            optimizer.param_groups[0]['lr'], global_step)
        
        global_step += 1
    
    avg_loss = np.mean(epoch_losses)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, config, writer):
    """Validate the model."""
    model.eval()
    
    metric_tracker = MetricTracker(metrics=config['metrics'])
    val_losses = []
    
    pbar = tqdm(val_loader, desc='Validation')
    
    for batch_idx, batch in enumerate(pbar):
        rainy = batch['rainy'].to(device)
        clean = batch['clean'].to(device)
        
        # Forward pass
        output, intermediates = model(rainy, return_intermediates=True)
        intermediates['input'] = rainy
        
        # Compute loss
        loss, loss_dict = criterion(output, clean, intermediates)
        val_losses.append(loss.item())
        
        # Update metrics
        metric_tracker.update(output, clean)
        
        # Save sample images
        if writer is not None and batch_idx < config['logging']['num_val_images']:
            # Save first image in batch
            writer.add_image(f'Val/rainy_{batch_idx}', rainy[0], epoch)
            writer.add_image(f'Val/output_{batch_idx}', output[0], epoch)
            writer.add_image(f'Val/clean_{batch_idx}', clean[0], epoch)
            
            # Save intermediate outputs
            if config['logging']['save_images']:
                writer.add_image(f'Val/rlp_output_{batch_idx}', 
                               intermediates['rlp_output'][0], epoch)
                writer.add_image(f'Val/nerd_output_{batch_idx}',
                               intermediates['nerd_output'][0], epoch)
                writer.add_image(f'Val/rlp_map_{batch_idx}',
                               intermediates['rlp_map'][0], epoch)
    
    # Get average metrics
    avg_loss = np.mean(val_losses)
    avg_metrics = metric_tracker.get_average()
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/loss', avg_loss, epoch)
        for metric, value in avg_metrics.items():
            writer.add_scalar(f'Val/{metric}', value, epoch)
    
    # Print results
    print(f"\nValidation - Epoch {epoch}")
    print(f"  Loss: {avg_loss:.4f}")
    for metric, value in avg_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train DNA-Net')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--name', type=str, default='',
                       help='Experiment name')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    if config.get('seed') is not None:
        set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_gpu'] 
                         else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_name = args.name if args.name else datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(config['checkpoint']['save_dir'], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config to experiment directory
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create model
    print("Creating model...")
    model = DNANet(
        in_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels'],
        scales=config['model']['scales']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = DNANetLoss(
        l1_weight=config['loss']['l1_weight'],
        l2_weight=config['loss']['l2_weight'],
        rlp_weight=config['loss']['rlp_weight'],
        perceptual_weight=config['loss']['perceptual_weight'],
        use_perceptual=(config['loss']['perceptual_weight'] > 0)
    )
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(config['data'])
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup tensorboard
    writer = None
    if config['logging']['tensorboard']:
        log_dir = os.path.join(config['logging']['log_dir'], exp_name)
        writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs: {log_dir}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    global_step = 0
    
    if config['resume']['enabled'] and config['resume']['checkpoint_path']:
        start_epoch, best_metric = load_checkpoint(
            model, optimizer, scheduler, config['resume']['checkpoint_path']
        )
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    num_epochs = config['training']['num_epochs']
    best_metric_name = config['checkpoint']['best_metric']
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, config, writer, global_step
        )
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % config['logging']['val_freq'] == 0:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, epoch, config, writer
            )
            
            # Check if best model
            current_metric = val_metrics.get(best_metric_name, 0.0)
            is_best = current_metric > best_metric
            
            if is_best:
                best_metric = current_metric
                print(f"New best {best_metric_name}: {best_metric:.4f}")
            
            # Save checkpoint
            if config['checkpoint']['save_best'] and is_best:
                save_path = os.path.join(exp_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path)
        
        # Save regular checkpoint
        if (epoch + 1) % config['checkpoint']['save_freq'] == 0:
            save_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get(best_metric_name, 0.0))
            else:
                scheduler.step()
    
    # Save final model
    final_path = os.path.join(exp_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, scheduler, num_epochs-1, best_metric, final_path)
    
    print("\nTraining completed!")
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
