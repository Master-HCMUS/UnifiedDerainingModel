"""
Dataset utilities for loading deraining datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import random


class DerainingDataset(Dataset):
    """
    Generic deraining dataset loader.
    Assumes paired rainy and clean images.
    """
    
    def __init__(self, rainy_dir, clean_dir, patch_size=256, 
                 augment=True, mode='train'):
        """
        Args:
            rainy_dir: Directory containing rainy images
            clean_dir: Directory containing clean images
            patch_size: Size of random crops for training
            augment: Whether to apply data augmentation
            mode: 'train', 'val', or 'test'
        """
        self.rainy_dir = rainy_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.augment = augment and (mode == 'train')
        self.mode = mode
        
        # Get image filenames
        self.rainy_images = sorted([f for f in os.listdir(rainy_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Loaded {len(self.rainy_images)} image pairs from {rainy_dir}")
    
    def __len__(self):
        return len(self.rainy_images)
    
    def augment_data(self, rainy, clean):
        """Apply random augmentation to image pair."""
        # Random horizontal flip
        if random.random() > 0.5:
            rainy = torch.flip(rainy, dims=[-1])
            clean = torch.flip(clean, dims=[-1])
        
        # Random vertical flip
        if random.random() > 0.5:
            rainy = torch.flip(rainy, dims=[-2])
            clean = torch.flip(clean, dims=[-2])
        
        # Random rotation (90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            rainy = torch.rot90(rainy, k, dims=[-2, -1])
            clean = torch.rot90(clean, k, dims=[-2, -1])
        
        return rainy, clean
    
    def __getitem__(self, idx):
        # Load images
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_name = self.rainy_images[idx]  # Assuming same name
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        rainy = Image.open(rainy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        
        # Convert to tensor
        rainy = transforms.ToTensor()(rainy)
        clean = transforms.ToTensor()(clean)
        
        # Ensure same size
        if rainy.shape != clean.shape:
            # Resize to match
            h, w = clean.shape[-2:]
            rainy = transforms.Resize((h, w))(rainy)
        
        # Random crop for training
        if self.mode == 'train' and self.patch_size > 0:
            h, w = rainy.shape[-2:]
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                rainy = rainy[:, top:top+self.patch_size, left:left+self.patch_size]
                clean = clean[:, top:top+self.patch_size, left:left+self.patch_size]
        
        # Apply augmentation
        if self.augment:
            rainy, clean = self.augment_data(rainy, clean)
        
        return {
            'rainy': rainy,
            'clean': clean,
            'filename': self.rainy_images[idx]
        }


class MixedDerainingDataset(Dataset):
    """
    Mixed dataset combining day and night deraining datasets.
    """
    
    def __init__(self, day_rainy_dir, day_clean_dir,
                 night_rainy_dir, night_clean_dir,
                 patch_size=256, augment=True, mode='train',
                 day_night_ratio=1.0):
        """
        Args:
            day_rainy_dir, day_clean_dir: Day scene directories
            night_rainy_dir, night_clean_dir: Night scene directories
            patch_size: Size of random crops
            augment: Whether to apply augmentation
            mode: 'train', 'val', or 'test'
            day_night_ratio: Ratio of day to night samples
        """
        self.day_dataset = DerainingDataset(
            day_rainy_dir, day_clean_dir, patch_size, augment, mode
        )
        self.night_dataset = DerainingDataset(
            night_rainy_dir, night_clean_dir, patch_size, augment, mode
        )
        
        self.day_night_ratio = day_night_ratio
        self.mode = mode
        
        # Calculate total length based on ratio
        self.day_len = len(self.day_dataset)
        self.night_len = len(self.night_dataset)
        
        print(f"Mixed dataset: {self.day_len} day + {self.night_len} night images")
    
    def __len__(self):
        # Return larger dataset length
        return max(self.day_len, self.night_len)
    
    def __getitem__(self, idx):
        # Randomly choose day or night based on ratio
        if random.random() < (self.day_night_ratio / (1 + self.day_night_ratio)):
            # Sample from day dataset
            day_idx = idx % self.day_len
            sample = self.day_dataset[day_idx]
            sample['scene_type'] = 'day'
        else:
            # Sample from night dataset
            night_idx = idx % self.night_len
            sample = self.night_dataset[night_idx]
            sample['scene_type'] = 'night'
        
        return sample


def create_dataloaders(config):
    """
    Create training and validation dataloaders from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader
    """
    # Check if using mixed dataset
    if config.get('use_mixed_dataset', False):
        train_dataset = MixedDerainingDataset(
            day_rainy_dir=config['train_day_rainy_dir'],
            day_clean_dir=config['train_day_clean_dir'],
            night_rainy_dir=config['train_night_rainy_dir'],
            night_clean_dir=config['train_night_clean_dir'],
            patch_size=config.get('patch_size', 256),
            augment=True,
            mode='train',
            day_night_ratio=config.get('day_night_ratio', 1.0)
        )
        
        val_dataset = MixedDerainingDataset(
            day_rainy_dir=config['val_day_rainy_dir'],
            day_clean_dir=config['val_day_clean_dir'],
            night_rainy_dir=config['val_night_rainy_dir'],
            night_clean_dir=config['val_night_clean_dir'],
            patch_size=-1,  # No cropping for validation
            augment=False,
            mode='val',
            day_night_ratio=config.get('day_night_ratio', 1.0)
        )
    else:
        # Single dataset
        train_dataset = DerainingDataset(
            rainy_dir=config['train_rainy_dir'],
            clean_dir=config['train_clean_dir'],
            patch_size=config.get('patch_size', 256),
            augment=True,
            mode='train'
        )
        
        val_dataset = DerainingDataset(
            rainy_dir=config['val_rainy_dir'],
            clean_dir=config['val_clean_dir'],
            patch_size=-1,
            augment=False,
            mode='val'
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 1),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    # Note: Update paths to actual dataset locations
    config = {
        'train_rainy_dir': 'data/train/rainy',
        'train_clean_dir': 'data/train/clean',
        'val_rainy_dir': 'data/val/rainy',
        'val_clean_dir': 'data/val/clean',
        'patch_size': 256,
        'batch_size': 4,
        'num_workers': 0  # Set to 0 for testing
    }
    
    print("Dataset loading test would run here with actual data paths")
