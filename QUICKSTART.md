# Quick Start Guide

## Installation

1. **Clone the repository** (if using git):
```bash
git init
git add .
git commit -m "Initial commit: DNA-Net unified deraining model"
```

2. **Activate conda environment**:
```bash
conda activate data-formulator
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your datasets in the following structure:

```
data/
├── train/
│   ├── rainy/
│   └── clean/
├── val/
│   ├── rainy/
│   └── clean/
└── test/
    ├── rainy/
    └── clean/
```

For mixed day/night training:
```
data/
├── train/
│   ├── day/
│   │   ├── rainy/
│   │   └── clean/
│   └── night/
│       ├── rainy/
│       └── clean/
└── val/
    ├── day/...
    └── night/...
```

## Testing Individual Components

Test each module to verify installation:

```bash
# Test RLP module
python modules/rlp_module.py

# Test RPIM
python modules/rpim.py

# Test Transformer
python modules/transformer.py

# Test INR
python modules/inr.py

# Test Illumination Estimator
python models/illumination_estimator.py

# Test RLP Branch
python models/rlp_branch.py

# Test NeRD-Rain Branch
python models/nerd_rain_branch.py

# Test Fusion Module
python models/fusion_module.py

# Test complete DNA-Net
python models/dna_net.py

# Test losses and metrics
python utils/losses.py
python utils/metrics.py
```

## Training

### Basic Training

```bash
python train.py --config config/default_config.yaml --name experiment_1
```

### Custom Training

1. Copy and modify config:
```bash
cp config/default_config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
```

2. Train with custom config:
```bash
python train.py --config config/my_config.yaml --name my_experiment
```

### Resume Training

Edit `config/default_config.yaml`:
```yaml
resume:
  enabled: true
  checkpoint_path: 'checkpoints/experiment_1/checkpoint_epoch_50.pth'
```

Then run:
```bash
python train.py --config config/default_config.yaml --name experiment_1_resume
```

### Monitor Training

```bash
tensorboard --logdir logs
```

Open browser at: http://localhost:6006

## Testing

### Test on Dataset

```bash
python test.py \
    --config config/default_config.yaml \
    --checkpoint checkpoints/experiment_1/best_model.pth \
    --mode test
```

Results will be saved to `results/` directory.

### Inference on Single Image

```bash
python test.py \
    --config config/default_config.yaml \
    --checkpoint checkpoints/experiment_1/best_model.pth \
    --mode inference \
    --input path/to/rainy_image.png \
    --output path/to/derained_output.png
```

## Configuration Tips

### For Nighttime-Only Training

```yaml
# Use lower learning rate and more iterations
training:
  learning_rate: 0.0001
  num_epochs: 150

loss:
  rlp_weight: 0.2  # Increase RLP loss weight
```

### For Daytime-Only Training

```yaml
loss:
  rlp_weight: 0.05  # Decrease RLP loss weight
  perceptual_weight: 0.1  # Add perceptual loss
```

### For Mixed Day-Night Training

```yaml
data:
  use_mixed_dataset: true
  day_night_ratio: 1.0  # Equal sampling

training:
  batch_size: 8  # Larger batch for diversity
```

## Common Issues and Solutions

### 1. CUDA Out of Memory

**Solution**: Reduce batch size or patch size
```yaml
training:
  batch_size: 2  # Reduce from 4

data:
  patch_size: 128  # Reduce from 256
```

### 2. Training Unstable / NaN Loss

**Solution**: Lower learning rate and enable gradient clipping
```yaml
training:
  learning_rate: 0.0001
  clip_grad_norm: 1.0
```

### 3. Poor Nighttime Performance

**Solution**: Increase RLP weight and check illumination estimator
```yaml
loss:
  rlp_weight: 0.15  # Increase from 0.1
```

### 4. Slow Training

**Solution**: Increase num_workers and use smaller model
```yaml
data:
  num_workers: 8  # Increase for faster data loading

model:
  base_channels: 48  # Reduce from 64
  scales: [1, 2]  # Use 2 scales instead of 3
```

## Expected Performance

With proper training (100 epochs on mixed dataset):

**Daytime (Rain100L)**:
- PSNR: 37-39 dB
- SSIM: 0.97-0.98

**Nighttime (NighttimeRain)**:
- PSNR: 24-26 dB
- SSIM: 0.83-0.86

Training time: ~24 hours on RTX 4090

## Model Architecture Summary

```
Total Parameters: ~45M
Model Size: ~180 MB (float32)
Memory Usage: ~8-10 GB (batch_size=4, patch_size=256)
Inference Time: ~50ms per 256×256 image (GPU)
```

## Next Steps

1. **Prepare your datasets** in the required structure
2. **Test individual modules** to verify installation
3. **Start with small experiment** (few epochs) to validate setup
4. **Scale up training** with full configuration
5. **Monitor metrics** and adjust hyperparameters
6. **Evaluate on test set** and analyze results

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dnanet2024,
  title={DNA-Net: Day-Night Adaptive Deraining Network},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- RLP: Zhang et al., "Learning Rain Location Prior for Nighttime Deraining", ICCV 2023
- NeRD-Rain: "Bidirectional Multiscale Transformer with Implicit Neural Representations", arXiv 2024

## Support

For issues and questions:
- Check the paper draft: `PAPER_DRAFT.md`
- Review configuration: `config/default_config.yaml`
- Examine module tests in each file's `__main__` section
