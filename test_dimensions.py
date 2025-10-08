"""
Test script to verify dimension flow through the model
"""

import torch
import sys

try:
    from models.dna_net import DNANet
    print("✓ Successfully imported DNANet")
except Exception as e:
    print(f"✗ Failed to import DNANet: {e}")
    sys.exit(1)

def test_model_dimensions():
    """Test model with various input sizes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create model (base_channels parameter is correct for DNANet)
    print("\nCreating DNA-Net model...")
    model = DNANet(in_channels=3, base_channels=64, scales=[1, 2, 4]).to(device)
    model.eval()
    
    # Test different input sizes
    test_sizes = [
        (128, 128),
        (256, 256),
        (384, 384),
        (512, 512),
    ]
    
    print("\n" + "="*60)
    print("Testing model with different input sizes:")
    print("="*60)
    
    for h, w in test_sizes:
        try:
            print(f"\nTesting size: {h}×{w}")
            x = torch.randn(2, 3, h, w).to(device)
            
            with torch.no_grad():
                output, intermediates = model(x, return_intermediates=True)
            
            # Check dimensions
            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
            
            print(f"  Input:  {x.shape}")
            print(f"  Output: {output.shape}")
            print(f"  RLP output: {intermediates['rlp_output'].shape}")
            print(f"  NeRD output: {intermediates['nerd_output'].shape}")
            print(f"  RLP map: {intermediates['rlp_map'].shape}")
            print(f"  ✓ PASSED")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*60)
    print("✓ All dimension tests passed!")
    print("="*60)
    return True


def test_rlp_branch():
    """Test RLP branch individually."""
    print("\n" + "="*60)
    print("Testing RLP Branch individually:")
    print("="*60)
    
    from models.rlp_branch import RLPBranch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RLPBranch(in_channels=3, base_channels=64).to(device)
    model.eval()
    
    test_sizes = [(128, 128), (256, 256), (384, 384)]
    
    for h, w in test_sizes:
        try:
            print(f"\nTesting size: {h}×{w}")
            x = torch.randn(2, 3, h, w).to(device)
            
            with torch.no_grad():
                output, rlp_map, features = model(x)
            
            print(f"  Input:  {x.shape}")
            print(f"  Output: {output.shape}")
            print(f"  RLP map: {rlp_map.shape}")
            print(f"  Features: enc1={features['enc1'].shape}, enc2={features['enc2'].shape}, enc3={features['enc3'].shape}")
            print(f"  ✓ PASSED")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✓ RLP Branch tests passed!")
    return True


def test_nerd_branch():
    """Test NeRD-Rain branch individually."""
    print("\n" + "="*60)
    print("Testing NeRD-Rain Branch individually:")
    print("="*60)
    
    from models.nerd_rain_branch import NeRDRainBranch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRDRainBranch(in_channels=3, base_dim=64, scales=[1, 2, 4]).to(device)
    model.eval()
    
    test_sizes = [(128, 128), (256, 256)]
    
    for h, w in test_sizes:
        try:
            print(f"\nTesting size: {h}×{w}")
            x = torch.randn(2, 3, h, w).to(device)
            
            with torch.no_grad():
                output, scale_outputs, features = model(x)
            
            print(f"  Input:  {x.shape}")
            print(f"  Output: {output.shape}")
            print(f"  Scale outputs: {len(scale_outputs)} scales")
            for i, out in enumerate(scale_outputs):
                print(f"    Scale {i}: {out.shape}")
            print(f"  Features: {list(features.keys())}")
            print(f"  ✓ PASSED")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✓ NeRD-Rain Branch tests passed!")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DNA-Net Dimension Test Suite")
    print("="*60)
    
    # Test individual branches first
    rlp_ok = test_rlp_branch()
    if not rlp_ok:
        print("\n✗ RLP Branch test failed. Fix this before testing full model.")
        sys.exit(1)
    
    nerd_ok = test_nerd_branch()
    if not nerd_ok:
        print("\n✗ NeRD-Rain Branch test failed. Fix this before testing full model.")
        sys.exit(1)
    
    # Test full model
    full_ok = test_model_dimensions()
    
    if full_ok:
        print("\n🎉 All tests passed! Model is ready for training.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
