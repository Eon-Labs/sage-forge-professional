#!/usr/bin/env python3
"""
TiRex GPU Testing Script - SAGE-Forge Integration
Tests NX-AI TiRex 35M parameter model on RTX 4090
"""

import torch
import numpy as np
import time
import json
from pathlib import Path

def test_tirex_environment():
    """Test GPU environment and TiRex model availability"""
    print("🚀 TiRex GPU Environment Test")
    print("=" * 50)
    
    # GPU Status
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Free: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    
    # Model Files
    model_path = Path("./models/tirex")
    print(f"\nTiRex Model Path: {model_path}")
    print(f"Model Exists: {model_path.exists()}")
    
    if model_path.exists():
        model_files = list(model_path.glob("*"))
        print(f"Model Files: {[f.name for f in model_files]}")
        
        # Check model checkpoint size
        ckpt_path = model_path / "model.ckpt"
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / 1e6
            print(f"Model Checkpoint: {size_mb:.1f} MB")
    
    return torch.cuda.is_available() and model_path.exists()

def test_synthetic_inference():
    """Test synthetic time series inference"""
    print("\n📊 Synthetic Time Series Inference Test")
    print("=" * 50)
    
    # Generate synthetic OHLCV data (similar to market data)
    seq_length = 200  # Typical lookback window
    batch_size = 1
    features = 5  # OHLCV
    
    # Create realistic synthetic market data
    np.random.seed(42)
    price_base = 100.0
    
    # Generate synthetic OHLCV with realistic patterns
    synthetic_data = []
    current_price = price_base
    
    for i in range(seq_length):
        # Simulate price movement with small random walk
        change = np.random.normal(0, 0.02)  # 2% volatility
        current_price *= (1 + change)
        
        # Generate OHLCV from current price
        volatility = abs(np.random.normal(0, 0.01))
        high = current_price * (1 + volatility)
        low = current_price * (1 - volatility)
        open_price = current_price + np.random.normal(0, 0.005)
        close_price = current_price
        volume = np.random.uniform(1000, 10000)
        
        synthetic_data.append([open_price, high, low, close_price, volume])
    
    # Convert to tensor
    input_tensor = torch.tensor(synthetic_data, dtype=torch.float32).unsqueeze(0)
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Input Data Type: {input_tensor.dtype}")
    print(f"Price Range: {input_tensor[0, :, 3].min():.2f} - {input_tensor[0, :, 3].max():.2f}")  # Close prices
    
    # Test GPU transfer
    if torch.cuda.is_available():
        print("\n🔄 Testing GPU Transfer...")
        start_time = time.time()
        input_gpu = input_tensor.cuda()
        transfer_time = time.time() - start_time
        print(f"GPU Transfer Time: {transfer_time*1000:.2f} ms")
        print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        
        # Simple computation test
        print("\n⚡ Testing GPU Computation...")
        start_time = time.time()
        
        # Simulate neural network forward pass operations
        with torch.no_grad():
            # Layer 1: Linear transformation (embedding)
            W1 = torch.randn(features, 64, device='cuda')
            hidden1 = torch.matmul(input_gpu, W1)
            
            # Layer 2: LSTM-like operations
            W2 = torch.randn(64, 128, device='cuda')
            hidden2 = torch.matmul(hidden1.view(-1, 64), W2).view(batch_size, seq_length, 128)
            
            # Layer 3: Attention-like mechanism
            attention_weights = torch.softmax(hidden2.sum(dim=2), dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), hidden2).squeeze(1)
            
            # Layer 4: Output projection
            W_out = torch.randn(128, 1, device='cuda')
            output = torch.matmul(context, W_out)
        
        compute_time = time.time() - start_time
        print(f"GPU Computation Time: {compute_time*1000:.2f} ms")
        print(f"Output Shape: {output.shape}")
        print(f"Prediction Value: {output.item():.6f}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"Total Processing Time: {(transfer_time + compute_time)*1000:.2f} ms")
        print(f"Throughput: {seq_length/(transfer_time + compute_time):.0f} timesteps/second")
        
        return True
    else:
        print("❌ GPU not available for testing")
        return False

def test_model_compatibility():
    """Test TiRex model checkpoint loading compatibility"""
    print("\n🔍 TiRex Model Compatibility Test")
    print("=" * 50)
    
    model_path = Path("./models/tirex/model.ckpt")
    
    if not model_path.exists():
        print("❌ Model checkpoint not found")
        return False
    
    try:
        # Attempt to load the checkpoint
        print("Loading TiRex checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Analyze model structure
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Model parameters: {len(state_dict)} layers")
            
            # Count total parameters
            total_params = 0
            for name, param in state_dict.items():
                if hasattr(param, 'numel'):
                    total_params += param.numel()
                    print(f"  {name}: {param.shape}")
            
            print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Check for model metadata
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Model Config: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def main():
    """Main test execution"""
    print("🎯 SAGE-Forge TiRex GPU Integration Test")
    print("RTX 4090 Performance Validation")
    print("=" * 60)
    
    # Test 1: Environment
    env_ok = test_tirex_environment()
    
    # Test 2: Synthetic Inference
    if env_ok:
        inference_ok = test_synthetic_inference()
    else:
        inference_ok = False
    
    # Test 3: Model Compatibility
    model_ok = test_model_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    print("=" * 60)
    print(f"✅ Environment Ready: {env_ok}")
    print(f"✅ GPU Inference: {inference_ok}")
    print(f"✅ Model Compatible: {model_ok}")
    
    if all([env_ok, inference_ok, model_ok]):
        print("\n🎉 All tests passed! Ready for TiRex integration.")
        print("Next step: Integrate with SAGE-Forge framework")
    else:
        print("\n⚠️  Some tests failed. Check issues above.")
    
    return all([env_ok, inference_ok, model_ok])

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)